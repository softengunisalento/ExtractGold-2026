#!/usr/bin/env python3
"""Selezione, congelamento e download dei modelli GGUF: un unico script.

Lanciato senza argomenti fa, in sequenza:
  1. SELEZIONE dai metadati del Hugging Face Hub: nessun criterio dipende dal nome del modello (parametri,
     chat template, lingue, data, contesto, relazioni con altri modelli, download); l'organizzazione che pubblica
     è l'unica scelta umana (ORG_PREDEFINITE o --orgs-file). Un modello è candidato solo se esiste un GGUF a 4 bit
     da un quantizzatore fidato.
  2. CONGELAMENTO in selezioni/, con la data nel nome e senza mai sovrascrivere:
       candidati_<AAAA-MM-GG>.json   criteri, versioni, contatori, candidati con il piano di download
                                     (repository GGUF, commit, etichetta, nome) ed esclusi con il motivo
       candidati_<AAAA-MM-GG>.txt    lo stesso, da consultare
  3. DOWNLOAD dei file GGUF scelti in models/<nome>/, con commit fissato (dal manifest o dalla selezione),
     verifica SHA-256, controllo dello spazio, lettura dell'intestazione GGUF e models/manifest.json;
     l'esito resta in selezioni/esito_download_<AAAA-MM-GG>.txt.

Perché il congelamento: le stesse query possono dare un risultato diverso fra un mese. Con --da-lista si
scarica esattamente la lista congelata, senza rifare la selezione.

Uso:
  python download_models.py                       # selezione + documenti congelati + download
  python download_models.py --solo-selezione      # solo selezione e documenti congelati
  python download_models.py --dry-run             # selezione e documenti, poi il piano di download senza scaricare
  python download_models.py --da-lista selezioni/candidati_2026-10-13.json
  python download_models.py --da-lista selezioni/candidati_2026-10-13.json --only llama-3.3-70b-instruct
  python download_models.py --max-modelli 10 --max-coding 5 --quant Q4_K_M,UD-Q4_K_XL

Il Hub limita le richieste anonime: con la variabile d'ambiente HF_TOKEN (o HUGGING_FACE_HUB_TOKEN) i limiti
sono più larghi. Il token non si scrive mai nello script né nei file di output.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    sys.exit("huggingface_hub non installato: pip install -U huggingface_hub")

VERSIONE = "2.0"


# Organizzazioni che pubblicano i modelli "ufficiali". È una scelta del gruppo: da rivedere.
# Un'organizzazione che non esiste sul Hub, o senza modelli adatti, produce semplicemente zero risultati.
ORG_PREDEFINITE = [
    "meta-llama", "Qwen", "google", "openai", "deepseek-ai", "nvidia", "microsoft", "mistralai",
    "ibm-granite", "moonshotai", "CohereLabs", "allenai", "HuggingFaceTB", "zai-org", "MiniMaxAI",
    "tiiuae", "LGAI-EXAONE", "01-ai", "internlm", "tencent", "ByteDance-Seed",
]


# Quantizzatori di cui ci si fida per i file GGUF (si aggiunge sempre l'organizzazione stessa).
QUANTIZZATORI_FIDATI = ["bartowski", "unsloth", "lmstudio-community", "ggml-org"]


# Compiti del Hub considerati: i modelli multimodali (es. Gemma) non hanno "text-generation".
COMPITI = ["text-generation", "image-text-to-text"]


LINGUE_COMUNI = {"en", "zh", "fr", "de", "es", "it", "pt", "ru", "ja", "ko", "ar", "hi", "tr", "vi", "th",
                 "id", "nl", "pl", "uk", "cs", "sv"}


CHIAVI_CONTESTO = ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length",
                   "max_seq_len", "model_max_length")


RE_ETICHETTA = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?I?Q\d[A-Za-z0-9_]*|BF16|F16|MXFP4)(?![A-Za-z0-9])")


# Etichette di quantizzazione preferite, nell'ordine: la classe dei 4 bit. La prima disponibile vince.
QUANT_PREFERITE = ["Q4_K_M", "UD-Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS"]


TOKEN = None  # token Hugging Face letto dall'ambiente in main(); None = richieste anonime


def leggi_token():
    """Stessa convenzione di download_models.py: variabile d'ambiente HF_TOKEN (o HUGGING_FACE_HUB_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


# Motivi di esclusione (stringhe stabili: finiscono nei contatori e nel file congelato)
R_RELAZIONE = "quantizzazione, adapter o merge di un altro modello"


R_PRESELEZIONE = "download propri sotto la soglia di pre-selezione"


R_PARAM_ASSENTI = "parametri non presenti nei metadati"


R_PARAM_FUORI = "parametri fuori dall'intervallo"


R_TEMPLATE = "nessun chat template"


R_LINGUA = "inglese assente tra le lingue dichiarate"


R_DATA = "creato prima del limite"


R_CONTESTO = "contesto dichiarato sotto il minimo"


R_DOWNLOADS = "download totali sotto la soglia"


R_FAMIGLIA = "oltre il massimo per organizzazione"


R_NO_GGUF = "nessun GGUF a 4 bit da quantizzatori fidati"


# ---------------------------------------------------------------------------
# Modello da scaricare e costanti del download
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    name: str           # nome breve: cartella models/<name>/ e alias di llama-server
    repo: str           # repository GGUF su Hugging Face (quantizzatore fidato)
    quant: str | None   # etichetta di quantizzazione scelta
    note: str = ""      # cosa significa davvero l'etichetta (finisce nel manifest come quant_note)
    commit: str | None = None      # commit del repository GGUF fissato dalla selezione congelata
    base_model: str | None = None  # modello originale da cui deriva il GGUF (finisce nel manifest)


SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


MARGIN = 2 * 1024**3  # margine di sicurezza sullo spazio libero (non si riempie mai del tutto il disco)


class DownloadError(Exception):
    """Errore atteso: quantizzazione non trovata, ambigua o spazio insufficiente."""


# ---------------------------------------------------------------------------
# Compatibilità con le versioni di huggingface_hub
# ---------------------------------------------------------------------------


_DIREZIONE_SUPPORTATA = None  # None = non ancora provato


def elenca_modelli(api, **kw):
    """api.list_models con compatibilità: le versioni recenti di huggingface_hub non hanno più il
    parametro 'direction' (l'ordine per 'sort' è già decrescente). Se manca, si riprova senza."""
    global _DIREZIONE_SUPPORTATA
    if _DIREZIONE_SUPPORTATA is False:
        kw.pop("direction", None)
    try:
        return api.list_models(**kw)
    except TypeError as exc:
        if "direction" in kw and "direction" in str(exc):
            _DIREZIONE_SUPPORTATA = False
            print("  Nota: questa versione di huggingface_hub non ha 'direction': si usa l'ordine predefinito del Hub.")
            kw.pop("direction")
            return api.list_models(**kw)
        raise


def verifica_compatibilita(api) -> None:
    """Si ferma subito, con un messaggio chiaro, se list_models non accetta i parametri indispensabili."""
    try:
        parametri = set(inspect.signature(api.list_models).parameters)
    except (TypeError, ValueError):
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(api.list_models).parameters.values()):
        return
    mancanti = [p for p in ("author", "pipeline_tag", "sort", "limit", "filter") if p not in parametri]
    if mancanti:
        sys.exit(f"huggingface_hub {huggingface_hub.__version__}: list_models non accetta i parametri {mancanti}. "
                 "Servono questi parametri: aggiorna o cambia versione della libreria (pip install -U huggingface_hub).")


# ---------------------------------------------------------------------------
# Lettura dei metadati del Hub
# ---------------------------------------------------------------------------


def get(obj, nome, default=None):
    """Legge un campo da un oggetto o da un dizionario, senza errori se manca."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(nome, default)
    valore = getattr(obj, nome, None)
    if valore is None and hasattr(obj, "get"):
        try:
            valore = obj.get(nome)
        except Exception:
            valore = None
    return default if valore is None else valore


def downloads_di(obj) -> int:
    return int(get(obj, "downloads", 0) or 0)


def parametri_b(info):
    """(miliardi di parametri, fonte) dai metadati; (None, None) se non ci sono."""
    for campo, nome in (("safetensors", "safetensors"), ("gguf", "gguf")):
        totale = get(get(info, campo), "total")
        if totale:
            return round(totale / 1e9, 2), nome
    return None, None


def lingue_di(info) -> list[str]:
    valore = get(get(info, "card_data"), "language")
    if isinstance(valore, str):
        valore = [valore]
    lingue = [str(x).lower() for x in (valore or [])]
    if not lingue:  # senza scheda: si guardano i tag che sono codici di lingua
        lingue = [t for t in (get(info, "tags") or []) if str(t).lower() in LINGUE_COMUNI]
    return lingue


def relazioni(tags) -> dict[str, list[str]]:
    """Tag del tipo base_model:<relazione>:<modello> -> {relazione: [modelli]}."""
    out: dict[str, list[str]] = defaultdict(list)
    for t in tags or []:
        m = re.match(r"base_model:(finetune|quantized|adapter|merge):(.+)$", str(t), re.IGNORECASE)
        if m:
            out[m.group(1).lower()].append(m.group(2))
    return out


def scarica_json(repo_id: str, nome_file: str):
    try:
        percorso = hf_hub_download(repo_id=repo_id, filename=nome_file, token=TOKEN)
        return json.loads(Path(percorso).read_text(encoding="utf-8"))
    except Exception:
        return None


def chat_template(api, info, repo_id):
    """Il segnale che dimostra un chat template nel repository, o None."""
    config = get(info, "config") or {}
    tc = config.get("tokenizer_config") if isinstance(config, dict) else None
    if isinstance(tc, dict) and tc.get("chat_template"):
        return "config.tokenizer_config"
    if get(get(info, "gguf"), "chat_template"):
        return "gguf"
    tc = scarica_json(repo_id, "tokenizer_config.json")
    if isinstance(tc, dict) and tc.get("chat_template"):
        return "tokenizer_config.json"
    for nome in ("chat_template.jinja", "chat_template.json"):
        try:
            if api.file_exists(repo_id, nome):
                return nome
        except Exception:
            pass
    return None


def contesto_dichiarato(repo_id: str):
    config = scarica_json(repo_id, "config.json")
    if not isinstance(config, dict):
        return None
    for blocco in (config, config.get("text_config"), config.get("llm_config"), config.get("language_config")):
        if isinstance(blocco, dict):
            for chiave in CHIAVI_CONTESTO:
                valore = blocco.get(chiave)
                if isinstance(valore, int) and valore > 0:
                    return valore
    return None


def modelli_derivati(api, model_id: str, relazione: str, tag_extra=None, limite=200):
    """Repository che dichiarano <relazione> di model_id (es. le sue quantizzazioni)."""
    voluto = f"base_model:{relazione}:{model_id}".lower()
    risultati = []
    for c in elenca_modelli(api, filter=[f"base_model:{relazione}:{model_id}"] + list(tag_extra or []),
                            sort="downloads", direction=-1, limit=limite):
        tags = [str(t).lower() for t in (get(c, "tags") or [])]
        if voluto in tags and all(x.lower() in tags for x in (tag_extra or [])):
            risultati.append(c)
    return risultati


def etichetta_quant(nome_file: str):
    m = RE_ETICHETTA.search(nome_file)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# File GGUF: etichette, risoluzione dei file, lettura dell'intestazione
# ---------------------------------------------------------------------------


def list_gguf(files: list[str]) -> list[str]:
    return sorted(f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower())


def resolve_files(repo_files: list[str], quant: str) -> list[str]:
    """Restituisce i file .gguf del repo che corrispondono alla quantizzazione.

    - confronto senza distinzione tra maiuscole e minuscole, con confini netti
      (Q4_K_M non corrisponde a Q4_K_L né a IQ4_K_M);
    - ignora i file mmproj (proiettore visivo dei modelli multimodali);
    - se la quantizzazione non inizia con "UD-", ignora le varianti Unsloth Dynamic;
    - accetta un solo file oppure una serie di shard -0000N-of-0000M.
    """
    token_re = re.compile(r"(?<![A-Za-z0-9])" + re.escape(quant) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    wants_ud = quant.upper().startswith("UD-")
    hits = [
        f for f in list_gguf(repo_files)
        if (wants_ud or "ud-" not in f.lower()) and token_re.search(f)
    ]

    if not hits:
        available = list_gguf(repo_files) or ["(nessuno)"]
        raise DownloadError(f"nessun file per '{quant}'. File .gguf nel repo:\n    " + "\n    ".join(available))
    if len(hits) > 1 and not all(SHARD_RE.search(h) for h in hits):
        raise DownloadError(f"più file per '{quant}', specifica un'etichetta più precisa:\n    " + "\n    ".join(hits))
    return hits


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def free_space(path: Path) -> int:
    """Spazio libero sul disco di `path`, anche se la cartella non esiste ancora."""
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def lfs_sha256(sibling) -> str | None:
    """SHA-256 dichiarato dal Hub per un file LFS (i .gguf lo sono sempre)."""
    lfs = getattr(sibling, "lfs", None)
    if not lfs:
        return None
    return lfs["sha256"] if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def sha256_of(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_manifest(out_dir: Path) -> dict:
    """Legge <out_dir>/manifest.json (vuoto se non esiste). Il manifest fissa i commit dei pesi."""
    path = out_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_manifest(out_dir: Path, record: dict) -> None:
    """Aggiunge o aggiorna la voce del modello in <out_dir>/manifest.json."""
    path = out_dir / "manifest.json"
    data = load_manifest(out_dir)
    data[record["name"]] = record
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Tipi di tensore (ggml) e tipi di file (llama.cpp) più comuni. Un id non presente qui viene registrato
# come type_<id>: in caso di dubbio si controlla con gguf_dump della versione di llama.cpp in uso.
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
              23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
              34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}


FILE_TYPES = {0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1", 7: "MOSTLY_Q8_0",
              8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M",
              13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S",
              17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS",
              21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS", 24: "MOSTLY_IQ1_S",
              25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S",
              29: "MOSTLY_IQ2_M", 30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16"}


_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _gguf_read(f, fmt: str) -> tuple:
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("file troncato")
    return struct.unpack("<" + fmt, data)


def _gguf_string(f) -> str:
    (n,) = _gguf_read(f, "Q")
    if n > 1 << 26:
        raise ValueError("stringa troppo lunga: file non valido")
    data = f.read(n)
    if len(data) != n:
        raise ValueError("file troncato")
    return data.decode("utf-8", "replace")


def _gguf_value(f, vtype: int):
    if vtype in _GGUF_SCALARS:
        return _gguf_read(f, _GGUF_SCALARS[vtype])[0]
    if vtype == 8:
        return _gguf_string(f)
    if vtype == 9:  # array: di scalari si salta; di stringhe si legge e si scarta
        (etype,) = _gguf_read(f, "I")
        (count,) = _gguf_read(f, "Q")
        if etype in _GGUF_SCALARS:
            f.seek(count * struct.calcsize("<" + _GGUF_SCALARS[etype]), 1)
        else:
            for _ in range(count):
                _gguf_value(f, etype)
        return None
    raise ValueError(f"tipo di metadato GGUF sconosciuto: {vtype}")


def inspect_gguf(path: Path, requested_quant: str | None = None) -> dict:
    """Legge solo l'intestazione di un file GGUF e riassume la quantizzazione reale dei tensori."""
    wanted = ("general.architecture", "general.file_type", "general.alignment")
    meta: dict = {}
    tensors: list[tuple[int, int, int]] = []  # (elementi, tipo ggml, offset)
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("non è un file GGUF")
        (version,) = _gguf_read(f, "I")
        if version not in (2, 3):
            raise ValueError(f"versione GGUF non gestita: {version}")
        tensor_count, kv_count = _gguf_read(f, "QQ")
        for _ in range(kv_count):
            key = _gguf_string(f)
            (vtype,) = _gguf_read(f, "I")
            value = _gguf_value(f, vtype)
            if key in wanted:
                meta[key] = value
        for _ in range(tensor_count):
            _gguf_string(f)
            (ndims,) = _gguf_read(f, "I")
            dims = _gguf_read(f, "Q" * ndims) if ndims else ()
            ttype, offset = _gguf_read(f, "IQ")
            elements = 1
            for d in dims:
                elements *= d
            tensors.append((elements, ttype, offset))
        alignment = int(meta.get("general.alignment") or 32)
        data_start = (f.tell() + alignment - 1) // alignment * alignment
    data_size = max(0, os.path.getsize(path) - data_start)
    offsets = sorted({t[2] for t in tensors})
    next_offset = {o: (offsets[i + 1] if i + 1 < len(offsets) else data_size) for i, o in enumerate(offsets)}
    by_type: dict = {}
    total_bytes = total_elements = 0
    for elements, ttype, offset in tensors:
        nbytes = next_offset[offset] - offset
        name = GGML_TYPES.get(ttype, f"type_{ttype}")
        d = by_type.setdefault(name, {"tensors": 0, "bytes": 0})
        d["tensors"] += 1
        d["bytes"] += nbytes
        total_bytes += nbytes
        total_elements += elements
    ftype = meta.get("general.file_type")
    ftype_name = FILE_TYPES.get(ftype) if ftype is not None else None
    result = {
        "architecture": meta.get("general.architecture"),
        "file_type": ftype,
        "file_type_name": ftype_name,
        "tensors": len(tensors),
        "bits_per_weight": round(8 * total_bytes / total_elements, 2) if total_elements else None,
        "tensor_types": {k: {"tensors": v["tensors"], "share_bytes_pct": round(100 * v["bytes"] / total_bytes, 1)}
                         for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["bytes"])} if total_bytes else {},
        "label_matches_file_type": None,
    }
    label = (requested_quant or "").upper()
    if label and not label.startswith("UD-") and ftype_name:  # le etichette UD non hanno un tipo di file proprio
        result["label_matches_file_type"] = ftype_name == "MOSTLY_" + label
    return result


def gguf_summary_line(g: dict) -> str:
    if "error" in g:
        return f"  tipo reale non leggibile ({g['error']})"
    types = ", ".join(f"{k} {v['share_bytes_pct']}%" for k, v in list(g["tensor_types"].items())[:4])
    return (f"  tipo di file: {g['file_type_name'] or g['file_type']} | tensori (quota in byte): {types} | "
            f"{g['bits_per_weight']} bit per peso")


# ---------------------------------------------------------------------------
# Dal modello al GGUF da scaricare, e liste congelate
# ---------------------------------------------------------------------------


def gguf_dalle_quantizzazioni(api, quantizzazioni, org, quantizzatori):
    """Repository GGUF, tra le quantizzazioni di un modello, pubblicati da quantizzatori fidati o dall'organizzazione."""
    ammessi = {q.lower() for q in quantizzatori} | {org.lower()}
    out = []
    for c in quantizzazioni:
        tags = {str(t).lower() for t in (get(c, "tags") or [])}
        autore = c.id.split("/")[0]
        if "gguf" not in tags or autore.lower() not in ammessi:
            continue
        try:
            file = list_gguf(list(api.list_repo_files(c.id)))
        except Exception:
            file = []
        out.append({"repo": c.id, "autore": autore, "downloads_30d": downloads_di(c), "file": sorted(file),
                    "etichette": sorted({e for e in map(etichetta_quant, file) if e})})
    return out


def scegli_gguf(gguf, preferenze):
    """Per ogni etichetta preferita, nell'ordine: il repository con più download che ha quel file (o quella serie
    di shard). Un'etichetta che non corrisponde a un file univoco (resolve_files) non vale."""
    for etichetta in preferenze:
        possibili = []
        for g in gguf:
            try:
                file = resolve_files(g["file"], etichetta)
            except DownloadError:
                continue
            possibili.append((g, file))
        if possibili:
            g, file = max(possibili, key=lambda x: (x[0]["downloads_30d"], x[0]["repo"]))
            return {"repo": g["repo"], "autore": g["autore"], "etichetta": etichetta, "file": file,
                    "downloads_30d": g["downloads_30d"]}
    return None


def commit_di(api, repo):
    """Commit attuale di un repository (si congela nella selezione)."""
    try:
        return get(api.model_info(repo), "sha")
    except Exception:
        return None


def nome_breve(model_id: str, usati: set) -> str:
    """Nome per la cartella models/<nome>/ e per --alias: il nome del repository, in minuscolo e senza simboli.
    Se è già usato da un altro modello si aggiunge l'organizzazione."""
    pulisci_nome = lambda t: re.sub(r"[^a-z0-9.]+", "-", t.lower()).strip("-")
    base = pulisci_nome(model_id.split("/")[-1])
    if base in usati:
        base = pulisci_nome(model_id)
    nome, k = base, 2
    while nome in usati:
        nome, k = f"{base}-{k}", k + 1
    usati.add(nome)
    return nome


def nota_quant(etichetta: str, autore: str) -> str:
    """Cosa significa l'etichetta, per il manifest (il tipo reale dei tensori lo registra inspect_gguf)."""
    if etichetta.upper().startswith("UD-"):
        return (f"Unsloth Dynamic (repository {autore}): precisione mista per strato, "
                f"non equivale a un {etichetta[3:]} uniforme")
    if etichetta.upper() == "MXFP4":
        return f"MXFP4 (repository {autore}): formato a 4 bit a blocchi, non un K-quant di llama.cpp"
    return f"etichetta {etichetta} richiesta al repository {autore}: il tipo reale dei tensori è nel manifest (files[].gguf)"


def modelli_da_candidati(candidati: list[dict]) -> list[Model]:
    """Dal piano di download di ogni candidato ('scarica') ai Model che scarica download()."""
    modelli = []
    for c in candidati:
        s = c.get("scarica") or {}
        if not (s.get("repo") and s.get("etichetta") and s.get("nome")):
            sys.exit(f"lista non valida: il candidato {c.get('id')} non ha un piano di download completo ('scarica')")
        modelli.append(Model(name=s["nome"], repo=s["repo"], quant=s["etichetta"], note=s.get("nota") or "",
                             commit=s.get("commit"), base_model=c.get("id")))
    return modelli


def leggi_lista_congelata(percorso: str):
    """Legge una selezione già congelata (candidati_<data>.json): si scarica senza rifare la selezione."""
    try:
        dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.exit(f"lista congelata non leggibile ({percorso}): {exc}")
    if not isinstance(dati, dict) or not dati.get("candidati"):
        sys.exit(f"lista congelata non valida o vuota: {percorso}")
    return dati, modelli_da_candidati(dati["candidati"])


def filtra_modelli(modelli: list[Model], only: str | None) -> list[Model]:
    if not only:
        return list(modelli)
    noti = {m.name: m for m in modelli}
    voluti = [x.strip() for x in only.split(",") if x.strip()]
    ignoti = [w for w in voluti if w not in noti]
    if ignoti:
        sys.exit(f"modelli sconosciuti: {', '.join(ignoti)}. Disponibili: {', '.join(noti)}")
    return [noti[w] for w in voluti]


# ---------------------------------------------------------------------------
# Selezione
# ---------------------------------------------------------------------------


CODING_TAG = {"code", "coding"}


def dominio_di(tags) -> str:
    """Euristica NON vincolante, dai tag dichiarati dal pubblicatore: se non li imposta, il modello risulta generalista."""
    return "coding" if CODING_TAG & {str(t).lower() for t in (tags or [])} else "generalista"


def universo(api, orgs, compiti, limite_per_org, soglia_pre, conta):
    """Tutti i modelli delle organizzazioni scelte, per compito, con download propri sopra la soglia di pre-selezione."""
    visti: dict[str, object] = {}
    for org in orgs:
        for compito in compiti:
            try:
                elenco = list(elenca_modelli(api, author=org, pipeline_tag=compito, sort="downloads", direction=-1,
                                             limit=limite_per_org))
                scaricamenti = [downloads_di(x) for x in elenco]
                if len(elenco) >= limite_per_org and scaricamenti and scaricamenti[0] < scaricamenti[-1]:
                    raise RuntimeError("l'elenco è in ordine crescente di download ed è troncato dal limite: "
                                       "il risultato sarebbe incompleto")
                for m in elenco:
                    if m.id in visti:
                        continue
                    if downloads_di(m) < soglia_pre:
                        continue  # (non si interrompe: non si dà per scontato l'ordinamento restituito dal Hub)
                    visti[m.id] = m
            except Exception as exc:
                print(f"  ATTENZIONE: elenco di {org} ({compito}) non riuscito: {type(exc).__name__}: {exc}")
                conta["elenchi_falliti"] += 1
    return list(visti.values())


def analizza(api, m, args, conta, esclusi):
    """Applica i criteri nell'ordine; restituisce il record del candidato o None (con il motivo registrato)."""
    def escludi(motivo, **extra):
        conta[motivo] += 1
        esclusi.append({"id": m.id, "motivo": motivo, "dominio": dominio_di(tags), **extra})
        return None

    def e_derivato(tags):
        rel = relazioni(tags)
        return bool(rel["quantized"] or rel["adapter"] or rel["merge"] or "gguf" in {str(t).lower() for t in tags})

    tags = [str(t) for t in (get(m, "tags") or [])]
    if e_derivato(tags):
        return escludi(R_RELAZIONE)

    info = api.model_info(m.id)
    tags = [str(t) for t in (get(info, "tags") or tags)]
    if e_derivato(tags):  # i tag dell'elenco possono essere incompleti: si ricontrolla con quelli del repository
        return escludi(R_RELAZIONE)
    rel = relazioni(tags)
    org = m.id.split("/")[0]

    miliardi, fonte_param = parametri_b(info)
    if miliardi is None:
        return escludi(R_PARAM_ASSENTI)
    if not (args.min_b <= miliardi <= args.max_b):
        return escludi(R_PARAM_FUORI, parametri_b=miliardi)

    segnale_template = chat_template(api, info, m.id)
    if not segnale_template:
        return escludi(R_TEMPLATE)

    lingue = lingue_di(info)
    if lingue and "en" not in lingue:
        return escludi(R_LINGUA, lingue=lingue)
    if not lingue:
        conta["non valutabile: lingue"] += 1

    creato = get(info, "created_at")
    if args.creati_dopo:
        if creato is None:
            conta["non valutabile: data di creazione"] += 1
        elif creato.date().isoformat() < args.creati_dopo:
            return escludi(R_DATA, creato=creato.date().isoformat())

    contesto = contesto_dichiarato(m.id)
    if contesto is None:
        conta["non valutabile: contesto"] += 1
    elif contesto < args.min_contesto:
        return escludi(R_CONTESTO, contesto=contesto)

    quantizzazioni = modelli_derivati(api, m.id, "quantized")
    dl_quant = sum(downloads_di(c) for c in quantizzazioni)
    dl_propri = downloads_di(info) or downloads_di(m)
    if dl_propri + dl_quant < args.min_downloads:
        return escludi(R_DOWNLOADS, downloads_totali=dl_propri + dl_quant)

    # Scaricabile: un GGUF con una delle etichette preferite, da un quantizzatore fidato o dall'organizzazione
    gguf = gguf_dalle_quantizzazioni(api, quantizzazioni, org, QUANTIZZATORI_FIDATI)
    scelta = scegli_gguf(gguf, args.quant)
    if scelta is None:
        return escludi(R_NO_GGUF, gguf_trovati=len(gguf))
    scelta["commit"] = commit_di(api, scelta["repo"])

    return {
        "id": m.id,
        "organizzazione": org,
        "compito": get(info, "pipeline_tag") or get(m, "pipeline_tag"),
        "multimodale": (get(info, "pipeline_tag") or get(m, "pipeline_tag")) == "image-text-to-text",
        "dominio": dominio_di(tags),
        "commit": get(info, "sha"),
        "creato": creato.date().isoformat() if creato else None,
        "ultima_modifica": get(info, "last_modified").date().isoformat() if get(info, "last_modified") else None,
        "parametri_miliardi": miliardi,
        "fonte_parametri": fonte_param,
        "contesto": contesto,
        "chat_template": segnale_template,
        "lingue": lingue,
        "derivato_da": rel["finetune"],
        "downloads_30d": dl_propri,
        "downloads_quantizzazioni_30d": dl_quant,
        "downloads_totali": dl_propri + dl_quant,
        "gguf": gguf,
        "scarica": scelta,
    }


def applica_tetto(candidati, massimo, ordine, conta, esclusi, priorita_coding=False):
    """EX-3: al massimo `massimo` modelli per organizzazione (con priorità ai modelli coding, se richiesto)."""
    per_org: dict[str, list] = defaultdict(list)
    for c in candidati:
        per_org[c["organizzazione"]].append(c)
    tenuti = []
    for org, lista in per_org.items():
        coding = (lambda c: c["dominio"] == "coding") if priorita_coding else (lambda c: False)
        if ordine == "recenti":
            lista.sort(key=lambda c: (coding(c), c["creato"] or "", c["downloads_totali"]), reverse=True)
        else:
            lista.sort(key=lambda c: (coding(c), c["downloads_totali"], c["creato"] or ""), reverse=True)
        tenuti += lista[:massimo]
        for c in lista[massimo:]:
            conta[R_FAMIGLIA] += 1
            esclusi.append({"id": c["id"], "motivo": R_FAMIGLIA, "dominio": c["dominio"], "organizzazione": org})
    return tenuti


def seleziona_finale(candidati, max_modelli, max_coding, conta, esclusi):
    """Lista finale. Con max_coding > 0: fino a max_coding modelli coding entrano per primi (i più scaricati),
    poi i non-coding più scaricati, fino a max_modelli in tutto. Con max_coding == 0 decidono solo i download."""
    per_download = lambda x: (-x["downloads_totali"], x["id"])

    def taglia(lista, motivo):
        for c in lista:
            conta[motivo] += 1
            esclusi.append({"id": c["id"], "motivo": motivo, "dominio": c["dominio"],
                            "downloads_totali": c["downloads_totali"]})

    if max_coding:
        coding = sorted((c for c in candidati if c["dominio"] == "coding"), key=per_download)
        altri = sorted((c for c in candidati if c["dominio"] != "coding"), key=per_download)
        taglia(coding[max_coding:], f"oltre il massimo di {max_coding} modelli coding")
        ordinati = coding[:max_coding] + altri
        motivo_limite = f"fuori dai {max_modelli} posti disponibili (priorità al coding, poi download totali)"
    else:
        ordinati = sorted(candidati, key=per_download)
        motivo_limite = f"fuori dai primi {max_modelli} per download totali"
    if max_modelli and len(ordinati) > max_modelli:
        taglia(ordinati[max_modelli:], motivo_limite)
        ordinati = ordinati[:max_modelli]
    return ordinati


def diagnostica_coding(candidati, esclusi) -> dict:
    """Quanti modelli esaminati risultano coding (dai tag) e che fine hanno fatto."""
    coding_esclusi = [e for e in esclusi if e.get("dominio") == "coding"]
    finali = [c for c in candidati if c["dominio"] == "coding"]
    return {
        "tag_usati": sorted(CODING_TAG),
        "esaminati_con_tag_coding": len(coding_esclusi) + len(finali),
        "nella_lista_finale": len(finali),
        "esclusi_per_motivo": dict(Counter(e["motivo"] for e in coding_esclusi)),
    }


# ---------------------------------------------------------------------------
# Documenti congelati
# ---------------------------------------------------------------------------


def percorso_libero(cartella: Path, prefisso: str, data: str) -> Path:
    """cartella/<prefisso>_<data> (con _2, _3... se esiste già): un file congelato non si sovrascrive."""
    base = cartella / f"{prefisso}_{data}"
    k = 1
    while True:
        candidato = base if k == 1 else Path(f"{base}_{k}")
        if not any(Path(f"{candidato}{ext}").exists() for ext in (".json", ".txt")):
            return candidato
        k += 1


def riga_candidato(c) -> str:
    quant = c["downloads_quantizzazioni_30d"]
    s = c.get("scarica") or {}
    piano = (f"scarica {s.get('repo')}@{str(s.get('commit'))[:10]} [{s.get('etichetta')}] in models/{s.get('nome')}/"
             if s else "nessun GGUF scelto")
    return (f"- {c['id']} | {c['parametri_miliardi']:g}B | contesto {c['contesto'] or 'n/d'} | creato {c['creato'] or 'n/d'} | "
            f"DL {c['downloads_30d']:,} (+{quant:,} quantizzazioni) | lingue {','.join(c['lingue']) or 'n/d'} | "
            f"commit {str(c['commit'])[:10]} | {piano} ({len(c.get('gguf') or [])} repository GGUF fidati)")


def scrivi_report(percorso: Path, dati: dict) -> None:
    c, e, cr = dati["candidati"], dati["esclusi"], dati["criteri"]
    righe = [
        "=== SELEZIONE MODELLI (da metadati del Hub) ===",
        f"Data di esecuzione (UTC): {dati['esecuzione']['data_ora_utc']}",
        f"Script: download_models.py versione {VERSIONE} (sha256 {dati['esecuzione']['sha256_script']})",
        f"huggingface_hub {dati['esecuzione']['huggingface_hub']} | Python {dati['esecuzione']['python']}",
        "I download sono quelli restituiti dal Hub (ultimi 30 giorni).",
        "",
        "Criteri applicati (nessuno usa il nome del modello):",
        f"- Universo: modelli delle organizzazioni elencate, compiti {', '.join(cr['compiti'])}, "
        f"con download propri >= {cr['soglia_preselezione']:,} (pre-selezione = 10% della soglia)",
        f"- Relazioni: esclusi i repository che dichiarano di essere quantizzazione, adapter o merge di un altro modello",
        f"- Parametri (IN-5): da safetensors/gguf, tra {cr['min_b']:g}B e {cr['max_b']:g}B; assenti nei metadati = escluso",
        "- Instruct/chat (IN-3): chat template presente (tokenizer_config, GGUF o file dedicato)",
        "- Lingua (EX-4): se le lingue sono dichiarate, 'en' deve esserci; se non lo sono, il criterio non è valutabile",
        f"- Data (EX-1): {('creato dal ' + cr['creati_dopo']) if cr['creati_dopo'] else 'nessun limite (la data è nel report)'}",
        f"- Contesto (IN-8): dichiarato >= {cr['min_contesto']:,} (config.json); se non dichiarato, non valutabile",
        f"- Adozione (IN-6): download propri + quelli delle quantizzazioni >= {cr['min_downloads']:,}",
        f"- Famiglia (EX-3): al massimo {cr['max_per_org']} per organizzazione, scelti per ordine '{cr['ordine_famiglia']}'",
        (f"- Priorità al coding: fino a {cr['max_coding']} modelli coding entrano per primi (anche nel tetto per organizzazione)"
         if cr["max_coding"] else "- Priorità al coding: nessuna (decidono solo i download)"),
        (f"- Limite finale: {cr['max_modelli']} modelli in tutto, poi per download totali (propri + quantizzazioni)"
         if cr["max_modelli"] else "- Limite finale: nessuno"),
        f"- Scaricabile: un GGUF con una di queste etichette, nell'ordine: {', '.join(cr['quant_preferite'])}; pubblicato da "
        f"un quantizzatore fidato ({', '.join(cr['quantizzatori_fidati'])}) o dall'organizzazione; tra più repository, quello con più download",
        "- Dominio: coding se i tag dichiarano 'code' o 'coding' (euristica non vincolante)",
        f"Organizzazioni ({len(cr['organizzazioni'])}): {', '.join(cr['organizzazioni'])}",
        "",
        "Conteggi:",
        f"  modelli esaminati: {dati['conteggi']['esaminati']} | candidati: {len(c)} | esclusi: {len(e)}",
    ]
    for motivo, n in sorted(dati["conteggi"]["per_motivo"].items(), key=lambda kv: -kv[1]):
        righe.append(f"  - {motivo}: {n}")
    for motivo, n in sorted(dati["conteggi"]["non_valutabili"].items()):
        righe.append(f"  - {motivo}: {n} (accettati senza il dato)")
    dg = dati["diagnostica_coding"]
    righe += ["", f"Diagnostica coding (dominio = tag {', '.join(repr(t) for t in dg['tag_usati'])} dichiarati dal pubblicatore):",
              f"  modelli esaminati con tag coding: {dg['esaminati_con_tag_coding']} | nella lista finale: {dg['nella_lista_finale']}"]
    for motivo, n in sorted(dg["esclusi_per_motivo"].items(), key=lambda kv: -kv[1]):
        righe.append(f"  - esclusi per «{motivo}»: {n}")
    if not dg["esaminati_con_tag_coding"]:
        righe.append("  Nessun modello esaminato ha un tag coding: se il pubblicatore non lo imposta, un modello coding risulta generalista.")
    for dominio, titolo in (("coding", "SPECIALIZZATI (CODING)"), ("generalista", "GENERALISTI")):
        lista = [x for x in c if x["dominio"] == dominio]
        righe += ["", f"--- CANDIDATI {titolo} ({len(lista)}) ---"] + [riga_candidato(x) for x in lista]
    righe += ["", "--- ESCLUSI, PER MOTIVO ---"]
    per_motivo: dict[str, list[str]] = defaultdict(list)
    for x in e:
        per_motivo[x["motivo"]].append(x["id"] + (" [coding]" if x.get("dominio") == "coding" else ""))
    for motivo, ids in sorted(per_motivo.items()):
        righe.append(f"[{motivo}] ({len(ids)})")
        righe += [f"  {i}" for i in sorted(ids)]
    percorso.write_text("\n".join(righe) + "\n", encoding="utf-8")


def scrivi_esito_download(percorso: Path, origine: str, out_dir: Path, risultati, record) -> None:
    """Documento leggibile con l'esito del download (si scrive dopo, accanto ai file congelati della selezione)."""
    righe = ["=== ESITO DEL DOWNLOAD ===",
             f"Data di esecuzione (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"Lista di origine: {origine}",
             f"Cartella dei modelli: {out_dir} | manifest: {out_dir / 'manifest.json'}", ""]
    for nome, stato, dettaglio in risultati:
        righe.append(f"[{stato}] {nome}")
        r = record.get(nome)
        if r:
            righe.append(f"    repository {r['repo']} @ {str(r['commit'])[:10]} | etichetta {r['quant']} | modello di origine {r.get('base_model')}")
            for f in r["files"]:
                righe.append(f"    {f['file']} ({human(f['size'])}) | {gguf_summary_line(f['gguf']).strip()}")
        elif dettaglio:
            righe.append(f"    {dettaglio}")
    percorso.write_text("\n".join(righe) + "\n", encoding="utf-8")


def leggi_orgs(percorso: str) -> list[str]:
    righe = Path(percorso).read_text(encoding="utf-8").splitlines()
    return [r.split("#")[0].strip() for r in righe if r.split("#")[0].strip()]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def local_state(target: Path, files: list[str], sizes: dict) -> tuple[int, int]:
    """(byte già presenti, byte ancora da scaricare) per i file di un modello.

    Un file è completo se esiste con la dimensione attesa. Un download interrotto lascia un file
    *.incomplete nella cache di huggingface_hub dentro la cartella del modello: se c'è, i suoi byte
    contano come già presenti (stima: la posizione dipende dalla versione della libreria).
    """
    present = missing = 0
    cache = target / ".cache" / "huggingface" / "download"
    for f in files:
        size = sizes[f]
        local = target / f
        if local.exists() and local.stat().st_size == size:
            present += size
            continue
        partial = 0
        if cache.exists():
            partial = max((p.stat().st_size for p in cache.rglob(Path(f).name + "*.incomplete")), default=0)
        partial = min(partial, size)
        present += partial
        missing += size - partial
    return present, missing


def choose_info(model: Model, api: HfApi, previous: dict | None, update: bool) -> tuple[object, str, list[str]]:
    """Sceglie il commit da scaricare. Restituisce (info del repository, origine, messaggi).

    Il commit è fissato, in quest'ordine: dal manifest (se ha già una voce per il modello), altrimenti dalla
    selezione congelata (model.commit). Senza nessuno dei due si scarica l'ultimo (vedi download()).
    """
    notes: list[str] = []
    info = api.model_info(model.repo, files_metadata=True)  # ultimo commit del repository
    pinned = None
    if previous and previous.get("commit"):
        if previous.get("repo") == model.repo:
            pinned = previous["commit"]
        else:
            notes.append(f"ATTENZIONE: nel manifest il repository era {previous.get('repo')}: il commit fissato non si applica")
    di_fonte, da_fonte = "del manifest", "dal manifest"
    if pinned is None and model.commit:
        pinned, di_fonte, da_fonte = model.commit, "della selezione congelata", "dalla selezione congelata"
    origin = "primo download" if not previous else "ultimo commit"
    if pinned and pinned != info.sha:
        if update:
            origin = "AGGIORNAMENTO"
            notes.append(f"AGGIORNAMENTO: commit {pinned[:10]} -> {info.sha[:10]}. I pesi cambiano: "
                         "le misure già fatte con il commit precedente non sono confrontabili.")
        else:
            notes.append(f"Il repository ha un commit più recente ({info.sha[:10]}): si usa quello {di_fonte} "
                         f"({pinned[:10]}). Usa --update per aggiornare.")
            try:
                info = api.model_info(model.repo, revision=pinned, files_metadata=True)
            except Exception as exc:
                raise DownloadError(f"il commit {di_fonte} ({pinned[:10]}) non è disponibile nel repository "
                                    f"({type(exc).__name__}): i pesi non sono riproducibili; "
                                    "con --update si scarica l'ultimo commit")
            origin = f"fissato {da_fonte}"
    elif pinned:
        origin = f"fissato {da_fonte} (coincide con l'ultimo)"
    return info, origin, notes


def plan_space(models: list[Model], out_dir: Path, api: HfApi, manifest: dict, update: bool) -> list[dict]:
    """Per ogni modello scaricabile: quanti byte sono già presenti e quanti mancano. Nessun download."""
    rows = []
    for m in models:
        if not m.quant:
            continue
        row = {"name": m.name, "present": 0, "missing": 0, "error": None}
        try:
            info, _, _ = choose_info(m, api, manifest.get(m.name), update)
            sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
            files = resolve_files(list(sizes), m.quant)
            row["present"], row["missing"] = local_state(out_dir / m.name, files, sizes)
        except Exception as exc:  # l'errore vero lo riporterà download()
            row["error"] = f"{type(exc).__name__}"
        rows.append(row)
    return rows


def print_space_summary(rows: list[dict], free: int) -> int:
    """Stampa il riepilogo dello spazio. Restituisce i byte che mancano (0 se basta)."""
    print("=== Spazio ===")
    for r in rows:
        extra = f"  (non calcolabile: {r['error']})" if r["error"] else ""
        print(f"  {r['name']:32} da scaricare {human(r['missing']):>10}   già presenti {human(r['present']):>10}{extra}")
    need = sum(r["missing"] for r in rows)
    deficit = max(0, need + MARGIN - free) if need else 0
    print(f"  Totale da scaricare: {human(need)} | spazio libero: {human(free)} | margine: {human(MARGIN)}")
    if deficit:
        print(f"  ATTENZIONE: spazio insufficiente, mancano {human(deficit)}. Libera spazio, oppure scarica "
              "pochi modelli alla volta con --only (--force salta questo controllo).")
    print()
    return deficit


def download(model: Model, out_dir: Path, api: HfApi, token: str | None,
             dry_run: bool, verify: bool = True, previous: dict | None = None,
             update: bool = False) -> tuple[str, dict | None]:
    """Scarica un modello. Restituisce (percorso per `llama-server -m`, voce del manifest).

    Il commit è fissato dal manifest (se ha già una voce per questo modello) o, in mancanza, dalla selezione
    congelata: un rilancio scarica quel commit, anche se il repository è cambiato nel frattempo. Con update=True si passa all'ultimo
    commit, conservando la voce precedente in "history".
    """
    info, origin, notes = choose_info(model, api, previous, update)
    for n in notes:
        print("  " + n)
    print(f"  commit {info.sha[:10]} ({origin})")
    siblings = {s.rfilename: s for s in info.siblings}
    sizes = {f: (s.size or 0) for f, s in siblings.items()}
    files = resolve_files(list(sizes), model.quant)
    total = sum(sizes[f] for f in files)

    target = out_dir / model.name
    free = free_space(target)
    present, missing = local_state(target, files, sizes)
    print(f"  {len(files)} file, {human(total)}: da scaricare {human(missing)}, già presenti {human(present)} "
          f"(spazio libero: {human(free)})")
    for f in files:
        print(f"    - {f}  ({human(sizes[f])})")
    if missing and missing + MARGIN > free:  # se non manca nulla non serve spazio
        raise DownloadError(f"spazio insufficiente: servono {human(missing)} più un margine di {human(MARGIN)}, "
                            f"liberi {human(free)}")
    if dry_run:
        return "dry-run", None

    target.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "name": model.name,
        "repo": model.repo,
        "quant": model.quant,
        "quant_note": model.note or None,
        "base_model": model.base_model,
        "commit_selezione": model.commit,
        "commit": info.sha,
        "downloaded_at": now,
        "files": [],
    }
    history = list((previous or {}).get("history", []))
    if previous and previous.get("repo") == model.repo and previous.get("commit") == info.sha:
        record["downloaded_at"] = previous.get("downloaded_at", now)  # stesso commit: si conserva la data originale
        record["last_checked_at"] = now
    elif previous:
        history.append({**{k: previous.get(k) for k in ("repo", "quant", "commit", "downloaded_at", "files")},
                        "replaced_at": now})
    if history:
        record["history"] = history
    for f in files:
        # revision=info.sha: scarica esattamente il commit registrato nel manifest
        hf_hub_download(repo_id=model.repo, filename=f, revision=info.sha,
                        local_dir=str(target), token=token)
        local = target / f
        expected = lfs_sha256(siblings[f])
        entry = {"file": f, "size": local.stat().st_size, "sha256": expected, "verified_locally": False}
        if verify:
            print(f"  verifica SHA-256: {f}")
            actual = sha256_of(local)
            if expected and actual != expected:
                local.unlink()  # così il prossimo avvio lo riscarica
                raise DownloadError(
                    f"SHA-256 diverso per {f} (atteso {expected[:12]}..., ottenuto {actual[:12]}...): "
                    "file rimosso, rilancia lo script"
                )
            entry["sha256"] = actual
            entry["verified_locally"] = expected is not None
        try:
            entry["gguf"] = inspect_gguf(local, model.quant)
        except Exception as exc:  # non deve mai interrompere il download
            entry["gguf"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(gguf_summary_line(entry["gguf"]))
        if entry["gguf"].get("label_matches_file_type") is False:
            print(f"  ATTENZIONE: l'etichetta richiesta ({model.quant}) non coincide con il tipo di file "
                  f"({entry['gguf']['file_type_name']})")
        record["files"].append(entry)
    return str(target / files[0]), record  # con gli shard, llama-server vuole il primo (-00001-of-...)


# ---------------------------------------------------------------------------
# Programma principale
# ---------------------------------------------------------------------------


def seleziona(api, args, orgs):
    """Fase 1 e 2: seleziona dai metadati e congela i documenti. Restituisce (candidati, dati, percorso base, esito)."""
    compiti = args.task or list(COMPITI)
    soglia_pre = args.min_downloads // 10
    conta: Counter = Counter()
    esclusi: list[dict] = []

    print(f"Elenco dei modelli di {len(orgs)} organizzazioni ({', '.join(compiti)})...")
    elenco = universo(api, orgs, compiti, args.limite_per_org, soglia_pre, conta)
    print(f"{len(elenco)} modelli sopra la soglia di pre-selezione: lettura dei metadati...")

    candidati = []
    for k, m in enumerate(elenco, 1):
        try:
            c = analizza(api, m, args, conta, esclusi)
        except Exception as exc:
            print(f"  ATTENZIONE: {m.id} saltato: {type(exc).__name__}: {exc}")
            conta["errori di lettura"] += 1
            esclusi.append({"id": m.id, "motivo": f"errore di lettura ({type(exc).__name__})"})
            continue
        if c:
            candidati.append(c)
        if k % 25 == 0:
            print(f"  ...{k}/{len(elenco)} esaminati, {len(candidati)} idonei finora")

    candidati = applica_tetto(candidati, args.max_per_org, args.ordine_famiglia, conta, esclusi,
                              priorita_coding=bool(args.max_coding))
    candidati = seleziona_finale(candidati, args.max_modelli, args.max_coding, conta, esclusi)
    candidati.sort(key=lambda x: (x["dominio"], -x["downloads_totali"], x["id"]))
    usati: set = set()
    for c in sorted(candidati, key=lambda x: x["id"]):  # nomi stabili: assegnati in ordine alfabetico
        c["scarica"]["nome"] = nome_breve(c["id"], usati)
        c["scarica"]["nota"] = nota_quant(c["scarica"]["etichetta"], c["scarica"]["autore"])
    diag = diagnostica_coding(candidati, esclusi)
    print(f"Modelli coding (dai tag {sorted(CODING_TAG)}): esaminati {diag['esaminati_con_tag_coding']}, "
          f"nella lista finale {diag['nella_lista_finale']}.")
    if not diag["nella_lista_finale"]:
        print("  Nota: nessun modello coding nella lista finale; il dettaglio è nella diagnostica del report.")

    ora = datetime.now(timezone.utc)
    non_valutabili = {k: v for k, v in conta.items() if k.startswith("non valutabile")}
    motivi = {k: v for k, v in conta.items() if not k.startswith("non valutabile") and k not in ("elenchi_falliti",)}
    dati = {
        "esecuzione": {
            "data_ora_utc": ora.isoformat(timespec="seconds"),
            "script": "download_models.py", "versione": VERSIONE,
            "sha256_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "huggingface_hub": huggingface_hub.__version__, "python": platform.python_version(),
            "token_usato": bool(TOKEN),  # solo se c'era: il token non viene mai scritto
        },
        "criteri": {
            "organizzazioni": orgs, "compiti": compiti, "soglia_preselezione": soglia_pre,
            "min_downloads": args.min_downloads, "min_b": args.min_b, "max_b": args.max_b,
            "min_contesto": args.min_contesto, "creati_dopo": args.creati_dopo,
            "max_per_org": args.max_per_org, "ordine_famiglia": args.ordine_famiglia,
            "max_modelli": args.max_modelli, "max_coding": args.max_coding,
            "quant_preferite": args.quant, "quantizzatori_fidati": QUANTIZZATORI_FIDATI,
        },
        "conteggi": {"esaminati": len(elenco), "per_motivo": motivi, "non_valutabili": non_valutabili,
                     "elenchi_falliti": conta.get("elenchi_falliti", 0)},
        "diagnostica_coding": diag,
        "candidati": candidati,
        "esclusi": esclusi,
    }
    cartella = Path(args.out_dir)
    cartella.mkdir(parents=True, exist_ok=True)
    base = percorso_libero(cartella, "candidati", ora.strftime("%Y-%m-%d"))
    Path(f"{base}.json").write_text(json.dumps(dati, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scrivi_report(Path(f"{base}.txt"), dati)
    print(f"\n{len(candidati)} modelli scelti, {len(esclusi)} esclusi. Documenti congelati: {base}.json e {base}.txt")
    incompleta = bool(conta.get("elenchi_falliti"))
    if incompleta:
        print(f"ATTENZIONE: {conta['elenchi_falliti']} elenchi non sono riusciti: la selezione potrebbe essere incompleta.")
    return candidati, dati, base, (1 if incompleta else 0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Seleziona i modelli dai metadati del Hugging Face Hub, congela la lista in documenti datati "
                    "e scarica i file GGUF. Senza argomenti fa tutte e tre le cose.")
    g = ap.add_argument_group("selezione")
    g.add_argument("--orgs-file", help="file di testo con un'organizzazione per riga (# per i commenti); sostituisce l'elenco predefinito")
    g.add_argument("--task", action="append", help=f"compito del Hub (ripetibile; predefiniti: {', '.join(COMPITI)})")
    g.add_argument("--min-downloads", type=int, default=100_000, help="soglia IN-6 sui download totali (default: 100000)")
    g.add_argument("--min-b", type=float, default=14.0, help="parametri minimi in miliardi (default: 14)")
    g.add_argument("--max-b", type=float, default=85.0, help="parametri massimi in miliardi (default: 85)")
    g.add_argument("--min-contesto", type=int, default=32_768, help="contesto minimo dichiarato (default: 32768)")
    g.add_argument("--creati-dopo", help="AAAA-MM-GG: esclude i modelli creati prima (default: nessun limite)")
    g.add_argument("--max-per-org", type=int, default=2, help="massimo di modelli per organizzazione (default: 2)")
    g.add_argument("--ordine-famiglia", choices=["recenti", "download"], default="recenti",
                   help="come scegliere i modelli quando un'organizzazione supera il massimo (default: i più recenti)")
    g.add_argument("--max-modelli", type=int, default=10, help="numero massimo di modelli nella lista finale (default: 10; 0 = nessun limite)")
    g.add_argument("--max-coding", type=int, default=5,
                   help="modelli coding che entrano per primi nella lista (default: 5); gli altri posti vanno ai "
                        "più scaricati per download totali (propri più quantizzazioni). 0 = nessuna priorità")
    g.add_argument("--quant", default=",".join(QUANT_PREFERITE),
                   help="etichette di quantizzazione preferite, in ordine, separate da virgola "
                        f"(default: {','.join(QUANT_PREFERITE)}); un modello senza un GGUF con una di queste non è candidato")
    g.add_argument("--limite-per-org", type=int, default=400, help="modelli letti per organizzazione e compito (default: 400)")
    g.add_argument("--out-dir", default="selezioni", help="cartella dei documenti congelati (default: selezioni)")
    g = ap.add_argument_group("fasi")
    g.add_argument("--solo-selezione", action="store_true", help="seleziona e scrive i documenti congelati, senza scaricare")
    g.add_argument("--da-lista", metavar="FILE.json",
                   help="non seleziona: scarica i modelli di una lista già congelata (candidati_<data>.json)")
    g.add_argument("--list", action="store_true", help="mostra il piano di download (nome, repository, etichetta) ed esce")
    g = ap.add_argument_group("download")
    g.add_argument("--only", help="scarica solo questi nomi, separati da virgola (vedi --list)")
    g.add_argument("--out", default="models", help="cartella di destinazione dei modelli (default: models)")
    g.add_argument("--dry-run", action="store_true",
                   help="seleziona e congela i documenti, poi mostra cosa verrebbe scaricato senza scaricare")
    g.add_argument("--no-verify", action="store_true",
                   help="non calcola lo SHA-256 locale (nel manifest resta quello dichiarato dal Hub)")
    g.add_argument("--update", action="store_true",
                   help="scarica l'ultimo commit anche se il manifest o la lista ne fissano uno diverso "
                        "(i pesi cambiano: le misure già fatte non sono confrontabili)")
    g.add_argument("--force", action="store_true",
                   help="prosegue anche se lo spazio libero non basta per tutti i modelli selezionati")
    args = ap.parse_args()
    args.quant = [x.strip() for x in args.quant.split(",") if x.strip()]
    if args.da_lista and args.solo_selezione:
        sys.exit("--da-lista e --solo-selezione non si possono usare insieme: la lista è già congelata")

    global TOKEN
    TOKEN = leggi_token()
    if TOKEN:
        print("Token Hugging Face trovato (variabile d'ambiente).")
    else:
        print("ATTENZIONE: HF_TOKEN non impostato: richieste anonime, con limiti più stretti.\n"
              "Token gratuito (scope read): https://huggingface.co/settings/tokens")
    api = HfApi(token=TOKEN)
    verifica_compatibilita(api)
    print(f"huggingface_hub {huggingface_hub.__version__}")

    # --- fasi 1 e 2: selezione e congelamento (oppure lettura di una lista già congelata) ---
    esito_selezione = 0
    if args.da_lista:
        dati, modelli = leggi_lista_congelata(args.da_lista)
        origine = str(args.da_lista)
        print(f"Lista congelata: {origine} ({len(modelli)} modelli, selezione del {dati['esecuzione']['data_ora_utc']})")
    else:
        orgs = leggi_orgs(args.orgs_file) if args.orgs_file else list(ORG_PREDEFINITE)
        candidati, dati, base, esito_selezione = seleziona(api, args, orgs)
        origine = f"{base}.json"
        modelli = modelli_da_candidati(candidati)
        if not modelli:
            print("Nessun modello selezionato: niente da scaricare.")
            return 1
    if args.list:
        print("\nPiano di download:")
        for m in modelli:
            print(f"{m.name:40} {m.repo:58} {m.quant}  (da {m.base_model})")
            if m.note:
                print(f"{'':40} nota: {m.note}")
        return esito_selezione
    if args.solo_selezione:
        return esito_selezione
    if esito_selezione:
        print("Selezione incompleta: download non avviato (correggi l'errore e rilancia).")
        return 1

    # --- fase 3: download ---
    modelli = filtra_modelli(modelli, args.only)
    out_dir = Path(args.out).resolve()
    manifest = load_manifest(out_dir)
    results: list[tuple[str, str, str]] = []
    records: dict[str, dict] = {}

    print()
    deficit = print_space_summary(plan_space(modelli, out_dir, api, manifest, args.update), free_space(out_dir))
    if deficit and not args.dry_run and not args.force:
        print("Nessun download avviato.")
        return 1

    for m in modelli:
        print(f"\n[{m.name}] {m.repo}  quant={m.quant}  (da {m.base_model})")
        try:
            path, record = download(m, out_dir, api, TOKEN, args.dry_run, verify=not args.no_verify,
                                    previous=manifest.get(m.name), update=args.update)
            if record:
                update_manifest(out_dir, record)
                records[m.name] = record
            results.append((m.name, "OK", path))
        except DownloadError as exc:
            print(f"  ERRORE: {exc}")
            results.append((m.name, "ERRORE", str(exc).splitlines()[0]))
        except KeyboardInterrupt:
            print("\nInterrotto dall'utente.")
            results.append((m.name, "INTERROTTO", ""))
            break
        except Exception as exc:  # rete, autenticazione, repo non trovato...
            print(f"  ERRORE: {type(exc).__name__}: {exc}")
            results.append((m.name, "ERRORE", type(exc).__name__))

    print("\n=== Riepilogo ===")
    for name, status, detail in results:
        print(f"{status:10} {name:40} {detail}")
    if not args.dry_run:
        cartella = Path(args.out_dir)
        cartella.mkdir(parents=True, exist_ok=True)
        esito = percorso_libero(cartella, "esito_download", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        scrivi_esito_download(Path(f"{esito}.txt"), origine, out_dir, results, records)
        print(f"Esito del download: {esito}.txt | manifest: {out_dir / 'manifest.json'}")
    ok = all(status == "OK" for _, status, _ in results) and not (args.dry_run and deficit)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
