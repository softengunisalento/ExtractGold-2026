#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    sys.exit("huggingface_hub non installato: pip install -U huggingface_hub")


@dataclass(frozen=True)
class Model:
    name: str           # nome breve, usato per la cartella di destinazione
    repo: str           # repository GGUF su Hugging Face
    quant: str | None   # etichetta di quantizzazione; None = da scegliere con --quant
    note: str = ""      # cosa significa davvero l'etichetta (finisce nel manifest come quant_note)


# Lista a 10 del documento "IN_EX Criteria e Lista Modelli". Le etichette sono quelle
# lette sulle pagine dei repository il 3 ottobre 2026; lo script le verifica comunque
# contro i file realmente presenti nel repo. Le note (quarto campo) vengono dalle stesse pagine e non
# sono verificate sul file: il tipo reale dei tensori lo registra inspect_gguf() nel manifest.
MODELS: list[Model] = [
    Model("llama-3.3-70b", "bartowski/Llama-3.3-70B-Instruct-GGUF", "Q4_K_M",
          "K-quant di llama.cpp (repository bartowski)"),
    Model("laguna-xs-2.1", "mudler/Laguna-XS-2.1-APEX-GGUF", None,
          "layout di quantizzazione APEX proprietario (repository mudler), senza Q4_K_M standard"),
    Model("gemma-4-31b", "bartowski/google_gemma-4-31B-it-GGUF", "Q4_K_M",
          "K-quant di llama.cpp (repository bartowski)"),
    Model("deepseek-r1-distill-qwen-32b", "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF", "Q4_K_M",
          "K-quant di llama.cpp (repository Unsloth, variante non Dynamic)"),
    Model("devstral-small-2-24b", "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF", "Q4_K_M",
          "K-quant di llama.cpp (repository Unsloth, variante non Dynamic)"),
    Model("qwen3.8-27b", "unsloth/Qwen3.8-27B-GGUF", "UD-Q4_K_M",
          "Unsloth Dynamic: precisione mista per strato, non equivale a un Q4_K_M uniforme"),
    Model("qwen3-coder-30b-a3b", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Q4_K_M",
          "K-quant di llama.cpp (repository Unsloth, variante non Dynamic)"),
    Model("deepseek-coder-v2-lite", "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF", "Q4_K_M",
          "K-quant di llama.cpp (repository bartowski)"),
    Model("gemma-4-26b-a4b", "unsloth/gemma-4-26B-A4B-it-qat-GGUF", "UD-Q4_K_XL",
          "Unsloth Dynamic su pesi QAT (quantization-aware training); etichetta di Unsloth, non un tipo standard di llama.cpp"),
    Model("gpt-oss-20b", "unsloth/gpt-oss-20b-GGUF", "Q4_K_M",
          "il modello è nativamente MXFP4: l'etichetta Q4_K_M è nominale"),
]
# Note:
# - Laguna-XS-2.1 (repo mudler APEX) usa un layout di quantizzazione proprietario senza
#   Q4_K_M standard: usa `--only laguna-xs-2.1 --quant <etichetta>`. Senza --quant lo
#   script elenca i file disponibili.
# - Per gpt-oss-20b l'etichetta Q4_K_M è nominale (il modello è nativamente MXFP4).

SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
MARGIN = 2 * 1024**3  # margine di sicurezza sullo spazio libero (non si riempie mai del tutto il disco)


class DownloadError(Exception):
    """Errore atteso: quantizzazione non trovata, ambigua o spazio insufficiente."""


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

    Se il manifest ha già una voce per il modello, il suo commit è fissato (vedi download()).
    """
    notes: list[str] = []
    info = api.model_info(model.repo, files_metadata=True)  # ultimo commit del repository
    pinned = None
    if previous and previous.get("commit"):
        if previous.get("repo") == model.repo:
            pinned = previous["commit"]
        else:
            notes.append(f"ATTENZIONE: nel manifest il repository era {previous.get('repo')}: il commit fissato non si applica")
    origin = "primo download" if not previous else "ultimo commit"
    if pinned and pinned != info.sha:
        if update:
            origin = "AGGIORNAMENTO"
            notes.append(f"AGGIORNAMENTO: commit {pinned[:10]} -> {info.sha[:10]}. I pesi cambiano: "
                         "le misure già fatte con il commit precedente non sono confrontabili.")
        else:
            notes.append(f"Il repository ha un commit più recente ({info.sha[:10]}): si usa quello del manifest "
                         f"({pinned[:10]}). Usa --update per aggiornare.")
            try:
                info = api.model_info(model.repo, revision=pinned, files_metadata=True)
            except Exception as exc:
                raise DownloadError(f"il commit del manifest ({pinned[:10]}) non è disponibile nel repository "
                                    f"({type(exc).__name__}): i pesi non sono riproducibili; "
                                    "con --update si scarica l'ultimo commit")
            origin = "fissato dal manifest"
    elif pinned:
        origin = "fissato dal manifest (coincide con l'ultimo)"
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

    Se il manifest ha già una voce per questo modello, il suo commit è fissato: un rilancio scarica
    quel commit, anche se il repository è cambiato nel frattempo. Con update=True si passa all'ultimo
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


def select_models(only: str | None) -> list[Model]:
    if not only:
        return list(MODELS)
    known = {m.name: m for m in MODELS}
    wanted = [x.strip() for x in only.split(",") if x.strip()]
    unknown = [w for w in wanted if w not in known]
    if unknown:
        sys.exit(f"modelli sconosciuti: {', '.join(unknown)}. Usa --list.")
    return [known[w] for w in wanted]


def main() -> int:
    ap = argparse.ArgumentParser(description="Scarica i modelli GGUF candidati da Hugging Face.")
    ap.add_argument("--only", help="nomi separati da virgola (vedi --list)")
    ap.add_argument("--quant", help="forza l'etichetta di quantizzazione (richiede un solo modello in --only)")
    ap.add_argument("--out", default="models", help="cartella di destinazione (default: models)")
    ap.add_argument("--dry-run", action="store_true", help="mostra cosa verrebbe scaricato, senza scaricare")
    ap.add_argument("--no-verify", action="store_true",
                    help="non calcola lo SHA-256 locale (nel manifest resta quello dichiarato dal Hub)")
    ap.add_argument("--update", action="store_true",
                    help="scarica l'ultimo commit anche se il manifest ne fissa uno diverso "
                         "(i pesi cambiano: le misure già fatte non sono confrontabili)")
    ap.add_argument("--force", action="store_true",
                    help="prosegue anche se lo spazio libero non basta per tutti i modelli selezionati")
    ap.add_argument("--list", action="store_true", help="elenca i modelli configurati ed esce")
    args = ap.parse_args()

    if args.list:
        for m in MODELS:
            print(f"{m.name:32} {m.repo:55} {m.quant or '(da scegliere)'}")
            if m.note:
                print(f"{'':32} nota: {m.note}")
        return 0

    models = select_models(args.only)
    if args.quant:
        if len(models) != 1:
            sys.exit("--quant richiede un solo modello in --only")
        models = [replace(models[0], quant=args.quant)]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("Token Hugging Face trovato.")
    else:
        print("ATTENZIONE: HF_TOKEN non impostato, download anonimo con limiti più stretti.\n"
              "Token gratuito (scope read): https://huggingface.co/settings/tokens")

    api = HfApi(token=token)
    out_dir = Path(args.out).resolve()
    manifest = load_manifest(out_dir)
    results: list[tuple[str, str, str]] = []

    deficit = print_space_summary(plan_space(models, out_dir, api, manifest, args.update), free_space(out_dir))
    if deficit and not args.dry_run and not args.force:
        print("Nessun download avviato.")
        return 1

    for m in models:
        print(f"\n[{m.name}] {m.repo}  quant={m.quant}")
        if not m.quant:
            print("  quantizzazione da scegliere (usa --quant, vedi nota nello script)")
            try:
                files = [s.rfilename for s in api.model_info(m.repo).siblings]
                print("  File .gguf nel repo:\n    " + "\n    ".join(list_gguf(files)))
            except Exception as exc:  # solo informativo
                print(f"  (elenco file non disponibile: {exc})")
            results.append((m.name, "SALTATO", "quant mancante"))
            continue
        try:
            path, record = download(m, out_dir, api, token, args.dry_run, verify=not args.no_verify,
                                    previous=manifest.get(m.name), update=args.update)
            if record:
                update_manifest(out_dir, record)
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
        print(f"{status:10} {name:32} {detail}")
    ok = all(status in ("OK", "SALTATO") for _, status, _ in results) and not (args.dry_run and deficit)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
