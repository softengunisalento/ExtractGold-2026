#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
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


# Lista a 10 del documento "IN_EX Criteria e Lista Modelli". Le etichette sono quelle
# lette sulle pagine dei repository il 3 ottobre 2026; lo script le verifica comunque
# contro i file realmente presenti nel repo.
MODELS: list[Model] = [
    Model("llama-3.3-70b", "bartowski/Llama-3.3-70B-Instruct-GGUF", "Q4_K_M"),
    Model("laguna-xs-2.1", "mudler/Laguna-XS-2.1-APEX-GGUF", None),
    Model("gemma-4-31b", "bartowski/google_gemma-4-31B-it-GGUF", "Q4_K_M"),
    Model("deepseek-r1-distill-qwen-32b", "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF", "Q4_K_M"),
    Model("devstral-small-2-24b", "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF", "Q4_K_M"),
    Model("qwen3.8-27b", "unsloth/Qwen3.8-27B-GGUF", "UD-Q4_K_M"),
    Model("qwen3-coder-30b-a3b", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Q4_K_M"),
    Model("deepseek-coder-v2-lite", "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF", "Q4_K_M"),
    Model("gemma-4-26b-a4b", "unsloth/gemma-4-26B-A4B-it-qat-GGUF", "UD-Q4_K_XL"),
    Model("gpt-oss-20b", "unsloth/gpt-oss-20b-GGUF", "Q4_K_M"),
]
# Note:
# - Laguna-XS-2.1 (repo mudler APEX) usa un layout di quantizzazione proprietario senza
#   Q4_K_M standard: usa `--only laguna-xs-2.1 --quant <etichetta>`. Senza --quant lo
#   script elenca i file disponibili.
# - Per gpt-oss-20b l'etichetta Q4_K_M è nominale (il modello è nativamente MXFP4).

SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


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


def update_manifest(out_dir: Path, record: dict) -> None:
    """Aggiunge o aggiorna la voce del modello in <out_dir>/manifest.json."""
    path = out_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[record["name"]] = record
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download(model: Model, out_dir: Path, api: HfApi, token: str | None,
             dry_run: bool, verify: bool = True) -> tuple[str, dict | None]:
    """Scarica un modello. Restituisce (percorso per `llama-server -m`, voce del manifest)."""
    info = api.model_info(model.repo, files_metadata=True)
    siblings = {s.rfilename: s for s in info.siblings}
    sizes = {f: (s.size or 0) for f, s in siblings.items()}
    files = resolve_files(list(sizes), model.quant)
    total = sum(sizes[f] for f in files)

    target = out_dir / model.name
    free = free_space(target)
    print(f"  {len(files)} file, {human(total)} (spazio libero: {human(free)})")
    for f in files:
        print(f"    - {f}  ({human(sizes[f])})")
    if total > free:
        raise DownloadError(f"spazio insufficiente: servono {human(total)}, liberi {human(free)}")
    if dry_run:
        return "dry-run", None

    target.mkdir(parents=True, exist_ok=True)
    record = {
        "name": model.name,
        "repo": model.repo,
        "quant": model.quant,
        "commit": info.sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [],
    }
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
    ap.add_argument("--list", action="store_true", help="elenca i modelli configurati ed esce")
    args = ap.parse_args()

    if args.list:
        for m in MODELS:
            print(f"{m.name:32} {m.repo:55} {m.quant or '(da scegliere)'}")
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
    results: list[tuple[str, str, str]] = []

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
            path, record = download(m, out_dir, api, token, args.dry_run, verify=not args.no_verify)
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
    return 0 if all(status in ("OK", "SALTATO") for _, status, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())