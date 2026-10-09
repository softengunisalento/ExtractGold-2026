#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

try:
    from datasets import load_dataset
    from conversation_stats import analyze, dataset_commit, get_messages, make_counter
except ImportError as exc:
    sys.exit(f"import non riuscito ({exc}): serve `pip install -U datasets` "
             "e conversation_stats.py nella stessa cartella")


def main() -> int:
    ap = argparse.ArgumentParser(description="Esporta in JSONL le conversazioni selezionate per il replay.")
    ap.add_argument("--dataset", default="Suzhen/CodeChat", help="dataset su Hugging Face (default: Suzhen/CodeChat)")
    ap.add_argument("--config", default=None, help="configurazione del dataset, se prevista")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-turns", type=int, default=10, help="turn minimi (default: 10)")
    ap.add_argument("--min-tokens", type=int, default=4000, help="token minimi di contesto finale (default: 4000)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="token massimi di contesto finale, per restare nella finestra dei modelli (0 = nessun limite)")
    ap.add_argument("--model-regex", default="^gpt-4",
                    help='tiene solo i modelli della colonna "model" che corrispondono (default: ^gpt-4; "" = tutti)')
    ap.add_argument("--n", type=int, default=100, help="conversazioni da esportare (0 = tutte le idonee)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan", type=int, default=0, help="legge solo le prime N conversazioni (0 = tutte; utile per prove)")
    ap.add_argument("--tokenizer", help="repo Hugging Face del tokenizer da usare (default: stima a caratteri/4)")
    ap.add_argument("--out", default="replay_set", help="cartella di output (default: replay_set)")
    args = ap.parse_args()

    model_re = re.compile(args.model_regex) if args.model_regex else None
    count, method = make_counter(args.tokenizer)
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    rng = random.Random(args.seed)
    reservoir: list[dict] = []
    scanned = matching = 0
    for i, row in enumerate(ds):
        if args.scan and scanned >= args.scan:
            break
        scanned += 1
        if scanned % 5000 == 0:
            print(f"  ...{scanned:,} conversazioni lette, {matching:,} idonee")
        if model_re and not model_re.search(str(row.get("model") or "")):
            continue
        msgs = get_messages(row)
        if not msgs:
            continue
        st = analyze(msgs, count)
        if st["turns"] < args.min_turns or st["context_last_turn"] < args.min_tokens:
            continue
        if args.max_tokens and st["context_last_turn"] > args.max_tokens:
            continue

        matching += 1
        rec = {
            "id": row.get("conversation_id") or row.get("conversation_hash") or row.get("id") or i,
            "model": row.get("model") or "",
            "turns": st["turns"],
            "total_tokens": st["total_tokens"],
            "context_last_turn": st["context_last_turn"],
            "messages": [{"role": r, "content": t} for r, t in msgs],
        }
        # reservoir sampling: campione uniforme tra tutte le idonee
        if args.n == 0 or matching <= args.n:
            reservoir.append(rec)
        else:
            j = rng.randint(0, matching - 1)
            if j < args.n:
                reservoir[j] = rec

    if not reservoir:
        sys.exit(f"nessuna conversazione idonea su {scanned:,} lette: abbassa --min-turns/--min-tokens, "
                 'controlla --model-regex (usa "" se il dataset non ha la colonna model) o --max-tokens')

    reservoir.sort(key=lambda r: str(r["id"]))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "conversazioni.jsonl", "w", encoding="utf-8") as fh:
        for rec in reservoir:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "dataset": args.dataset, "commit": dataset_commit(args.dataset), "split": args.split,
        "parametri": {"min_turns": args.min_turns, "min_tokens": args.min_tokens, "max_tokens": args.max_tokens,
                      "model_regex": args.model_regex, "n": args.n, "seed": args.seed, "scan": args.scan},
        "conteggio_token": method,
        "lette": scanned, "idonee": matching, "esportate": len(reservoir),
        "turn_mediana": median(r["turns"] for r in reservoir),
        "contesto_finale_mediana": median(r["context_last_turn"] for r in reservoir),
        "data_estrazione": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attribuzione": "Dati derivati da CodeChat (Zhong, Zou, Adams, 2025) e WildChat (Zhao et al., 2024), "
                        "licenza ODC-BY: citare le fonti e non tentare di identificare le persone.",
    }
    (out_dir / "selezione.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{scanned:,} conversazioni lette, {matching:,} idonee, {len(reservoir):,} esportate - {method}")
    print(f"Turn: mediana {meta['turn_mediana']:.0f}; contesto finale: mediana {meta['contesto_finale_mediana']:,.0f} token")
    print(f"Salvati: {out_dir / 'conversazioni.jsonl'}, {out_dir / 'selezione.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())