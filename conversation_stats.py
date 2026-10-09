#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets non installato: pip install -U datasets")

ROLE_MAP = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant",
            "model": "assistant", "bot": "assistant", "system": "system"}
MSG_KEYS = ("conversation", "messages", "conversations", "dialogue")  # nomi possibili della colonna
TURN_GRID = (2, 5, 10, 15, 20, 30)
TOKEN_GRID = (2_000, 4_000, 8_000, 16_000, 32_000)


# ---------------------------------------------------------------------------
# Lettura e analisi delle conversazioni
# ---------------------------------------------------------------------------
def flatten(node) -> list[dict]:
    """Appiattisce liste annidate di messaggi (alcuni dataset, come CodeChat, li raggruppano per turn)."""
    if isinstance(node, dict):
        return [node]
    if isinstance(node, list):
        return [m for item in node for m in flatten(item)]
    return []


def get_messages(row: dict) -> list[tuple[str, str]] | None:
    """Estrae la lista (ruolo, testo) dalla riga, o None se non riconosce il formato.

    Riconosce sia il formato {"role", "content"} (WildChat, CodeChat, LMSYS) sia il
    formato ShareGPT {"from", "value"}, anche se i messaggi sono in liste annidate.
    """
    for key in MSG_KEYS:
        msgs = flatten(row.get(key))
        if msgs:
            out = []
            for m in msgs:
                role = ROLE_MAP.get(str(m.get("role") or m.get("from") or "").lower(), "other")
                text = m.get("content") if m.get("content") is not None else m.get("value")
                out.append((role, text if isinstance(text, str) else ""))
            return out
    return None


def make_counter(tokenizer: str | None):
    """Restituisce (funzione che conta i token di un testo, descrizione del metodo)."""
    if not tokenizer:
        sys.exit("Errore: Specify a real tokenizer using --tokenizer")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("tokenizers non installato: pip install -U tokenizers (oppure ometti --tokenizer)")
    token = os.environ.get("HF_TOKEN")
    try:
        tok = Tokenizer.from_pretrained(tokenizer, token=token)
    except TypeError:  # versioni vecchie di tokenizers
        tok = Tokenizer.from_pretrained(tokenizer)
    return (lambda text: len(tok.encode(text, add_special_tokens=False).ids) if text else 0), f"tokenizer {tokenizer}"


def analyze(messages: list[tuple[str, str]], count) -> dict:
    ctx = turns = prefill = ctx_last = 0
    for role, text in messages:
        ctx += count(text)
        if role == "user":
            turns += 1
            prefill += ctx
            ctx_last = ctx
    return {"turns": turns, "total_tokens": ctx, "context_last_turn": ctx_last, "prefill_no_cache": prefill}


# ---------------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------------
def percentile(sorted_vals: list[float], p: float) -> float:
    k = (len(sorted_vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def describe(vals: list[float]) -> dict:
    s = sorted(vals)
    return {"min": s[0], "p25": percentile(s, 25), "mediana": percentile(s, 50),
            "p75": percentile(s, 75), "p90": percentile(s, 90), "p95": percentile(s, 95),
            "max": s[-1], "media": sum(s) / len(s)}


def share_at_least(vals: list[float], thresholds) -> dict:
    return {str(t): {"n": sum(v >= t for v in vals), "percentuale": 100 * sum(v >= t for v in vals) / len(vals)}
            for t in thresholds}


def print_describe(title: str, d: dict) -> None:
    print(f"  {title:24} " + "  ".join(f"{k}={d[k]:,.0f}" for k in ("min", "p25", "mediana", "p75", "p90", "p95", "max", "media")))


def print_shares(title: str, shares: dict) -> None:
    print(f"  {title}")
    for t, v in shares.items():
        print(f"    >= {int(t):>6,}: {v['n']:>6,} conversazioni ({v['percentuale']:.1f}%)")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def open_dataset(args):
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=not args.no_streaming)
    if args.lang:
        first_row = next(iter(ds))
        if "language" not in first_row:
            sys.exit(f"Error: '{args.dataset}' does not have a column named 'language'. Remove the flag --lang ")
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=not args.no_streaming)
        ds = ds.filter(lambda r: r.get("language") == args.lang)
    if args.no_streaming:
        ds = ds.shuffle(seed=args.seed)
        if args.sample:
            ds = ds.select(range(min(args.sample, len(ds))))
    else:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        if args.sample:
            ds = ds.take(args.sample)
    return ds


def inspect(ds) -> None:
    try:
        row = next(iter(ds))
    except StopIteration:
        sys.exit("nessun record (controlla --split, --config e --lang)")
    print("Colonne del primo record:")
    for k, v in row.items():
        r = repr(v)
        print(f"  {k:24} {type(v).__name__:10} {r[:100] + ('...' if len(r) > 100 else '')}")
    msgs = get_messages(row)
    if msgs:
        print(f"\nConversazione riconosciuta: {len(msgs)} messaggi, primi ruoli: {[r for r, _ in msgs][:6]}")
    else:
        print("\nNessuna colonna di messaggi riconosciuta (attese: " + ", ".join(MSG_KEYS) + ")")


def dataset_commit(name: str) -> str | None:
    """Commit del dataset al momento dell'analisi, per la riproducibilità."""
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(name).sha
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Grafico
# ---------------------------------------------------------------------------
def make_plot(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non installato: grafico saltato (pip install matplotlib)")
        return
    ctx = sorted(r["context_last_turn"] for r in rows if r["context_last_turn"] > 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist([min(r["turns"], 50) for r in rows], bins=range(1, 52))
    a1.set_xlabel("turn per conversazione (tagliato a 50)")
    a1.set_ylabel("conversazioni")
    a2.plot(ctx, [(i + 1) / len(ctx) for i in range(len(ctx))])
    a2.set_xscale("log")
    a2.set_xlabel("contesto finale (token)")
    a2.set_ylabel("frazione cumulata")
    for t in (4_000, 8_000, 32_000):
        a2.axvline(t, linestyle="--", linewidth=0.8, color="gray")
    fig.tight_layout()
    fig.savefig(out_dir / "distribuzioni.png", dpi=150)
    print(f"Grafico: {out_dir / 'distribuzioni.png'}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Statistiche su turn e token di dataset di conversazioni con LLM.")
    ap.add_argument("--dataset", default="Suzhen/CodeChat", help="dataset su Hugging Face (default: Suzhen/CodeChat)")
    ap.add_argument("--config", default=None, help="configurazione del dataset, se prevista")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=5000, help="numero di conversazioni da analizzare (0 = tutte)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000, help="buffer del rimescolamento in streaming")
    ap.add_argument("--no-streaming", action="store_true", help="scarica tutto il dataset (campione davvero casuale)")
    ap.add_argument("--lang", help="tiene solo le conversazioni con questo valore della colonna 'language' (es. English)")
    ap.add_argument("--tokenizer", required=True, help="repo Hugging Face del tokenizer da usare (es. Qwen/Qwen3-Coder-30B-A3B-Instruct)")
    ap.add_argument("--min-turns", type=int, required=True, help="turn minimi per una conversazione candidata al replay")
    ap.add_argument("--min-tokens", type=int, required=True, help="token minimi di contesto finale per una candidata")
    ap.add_argument("--out", default="stats_out", help="cartella di output (default: stats_out)")
    ap.add_argument("--plot", action="store_true", help="salva un grafico delle distribuzioni (richiede matplotlib)")
    ap.add_argument("--inspect", action="store_true", help="mostra le colonne del primo record ed esce")
    args = ap.parse_args()

    ds = open_dataset(args)
    if args.inspect:
        inspect(ds)
        return 0

    count, method = make_counter(args.tokenizer)
    rows: list[dict] = []
    skipped = 0
    languages: Counter = Counter()
    models: Counter = Counter()
    for i, row in enumerate(ds):
        msgs = get_messages(row)
        if msgs is None:
            skipped += 1
            continue
        stats = analyze(msgs, count)
        if stats["turns"] == 0:
            skipped += 1
            continue
        stats["id"] = row.get("conversation_hash") or row.get("conversation_id") or row.get("id") or i
        stats["language"] = row.get("language") or ""
        stats["model"] = row.get("model") or ""
        stats["eligible"] = stats["turns"] >= args.min_turns and stats["context_last_turn"] >= args.min_tokens
        languages[stats["language"] or "(n/d)"] += 1
        models[stats["model"] or "(n/d)"] += 1
        rows.append(stats)
        if len(rows) % 1000 == 0:
            print(f"  ...{len(rows):,} conversazioni analizzate")

    if not rows:
        sys.exit("nessuna conversazione analizzabile: usa --inspect per vedere il formato del dataset "
                 "(o controlla --lang, che richiede una colonna 'language')")

    n = len(rows)
    turns = [r["turns"] for r in rows]
    ctx = [r["context_last_turn"] for r in rows]
    summary = {
        "dataset": args.dataset, "commit": dataset_commit(args.dataset), "split": args.split,
        "campione_richiesto": args.sample, "analizzate": n, "scartate": skipped, "seed": args.seed,
        "streaming": not args.no_streaming, "filtro_lingua": args.lang, "conteggio_token": method,
        "data_analisi": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turn": describe(turns),
        "contesto_finale_token": describe(ctx),
        "token_totali": describe([r["total_tokens"] for r in rows]),
        "prefill_totale_senza_cache": describe([r["prefill_no_cache"] for r in rows]),
        "turn_almeno": share_at_least(turns, TURN_GRID),
        "contesto_finale_almeno": share_at_least(ctx, TOKEN_GRID),
        "candidate_replay": {"min_turns": args.min_turns, "min_tokens": args.min_tokens,
                             "n": sum(r["eligible"] for r in rows),
                             "percentuale": 100 * sum(r["eligible"] for r in rows) / n},
        "lingue_top5": languages.most_common(5),
        "modelli_top5": models.most_common(5),
    }

    print(f"\n=== {args.dataset}: {n:,} conversazioni analizzate ({skipped:,} scartate) - {method} ===")
    print_describe("turn", summary["turn"])
    print_describe("contesto finale (token)", summary["contesto_finale_token"])
    print_describe("token totali", summary["token_totali"])
    print_describe("prefill totale (token)", summary["prefill_totale_senza_cache"])
    print()
    print_shares("Turn per conversazione:", summary["turn_almeno"])
    print_shares("Contesto finale (token):", summary["contesto_finale_almeno"])
    c = summary["candidate_replay"]
    print(f"\nCandidate al replay (>= {c['min_turns']} turn e >= {c['min_tokens']:,} token): "
          f"{c['n']:,} ({c['percentuale']:.1f}%)")
    print("Lingue più frequenti: " + ", ".join(f"{k} ({v})" for k, v in summary["lingue_top5"]))
    print("Modelli che hanno generato le risposte: " + ", ".join(f"{k} ({v})" for k, v in summary["modelli_top5"]))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["id", "language", "model", "turns", "total_tokens", "context_last_turn", "prefill_no_cache", "eligible"]
    with open(out_dir / "conversazioni.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSalvati: {out_dir / 'conversazioni.csv'}, {out_dir / 'summary.json'}")
    if args.plot:
        make_plot(rows, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())