"""Tempi del run di prova: legge i file dei risultati senza modificarli.
Uso: python3 tempi_pilota.py risultati/<nome dell'esecuzione>"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

res = Path(sys.argv[1])


def righe(p):
    with open(p, encoding="utf-8") as fh:
        for r in fh:
            if r.strip():
                yield json.loads(r)


def ore(ms):
    return f"{ms / 3_600_000:.1f} h"


# --- misure ---
tot_ms = 0.0
per_tipo = defaultdict(lambda: [0, 0.0])          # tipo -> [richieste, ms]
per_strat = defaultdict(float)
tok_gen = dec_ms = 0.0
inizi = []
for r in righe(res / "richieste.jsonl"):
    ms = r.get("wall_ms") or 0.0
    tot_ms += ms
    t = r.get("tipo")
    per_tipo[t][0] += 1
    per_tipo[t][1] += ms
    per_strat[r.get("strategia")] += ms
    if r.get("inizio_utc"):
        inizi.append(r["inizio_utc"])
    if t != "compressione" and t != "embedding" and r.get("token_generati") and r.get("decode_ms"):
        tok_gen += r["token_generati"]
        dec_ms += r["decode_ms"]
print("MISURE")
print(f"  richieste: {sum(v[0] for v in per_tipo.values())}, tempo totale delle richieste: {ore(tot_ms)}")
print(f"  prima richiesta: {min(inizi)}  ultima: {max(inizi)}")
print(f"  velocità media di generazione: {1000 * tok_gen / dec_ms:.1f} token/s ({int(tok_gen):,} token generati)")
for t, (n, ms) in sorted(per_tipo.items(), key=lambda x: -x[1][1]):
    print(f"  tipo {t:28s} {n:5d} richieste  {ore(ms):>8s}  media {ms / n / 1000:6.1f} s")
for s in sorted(per_strat):
    print(f"  strategia {s}: {ore(per_strat[s])}")

# --- giudice ---
for gfile in sorted(res.glob("giudizi/*/giudizi.jsonl")):
    g_ms, n, eseguiti, ore_g = 0.0, 0, 0, []
    for g in righe(gfile):
        n += 1
        if g.get("wall_ms"):
            eseguiti += 1
            g_ms += g["wall_ms"]
        if g.get("ora"):
            ore_g.append(g["ora"])
    print("GIUDICE")
    print(f"  giudizi: {n} (eseguiti con il modello: {eseguiti}), tempo totale delle richieste: {ore(g_ms)}")
    if eseguiti:
        print(f"  media per giudizio: {g_ms / eseguiti / 1000:.1f} s")
    if ore_g:
        print(f"  primo: {min(ore_g)}  ultimo: {max(ore_g)}")
