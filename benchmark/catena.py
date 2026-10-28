#!/usr/bin/env python3
"""Comando unico (PC di controllo): benchmark → giudice → elaborazione.

    python catena.py --modelli qwen2.5-coder-32b-instruct [--config config.toml] [--misura http://<ip>:<porta>]

Esegue in ordine runner.py, giudice.py ed elabora.py sugli stessi modelli. Ogni fase riprende da dove si era
fermata: se la catena si interrompe, si rilancia lo stesso comando e le parti già fatte si saltano.
Prima di partire controlla che le librerie dell'elaborazione siano installate, così un errore non arriva dopo
giorni di benchmark. Sulla macchina di misura deve essere attivo il servizio (python misura.py esegui).
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent
LIBRERIE_ELABORAZIONE = ("matplotlib", "numpy", "pandas", "scipy", "docx")


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark, giudice ed elaborazione in un solo comando.")
    ap.add_argument("--config", default=str(QUI / "config.toml"))
    ap.add_argument("--misura", help="indirizzo del servizio di misura (sostituisce rete.indirizzo_misura)")
    ap.add_argument("--modelli", help="solo questi modelli (nomi brevi, separati da virgola); senza: tutti")
    ap.add_argument("--max-conversazioni", type=int, default=None,
                    help="le prime N conversazioni nell'ordine a giri; rilanciando con un N più grande si allunga "
                         "riusando il già fatto. Senza: tutte per un'esecuzione nuova, lo stesso N per una già avviata")
    args = ap.parse_args()

    mancanti = [m for m in LIBRERIE_ELABORAZIONE if importlib.util.find_spec(m) is None]
    if mancanti:
        sys.exit(f"mancano le librerie dell'elaborazione ({', '.join(mancanti)}): "
                 f"pip install -r {QUI / 'requirements_runner.txt'}")

    comune = ["--config", args.config]
    if args.misura:
        comune += ["--misura", args.misura]
    if args.modelli:
        comune += ["--modelli", args.modelli]
    fasi = [
        ("1/3 benchmark", ["runner.py", *comune] + (["--max-conversazioni", str(args.max_conversazioni)]
                                                   if args.max_conversazioni is not None else [])),
        ("2/3 giudice", ["giudice.py", *comune]),
        ("3/3 elaborazione", ["elabora.py", "--config", args.config] + (["--modelli", args.modelli] if args.modelli else [])),
    ]
    for nome, cmd in fasi:
        print(f"\n===== {nome} =====", flush=True)
        r = subprocess.run([sys.executable, str(QUI / cmd[0]), *cmd[1:]], cwd=QUI)
        if r.returncode:
            print(f"\nLa fase «{nome}» si è fermata (codice {r.returncode}). "
                  "Rilancia lo stesso comando per riprendere da dove si era fermata.")
            return r.returncode
    print("\nCatena conclusa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
