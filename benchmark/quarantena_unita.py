"""Quarantena delle unità (modello, conversazione, strategia) con errori di compressione registrati.

Con il codice precedente alla correzione del 2026-11-05, un errore durante una compressione lasciava il testo intero
e l'unità risultava completata. Questo script individua quelle unità dalle righe di richieste.jsonl con il campo
"errori" non vuoto, e ne sposta tutte le righe (richieste, testi, intermedi, completate e giudizi) in
<cartella>/quarantena/<data-ora>/, dopo aver copiato lì anche i file originali interi. Alla ripresa il runner rifà
le unità tolte e il giudice rifà i loro giudizi. Non cancella niente.

    python quarantena_unita.py --cartella risultati/pilota_2026-10-27            # mostra cosa sposterebbe
    python quarantena_unita.py --cartella risultati/pilota_2026-10-27 --applica  # sposta
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cartella", required=True, help="cartella dell'esecuzione, per esempio risultati/pilota_2026-10-27")
ap.add_argument("--applica", action="store_true", help="sposta davvero le righe (senza: mostra solo cosa farebbe)")
args = ap.parse_args()

cartella = Path(args.cartella)
if not (cartella / "richieste.jsonl").exists():
    sys.exit(f"{cartella / 'richieste.jsonl'} non trovato")


def leggi(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines(keepends=True) if p.exists() else []


def unita_di(riga: dict, nome_file: str):
    if nome_file == "completate.jsonl":
        parti = str(riga.get("unita", "")).split("|")
        return tuple(parti) if len(parti) == 3 else None
    return (riga.get("modello"), riga.get("conversazione"), riga.get("strategia"))


# 1. unità con errori di compressione
contaminate, n_err = set(), 0
for riga in leggi(cartella / "richieste.jsonl"):
    if riga.strip():
        r = json.loads(riga)
        if r.get("errori"):
            n_err += 1
            contaminate.add((r.get("modello"), r.get("conversazione"), r.get("strategia")))
print(f"richieste con errori di compressione: {n_err}; unità contaminate: {len(contaminate)}")
for u in sorted(contaminate):
    print("  ", "|".join(map(str, u)))
if not contaminate:
    sys.exit("niente da mettere in quarantena")

# 2. righe da spostare, file per file
file_unita = [cartella / n for n in ("richieste.jsonl", "testi.jsonl", "intermedi.jsonl", "completate.jsonl")]
file_unita += sorted(cartella.glob("giudizi/*/giudizi.jsonl"))
piano = []
for p in file_unita:
    righe = leggi(p)
    tenute, tolte = [], []
    for riga in righe:
        if riga.strip() and unita_di(json.loads(riga), p.name) in contaminate:
            tolte.append(riga)
        else:
            tenute.append(riga)
    piano.append((p, tenute, tolte))
    print(f"{p.relative_to(cartella)}: {len(righe)} righe, da spostare {len(tolte)}")

completate_tolte = next(len(t) for p, _, t in piano if p.name == "completate.jsonl")
if completate_tolte != len(contaminate):
    sys.exit(f"ATTENZIONE: {completate_tolte} righe di completamento per {len(contaminate)} unità: non procedo")

if not args.applica:
    print("\nprova a vuoto: niente è stato modificato. Rilancia con --applica per spostare.")
    sys.exit(0)

# 3. copia di sicurezza degli originali, righe tolte a parte, poi riscrittura atomica
dest = cartella / "quarantena" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
for p, tenute, tolte in piano:
    rel = p.relative_to(cartella)
    (dest / "originali" / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest / "originali" / rel)
    (dest / "tolte" / rel).parent.mkdir(parents=True, exist_ok=True)
    (dest / "tolte" / rel).write_text("".join(tolte), encoding="utf-8")
for p, tenute, tolte in piano:
    if tolte:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("".join(tenute), encoding="utf-8")
        os.replace(tmp, p)
(dest / "LEGGIMI.json").write_text(json.dumps({
    "motivo": "errori di compressione con testo lasciato intero (codice precedente alla correzione del 2026-11-05)",
    "creata_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "richieste_con_errori": n_err,
    "unita": sorted("|".join(map(str, u)) for u in contaminate),
    "righe_spostate": {str(p.relative_to(cartella)): len(t) for p, _, t in piano},
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\nfatto: righe spostate in {dest}; copie integrali degli originali in {dest / 'originali'}")
