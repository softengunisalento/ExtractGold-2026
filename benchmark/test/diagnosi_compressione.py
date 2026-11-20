"""Diagnosi dell'errore CUDA su /comprimi (S5, S6): prende i testi che S5 e S6 comprimono (i pezzi di testo,
senza codice, di ogni messaggio della storia) per le prime N conversazioni nell'ordine a giri, ne conta i token
con i tokenizer di GPT-2 (Selective Context, massimo 1024 posizioni) e di XLM-RoBERTa (LLMLingua-2) e, con
--comprimi, li comprime uno alla volta con il codice del servizio, fermandosi al primo errore.
Non tocca il servizio di misura né i risultati.

    python test/diagnosi_compressione.py                                      # solo conteggi (veloce)
    CUDA_VISIBLE_DEVICES="" python test/diagnosi_compressione.py --comprimi --solo-lunghi   # su CPU
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import configurazione  # noqa: E402
import dati  # noqa: E402
import strategie  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.toml"))
ap.add_argument("--max-conversazioni", type=int, default=20)
ap.add_argument("--comprimi", action="store_true", help="comprime davvero i testi con il codice del servizio")
ap.add_argument("--solo-lunghi", action="store_true", help="con --comprimi: solo i testi oltre 1024 token GPT-2")
args = ap.parse_args()

cfg = configurazione.carica(args.config)
a = dict(cfg["ausiliari"])
a["dispositivo"] = os.environ.get("DISPOSITIVO_DIAGNOSI", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
quota = float(cfg["strategie"]["quota_token_tenuti"])
conv = dati.conversazioni_ordinate(cfg)[: args.max_conversazioni]

from transformers import AutoTokenizer  # noqa: E402

tok_gpt2 = AutoTokenizer.from_pretrained(a["selective_context_modello"])
tok_xlmr = AutoTokenizer.from_pretrained(a["llmlingua2"])

testi = []  # (conversazione, posizione nella conversazione, ruolo, testo)
for n, c in enumerate(conv, 1):
    for i, m in enumerate(c.messaggi[:-1]):       # l'ultimo messaggio dell'utente non si comprime mai
        for codice, s in strategie.segmenti(m["content"]):
            if not codice and s.strip():
                testi.append((n, c.id, i, m["role"], s))

lunghi = 0
righe = []
for n, cid, i, ruolo, s in testi:
    g = len(tok_gpt2(s, add_special_tokens=False)["input_ids"]) + 1   # Selective Context aggiunge <|endoftext|>
    x = len(tok_xlmr(s, add_special_tokens=False)["input_ids"])
    righe.append((g, x, n, cid, i, ruolo, s))
    if g > 1024:
        lunghi += 1
        print(f"OLTRE 1024 (GPT-2): conversazione {n} {cid}, messaggio {i} ({ruolo}): {g} token GPT-2, "
              f"{x} token XLM-R, {len(s)} caratteri")
print(f"\n{len(testi)} testi; più lungo: {max(r[0] for r in righe)} token GPT-2, {max(r[1] for r in righe)} XLM-R; "
      f"oltre 1024 token GPT-2: {lunghi}", flush=True)

if not args.comprimi:
    sys.exit(0)

from misura import Ausiliari  # noqa: E402

aus = Ausiliari(a)
aus.precarica()
da_fare = [r for r in righe if r[0] > 1024] if args.solo_lunghi else righe
print(f"\ncompressione di {len(da_fare)} testi, uno alla volta, con LLMLingua-2 e Selective Context...", flush=True)
for g, x, n, cid, i, ruolo, s in da_fare:
    for metodo in ("llmlingua2", "selective_context"):
        try:
            _, errori = aus.comprimi(metodo, [s], quota)
        except Exception as exc:
            print(f"\nERRORE {metodo}: conversazione {n} {cid}, messaggio {i} ({ruolo}), {g} token GPT-2, {x} XLM-R:"
                  f"\n  {type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(1)
        if errori:
            print(f"ERRORE GESTITO {metodo}: conversazione {n} {cid}, messaggio {i} ({ruolo}), {g} token GPT-2, "
                  f"{x} XLM-R: {errori[0][:300]}")
print("\ncompressioni concluse")
