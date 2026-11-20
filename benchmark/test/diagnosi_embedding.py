"""Diagnosi dell'errore CUDA su /embedding (S7): rifà, con la stessa logica di misura.py, gli embedding che S7
chiede per le prime N conversazioni nell'ordine a giri. Una chiamata per turno, con tutti i turni candidati e la
query nello stesso encode, come nel servizio. Non tocca il servizio di misura né i risultati.

    CUDA_VISIBLE_DEVICES="" python test/diagnosi_embedding.py                      # su CPU
    CUDA_LAUNCH_BLOCKING=1 python test/diagnosi_embedding.py --dispositivo cuda     # su GPU
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import configurazione  # noqa: E402
import dati  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.toml"))
ap.add_argument("--max-conversazioni", type=int, default=20)
ap.add_argument("--dispositivo", default="cpu")
args = ap.parse_args()

cfg = configurazione.carica(args.config)
a = cfg["ausiliari"]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
conv = dati.conversazioni_ordinate(cfg)[: args.max_conversazioni]
recenti = int(cfg["strategie"]["turni_recenti"])

from sentence_transformers import SentenceTransformer  # noqa: E402

st = SentenceTransformer(a["embedding"], device=args.dispositivo)
st.max_seq_length = int(a["embedding_max_token"])
tok = st.tokenizer
quanti = int(a["embedding_max_token"]) - 2
limite_pos = st[0].auto_model.config.max_position_embeddings
vocabolario = st[0].auto_model.config.vocab_size
print(f"modello {a['embedding']} su {args.dispositivo}: posizioni {limite_pos}, vocabolario {vocabolario}, "
      f"max_seq_length {st.max_seq_length}")


def pezzi_di(testo, tipo):
    if tipo == "query":
        return [a["embedding_prefisso_query"] + testo]
    ids = tok(testo, add_special_tokens=False)["input_ids"]
    return [tok.decode(ids[x:x + quanti]) for x in range(0, len(ids), quanti)] or [testo]


problemi, chiamate, max_len = 0, 0, 0
for c in conv:
    for k in range(1, c.n_turni + 1):
        turni = dati.dividi_turni(c.storia_prima(k))
        voci = [("documento", j + 1, "\n\n".join(m["content"] for m in turni[j]))
                for j in range(max(0, len(turni) - recenti))]
        if not voci:
            continue
        voci.append(("query", k, c.utente(k)["content"]))
        pezzi = [p for tipo, _, testo in voci for p in pezzi_di(testo, tipo)]
        for p in pezzi:
            f = st.tokenize([p])
            lung, id_max = int(f["input_ids"].shape[1]), int(f["input_ids"].max())
            max_len = max(max_len, lung)
            if lung > limite_pos or id_max >= vocabolario:
                problemi += 1
                print(f"\nPROBLEMA {c.id}, turno {k}: pezzo di {lung} token, id massimo {id_max}: {p[:200]!r}")
        chiamate += 1
        try:
            st.encode(pezzi, normalize_embeddings=True, convert_to_numpy=True)
        except Exception as exc:
            problemi += 1
            print(f"\nPROBLEMA {c.id}, turno {k}: encode di {len(pezzi)} pezzi fallito: "
                  f"{type(exc).__name__}: {str(exc)[:400]}")
            sys.exit(f"\nfermato alla prima eccezione ({chiamate} chiamate, pezzo più lungo {max_len} token)")
    print(f"conversazione {c.id}: fatta ({c.n_turni} turni)", flush=True)

print(f"\n{chiamate} chiamate, pezzo più lungo {max_len} token, problemi trovati: {problemi}")
