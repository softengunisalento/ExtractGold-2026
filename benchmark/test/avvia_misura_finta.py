"""Avvia il servizio di misura con modelli ausiliari FINTI (solo per la prova simulata).

Il codice del servizio è quello vero: si sostituisce solo la classe Ausiliari, che sulla macchina vera carica
LLMLingua-2, Selective Context e bge-small-en-v1.5.
"""
import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import misura  # noqa: E402


class AusiliariFinti(misura.Ausiliari):
    def precarica(self):
        self.info = {"caricati": True, "finti": True, "selective_context_unita": self.c["selective_context_unita"]}

    def comprimi(self, metodo, testi, quota):
        if metodo not in ("llmlingua2", "selective_context"):
            raise misura.Errore(f"metodo sconosciuto: {metodo}")
        passo = 2 if metodo == "llmlingua2" else 3
        return [" ".join(t.split()[::passo]) or t for t in testi], []

    def embedding(self, voci):
        def vettore(testo):
            v = [0.0] * 32
            for p in testo.lower().split():
                v[int(hashlib.md5(p.encode()).hexdigest(), 16) % 32] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            return [x / n for x in v]
        out = []
        for voce in voci:
            parole = voce["testo"].split()
            pezzi = [" ".join(parole[i:i + 50]) for i in range(0, len(parole), 50)] or [voce["testo"]]
            out.append([vettore(p) for p in (pezzi if voce["tipo"] == "documento" else [voce["testo"]])])
        return out


misura.Ausiliari = AusiliariFinti
sys.exit(misura.main())
