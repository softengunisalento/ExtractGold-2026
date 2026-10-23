"""Strategie di gestione del contesto S0–S7.

Ogni strategia riceve, a ogni turno, la storia dei messaggi prima del turno e il messaggio dell'utente, e
restituisce il contesto da mandare al modello. Non legge il dataset: la storia può essere quella originale
(replay a storia fissa) o una storia generata (utente simulato, seconda versione).

Le operazioni che costano energia (riassunti, compressioni, embedding) passano dall'oggetto `servizi`, che le
esegue sulla macchina di misura e le registra come overhead. Interfaccia richiesta a `servizi`:

    chiedi(messaggi, tipo) -> str | None        richiesta al modello in prova (None se fallita)
    comprimi(metodo, testi, quota) -> list[str]   "llmlingua2" o "selective_context"; quota = token tenuti
    embedding(voci) -> list[list[list[float]]]    voci = [{"testo", "tipo": "documento"|"query"}]
    intermedio(tipo, contenuto)                   salva un testo intermedio (riassunto, gold prompt...)

Una strategia si crea nuova per ogni coppia (modello, conversazione): conserva lo stato tra un turno e l'altro.
Prima del trigger (attiva=False) ogni strategia coincide con S0. Le scelte sono quelle del documento
«Strategie di gestione del contesto: scelte di utilizzo».
"""
from __future__ import annotations

import re

from dati import dividi_turni


def appiattisci(turni: list[list[dict]]) -> list[dict]:
    return [dict(m) for t in turni for m in t]


def trascrizione(turni: list[list[dict]]) -> str:
    etichette = {"user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"{etichette.get(m['role'], m['role'])}: {m['content']}" for t in turni for m in t)


def unisci_riassunto(prefisso: str, riassunto: str, messaggi_dopo: list[dict], utente: dict) -> list[dict]:
    """Il riassunto va nel primo messaggio dell'utente che lo segue, dopo il prefisso e una riga vuota.

    Così il contesto non ha mai due messaggi dell'utente consecutivi (alcuni chat template li rifiutano).
    """
    resto = [dict(m) for m in messaggi_dopo] + [dict(utente)]
    testa = f"{prefisso}\n\n{riassunto or ''}"
    if resto[0].get("role") == "user":
        resto[0]["content"] = f"{testa}\n\n{resto[0]['content']}"
        return resto
    return [{"role": "user", "content": testa}] + resto


# Codice protetto dalla compressione: blocchi delimitati da ``` o ~~~ (anche non chiusi) e codice in linea.
RE_CODICE = re.compile(r"(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]+`)")


def segmenti(testo: str) -> list[tuple[bool, str]]:
    """[(è_codice, pezzo), ...] nell'ordine del testo."""
    return [(i % 2 == 1, pezzo) for i, pezzo in enumerate(RE_CODICE.split(testo)) if pezzo]


class Strategia:
    nome = "S0"

    def __init__(self, parametri: dict, servizi):
        self.p = parametri
        self.servizi = servizi

    def contesto(self, storia: list[dict], utente: dict, attiva: bool) -> list[dict]:
        return [dict(m) for m in storia] + [dict(utente)]


class S0(Strategia):
    """Storia completa."""
    nome = "S0"


class S1(Strategia):
    """Finestra scorrevole: dopo il trigger, solo gli ultimi N turni più il messaggio corrente."""
    nome = "S1"

    def contesto(self, storia, utente, attiva):
        if not attiva:
            return super().contesto(storia, utente, attiva)
        turni = dividi_turni(storia)
        return appiattisci(turni[-self.p["turni_recenti"]:]) + [dict(utente)]


class S2(Strategia):
    """Gold Extraction: una sola volta, al trigger, il modello genera il gold prompt dalla storia; poi si apre
    una nuova sessione con il gold prompt come primo messaggio dell'utente e si continua da lì."""
    nome = "S2"

    def __init__(self, parametri, servizi):
        super().__init__(parametri, servizi)
        self.fatto = False
        self.sessione: list[dict] = []
        self.coperti = 0

    def contesto(self, storia, utente, attiva):
        turni = dividi_turni(storia)
        if not attiva or (not self.fatto and not turni):  # senza storia non c'è niente da estrarre
            return super().contesto(storia, utente, attiva)
        if not self.fatto:
            self.fatto = True
            gold = self.servizi.chiedi(appiattisci(turni) + [{"role": "user", "content": self.p["prompt_ge"]}],
                                       "ge_gold_prompt")
            self.servizi.intermedio("ge_gold_prompt", gold)
            if gold is None:  # richiesta fallita: la conversazione prosegue come S0, e resta registrato
                self.servizi.intermedio("ge_fallita", "gold prompt non generato: si prosegue con la storia completa")
                self.sessione = None
                return super().contesto(storia, utente, attiva)
            risposta = self.servizi.chiedi([{"role": "user", "content": gold}], "ge_nuova_sessione")
            self.servizi.intermedio("ge_risposta_nuova_sessione", risposta)
            self.sessione = [{"role": "user", "content": gold}, {"role": "assistant", "content": risposta or ""}]
            self.coperti = len(turni)
        if self.sessione is None:
            return super().contesto(storia, utente, attiva)
        return [dict(m) for m in self.sessione] + appiattisci(turni[self.coperti:]) + [dict(utente)]


class S3(Strategia):
    """Riassunto ricorsivo: al trigger, riassunto di tutta la storia; poi, ogni N turni, il riassunto si
    aggiorna con i turni arrivati dopo. Alla lettera restano solo i turni dopo l'ultimo aggiornamento."""
    nome = "S3"

    def __init__(self, parametri, servizi):
        super().__init__(parametri, servizi)
        self.riassunto: str | None = None
        self.coperti = 0
        self.prossimo = None

    def contesto(self, storia, utente, attiva):
        turni = dividi_turni(storia)
        if not attiva or (self.riassunto is None and not turni):
            return super().contesto(storia, utente, attiva)
        n = len(turni)
        if self.riassunto is None:
            r = self.servizi.chiedi(appiattisci(turni) + [{"role": "user", "content": self.p["prompt_s3_primo"]}],
                                    "s3_riassunto")
            self.servizi.intermedio("s3_riassunto", r)
            self.riassunto, self.coperti, self.prossimo = (r or ""), n, n + self.p["s3_ogni_turni"]
        elif n >= self.prossimo:
            testo = self.p["formato_s3_aggiornamento"].format(
                prompt=self.p["prompt_s3_aggiornamento"], riassunto=self.riassunto,
                messaggi=trascrizione(turni[self.coperti:n]))
            r = self.servizi.chiedi([{"role": "user", "content": testo}], "s3_aggiornamento")
            self.servizi.intermedio("s3_aggiornamento", r)
            if r is not None:  # se l'aggiornamento fallisce si tiene il riassunto precedente con i turni alla lettera
                self.riassunto, self.coperti = r, n
            self.prossimo = n + self.p["s3_ogni_turni"]
        return unisci_riassunto(self.p["prefisso_riassunto"], self.riassunto, appiattisci(turni[self.coperti:]), utente)


class S4(Strategia):
    """Keep-tail: una sola volta, riassume i turni più vecchi degli ultimi N e tiene questi alla lettera.
    Se al trigger non ci sono turni più vecchi degli ultimi N, la compattazione avviene al primo turno in cui
    ci sono (sempre una sola volta)."""
    nome = "S4"

    def __init__(self, parametri, servizi):
        super().__init__(parametri, servizi)
        self.riassunto: str | None = None
        self.inizio_coda = 0
        self.fatto = False

    def contesto(self, storia, utente, attiva):
        turni = dividi_turni(storia)
        recenti = self.p["turni_recenti"]
        if not attiva or (not self.fatto and len(turni) <= recenti):
            return super().contesto(storia, utente, attiva)
        if not self.fatto:
            self.fatto = True
            vecchi = turni[: len(turni) - recenti]
            r = self.servizi.chiedi(appiattisci(vecchi) + [{"role": "user", "content": self.p["prompt_s4"]}],
                                    "s4_riassunto")
            self.servizi.intermedio("s4_riassunto", r)
            if r is None:
                self.riassunto = None
            else:
                self.riassunto, self.inizio_coda = r, len(vecchi)
        if self.riassunto is None:
            return super().contesto(storia, utente, attiva)
        return unisci_riassunto(self.p["prefisso_riassunto"], self.riassunto,
                                appiattisci(turni[self.inizio_coda:]), utente)


class _Compressione(Strategia):
    """S5/S6: ogni messaggio della storia si comprime una volta sola, separatamente, tenendo il ruolo; il codice
    (blocchi e in linea) resta intatto; l'ultimo messaggio dell'utente non si comprime mai."""
    metodo = ""

    def __init__(self, parametri, servizi):
        super().__init__(parametri, servizi)
        self.cache: dict[tuple[int, str], str] = {}

    def contesto(self, storia, utente, attiva):
        if not attiva:
            return super().contesto(storia, utente, attiva)
        da_fare = [(i, m) for i, m in enumerate(storia) if (i, m["content"]) not in self.cache]
        if da_fare:
            pezzi, testi = [], []
            for i, m in da_fare:
                segs = segmenti(m["content"])
                pezzi.append((i, m, segs))
                testi += [s for codice, s in segs if not codice and s.strip()]
            compressi = iter(self.servizi.comprimi(self.metodo, testi, self.p["quota_token_tenuti"]) if testi else [])
            for i, m, segs in pezzi:
                nuovo = "".join(s if (codice or not s.strip()) else next(compressi) for codice, s in segs)
                self.cache[(i, m["content"])] = nuovo
            self.servizi.intermedio(f"{self.nome.lower()}_compressi",
                                    [{"indice": i, "ruolo": m["role"], "testo": self.cache[(i, m["content"])]}
                                     for i, m, _ in pezzi])
        return [{"role": m["role"], "content": self.cache[(i, m["content"])]} for i, m in enumerate(storia)] + [dict(utente)]


class S5(_Compressione):
    """LLMLingua-2."""
    nome = "S5"
    metodo = "llmlingua2"


class S6(_Compressione):
    """Selective Context."""
    nome = "S6"
    metodo = "selective_context"


class S7(Strategia):
    """Retrieval: gli ultimi N turni alla lettera più i turni più vecchi più simili al messaggio corrente
    (embedding con i turni lunghi divisi in pezzi; un turno vale quanto il suo pezzo più simile), in ordine
    cronologico, senza segnalare il salto."""
    nome = "S7"

    def __init__(self, parametri, servizi):
        super().__init__(parametri, servizi)
        self.vettori: dict[tuple[int, str], list[list[float]]] = {}

    def contesto(self, storia, utente, attiva):
        if not attiva:
            return super().contesto(storia, utente, attiva)
        turni = dividi_turni(storia)
        recenti = self.p["turni_recenti"]
        candidati = list(range(max(0, len(turni) - recenti)))
        if not candidati:
            return appiattisci(turni) + [dict(utente)]
        testi = {j: "\n\n".join(m["content"] for m in turni[j]) for j in candidati}
        nuovi = [j for j in candidati if (j, testi[j]) not in self.vettori]
        voci = [{"testo": testi[j], "tipo": "documento"} for j in nuovi] + [{"testo": utente["content"], "tipo": "query"}]
        risultato = self.servizi.embedding(voci)
        for j, v in zip(nuovi, risultato[:-1]):
            self.vettori[(j, testi[j])] = v
        query = risultato[-1][0] if risultato and risultato[-1] else []
        punteggi = {j: max((sum(a * b for a, b in zip(query, pezzo)) for pezzo in self.vettori[(j, testi[j])]),
                           default=float("-inf")) for j in candidati}
        scelti = sorted(sorted(candidati, key=lambda j: (-punteggi[j], j))[: self.p["s7_recuperati"]])
        self.servizi.intermedio("s7_recuperati", [{"turno": j + 1, "punteggio": round(punteggi[j], 6)} for j in scelti])
        return appiattisci([turni[j] for j in scelti]) + appiattisci(turni[-recenti:]) + [dict(utente)]


CLASSI = {c.nome: c for c in (S0, S1, S2, S3, S4, S5, S6, S7)}


def crea(nome: str, parametri: dict, servizi) -> Strategia:
    return CLASSI[nome](parametri, servizi)
