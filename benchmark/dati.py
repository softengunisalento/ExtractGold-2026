"""Lettura dei file di input congelati: conversazioni (JSONL), selezione delle conversazioni e lista dei modelli.

Una conversazione è una lista di messaggi {"role", "content"}. Il turno k è il k-esimo messaggio dell'utente
con le risposte che lo seguono, fino al messaggio dell'utente successivo.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


def sha256_file(percorso: Path) -> str:
    h = hashlib.sha256()
    with open(percorso, "rb") as fh:
        for blocco in iter(lambda: fh.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def dividi_turni(messaggi: list[dict]) -> list[list[dict]]:
    """Divide una storia in turni: ogni turno comincia con un messaggio dell'utente.

    Gli eventuali messaggi prima del primo messaggio dell'utente formano un turno a sé (non succede nei
    nostri dati, ma una storia generata da altri script potrebbe averli).
    """
    turni: list[list[dict]] = []
    for m in messaggi:
        if m.get("role") == "user" or not turni:
            turni.append([m])
        else:
            turni[-1].append(m)
    return turni


@dataclass
class Conversazione:
    id: str
    fonte: str
    messaggi: list[dict]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self._utenti = [i for i, m in enumerate(self.messaggi) if m.get("role") == "user"]

    @property
    def n_turni(self) -> int:
        return len(self._utenti)

    def storia_prima(self, k: int) -> list[dict]:
        """La storia originale prima del turno k (k parte da 1)."""
        return [dict(m) for m in self.messaggi[: self._utenti[k - 1]]]

    def utente(self, k: int) -> dict:
        return dict(self.messaggi[self._utenti[k - 1]])

    def risposta_originale(self, k: int) -> str:
        fine = self._utenti[k] if k < self.n_turni else len(self.messaggi)
        return "\n\n".join(str(m.get("content") or "") for m in self.messaggi[self._utenti[k - 1] + 1: fine]
                           if m.get("role") == "assistant")


def carica_conversazioni(percorso: Path, sha_atteso: str | None = None) -> list[Conversazione]:
    if not percorso.exists():
        sys.exit(f"file delle conversazioni non trovato: {percorso}")
    if sha_atteso:
        reale = sha256_file(percorso)
        if reale != sha_atteso:
            sys.exit(f"le conversazioni non sono quelle congelate: sha256 {reale[:12]}… invece di {sha_atteso[:12]}…")
    conversazioni, visti = [], set()
    with open(percorso, encoding="utf-8") as fh:
        for n, riga in enumerate(fh, 1):
            if not riga.strip():
                continue
            try:
                r = json.loads(riga)
            except json.JSONDecodeError as exc:
                sys.exit(f"{percorso}, riga {n}: JSON non valido ({exc})")
            if r.get("id") in visti:
                sys.exit(f"{percorso}, riga {n}: id ripetuto {r.get('id')}")
            visti.add(r.get("id"))
            messaggi = [{"role": str(m.get("role")), "content": m.get("content") if isinstance(m.get("content"), str)
                         else json.dumps(m.get("content"), ensure_ascii=False)} for m in r.get("messages") or []]
            c = Conversazione(id=str(r["id"]), fonte=str(r.get("fonte") or ""), messaggi=messaggi,
                              meta={k: r.get(k) for k in ("turni", "token_prompt", "fascia_turni", "fascia_token")})
            if c.n_turni == 0:
                sys.exit(f"{percorso}, riga {n}: conversazione {c.id} senza messaggi dell'utente")
            conversazioni.append(c)
    return conversazioni


def _kwargs_da_interruttori(interruttori: list[str]) -> dict:
    """["enable_thinking=false", "thinking_budget=0"] -> {"enable_thinking": False, "thinking_budget": 0}"""
    out = {}
    for voce in interruttori or []:
        nome, _, valore = str(voce).partition("=")
        v = valore.strip().lower()
        out[nome.strip()] = False if v == "false" else True if v == "true" else int(v) if v.lstrip("-").isdigit() else valore
    return out


def carica_modelli(percorso_candidati: Path, percorso_selezione: Path) -> list[dict]:
    """I modelli della lista congelata di download_models.py, controllati 1:1 contro selezione.json.

    Restituisce, nell'ordine della lista: nome breve, repository originale, piano di download e parametri del
    chat template che spengono il reasoning.
    """
    try:
        cand = json.loads(percorso_candidati.read_text(encoding="utf-8"))
        sel = json.loads(percorso_selezione.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.exit(f"lista dei modelli o selezione non leggibile: {exc}")
    sel_modelli = sel.get("modelli") or {}
    modelli, problemi = [], []
    for c in cand.get("candidati") or []:
        s = c.get("scarica") or {}
        nome = s.get("nome")
        voce = sel_modelli.get(nome)
        if voce is None:
            problemi.append(f"{nome}: assente in selezione.json")
            continue
        if voce.get("repo") != c.get("id"):
            problemi.append(f"{nome}: repository diverso ({voce.get('repo')} / {c.get('id')})")
        if voce.get("revision") != c.get("commit"):
            problemi.append(f"{nome}: commit del modello originale diverso")
        kwargs_sel = voce.get("template_kwargs") or {}
        kwargs_cand = _kwargs_da_interruttori((c.get("reasoning") or {}).get("interruttori"))
        if kwargs_sel != kwargs_cand:
            problemi.append(f"{nome}: interruttori del reasoning diversi ({kwargs_sel} / {kwargs_cand})")
        modelli.append({"nome": nome, "repo_originale": c.get("id"), "commit_originale": c.get("commit"),
                        "gguf_repo": s.get("repo"), "gguf_etichetta": s.get("etichetta"),
                        "gguf_commit": s.get("commit"), "chat_template_kwargs": kwargs_sel})
    nomi = {m["nome"] for m in modelli}
    problemi += [f"{n}: presente in selezione.json ma non nella lista dei modelli" for n in sel_modelli if n not in nomi]
    if problemi:
        sys.exit("lista dei modelli e selezione delle conversazioni non corrispondono:\n  " + "\n  ".join(problemi))
    return modelli
