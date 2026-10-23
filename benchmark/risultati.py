"""Scrittura dei risultati e ripresa dopo un'interruzione.

Cartella di un'esecuzione: <cartella_risultati>/<nome>/
  esecuzione.json     configurazione (impronta), file di input (impronte), primo avvio
  sessioni.jsonl      un record per ogni avvio del runner (stato del servizio di misura, versioni)
  modelli.jsonl       caricamento di ogni modello (OK/FAIL), riscaldamento e consumo a riposo
  trigger/<modello>.json   turno di attivazione di ogni conversazione, contato con il tokenizer del modello
  verifiche/<modello>.json conteggi automatici sul trigger
  richieste.jsonl     una riga per richiesta misurata (chiavi stabili + token, tempi, energia lorda e netta)
  testi.jsonl         testo delle risposte, con le stesse chiavi
  intermedi.jsonl     testi intermedi delle strategie (gold prompt, riassunti, compressi, turni recuperati)
  completate.jsonl    unità (modello, conversazione, strategia) concluse: alla ripresa si saltano

Le righe di un'unità si scrivono tutte insieme a unità conclusa, poi la riga di completamento: un'unità
interrotta a metà non lascia righe e si rifà da capo.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def adesso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def chiave_unita(modello: str, conversazione: str, strategia: str) -> str:
    return f"{modello}|{conversazione}|{strategia}"


class Registro:
    def __init__(self, cartella: Path, meta: dict):
        self.cartella = cartella
        cartella.mkdir(parents=True, exist_ok=True)
        (cartella / "trigger").mkdir(exist_ok=True)
        (cartella / "verifiche").mkdir(exist_ok=True)
        p = cartella / "esecuzione.json"
        if p.exists():
            vecchio = json.loads(p.read_text(encoding="utf-8"))
            diversi = [k for k in ("config_sha256", "input") if vecchio.get(k) != meta.get(k)]
            if diversi:
                sys.exit(f"la cartella {cartella} contiene un'esecuzione con {', '.join(diversi)} diversi: "
                         "per non mescolare i risultati usa un altro esecuzione.nome nella configurazione")
            self.meta = vecchio
        else:
            self.meta = {**meta, "creata": adesso()}
            p.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.completate = set()
        pc = cartella / "completate.jsonl"
        if pc.exists():
            for riga in pc.read_text(encoding="utf-8").splitlines():
                if riga.strip():
                    self.completate.add(json.loads(riga)["unita"])

    def _accoda(self, nome: str, record_list: list[dict]) -> None:
        if not record_list:
            return
        with open(self.cartella / nome, "a", encoding="utf-8") as fh:
            for r in record_list:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def scrivi(self, nome: str, record: dict) -> None:
        self._accoda(nome, [record])

    def scrivi_json(self, relativo: str, dati) -> None:
        p = self.cartella / relativo
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(dati, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(p)

    def leggi_json(self, relativo: str):
        p = self.cartella / relativo
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def conclusa(self, chiave: str) -> bool:
        return chiave in self.completate

    def scrivi_unita(self, chiave: str, richieste: list[dict], testi: list[dict], intermedi: list[dict],
                     esito: dict) -> None:
        self._accoda("richieste.jsonl", richieste)
        self._accoda("testi.jsonl", testi)
        self._accoda("intermedi.jsonl", intermedi)
        self._accoda("completate.jsonl", [{"unita": chiave, "conclusa": adesso(), **esito}])
        self.completate.add(chiave)
