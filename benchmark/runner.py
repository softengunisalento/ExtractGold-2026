#!/usr/bin/env python3
"""Runner del benchmark (PC di controllo): replay a storia fissa delle conversazioni con le strategie S0–S7.

    python runner.py [--config config.toml] [--misura http://<ip>:<porta>] [--modelli a,b] [--solo-verifiche]

Per ogni modello della lista congelata: lo fa caricare alla macchina di misura (FAIL se non entra in memoria),
fa il riscaldamento, misura il consumo a riposo, calcola il turno di attivazione (trigger) di ogni conversazione
con il tokenizer del modello, poi, conversazione per conversazione e strategia per strategia (in ordine mescolato
con seed fisso), turno per turno: costruisce il contesto con la strategia, lo manda alla macchina di misura e
registra token, tempi ed energia. La risposta generata si salva ma non entra nella storia: il turno successivo
riparte dalla storia originale.

Ripresa: rilanciato con la stessa configurazione, salta le unità (modello, conversazione, strategia) già concluse.
Usa solo la libreria standard di Python (3.11 o superiore).
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import configurazione
import dati
import strategie
from client_misura import ClientMisura, ErroreServizio
from risultati import Registro, adesso, chiave_unita

COMPONENTI = ("gpu", "cpu", "ram", "totale")


def energia_netta(lorda: dict | None, potenze: dict | None) -> dict | None:
    """Energia netta = lorda meno (potenza media a riposo x durata della finestra), per componente."""
    if not lorda:
        return None
    out = {}
    for finestra, e in lorda.items():
        if e is None:
            out[finestra] = None
            continue
        durata = e.get("durata_s") or 0
        out[finestra] = {}
        for c in COMPONENTI:
            v, p = e.get(f"{c}_wh"), (potenze or {}).get(f"{c}_w")
            out[finestra][f"{c}_wh"] = None if v is None or p is None else v - p * durata / 3600
    return out


class Servizi:
    """Le operazioni che le strategie chiedono alla macchina di misura, registrate con chiavi stabili."""

    def __init__(self, client: ClientMisura, chiavi: dict, potenze: dict | None):
        self.client = client
        self.chiavi = chiavi  # esecuzione, sessione, modello, conversazione, fonte, strategia, turno_trigger
        self.potenze = potenze
        self.turno: int | None = None
        self.pre_trigger = False
        self.indice = 0
        self.richieste: list[dict] = []
        self.testi: list[dict] = []
        self.intermedi: list[dict] = []

    def _base(self, tipo: str) -> dict:
        self.indice += 1
        return {**self.chiavi, "turno": self.turno, "pre_trigger": self.pre_trigger, "tipo": tipo, "indice": self.indice}

    def _etichetta(self, tipo: str) -> str:
        c = self.chiavi
        return f"{c['modello']}|{c['conversazione']}|{c['strategia']}|{self.turno}|{tipo}"

    def chiedi(self, messaggi: list[dict], tipo: str) -> str | None:
        r = self.client.chat(messaggi, misura=True, etichetta=self._etichetta(tipo))
        usage, tim = r.get("usage") or {}, r.get("timings") or {}
        riga = {**self._base(tipo), "esito": r["esito"], "errore": r.get("errore"), "n_messaggi": len(messaggi),
                "finish_reason": r.get("finish_reason"), "fine_finestra": r.get("fine_finestra"),
                "token_prompt": usage.get("prompt_tokens"), "token_valutati": tim.get("prompt_n"),
                "token_in_cache": tim.get("cache_n"), "token_generati": usage.get("completion_tokens"),
                "prefill_ms": tim.get("prompt_ms"), "decode_ms": tim.get("predicted_ms"),
                "ttft_ms": r.get("ttft_ms"), "wall_ms": r.get("wall_ms"), "inizio_utc": r.get("inizio_utc"),
                "metodo_finestre": r.get("metodo_finestre"), "energia_lorda": r.get("energia"),
                "energia_netta": energia_netta(r.get("energia"), self.potenze)}
        self.richieste.append(riga)
        self.testi.append({**{k: riga[k] for k in ("esecuzione", "modello", "conversazione", "strategia", "turno",
                                                   "tipo", "indice")},
                           "testo": r.get("testo"), "ragionamento": r.get("ragionamento") or None})
        return r.get("testo") if r["esito"] == "ok" else None

    def _overhead(self, tipo: str, r: dict, extra: dict) -> None:
        self.richieste.append({**self._base(tipo), "esito": "ok", **extra, "wall_ms": r.get("wall_ms"),
                               "inizio_utc": r.get("inizio_utc"), "metodo_finestre": r.get("metodo_finestre"),
                               "energia_lorda": r.get("energia"),
                               "energia_netta": energia_netta(r.get("energia"), self.potenze)})

    def comprimi(self, metodo: str, testi: list[str], quota: float) -> list[str]:
        r = self.client.comprimi(metodo, testi, quota, misura=True, etichetta=self._etichetta("compressione"))
        self._overhead("compressione", r, {"metodo": metodo, "n_testi": len(testi), "errori": r.get("errori") or None,
                                           "caratteri_prima": sum(map(len, testi)),
                                           "caratteri_dopo": sum(map(len, r["testi"]))})
        return r["testi"]

    def embedding(self, voci: list[dict]):
        r = self.client.embedding(voci, misura=True, etichetta=self._etichetta("embedding"))
        self._overhead("embedding", r, {"n_voci": len(voci), "n_pezzi": sum(len(v) for v in r["vettori"])})
        return r["vettori"]

    def intermedio(self, tipo: str, contenuto) -> None:
        c = self.chiavi
        self.intermedi.append({"esecuzione": c["esecuzione"], "modello": c["modello"], "conversazione": c["conversazione"],
                               "strategia": c["strategia"], "turno": self.turno, "tipo": tipo, "contenuto": contenuto})


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

def calcola_trigger(client: ClientMisura, conversazioni: list[dati.Conversazione], soglia: int,
                    reg: Registro, modello: str, turni_recenti: int) -> dict:
    """Per ogni conversazione, il primo turno k in cui il contesto di S0 (storia originale fino a k-1 più il
    messaggio dell'utente k) supera la soglia, contato con il tokenizer del modello tramite llama-server."""
    salvato = reg.leggi_json(f"trigger/{modello}.json")
    if salvato and salvato.get("soglia") == soglia and set(salvato["conversazioni"]) >= {c.id for c in conversazioni}:
        return salvato
    print(f"  calcolo del trigger ({soglia} token) per {len(conversazioni)} conversazioni...")
    per_conv, metodi = {}, set()
    for i, c in enumerate(conversazioni, 1):
        trigger, token = None, None
        for k in range(1, c.n_turni + 1):
            r = client.conta_token(c.storia_prima(k) + [c.utente(k)])
            metodi.add(r["metodo"])
            if r["token"] > soglia:
                trigger, token = k, r["token"]
                break
        per_conv[c.id] = {"turno_trigger": trigger, "token_al_trigger": token, "turni": c.n_turni}
        if i % 50 == 0:
            print(f"    {i}/{len(conversazioni)}")
    risultato = {"modello": modello, "soglia": soglia, "metodo": sorted(metodi), "calcolato": adesso(),
                 "conversazioni": per_conv}
    reg.scrivi_json(f"trigger/{modello}.json", risultato)
    reg.scrivi_json(f"verifiche/{modello}.json", verifiche(risultato, turni_recenti))
    return risultato


def verifiche(trig: dict, turni_recenti: int) -> dict:
    voci = list(trig["conversazioni"].values())
    con = [v for v in voci if v["turno_trigger"]]
    dopo = [v["turni"] - v["turno_trigger"] + 1 for v in con]
    return {
        "modello": trig["modello"], "soglia": trig["soglia"], "conversazioni": len(voci),
        "con_trigger": len(con), "senza_trigger": len(voci) - len(con),
        "turni_misurati_dopo_il_trigger": ({"min": min(dopo), "mediana": statistics.median(dopo), "max": max(dopo)}
                                          if dopo else None),
        "trigger_al_turno_1": sum(v["turno_trigger"] == 1 for v in con),
        "trigger_con_al_massimo_turni_recenti_prima": sum(v["turno_trigger"] - 1 <= turni_recenti for v in con),
    }


# ---------------------------------------------------------------------------
# Esecuzione
# ---------------------------------------------------------------------------

def esegui_unita(client, reg, chiavi, conv: dati.Conversazione, strategia: str, trigger: int | None,
                 parametri: dict, potenze: dict | None) -> None:
    chiave = chiave_unita(chiavi["modello"], conv.id, strategia)
    if strategia != "S0" and trigger is None:
        reg.scrivi_unita(chiave, [], [], [], {"esito": "nessun_trigger"})
        return
    serv = Servizi(client, {**chiavi, "conversazione": conv.id, "fonte": conv.fonte, "strategia": strategia,
                            "turno_trigger": trigger}, potenze)
    strat = strategie.crea(strategia, parametri, serv)
    inizio = 1 if strategia == "S0" else trigger
    for k in range(inizio, conv.n_turni + 1):
        attiva = trigger is not None and k >= trigger
        serv.turno, serv.pre_trigger = k, not attiva
        contesto = strat.contesto(conv.storia_prima(k), conv.utente(k), attiva)
        serv.chiedi(contesto, "turno")
    esiti = [r["esito"] for r in serv.richieste]
    reg.scrivi_unita(chiave, serv.richieste, serv.testi, serv.intermedi,
                     {"esito": "ok", "richieste": len(esiti), "contesto_superato": esiti.count("contesto_superato"),
                      "errori": esiti.count("errore")})


def main() -> int:
    ap = argparse.ArgumentParser(description="Runner del benchmark delle strategie di gestione del contesto.")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    ap.add_argument("--misura", help="indirizzo del servizio di misura (sostituisce rete.indirizzo_misura)")
    ap.add_argument("--modelli", help="solo questi modelli (nomi brevi, separati da virgola)")
    ap.add_argument("--solo-verifiche", action="store_true",
                    help="carica i modelli e calcola solo trigger e verifiche, senza misure")
    ap.add_argument("--max-conversazioni", type=int, default=0, help="solo le prime N conversazioni (prove)")
    args = ap.parse_args()

    cfg = configurazione.carica(args.config)
    inp = cfg["input"]
    p_conv, p_sel, p_mod = (cfg.percorso(inp[k]) for k in ("conversazioni", "selezione", "modelli"))
    conversazioni = dati.carica_conversazioni(p_conv, inp.get("sha256_conversazioni") or None)
    if args.max_conversazioni:
        conversazioni = conversazioni[: args.max_conversazioni]
    modelli = dati.carica_modelli(p_mod, p_sel)
    if args.modelli:
        voluti = [x.strip() for x in args.modelli.split(",") if x.strip()]
        ignoti = [v for v in voluti if v not in {m["nome"] for m in modelli}]
        if ignoti:
            sys.exit(f"modelli sconosciuti: {ignoti}")
        modelli = [m for m in modelli if m["nome"] in voluti]
    print(f"{len(conversazioni)} conversazioni, {len(modelli)} modelli, strategie {cfg['strategie']['elenco']}")

    client = ClientMisura(args.misura or cfg["rete"]["indirizzo_misura"], cfg["rete"].get("chiave", ""))
    try:
        stato = client.stato()
    except ErroreServizio as exc:
        sys.exit(f"servizio di misura non raggiungibile: {exc}")
    if stato.get("config_sha256") != cfg["_sha256"]:
        sys.exit("la configurazione della macchina di misura è diversa da quella del runner: usate lo stesso config.toml")
    if stato.get("llama_commit_compilato") != cfg["llama"]["commit"]:
        sys.exit(f"llama.cpp sulla macchina di misura non è al commit congelato ({stato.get('llama_commit_compilato')})")

    nome = cfg["esecuzione"]["nome"]
    meta = {"nome": nome, "config_benchmark_sha256": cfg["_sha256_benchmark"],
            "input": {"conversazioni": dati.sha256_file(p_conv), "selezione": dati.sha256_file(p_sel),
                      "modelli": dati.sha256_file(p_mod), "max_conversazioni": args.max_conversazioni or None}}
    reg = Registro(cfg.percorso(cfg["esecuzione"]["cartella_risultati"]) / nome, meta)
    sessione = adesso()
    reg.scrivi("sessioni.jsonl", {"sessione": sessione, "stato_servizio": stato, "python": sys.version,
                                  "modelli": [m["nome"] for m in modelli], "solo_verifiche": args.solo_verifiche})
    par = dict(cfg["strategie"])
    elenco = list(par["elenco"])
    seed = cfg["esecuzione"]["seed"]

    for m in modelli:
        nm = m["nome"]
        chiavi_modello = [chiave_unita(nm, c.id, s) for c in conversazioni for s in elenco]
        if not args.solo_verifiche and all(reg.conclusa(k) for k in chiavi_modello):
            print(f"\n[{nm}] già concluso")
            continue
        print(f"\n[{nm}] caricamento...")
        r = client.carica_modello(nm, m["chat_template_kwargs"])
        reg.scrivi("modelli.jsonl", {"sessione": sessione, "evento": "caricamento", "modello": nm, **r})
        if r.get("esito") != "OK":
            print(f"  FAIL: {r.get('motivo')} (log: {r.get('log')})")
            continue
        print(f"  pronto in {r.get('secondi_caricamento')} s; riscaldamento ({cfg['misura']['riscaldamento_richieste']} richieste)")
        for _ in range(int(cfg["misura"]["riscaldamento_richieste"])):
            client.chat([{"role": "user", "content": cfg["misura"]["riscaldamento_messaggio"]}], misura=False,
                        etichetta=f"{nm}|riscaldamento")
        trig = calcola_trigger(client, conversazioni, int(par["soglia_token"]), reg, nm, int(par["turni_recenti"]))
        v = verifiche(trig, int(par["turni_recenti"]))
        print(f"  trigger: {v['con_trigger']}/{v['conversazioni']} conversazioni; turni misurati dopo il trigger "
              f"{v['turni_misurati_dopo_il_trigger']}")
        if args.solo_verifiche:
            client.scarica_modello()
            continue
        print(f"  consumo a riposo ({cfg['misura']['riposo_secondi']} s)...")
        rip = client.riposo(float(cfg["misura"]["riposo_secondi"]))
        reg.scrivi("modelli.jsonl", {"sessione": sessione, "evento": "riposo", "modello": nm, **rip})
        potenze = rip.get("potenza_media")
        chiavi = {"esecuzione": nome, "sessione": sessione, "modello": nm}
        for i, c in enumerate(conversazioni, 1):
            ordine = list(elenco)
            random.Random(f"{seed}|{nm}|{c.id}").shuffle(ordine)
            for s in ordine:
                if reg.conclusa(chiave_unita(nm, c.id, s)):
                    continue
                esegui_unita(client, reg, chiavi, c, s, trig["conversazioni"][c.id]["turno_trigger"], par, potenze)
            print(f"  {i}/{len(conversazioni)} conversazioni")
        client.scarica_modello()
    print(f"\nRisultati in {reg.cartella}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ErroreServizio as exc:
        sys.exit(f"\nInterrotto: {exc}\nRilancia lo stesso comando per riprendere dalle unità non concluse.")
