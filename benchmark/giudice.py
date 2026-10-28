#!/usr/bin/env python3
"""Giudice LLM (PC di controllo): giudica le risposte del benchmark con il modello giudice sulla macchina di misura.

    python giudice.py [--config config.toml] [--misura http://<ip>:<porta>] [--modelli a,b]

Si giudicano solo i turni dal trigger in poi (prima del trigger tutte le strategie coincidono con S0).
  Variante A (voti relativi), per S1–S7: storia originale fino al turno precedente, messaggio dell'utente,
             risposta S0 come «Reference answer» e risposta della tecnica.
  Variante B (voti assoluti), per S0–S7: storia, messaggio e una sola risposta, senza riferimento.
Controlli dell'affidabilità (stessi testi, il giudice non sa di essere testato), su campioni scelti con seed:
  stabilità (A e B ripetuti con un altro seed), identità (A con la risposta S0 al posto della tecnica),
  degrado (A e B con la risposta S0 di un'altra conversazione, a un turno dopo il trigger).

La misura dell'energia è spenta. I giudizi si salvano uno per uno in risultati/<nome>/giudizi/<impronta del
giudice>/giudizi.jsonl: rilanciato, il giudice salta quelli già fatti. Un modello si giudica solo quando il suo
benchmark è concluso. Avanzamento: una riga con la percentuale dei giudizi del modello (quelli già salvati
contano come fatti). Usa solo la libreria standard di Python (3.11 o superiore).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import configurazione
import dati
from avanzamento import Avanzamento
from client_misura import ClientMisura, ErroreServizio
from risultati import adesso, cartella_esecuzione, chiave_unita

VOTI = {"A": ("relative_correctness", "relative_coherence"),
        "B": ("absolute_correctness", "absolute_coherence")}
RUOLI = {"user": "User", "assistant": "Assistant"}


# ---------------------------------------------------------------------------
# Strumenti
# ---------------------------------------------------------------------------

def schema(tipo: str) -> dict:
    """Schema JSON dell'uscita: prima la motivazione, poi i due voti interi da 1 a 10."""
    props = {"motivation": {"type": "string"}}
    for v in VOTI[tipo]:
        props[v] = {"type": "integer", "minimum": 1, "maximum": 10}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def frazione(*parti) -> float:
    """Numero in [0, 1) ricavato dalle parti: stesso input, stesso numero, in qualunque ordine si lavori."""
    return int(hashlib.sha256("|".join(map(str, parti)).encode()).hexdigest()[:13], 16) / 16 ** 13


def seed_di(*parti) -> int:
    return int(hashlib.sha256("|".join(map(str, parti)).encode()).hexdigest()[:8], 16) % 2 ** 31


def formatta_storia(messaggi: list[dict], vuota: str) -> str:
    righe = [f"{RUOLI.get(m.get('role'), str(m.get('role')).capitalize())}: {m.get('content') or ''}" for m in messaggi]
    return "\n\n".join(righe) if righe else vuota


def riempi(modello: str, valori: dict) -> str:
    """Sostituisce i segnaposto in un solo passaggio: il testo inserito non viene mai riletto."""
    return configurazione.SEGNAPOSTO.sub(lambda m: valori[m.group(1)], modello)


def leggi_voti(testo: str, tipo: str) -> tuple[dict | None, str | None, str | None]:
    """(voti, motivazione, problema). I voti devono essere interi tra 1 e 10."""
    t = (testo or "").strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < a:
        return None, None, "nessun oggetto JSON nella risposta"
    try:
        obj = json.loads(t[a:b + 1])
    except ValueError as exc:
        return None, None, f"JSON non valido: {exc}"
    if not isinstance(obj, dict):
        return None, None, "il JSON non è un oggetto"
    voti = {}
    for v in VOTI[tipo]:
        x = obj.get(v)
        if isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= 10:
            return None, None, f"voto {v} mancante o fuori scala: {x!r}"
        voti[v] = x
    mot = obj.get("motivation")
    return voti, (mot if isinstance(mot, str) else None), None


def leggi_jsonl(p: Path):
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        for riga in fh:
            if riga.strip():
                yield json.loads(riga)


# ---------------------------------------------------------------------------
# Risposte del benchmark
# ---------------------------------------------------------------------------

def risposte_del_modello(res: Path, modello: str) -> dict:
    """(conversazione, strategia, turno) -> esito, testo e flag della risposta al turno."""
    out = {}
    for r in leggi_jsonl(res / "richieste.jsonl"):
        if r.get("modello") == modello and r.get("tipo") == "turno":
            out[(r["conversazione"], r["strategia"], r["turno"])] = {
                "esito": r.get("esito"), "fine_finestra": bool(r.get("fine_finestra")), "indice": r.get("indice"),
                "token_generati": r.get("token_generati"), "testo": None}
    for t in leggi_jsonl(res / "testi.jsonl"):
        if t.get("modello") == modello and t.get("tipo") == "turno":
            v = out.get((t["conversazione"], t["strategia"], t["turno"]))
            if v is not None and v["indice"] == t.get("indice"):
                v["testo"] = t.get("testo") or ""
    return out


def lavori_del_modello(cfg, res: Path, modello: str, conversazioni: list[dati.Conversazione],
                       completate: set) -> tuple[list[dict], str | None]:
    """L'elenco ordinato dei giudizi di un modello (base, controlli, non giudicabili), o il motivo per saltarlo."""
    elenco = list(cfg["strategie"]["elenco"])
    if not any(chiave_unita(modello, c.id, s) in completate for c in conversazioni for s in elenco):
        return [], "nessun risultato del benchmark (modello non caricato o non ancora eseguito)"
    if not all(chiave_unita(modello, c.id, s) in completate for c in conversazioni for s in elenco):
        return [], "benchmark non concluso per questo modello"
    p_trig = res / "trigger" / f"{modello}.json"
    if not p_trig.exists():
        return [], "trigger non calcolato"
    trig = json.loads(p_trig.read_text(encoding="utf-8"))["conversazioni"]
    risp = risposte_del_modello(res, modello)
    g = cfg["giudice"]
    gseed = g["seed"]
    # risposte S0 dopo il trigger, per il controllo di degrado (da un'altra conversazione)
    pool = [(c.id, k) for c in conversazioni if trig[c.id]["turno_trigger"]
            for k in range(trig[c.id]["turno_trigger"], c.n_turni + 1)
            if (risp.get((c.id, "S0", k)) or {}).get("esito") == "ok"]
    lavori = []
    for c in conversazioni:
        k0 = trig[c.id]["turno_trigger"]
        if not k0:
            continue
        per_tipo = {"A": [], "B": []}
        for k in range(k0, c.n_turni + 1):
            s0 = risp.get((c.id, "S0", k)) or {}
            for s in elenco:
                r = risp.get((c.id, s, k)) or {}
                base = {"modello": modello, "conversazione": c.id, "fonte": c.fonte, "turno": k, "turno_trigger": k0,
                        "distanza": k - k0, "strategia": s,
                        "risposta_fine_finestra": r.get("fine_finestra"),
                        "lunghezza_risposta": len(r.get("testo") or ""), "token_risposta": r.get("token_generati")}
                for tipo in ("A", "B"):
                    if tipo == "A" and s == "S0":
                        continue
                    lid = f"{modello}|{c.id}|{k}|{s}|{tipo}"
                    lav = {**base, "id": lid + "|base", "tipo": tipo, "controllo": None,
                           "riferimento_fine_finestra": s0.get("fine_finestra") if tipo == "A" else None}
                    if r.get("esito") != "ok":
                        per_tipo[tipo].append({**lav, "non_giudicabile": "risposta_mancante",
                                               "esito_risposta": r.get("esito")})
                        continue
                    if tipo == "A" and s0.get("esito") != "ok":
                        per_tipo[tipo].append({**lav, "non_giudicabile": "tecnica_senza_S0",
                                               "esito_risposta": s0.get("esito")})
                        continue
                    lav.update(storia=(c.id, k), risposta=r["testo"], riferimento=s0.get("testo") if tipo == "A" else None)
                    per_tipo[tipo].append(lav)
                    # controlli, su campioni scelti con seed
                    if frazione(gseed, "stabilita", lav["id"]) < g["campione_stabilita"]:
                        per_tipo[tipo].append({**lav, "id": lid + "|stabilita", "controllo": "stabilita",
                                               "originale": lav["id"]})
                    if tipo == "A" and frazione(gseed, "identita", lav["id"]) < g["campione_identita"]:
                        per_tipo[tipo].append({**lav, "id": lid + "|identita", "controllo": "identita",
                                               "risposta": s0["testo"], "risposta_fine_finestra": s0.get("fine_finestra"),
                                               "lunghezza_risposta": len(s0["testo"] or ""), "originale": lav["id"]})
                    if frazione(gseed, "degrado", lav["id"]) < g["campione_degrado"]:
                        altri = [x for x in pool if x[0] != c.id]
                        if altri:
                            c2, k2 = random.Random(f"{gseed}|degrado|{lav['id']}").choice(altri)
                            t2 = risp[(c2, "S0", k2)]
                            per_tipo[tipo].append({**lav, "id": lid + "|degrado", "controllo": "degrado",
                                                   "risposta": t2["testo"], "risposta_fine_finestra": t2["fine_finestra"],
                                                   "lunghezza_risposta": len(t2["testo"] or ""),
                                                   "degrado_da": {"conversazione": c2, "turno": k2},
                                                   "originale": lav["id"]})
        # prima tutti i giudizi A della conversazione, poi tutti i B: la storia cresce turno dopo turno e
        # llama-server riusa dalla cache la parte iniziale del prompt (istruzioni + storia comune)
        lavori += per_tipo["A"] + per_tipo["B"]
    return lavori, None


# ---------------------------------------------------------------------------
# Archivio dei giudizi
# ---------------------------------------------------------------------------

class Archivio:
    def __init__(self, cartella: Path, cfg):
        self.cartella = cartella
        cartella.mkdir(parents=True, exist_ok=True)
        p = cartella / "giudice.json"
        if not p.exists():
            info = {"impronta_giudice": cfg["_sha256_giudice"], "creato": adesso(), "giudice": cfg["giudice"]}
            p.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.fatti = {r["id"] for r in leggi_jsonl(cartella / "giudizi.jsonl")}

    def scrivi(self, nome: str, riga: dict) -> None:
        with open(self.cartella / nome, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(riga, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def cartella_giudizi(cfg, res: Path) -> Path:
    return res / "giudizi" / cfg["_sha256_giudice"][:12]


# ---------------------------------------------------------------------------
# Esecuzione
# ---------------------------------------------------------------------------

class Giudice:
    def __init__(self, cfg, client: ClientMisura, archivio: Archivio, esecuzione: str):
        self.cfg, self.client, self.arch, self.esecuzione = cfg, client, archivio, esecuzione
        self.g = cfg["giudice"]
        self.caricato = False

    def carica(self) -> None:
        print(f"  caricamento del giudice {self.g['nome']} (contesto {self.g['contesto']})...")
        r = self.client.carica_giudice()
        self.arch.scrivi("sessioni.jsonl", {"evento": "caricamento", "ora": adesso(), **r})
        if r.get("esito") != "OK":
            raise SystemExit(f"il giudice non si carica: {r.get('motivo')} (log: {r.get('log')})")
        print(f"  pronto in {r.get('secondi_caricamento')} s")
        self.caricato = True
        self.primo = True

    def campionamento(self, seed: int) -> dict:
        g = self.g
        return {"temperature": g["temperatura"], "top_p": g["top_p"], "top_k": g["top_k"], "min_p": g["min_p"],
                "seed": seed}

    def prompt(self, lav: dict, conversazioni: dict) -> str:
        cid, k = lav["storia"]
        c = conversazioni[cid]
        valori = {"storia": formatta_storia(c.storia_prima(k), self.g["storia_vuota"]),
                  "messaggio": str(c.utente(k).get("content") or ""), "risposta": lav["risposta"] or "",
                  "riferimento": lav.get("riferimento") or ""}
        return riempi(self.g["prompt_relativo"] if lav["tipo"] == "A" else self.g["prompt_assoluto"], valori)

    def giudica(self, lav: dict, conversazioni: dict) -> dict:
        chiavi = {k: lav.get(k) for k in ("id", "modello", "conversazione", "fonte", "turno", "turno_trigger",
                                          "distanza", "strategia", "tipo", "controllo", "originale", "degrado_da",
                                          "risposta_fine_finestra", "riferimento_fine_finestra",
                                          "lunghezza_risposta", "token_risposta")}
        riga = {"esecuzione": self.esecuzione, "impronta_giudice": self.cfg["_sha256_giudice"], **chiavi}
        if lav.get("non_giudicabile"):
            return {**riga, "esito": "non_giudicabile", "motivo": lav["non_giudicabile"],
                    "esito_risposta": lav.get("esito_risposta"), "ora": adesso()}
        if not self.caricato:
            self.carica()
        messaggi = [{"role": "user", "content": self.prompt(lav, conversazioni)}]
        problemi = []
        r, voti, motivazione, seed = {}, None, None, None
        for tentativo in range(1 + int(self.g["tentativi_extra"])):
            seed = seed_di(self.g["seed"], lav["id"], tentativo)
            r = self.client.chat(messaggi, misura=False, etichetta=f"giudice|{lav['id']}|{tentativo}",
                                 campionamento=self.campionamento(seed),
                                 schema=schema(lav["tipo"]) if self.g["schema_json"] else None,
                                 leggi_slot=self.primo)  # alla prima risposta dopo il caricamento
            if self.primo and r["esito"] == "ok":
                self.primo = False
                voluti = self.campionamento(seed)
                usati = r.get("parametri_slot") or {}
                diversi = {k: (voluti[k], usati.get(k)) for k in voluti
                           if usati.get(k) is None or abs(float(usati[k]) - float(voluti[k])) > 1e-4}
                self.arch.scrivi("sessioni.jsonl", {"evento": "parametri_dichiarati", "ora": adesso(),
                                                    "richiesti": voluti, "dichiarati": usati,
                                                    "coerenti": not diversi, "diversi": diversi})
            if r["esito"] == "contesto_superato":
                problemi.append("contesto del giudice superato")
                break
            if r["esito"] != "ok":
                problemi.append(f"errore: {r.get('errore')}")
                continue
            voti, motivazione, problema = leggi_voti(r.get("testo"), lav["tipo"])
            if voti:
                break
            problemi.append(problema)
        usage = r.get("usage") or {}
        esito = "ok" if voti else ("contesto_superato_giudice" if r.get("esito") == "contesto_superato" else "non_valido")
        return {**riga, "esito": esito, "voti": voti, "motivazione": motivazione, "testo": r.get("testo"),
                "ragionamento": r.get("ragionamento") or None, "tentativi": tentativo + 1, "problemi": problemi or None,
                "seed": seed, "finish_reason": r.get("finish_reason"), "token_prompt": usage.get("prompt_tokens"),
                "token_generati": usage.get("completion_tokens"), "token_in_cache": (r.get("timings") or {}).get("cache_n"),
                "wall_ms": r.get("wall_ms"), "ora": adesso()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Giudice LLM delle risposte del benchmark.")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    ap.add_argument("--misura", help="indirizzo del servizio di misura (sostituisce rete.indirizzo_misura)")
    ap.add_argument("--modelli", help="solo questi modelli (nomi brevi, separati da virgola)")
    args = ap.parse_args(argv)

    cfg = configurazione.carica(args.config)
    nome = cfg["esecuzione"]["nome"]
    res = cartella_esecuzione(cfg)
    p_meta = res / "esecuzione.json"
    if not p_meta.exists():
        sys.exit(f"nessun risultato del benchmark in {res}: lancia prima runner.py")
    meta = json.loads(p_meta.read_text(encoding="utf-8"))
    if meta.get("config_benchmark_sha256") != cfg["_sha256_benchmark"]:
        sys.exit("i risultati del benchmark sono stati prodotti con una configurazione diversa da questa")
    inp = cfg["input"]
    # le stesse conversazioni del benchmark: le prime N nell'ordine a giri
    conversazioni = dati.conversazioni_ordinate(cfg)[:(meta.get("input") or {}).get("conversazioni_del_run")]
    per_id = {c.id: c for c in conversazioni}
    modelli = [m["nome"] for m in dati.carica_modelli(cfg.percorso(inp["modelli"]), cfg.percorso(inp["selezione"]))]
    if args.modelli:
        voluti = [x.strip() for x in args.modelli.split(",") if x.strip()]
        ignoti = [v for v in voluti if v not in modelli]
        if ignoti:
            sys.exit(f"modelli sconosciuti: {ignoti}")
        modelli = [m for m in modelli if m in voluti]
    completate = {r["unita"] for r in leggi_jsonl(res / "completate.jsonl")}

    client = ClientMisura(args.misura or cfg["rete"]["indirizzo_misura"], cfg["rete"].get("chiave", ""))
    try:
        stato = client.stato()
    except ErroreServizio as exc:
        sys.exit(f"servizio di misura non raggiungibile: {exc}")
    if stato.get("config_sha256") != cfg["_sha256"]:
        sys.exit("la configurazione della macchina di misura è diversa da quella del PC di controllo: "
                 "usate lo stesso config.toml")

    arch = Archivio(cartella_giudizi(cfg, res), cfg)
    arch.scrivi("sessioni.jsonl", {"evento": "avvio", "ora": adesso(), "modelli": modelli,
                                   "stato_servizio": stato})
    giudice = Giudice(cfg, client, arch, nome)
    print(f"Giudice {cfg['giudice']['nome']}: giudizi in {arch.cartella}")
    for n_modello, m in enumerate(modelli, 1):
        lavori, motivo = lavori_del_modello(cfg, res, m, conversazioni, completate)
        if motivo:
            print(f"\n[{m}] saltato: {motivo}")
            continue
        da_fare = [x for x in lavori if x["id"] not in arch.fatti]
        n_base = sum(1 for x in lavori if not x["controllo"] and not x.get("non_giudicabile"))
        print(f"\n[{m}] {len(lavori)} voci ({n_base} giudizi base, "
              f"{sum(1 for x in lavori if x['controllo'])} controlli, "
              f"{sum(1 for x in lavori if x.get('non_giudicabile'))} non giudicabili); da fare: {len(da_fare)}")
        if any(not x.get("non_giudicabile") for x in da_fare) and not giudice.caricato:
            giudice.carica()  # prima della riga di avanzamento, così i messaggi di caricamento non la spezzano
        etichetta = f"  [{m}]" + (f" modello {n_modello}/{len(modelli)}" if len(modelli) > 1 else "")
        avanz = Avanzamento(etichetta, len(lavori), len(lavori) - len(da_fare), "giudizi")
        for lav in da_fare:
            riga = giudice.giudica(lav, per_id)
            arch.scrivi("giudizi.jsonl", riga)
            arch.fatti.add(lav["id"])
            avanz.avanza()
        avanz.chiudi()
    if giudice.caricato:
        client.scarica_modello()
    print(f"\nGiudizi in {arch.cartella / 'giudizi.jsonl'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ErroreServizio as exc:
        sys.exit(f"\nInterrotto: {exc}\nRilancia lo stesso comando per riprendere dai giudizi non fatti.")
