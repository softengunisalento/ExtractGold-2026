#!/usr/bin/env python3
"""Prova simulata del benchmark, senza GPU né rete: python test/prova_simulata.py

Parte A: le strategie, controllate sui contesti esatti dell'esempio del documento (trigger al turno 6).
Parte B: servizio di misura vero (con llama-server, CodeCarbon e modelli ausiliari FINTI) e runner vero, su
conversazioni inventate: caricamento con FAIL, riscaldamento, consumo a riposo, trigger, tutte le strategie,
contesto superato, energia lorda e netta, ripresa dopo l'interruzione.
Parte C: giudice vero sullo stesso servizio (il llama-server finto fa anche da giudice): varianti A e B, turni
giudicati, non giudicabili, controlli, parametri, nuovi tentativi, ripresa.
Parte D: elaborazione vera (tabelle, grafici, rapporto Word). Parte E: comando unico catena.py.
"""
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

QUI = Path(__file__).resolve().parent
BENCH = QUI.parent
sys.path.insert(0, str(BENCH))
import strategie  # noqa: E402
from dati import Conversazione  # noqa: E402

ERRORI = []


def controlla(cond, descrizione):
    print(("  OK   " if cond else "  ERR  ") + descrizione)
    if not cond:
        ERRORI.append(descrizione)


# ---------------------------------------------------------------------------
# Parte A: strategie
# ---------------------------------------------------------------------------

PAR = {"turni_recenti": 3, "s3_ogni_turni": 3, "s7_recuperati": 2, "quota_token_tenuti": 0.5,
       "prefisso_riassunto": "Here is a summary of the conversation to date:",
       "prompt_ge": "PROMPT_GE", "prompt_s3_primo": "PROMPT_S3_1", "prompt_s3_aggiornamento": "PROMPT_S3_N",
       "formato_s3_aggiornamento": "{prompt}\n\nCurrent summary:\n{riassunto}\n\nNew messages:\n{messaggi}",
       "prompt_s4": "PROMPT_S4"}
PREF = PAR["prefisso_riassunto"]


class ServiziFinti:
    def __init__(self):
        self.richieste, self.compressi, self.embedding_voci, self.intermedi = [], [], [], []
        self.n = 0

    def chiedi(self, messaggi, tipo):
        self.n += 1
        self.richieste.append((tipo, messaggi))
        return f"{tipo.upper()}#{self.n}"

    def comprimi(self, metodo, testi, quota):
        self.compressi.append((metodo, list(testi)))
        return [f"[{metodo}:{t[:10]}]" for t in testi]

    def embedding(self, voci):
        self.embedding_voci.append(voci)
        # il turno che contiene "alpha" è il più simile alla query
        return [[[1.0, 0.0]] if "alpha" in v["testo"] else [[0.0, 1.0]] for v in voci[:-1]] + [[[1.0, 0.0]]]

    def intermedio(self, tipo, contenuto):
        self.intermedi.append((tipo, contenuto))


def conv_esempio(n=10):
    m = []
    for k in range(1, n + 1):
        m.append({"role": "user", "content": f"U{k}" + (" alpha" if k == 2 else "")})
        m.append({"role": "assistant", "content": f"A{k}" + (" ```code k``` and `x`" if k == 1 else "")})
    return Conversazione(id="c", fonte="prova", messaggi=m)


def esegui(strat_nome, conv, trigger, fino_a):
    serv = ServiziFinti()
    s = strategie.crea(strat_nome, PAR, serv)
    contesti = {}
    for k in range(trigger, fino_a + 1):
        contesti[k] = s.contesto(conv.storia_prima(k), conv.utente(k), k >= trigger)
    return serv, contesti


def testi(contesto):
    return [(m["role"][0], m["content"]) for m in contesto]


def parte_a():
    print("Parte A: strategie (trigger al turno 6)")
    c = conv_esempio()
    # S0
    _, ctx = esegui("S0", c, 6, 6)
    controlla(len(ctx[6]) == 11 and ctx[6][-1]["content"] == "U6", "S0: storia completa più U6")
    # S1
    _, ctx = esegui("S1", c, 6, 7)
    controlla(testi(ctx[6]) == [("u", "U3"), ("a", "A3"), ("u", "U4"), ("a", "A4"), ("u", "U5"), ("a", "A5"), ("u", "U6")],
              "S1: ultimi 3 turni (3–5) più U6")
    controlla(testi(ctx[7])[0] == ("u", "U4"), "S1: al turno 7 la finestra scorre (4–6)")
    # S2
    serv, ctx = esegui("S2", c, 6, 7)
    tipi = [t for t, _ in serv.richieste]
    controlla(tipi == ["ge_gold_prompt", "ge_nuova_sessione"], "S2: una sola GE (gold prompt + nuova sessione)")
    controlla(serv.richieste[0][1][-1]["content"] == "PROMPT_GE" and len(serv.richieste[0][1]) == 11,
              "S2: il gold prompt si chiede su tutta la storia (turni 1–5) + prompt GE")
    controlla(serv.richieste[1][1] == [{"role": "user", "content": "GE_GOLD_PROMPT#1"}],
              "S2: la nuova sessione inizia con il gold prompt come messaggio dell'utente, senza sistema")
    controlla(testi(ctx[6]) == [("u", "GE_GOLD_PROMPT#1"), ("a", "GE_NUOVA_SESSIONE#2"), ("u", "U6")],
              "S2 turno 6: gold prompt + risposta + U6")
    controlla(testi(ctx[7]) == [("u", "GE_GOLD_PROMPT#1"), ("a", "GE_NUOVA_SESSIONE#2"), ("u", "U6"), ("a", "A6"), ("u", "U7")],
              "S2 turno 7: come sopra + A6 + U7")
    # S3
    serv, ctx = esegui("S3", c, 6, 10)
    tipi = [t for t, _ in serv.richieste]
    controlla(tipi == ["s3_riassunto", "s3_aggiornamento"], "S3: primo riassunto al turno 6, aggiornamento al turno 9")
    controlla(serv.richieste[0][1][-1]["content"] == "PROMPT_S3_1" and len(serv.richieste[0][1]) == 11,
              "S3: il primo riassunto copre tutta la storia (turni 1–5)")
    controlla(testi(ctx[6]) == [("u", f"{PREF}\n\nS3_RIASSUNTO#1\n\nU6")], "S3 turno 6: riassunto unito a U6")
    controlla(testi(ctx[7]) == [("u", f"{PREF}\n\nS3_RIASSUNTO#1\n\nU6"), ("a", "A6"), ("u", "U7")], "S3 turno 7")
    agg = serv.richieste[1][1]
    controlla(len(agg) == 1 and agg[0]["content"].startswith("PROMPT_S3_N\n\nCurrent summary:\nS3_RIASSUNTO#1")
              and "User: U6" in agg[0]["content"] and "Assistant: A8" in agg[0]["content"]
              and "U9" not in agg[0]["content"], "S3: l'aggiornamento contiene riassunto + turni 6–8 dopo il prompt")
    controlla(testi(ctx[9]) == [("u", f"{PREF}\n\nS3_AGGIORNAMENTO#2\n\nU9")], "S3 turno 9: riassunto aggiornato unito a U9")
    # S4
    serv, ctx = esegui("S4", c, 6, 8)
    controlla([t for t, _ in serv.richieste] == ["s4_riassunto"], "S4: una sola compattazione")
    controlla([m["content"] for m in serv.richieste[0][1]] == ["U1", c.messaggi[1]["content"], "U2 alpha", "A2", "PROMPT_S4"],
              "S4: si riassumono solo i turni vecchi (1–2)")
    controlla(testi(ctx[6]) == [("u", f"{PREF}\n\nS4_RIASSUNTO#1\n\nU3"), ("a", "A3"), ("u", "U4"), ("a", "A4"),
                                ("u", "U5"), ("a", "A5"), ("u", "U6")], "S4 turno 6: riassunto unito a U3, poi turni 3–5 e U6")
    controlla(len(ctx[8]) == 11 and ctx[8][-1]["content"] == "U8", "S4 turno 8: la coda cresce (nessuna nuova compattazione)")
    serv, ctx = esegui("S4", c, 3, 5)
    controlla([t for t, _ in serv.richieste] == ["s4_riassunto"] and len(ctx[3]) == 5 and len(ctx[4]) == 7,
              "S4 con trigger al turno 3: compattazione rinviata al turno 5 (primo con turni più vecchi degli ultimi 3)")
    # S5 / S6
    for nome, metodo in (("S5", "llmlingua2"), ("S6", "selective_context")):
        serv, ctx = esegui(nome, c, 6, 7)
        controlla(serv.compressi[0][0] == metodo, f"{nome}: metodo {metodo}")
        tutti = [t for _, lst in serv.compressi for t in lst]
        controlla(not any("```" in t or "`x`" in t for t in tutti), f"{nome}: il codice non va al compressore")
        controlla("```code k```" in ctx[6][1]["content"] and "`x`" in ctx[6][1]["content"],
                  f"{nome}: il codice resta intatto nel messaggio compresso")
        controlla(ctx[6][-1] == {"role": "user", "content": "U6"}, f"{nome}: l'ultimo messaggio dell'utente non si comprime")
        controlla(len(serv.compressi) == 2 and len(serv.compressi[1][1]) == 2,
                  f"{nome}: al turno 7 si comprimono solo i 2 messaggi nuovi (U6, A6)")
    # S7
    serv, ctx = esegui("S7", c, 7, 7)
    controlla(testi(ctx[7]) == [("u", "U1"), ("a", c.messaggi[1]["content"]), ("u", "U2 alpha"), ("a", "A2"),
                                ("u", "U4"), ("a", "A4"), ("u", "U5"), ("a", "A5"), ("u", "U6"), ("a", "A6"), ("u", "U7")],
              "S7 turno 7: 2 turni recuperati (il 2 più simile, poi l'1 a parità), in ordine cronologico, + ultimi 3 (4–6) + U7")
    controlla(serv.intermedi[0][0] == "s7_recuperati" and [v["turno"] for v in serv.intermedi[0][1]] == [1, 2],
              "S7: turni recuperati registrati (1, 2)")
    # ordine a giri
    import dati
    def finta(cid, n):
        return Conversazione(id=cid, fonte="prova", messaggi=[{"role": r, "content": "x"} for _ in range(n)
                                                                  for r in ("user", "assistant")])
    convs = [finta(f"a{i}", 10) for i in range(4)] + [finta(f"b{i}", 13) for i in range(3)] + [finta("c0", 40)]
    o = dati.ordina_a_giri(list(convs), [10, 12, 15, 20, 30], 42)
    fasce = [c.meta["fascia"] for c in o]
    controlla(fasce == ["10–11", "12–14", "≥30", "10–11", "12–14", "10–11", "12–14", "10–11"],
              f"ordine a giri: una per fascia dalla più corta, fasce esaurite saltate ({' '.join(fasce)})")
    o2 = dati.ordina_a_giri(list(reversed(convs)), [10, 12, 15, 20, 30], 42)
    controlla([c.id for c in o2] == [c.id for c in o], "ordine a giri: stesso seed, stesso ordine (anche con il file in ordine diverso)")
    controlla([c.meta["giro"] for c in o] == [1, 1, 1, 2, 2, 3, 3, 4], "ordine a giri: numeri di giro")
    # cartella dei risultati: una configurazione del benchmark diversa va in una cartella nuova
    from risultati import cartella_esecuzione
    tmp = Path(tempfile.mkdtemp(prefix="cartella_"))
    cfg_finta = {"esecuzione": {"cartella_risultati": str(tmp), "nome": "v1"}, "_sha256_benchmark": "a" * 64}
    cfg_finta = type("C", (dict,), {"percorso": lambda self, v: Path(v)})(cfg_finta)
    c1 = cartella_esecuzione(cfg_finta)
    (tmp / "v1").mkdir()
    (tmp / "v1" / "esecuzione.json").write_text(json.dumps({"config_benchmark_sha256": "a" * 64}))
    c2 = cartella_esecuzione(cfg_finta)
    cfg_finta["_sha256_benchmark"] = "b" * 64
    c3 = cartella_esecuzione(cfg_finta)
    controlla(c1 == c2 == tmp / "v1" and c3 == tmp / ("v1_" + "b" * 12),
              "configurazione del benchmark diversa (es. fasce o seed): cartella nuova <nome>_<impronta>")
    shutil.rmtree(tmp, ignore_errors=True)
    voci = serv.embedding_voci[0]
    controlla(voci[-1] == {"testo": "U7", "tipo": "query"} and len(voci) == 4,
              "S7: embedding dei 3 turni candidati (1–3) + query")


# ---------------------------------------------------------------------------
# Parte B: servizio di misura + runner
# ---------------------------------------------------------------------------

def porta_libera():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parole(n, tema="word"):
    return " ".join(f"{tema}{i}" for i in range(n))


def crea_dati(d: Path):
    def conv(cid, turni):
        msgs = []
        for u, a in turni:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        return {"id": cid, "fonte": "prova", "turni": len(turni), "messages": msgs}
    righe = [
        conv("prova:normale", [(f"question {k} " + parole(15, f"q{k}_"), f"answer {k} ```python\nprint({k})\n``` "
                                + parole(40, f"a{k}_")) for k in range(1, 13)]),
        conv("prova:corta", [(f"hi {k}", f"hello {k}") for k in range(1, 11)]),
        conv("prova:trigger1", [(parole(600, "big"), "ok")] + [(f"next {k}", f"reply {k} " + parole(20)) for k in range(2, 11)]),
        conv("prova:lunghissima", [(f"long {k} " + parole(30), parole(160, f"l{k}_")) for k in range(1, 15)]),
    ]
    (d / "conversazioni.jsonl").write_text("".join(json.dumps(r) + "\n" for r in righe), encoding="utf-8")
    modelli = {"modello-a": {"enable_thinking": False}, "modello-b": {}, "modello-fail": {}}
    sel = {"modelli": {n: {"repo": f"org/{n}", "revision": "c" * 40, "template_kwargs": k} for n, k in modelli.items()}}
    (d / "selezione.json").write_text(json.dumps(sel), encoding="utf-8")
    cand = {"candidati": [{"id": f"org/{n}", "commit": "c" * 40,
                           "reasoning": {"interruttori": [f"{a}={str(b).lower()}" for a, b in k.items()]},
                           "scarica": {"nome": n, "repo": f"q/{n}-GGUF", "etichetta": "Q4_K_M", "commit": "d" * 40}}
                          for n, k in modelli.items()]}
    (d / "candidati.json").write_text(json.dumps(cand), encoding="utf-8")
    manifest = {}
    for n in modelli:
        (d / "models" / n).mkdir(parents=True)
        f = f"{n}-fail-Q4_K_M.gguf" if "fail" in n else f"{n}-Q4_K_M.gguf"
        (d / "models" / n / f).write_bytes(b"GGUF")
        manifest[n] = {"files": [{"file": f}]}
    (d / "models" / "gpt-oss-120b").mkdir(parents=True)
    (d / "models" / "gpt-oss-120b" / "gpt-oss-120b-MXFP4.gguf").write_bytes(b"GGUF")
    (d / "models" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    commit = "def4d406ae2c2f39573120d68730fbb7760b24bf"
    (d / "llama.cpp" / "build" / "bin").mkdir(parents=True)
    (d / "llama.cpp" / "build" / ".commit_benchmark").write_text(commit + "\n")
    eseguibile = d / "llama.cpp" / "build" / "bin" / "llama-server"
    eseguibile.write_text(f"#!/bin/sh\nexec {sys.executable} {QUI / 'finto_llama_server.py'} \"$@\"\n")
    eseguibile.chmod(0o755)
    cfg = (BENCH / "config.toml").read_text(encoding="utf-8")
    porta = porta_libera()
    sostituzioni = {
        'conversazioni = "../replay_set/conversazioni.jsonl"': f'conversazioni = "{d / "conversazioni.jsonl"}"',
        'selezione = "../replay_set/selezione.json"': f'selezione = "{d / "selezione.json"}"',
        'modelli = "../selezioni/candidati_2026-10-19.json"': f'modelli = "{d / "candidati.json"}"',
        'sha256_conversazioni = "44b790a91ac4fc903da8282fafd444d13835ec3a8a707f16df3bacde386a276d"': 'sha256_conversazioni = ""',
        'cartella_risultati = "risultati"': f'cartella_risultati = "{d / "risultati"}"',
        'host = "0.0.0.0"': 'host = "127.0.0.1"',
        "porta = 9100": f"porta = {porta}",
        'indirizzo_misura = "http://127.0.0.1:9100"': f'indirizzo_misura = "http://127.0.0.1:{porta}"',
        'sorgenti = "../llama.cpp"': f'sorgenti = "{d / "llama.cpp"}"',
        'eseguibile = "../llama.cpp/build/bin/llama-server"': f'eseguibile = "{eseguibile}"',
        "porta = 8080": f"porta = {porta_libera()}",
        "contesto = 32768": "contesto = 2500",
        'cartella_modelli = "../models"': f'cartella_modelli = "{d / "models"}"',
        "riposo_secondi = 120": "riposo_secondi = 0.3",
        "riscaldamento_richieste = 3": "riscaldamento_richieste = 1",
        "soglia_token = 3500": "soglia_token = 500",
        "contesto = 65536 ": "contesto = 2400 ",
        "campione_stabilita = 0.05": "campione_stabilita = 0.3",
        "campione_identita = 0.05": "campione_identita = 0.3",
        "campione_degrado = 0.05": "campione_degrado = 0.3",
        "# degrado_media_max, bias_lunghezza_max, coerenza_interna_min":
            "# degrado_media_max, bias_lunghezza_max, coerenza_interna_min\nnon_validi_max = 0.5",
        "min_conversazioni_per_punto = 20": "min_conversazioni_per_punto = 1",
    }
    for a, b in sostituzioni.items():
        assert cfg.count(a) == 1, a
        cfg = cfg.replace(a, b)
    (d / "config.toml").write_text(cfg, encoding="utf-8")
    return d / "config.toml", porta


def pct(riga):
    """La percentuale di una riga di avanzamento, come numero (None se manca)."""
    import re
    m = re.search(r"(\d+,\d)%", riga or "")
    return float(m.group(1).replace(",", ".")) if m else None


def leggi(p: Path):
    return [json.loads(r) for r in p.read_text(encoding="utf-8").splitlines() if r.strip()] if p.exists() else []


def parte_b(d, env, config):
    print("\nParte B: servizio di misura + runner (llama-server, CodeCarbon e ausiliari finti)")
    if True:
        runner = [sys.executable, str(BENCH / "runner.py"), "--config", str(config)]
        # primo passo: le prime 2 conversazioni nell'ordine a giri (un giro: una per fascia)
        r0 = subprocess.run(runner + ["--max-conversazioni", "2"], cwd=d, env=env, capture_output=True, text=True)
        if r0.returncode:
            print(r0.stdout[-2000:], r0.stderr[-3000:])
        res0 = d / "risultati" / "v1"
        comp0 = leggi(res0 / "completate.jsonl")
        conv0 = sorted({c["unita"].split("|")[1] for c in comp0})
        controlla(r0.returncode == 0 and len(comp0) == 2 * 2 * 8 and "fino al giro 1" in r0.stdout,
                  f"primo passo: 2 conversazioni (giro 1), {len(comp0)} unità")
        controlla(len(conv0) == 2 and any(c in conv0 for c in ("prova:corta", "prova:trigger1"))
                  and any(c in conv0 for c in ("prova:normale", "prova:lunghissima")),
                  f"giro 1: una conversazione per fascia di turni ({', '.join(conv0)})")
        av0 = [x for x in r0.stdout.splitlines() if "[modello-a]" in x and "%" in x]
        controlla(av0 and pct(av0[0]) == 0 and pct(av0[-1]) == 100, "avanzamento del primo passo da 0,0% a 100,0%")
        # allungamento: da 2 a 4 conversazioni, riusando il già fatto
        r1 = subprocess.run(runner + ["--max-conversazioni", "4"], cwd=d, env=env, capture_output=True, text=True)
        print("  --- uscita del runner ---\n" + "\n".join("  | " + x for x in r1.stdout.splitlines()[-12:]))
        if r1.returncode:
            print(r1.stderr[-3000:])
        controlla(r1.returncode == 0, "runner concluso senza errori")
        av = [x for x in r1.stdout.splitlines() if "[modello-a]" in x and "%" in x]
        controlla(av and 0 < pct(av[0]) < 100 and pct(av[-1]) == 100 and "turni" in av[-1]
                  and "stima residua" in av[-1] and "giro" in av[-1],
                  f"allungamento: avanzamento riparte da {pct(av[0]) if av else '?'}% e arriva al 100,0%")
        comp1 = leggi(d / "risultati" / "v1" / "completate.jsonl")
        controlla(len(comp1) == len({c["unita"] for c in comp1}) and
                  all(c["unita"] in {x["unita"] for x in comp1[:len(comp0)]} for c in comp0),
                  "allungamento: le unità del primo passo non si rifanno")
        meta = json.loads((d / "risultati" / "v1" / "esecuzione.json").read_text())
        controlla(meta["input"]["conversazioni_del_run"] == 4 and meta["allungamenti"][-1]["da"] == 2,
                  "allungamento registrato in esecuzione.json (da 2 a 4)")
        r_meno = subprocess.run(runner + ["--max-conversazioni", "3"], cwd=d, env=env, capture_output=True, text=True)
        controlla(r_meno.returncode != 0 and "non accorciare" in (r_meno.stderr + r_meno.stdout),
                  "accorciare (da 4 a 3) è rifiutato")
        # totale atteso: S0 tutti i turni, le altre strategie dal trigger alla fine
        tr_a = json.loads((d / "risultati" / "v1" / "trigger" / "modello-a.json").read_text())["conversazioni"]
        atteso = sum(v["turni"] + (7 * (v["turni"] - v["turno_trigger"] + 1) if v["turno_trigger"] else 0)
                     for v in tr_a.values())
        controlla(f"/{atteso} turni)" in av[-1].replace(".", ""), f"avanzamento: totale di {atteso} risposte ai turni")
        res = d / "risultati" / "v1"
        mod = leggi(res / "modelli.jsonl")
        controlla(any(m["modello"] == "modello-fail" and m.get("esito") == "FAIL" for m in mod), "modello-fail: FAIL registrato")
        controlla(sum(m.get("evento") == "riposo" for m in mod) == 4,
                  "consumo a riposo misurato per i 2 modelli caricati, in ognuna delle 2 sessioni")
        rich = leggi(res / "richieste.jsonl")
        comp = leggi(res / "completate.jsonl")
        controlla(len(comp) == 2 * 4 * 8, "64 unità concluse (2 modelli x 4 conversazioni x 8 strategie)")
        controlla(sum(c["esito"] == "nessun_trigger" for c in comp) == 2 * 7, "conversazione corta: nessun trigger, solo S0")
        trig = json.loads((res / "trigger" / "modello-a.json").read_text())["conversazioni"]
        kt = trig["prova:normale"]["turno_trigger"]
        controlla(kt is not None and 1 < kt < 12, f"conversazione normale: trigger al turno {kt}")
        controlla(trig["prova:trigger1"]["turno_trigger"] == 1, "conversazione con primo messaggio enorme: trigger al turno 1")
        ge1 = [r for r in rich if r["conversazione"] == "prova:trigger1" and r["tipo"] == "ge_gold_prompt"]
        controlla(len(ge1) == 2 and all(r["turno"] == 2 for r in ge1),
                  "trigger al turno 1: la GE avviene al turno 2 (al turno 1 non c'è storia da riassumere)")
        def righe(mod_, conv, strat, tipo=None):
            return [r for r in rich if r["modello"] == mod_ and r["conversazione"] == conv and r["strategia"] == strat
                    and (tipo is None or r["tipo"] == tipo)]
        s0 = righe("modello-a", "prova:normale", "S0", "turno")
        controlla(len(s0) == 12 and sum(r["pre_trigger"] for r in s0) == kt - 1, "S0: tutti i turni, pre_trigger prima del trigger")
        for s in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
            t = righe("modello-a", "prova:normale", s, "turno")
            controlla(len(t) == 12 - kt + 1 and t[0]["turno"] == kt, f"{s}: turni dal trigger alla fine")
        controlla(len(righe("modello-a", "prova:normale", "S2", "ge_gold_prompt")) == 1
                  and len(righe("modello-a", "prova:normale", "S2", "ge_nuova_sessione")) == 1, "S2: una GE")
        n_agg = len(righe("modello-a", "prova:normale", "S3", "s3_aggiornamento"))
        controlla(n_agg == (12 - kt) // 3, f"S3: {n_agg} aggiornamenti ogni 3 turni")
        controlla(len(righe("modello-a", "prova:normale", "S4", "s4_riassunto")) == 1, "S4: una compattazione")
        controlla(len(righe("modello-a", "prova:normale", "S5", "compressione")) == 12 - kt + 1, "S5: compressione a ogni turno attivo")
        controlla(len(righe("modello-a", "prova:normale", "S7", "embedding")) >= 1, "S7: embedding registrati")
        cs = [r for r in rich if r["esito"] == "contesto_superato"]
        controlla(len(cs) > 0 and all(r["conversazione"] == "prova:lunghissima" for r in cs),
                  f"contesto superato registrato ({len(cs)} richieste) senza fermare il benchmark")
        t0 = s0[0]
        e = t0["energia_lorda"]
        controlla(e["prefill"]["gpu_wh"] > 0 and e["decode"]["gpu_wh"] > 0 and e["prefill"]["cpu_wh"] > 0,
                  "energia lorda: prefill e decode con GPU, CPU e RAM")
        controlla(t0["energia_netta"]["prefill"]["gpu_wh"] is not None, "energia netta calcolata")
        controlla(any(r["token_in_cache"] for r in rich if r["tipo"] == "turno"), "token in cache registrati")
        log = leggi(d / "llama_richieste.jsonl")
        a = [x for x in log if x["alias"] == "modello-a"]
        b = [x for x in log if x["alias"] == "modello-b"]
        controlla(all(x.get("chat_template_kwargs") == {"enable_thinking": False} for x in a) and
                  all("chat_template_kwargs" not in x for x in b), "reasoning spento solo dove serve (modello-a)")
        controlla(all(x["temperature"] == 0.7 and x["seed"] == 42 and "max_tokens" not in x for x in log),
                  "temperatura 0,7, seed 42, nessun tetto alla risposta")
        controlla(all(x["messages"][0]["role"] != "system" for x in log), "nessun messaggio di sistema")
        inter = leggi(res / "intermedi.jsonl")
        tipi = {i["tipo"] for i in inter}
        controlla({"ge_gold_prompt", "s3_riassunto", "s4_riassunto", "s5_compressi", "s6_compressi", "s7_recuperati"} <= tipi,
                  "testi intermedi salvati per tutte le strategie")
        s5 = [i for i in inter if i["tipo"] == "s5_compressi" and i["conversazione"] == "prova:normale"]
        controlla(any("```python\nprint(1)\n```" in m["testo"] for i in s5 for m in i["contenuto"]),
                  "S5: blocchi di codice intatti nei messaggi compressi")
        n_righe = len(rich)
        r2 = subprocess.run(runner, cwd=d, env=env, capture_output=True, text=True)
        controlla(r2.returncode == 0 and len(leggi(res / "richieste.jsonl")) == n_righe and "già concluso" in r2.stdout,
                  "ripresa: un secondo avvio non rifà le unità concluse")
        # ripresa dopo un'interruzione: si cancella l'ultima unità conclusa e si rilancia
        comp_righe = (res / "completate.jsonl").read_text().splitlines()
        tolta = json.loads(comp_righe[-1])["unita"]
        (res / "completate.jsonl").write_text("\n".join(comp_righe[:-1]) + "\n")
        r3 = subprocess.run(runner, cwd=d, env=env, capture_output=True, text=True)
        nuove = [x for x in leggi(res / "completate.jsonl") if x["unita"] == tolta]
        controlla(r3.returncode == 0 and len(nuove) == 1, f"ripresa: l'unità non conclusa ({tolta}) si rifà")
        av3 = [x for x in r3.stdout.splitlines() if "[modello-b]" in x and "%" in x]
        controlla(av3 and 0 < pct(av3[0]) < 100 and pct(av3[-1]) == 100,
                  f"avanzamento alla ripresa: riparte da {pct(av3[0]) if av3 else '?'}%")


def parte_c(d, env, config):
    print("\nParte C: giudice")
    import configurazione
    cfg = configurazione.carica(config)
    res = d / "risultati" / "v1"
    cg = res / "giudizi" / cfg["_sha256_giudice"][:12]
    giudice = [sys.executable, str(BENCH / "giudice.py"), "--config", str(config)]
    n_log = len(leggi(d / "llama_richieste.jsonl"))
    r = subprocess.run(giudice, cwd=d, env=env, capture_output=True, text=True)
    print("  --- uscita del giudice ---\n" + "\n".join("  | " + x for x in r.stdout.splitlines()[-8:]))
    if r.returncode:
        print(r.stderr[-3000:])
    controlla(r.returncode == 0, "giudice concluso senza errori")
    avg = [x for x in r.stdout.splitlines() if "[modello-a]" in x and "%" in x]
    controlla(avg and pct(avg[0]) == 0 and pct(avg[-1]) == 100 and "giudizi" in avg[-1],
              f"avanzamento del giudice da 0,0% a 100,0% ({len(avg)} righe)")
    g = leggi(cg / "giudizi.jsonl")
    controlla("[modello-fail] saltato: nessun risultato" in r.stdout, "modello in FAIL saltato dal giudice")
    base = [x for x in g if not x["controllo"]]
    trig = json.loads((res / "trigger" / "modello-a.json").read_text())["conversazioni"]
    attese = set()
    for cid, v in trig.items():
        if v["turno_trigger"]:
            for k in range(v["turno_trigger"], v["turni"] + 1):
                for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"):
                    attese.add((cid, k, s, "B"))
                    if s != "S0":
                        attese.add((cid, k, s, "A"))
    trovate = {(x["conversazione"], x["turno"], x["strategia"], x["tipo"]) for x in base if x["modello"] == "modello-a"}
    controlla(trovate == attese, f"modello-a: una voce A per S1–S7 e una B per S0–S7 per ogni turno dal trigger ({len(attese)})")
    controlla(not any(x["turno"] < x["turno_trigger"] for x in g), "nessun giudizio prima del trigger")
    controlla(any(x.get("motivo") == "tecnica_senza_S0" for x in g), "casi «tecnica senza S0» registrati come tali")
    controlla(any(x.get("motivo") == "risposta_mancante" for x in g), "risposte mancanti: non giudicabili")
    controlla(any(x["esito"] == "contesto_superato_giudice" for x in g), "contesto del giudice superato registrato")
    ok = [x for x in base if x["esito"] == "ok"]
    controlla(all(set(x["voti"]) == {"relative_correctness", "relative_coherence"} for x in ok if x["tipo"] == "A") and
              all(set(x["voti"]) == {"absolute_correctness", "absolute_coherence"} for x in ok if x["tipo"] == "B"),
              "variante A: solo voti relativi; variante B: solo voti assoluti")
    controlla(any(x["tentativi"] > 1 for x in ok), "uscite non valide: giudizio riuscito con un nuovo tentativo")
    controlla(all(x["motivazione"] and x["ragionamento"] for x in ok), "motivazione e ragionamento salvati")
    ctrl = {c: [x for x in g if x["controllo"] == c and x["esito"] == "ok"] for c in ("stabilita", "identita", "degrado")}
    controlla(all(ctrl.values()), "controlli presenti: " + ", ".join(f"{k} {len(v)}" for k, v in ctrl.items()))
    controlla(all(x["tipo"] == "A" for x in ctrl["identita"]) and
              sum(v for x in ctrl["identita"] for v in x["voti"].values()) / (2 * len(ctrl["identita"])) >= 9,
              "identità: solo variante A, voto relativo vicino a 10")
    deg_b = [x for x in ctrl["degrado"] if x["tipo"] == "B"]
    controlla(deg_b and all(x["degrado_da"]["conversazione"] != x["conversazione"] for x in ctrl["degrado"]) and
              max(x["voti"]["absolute_coherence"] for x in deg_b) <= 3,
              "degrado: risposta da un'altra conversazione, voti assoluti bassi")
    log = leggi(d / "llama_richieste.jsonl")[n_log:]
    gl = [x for x in log if x["alias"] == "gpt-oss-120b"]
    controlla(gl and all(x["temperature"] == 1.0 and x["top_p"] == 1.0 and x["top_k"] == 0 and x["min_p"] == 0.0
                         and x["chat_template_kwargs"] == {"reasoning_effort": "low"}
                         and x["response_format"]["type"] == "json_schema" for x in gl),
              "giudice: temperatura 1.0, top_p 1.0, top_k 0, min_p 0, ragionamento low, schema JSON")
    controlla(len({x["seed"] for x in gl}) > len(gl) * 0.9, "seed diverso per ogni giudizio")
    a_prompt = next(x["messages"][0]["content"] for x in gl if "REFERENCE ANSWER START" in x["messages"][0]["content"])
    b_prompt = next(x["messages"][0]["content"] for x in gl if "=== ANSWER START ===" in x["messages"][0]["content"])
    controlla(len(gl[0]["messages"]) == 1 and "User: " in a_prompt and "Assistant: " in a_prompt
              and "REFERENCE ANSWER" not in b_prompt and "{" + "storia}" not in a_prompt,
              "prompt: un solo messaggio, storia citata come User/Assistant, B senza riferimento")
    sess = leggi(cg / "sessioni.jsonl")
    car = [x for x in sess if x["evento"] == "caricamento"]
    controlla(car and "2400" in car[-1]["comando"], "giudice caricato con il suo contesto")
    par = [x for x in sess if x["evento"] == "parametri_dichiarati"]
    controlla(par and par[-1]["coerenti"], "parametri dichiarati da llama-server uguali a quelli richiesti")
    stato = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{cfg['rete']['porta']}/stato").read())
    controlla(stato["modello"] is None, "giudice scaricato alla fine")
    # ripresa
    n = len(g)
    r2 = subprocess.run(giudice, cwd=d, env=env, capture_output=True, text=True)
    controlla(r2.returncode == 0 and len(leggi(cg / "giudizi.jsonl")) == n, "ripresa: un secondo avvio non rifà i giudizi")
    righe = (cg / "giudizi.jsonl").read_text().splitlines()
    (cg / "giudizi.jsonl").write_text("\n".join(righe[:-5]) + "\n")
    r3 = subprocess.run(giudice, cwd=d, env=env, capture_output=True, text=True)
    nuovi = leggi(cg / "giudizi.jsonl")
    avr = [x for x in r3.stdout.splitlines() if "[modello-b]" in x and "%" in x]  # le 5 righe tolte sono di modello-b
    controlla(avr and 95 < pct(avr[0]) < 100 and pct(avr[-1]) == 100,
              f"avanzamento del giudice alla ripresa: riparte da {pct(avr[0]) if avr else '?'}%")
    controlla(r3.returncode == 0 and len(nuovi) == n and [x["id"] for x in nuovi[-5:]] == [json.loads(x)["id"] for x in righe[-5:]],
              "ripresa: i giudizi mancanti si rifanno")
    return cfg


def parte_d(d, env, config):
    print("\nParte D: elaborazione")
    r = subprocess.run([sys.executable, str(BENCH / "elabora.py"), "--config", str(config)], cwd=d, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
    controlla(r.returncode == 0, "elaborazione conclusa senza errori")
    el = d / "risultati" / "v1" / "elaborazione"
    controlla((el / "rapporto.docx").exists(), "rapporto Word scritto")
    tab = el / "tabelle"
    nomi = ["A_quadro", "B_energia_per_strategia", "B_token_per_strategia", "B_pareggio", "C_qualita_per_strategia",
            "C_per_fonte", "D_compromesso", "E_test_contro_S0", "F_referto_giudice", "F_robustezza"]
    mancano = [n for n in nomi if not (tab / f"modello-a__{n}.csv").exists()]
    controlla(not mancano, f"tabelle A–F per modello-a ({'mancano ' + ', '.join(mancano) if mancano else 'tutte'})")
    controlla((tab / "G_punteggio_medio_per_modello.csv").exists(), "tabella G di confronto tra modelli")
    pesate = list(csv.DictReader(open(tab / "modello-a__D_medie_pesate.csv", encoding="utf-8")))
    per_f = list(csv.DictReader(open(tab / "modello-a__D_per_fascia.csv", encoding="utf-8")))
    comp_f = list(csv.DictReader(open(tab / "modello-a__A_fasce.csv", encoding="utf-8")))
    controlla(len(pesate) == 8 and all(k in pesate[1] for k in ("risparmio_semplice", "risparmio_pesata",
                                                              "punteggio_semplice", "punteggio_pesata"))
              and pesate[1]["punteggio_pesata"] not in ("", "nan"),
              "medie semplici e pesate per strategia (risparmio, punteggio, voti assoluti)")
    controlla({x["fascia"] for x in per_f} >= {"10–11", "12–14"} and len(comp_f) == 5
              and abs(sum(float(x["quota_dataset"]) for x in comp_f) - 1) < 1e-9,
              "risultati per fascia di turni e composizione dataset/run (5 fasce)")
    controlla(len(list((el / "grafici").glob("modello-a__*.png"))) >= 8, "grafici PNG scritti")
    ref = list(csv.DictReader(open(tab / "modello-a__F_referto_giudice.csv", encoding="utf-8")))
    stati = {}
    for x in ref:
        stati.setdefault(x["indicatore"], x["stato"])
    controlla({str(i) for i in range(1, 8)} <= set(stati), "referto: 7 indicatori")
    controlla(stati["1"] in ("ok", "avviso") and stati["4"] == "soglia da fissare", "referto: soglie applicate solo se fissate")
    en = list(csv.DictReader(open(tab / "modello-a__B_energia_per_strategia.csv", encoding="utf-8")))
    controlla(len(en) == 8 and all(float(x["Wh_per_turno"]) > 0 for x in en)
              and all(x["risparmio_medio"] not in ("", "nan") for x in en), "energia netta e risparmio calcolati")
    rif = {x["voce"][:40]: x["valore"] for x in ref}
    controlla(any(v == "sì" for k, v in rif.items() if k.startswith("parametri di campionamento")),
              "referto: parametri del giudice verificati")
    from docx import Document
    testo = "\n".join(p.text for p in Document(el / "rapporto.docx").paragraphs)
    controlla(all(t in testo for t in ("A. Quadro dei dati", "B. Energia", "C. Qualità", "D. Compromesso",
                                       "E. Statistica", "F. Affidabilità", "G. Confronto")), "rapporto: sezioni A–G")


def parte_e(d, env, config):
    print("\nParte E: comando unico")
    r = subprocess.run([sys.executable, str(BENCH / "catena.py"), "--config", str(config), "--modelli", "modello-a"],
                       cwd=d, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
    controlla(r.returncode == 0 and "già concluso" in r.stdout and "Catena conclusa" in r.stdout,
              "catena.py: benchmark e giudice già fatti si saltano, elaborazione rifatta")


if __name__ == "__main__":
    parte_a()
    d = Path(tempfile.mkdtemp(prefix="bench_"))
    config, porta = crea_dati(d)
    env = {**os.environ, "PYTHONPATH": str(QUI / "finti"), "FINTO_LOG": str(d / "llama_richieste.jsonl")}
    servizio = subprocess.Popen([sys.executable, str(QUI / "avvia_misura_finta.py"), "esegui", "--config", str(config)],
                                cwd=d, env=env, stdout=open(d / "servizio.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{porta}/stato", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        parte_b(d, env, config)
        parte_c(d, env, config)
        parte_d(d, env, config)
        parte_e(d, env, config)
    finally:
        servizio.terminate()
        servizio.wait()
    if ERRORI or os.environ.get("CONSERVA_PROVA"):
        print(f"\n(cartella della prova conservata: {d})")
    else:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{'TUTTO OK' if not ERRORI else f'{len(ERRORI)} CONTROLLI FALLITI'}")
    sys.exit(1 if ERRORI else 0)
