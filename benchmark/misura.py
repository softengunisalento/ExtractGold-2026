#!/usr/bin/env python3
"""Macchina di misura (M1): preparazione e servizio di misura del benchmark.

    python misura.py prepara [--config config.toml] [--modelli a,b]
                                                      una volta, con internet: llama.cpp al commit congelato,
                                                      librerie, modelli della lista congelata (o solo quelli
                                                      indicati), giudice, modelli ausiliari
    python misura.py esegui  [--config config.toml]   servizio di misura, anche senza internet

Il servizio offre poche operazioni generiche (stato, carica modello o giudice, richiesta al modello, conteggio
dei token, compressione, embedding, consumo a riposo). Ogni operazione dice se la misura è accesa o spenta. Non conosce
strategie né conversazioni. Esegue UNA richiesta alla volta: con CodeCarbon in modalità machine due richieste
sovrapposte finirebbero nella stessa misura.

Misura: CodeCarbon (GPU + CPU + RAM). Per ogni richiesta al modello, due finestre: prefill (dall'invio al primo
token) e decode (dal primo all'ultimo token). Compressione ed embedding: una finestra di overhead.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import configurazione

VERSIONE = "1.1"
CARTELLA_LOG = "log_misura"


def adesso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Errore(Exception):
    """Errore atteso, restituito al chiamante come risposta HTTP 400."""


# ---------------------------------------------------------------------------
# Controlli dell'hardware
# ---------------------------------------------------------------------------

def file_rapl() -> list[str]:
    return sorted(set(glob.glob("/sys/class/powercap/intel-rapl*/energy_uj")
                      + glob.glob("/sys/class/powercap/intel-rapl*/*/energy_uj")))


def rapl_leggibile() -> dict:
    """I contatori RAPL della CPU: CodeCarbon li usa se può leggerli, altrimenti stima la CPU."""
    trovati = file_rapl()
    leggibili = []
    for f in trovati:
        try:
            Path(f).read_text()
            leggibili.append(f)
        except OSError:
            pass
    return {"trovati": len(trovati), "leggibili": len(leggibili), "ok": bool(trovati) and len(leggibili) == len(trovati)}


def gpu_nvidia() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
        return [r.strip() for r in out.splitlines() if r.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


# ---------------------------------------------------------------------------
# Misura dell'energia (CodeCarbon)
# ---------------------------------------------------------------------------

def _wh(valore) -> float | None:
    return None if valore is None else float(valore) * 1000  # CodeCarbon restituisce kWh


def _dati_energia(dati, durata_misurata: float) -> dict:
    return {
        "gpu_wh": _wh(getattr(dati, "gpu_energy", None)),
        "cpu_wh": _wh(getattr(dati, "cpu_energy", None)),
        "ram_wh": _wh(getattr(dati, "ram_energy", None)),
        "totale_wh": _wh(getattr(dati, "energy_consumed", None)),
        "durata_codecarbon_s": getattr(dati, "duration", None),
        "durata_s": durata_misurata,
        "emissioni_kg": getattr(dati, "emissions", None),
    }


class Finestra:
    """Una finestra di misura. In modalità 'tracker' il tracker si crea PRIMA della finestra, così la sua
    preparazione resta fuori dalla misura; in modalità 'task' si usa il tracker sempre attivo del servizio."""

    def __init__(self, misuratore: "Misuratore", nome: str):
        self.m, self.nome = misuratore, nome
        self.tracker = None if misuratore.metodo == "task" else misuratore._nuovo_tracker()
        self.t0 = None

    def avvia(self) -> None:
        self.t0 = time.perf_counter()
        if self.tracker is None:
            self.m.principale.start_task(self.nome)
        else:
            self.tracker.start()

    def ferma(self) -> dict:
        if self.tracker is None:
            dati = self.m.principale.stop_task()
        else:
            self.tracker.stop()
            dati = getattr(self.tracker, "final_emissions_data", None)
        return _dati_energia(dati, time.perf_counter() - self.t0)


class Misuratore:
    def __init__(self, cfg_misura: dict):
        import codecarbon
        from codecarbon import EmissionsTracker
        self.versione = getattr(codecarbon, "__version__", None)
        self._classe = EmissionsTracker
        self.modalita = cfg_misura["modalita"]
        metodo = cfg_misura["metodo_finestre"]
        if metodo == "auto":
            metodo = "task" if hasattr(EmissionsTracker, "start_task") else "tracker"
        self.metodo = metodo
        self.principale = None
        if metodo == "task":
            self.principale = self._nuovo_tracker()
            self.principale.start()
        campione = self.principale or self._nuovo_tracker()
        self.hardware = descrivi_hardware(campione)

    def _nuovo_tracker(self):
        voluti = {"tracking_mode": self.modalita, "save_to_file": False, "log_level": "error",
                  "allow_multiple_runs": True, "project_name": "benchmark-compressione"}
        try:
            ammessi = set(inspect.signature(self._classe.__init__).parameters)
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in inspect.signature(self._classe.__init__).parameters.values()):
                voluti = {k: v for k, v in voluti.items() if k in ammessi}
        except (TypeError, ValueError):
            pass
        return self._classe(**voluti)

    def finestra(self, nome: str) -> Finestra:
        return Finestra(self, nome)


def descrivi_hardware(tracker) -> dict:
    """Come CodeCarbon misura ogni componente (es. CPU con RAPL o stimata): si registra con i risultati."""
    out = {}
    for h in getattr(tracker, "_hardware", None) or []:
        nome = type(h).__name__
        dettaglio = getattr(h, "_mode", None) or getattr(h, "_tracking_method", None)
        out[nome] = str(dettaglio) if dettaglio is not None else "n/d"
    return out


# ---------------------------------------------------------------------------
# llama-server
# ---------------------------------------------------------------------------

class LlamaServer:
    def __init__(self, cfg: configurazione.Configurazione):
        c = cfg["llama"]
        self.eseguibile = cfg.percorso(c["eseguibile"])
        self.porta = int(c["porta"])
        self.contesto = int(c["contesto"])
        self.argomenti = [str(a) for a in c["argomenti"]]
        self.attesa = float(c["attesa_caricamento_s"])
        self.cartella_modelli = cfg.percorso(c["cartella_modelli"])
        self.log_dir = cfg.cartella / CARTELLA_LOG
        self.proc: subprocess.Popen | None = None
        self.modello: str | None = None
        self.kwargs: dict = {}
        self.props: dict = {}

    def url(self, percorso: str) -> str:
        return f"http://127.0.0.1:{self.porta}{percorso}"

    def percorso_gguf(self, nome: str) -> Path:
        """Il file da passare a -m: quello registrato nel manifest di download_models.py (con più parti, il
        primo); se il manifest non c'è, il primo .gguf nella cartella del modello."""
        manifest = self.cartella_modelli / "manifest.json"
        if manifest.exists():
            voce = json.loads(manifest.read_text(encoding="utf-8")).get(nome) or {}
            files = [f.get("file") for f in voce.get("files") or [] if f.get("file")]
            if files:
                p = self.cartella_modelli / nome / sorted(files)[0]
                if p.exists():
                    return p
        trovati = sorted(p for p in (self.cartella_modelli / nome).rglob("*.gguf") if "mmproj" not in p.name.lower())
        if not trovati:
            raise Errore(f"nessun file GGUF per {nome} in {self.cartella_modelli / nome}")
        return trovati[0]

    def ferma(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc, self.modello, self.kwargs, self.props = None, None, {}, {}

    def avvia(self, nome: str, kwargs: dict, gguf: Path | None = None, contesto: int | None = None,
              argomenti: list[str] | None = None) -> dict:
        """Avvia llama-server. Per il giudice si passano file, contesto e argomenti della sezione [giudice]."""
        self.ferma()
        if not self.eseguibile.exists():
            raise Errore(f"llama-server non trovato: {self.eseguibile} (lancia prima: python misura.py prepara)")
        if gguf is None:
            gguf = self.percorso_gguf(nome)
        elif not gguf.exists():
            raise Errore(f"file GGUF non trovato: {gguf} (lancia prima: python misura.py prepara)")
        contesto = int(contesto or self.contesto)
        argomenti = [str(a) for a in argomenti] if argomenti is not None else self.argomenti
        self.log_dir.mkdir(exist_ok=True)
        log = self.log_dir / f"llama_{nome}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        cmd = [str(self.eseguibile), "-m", str(gguf), "--alias", nome, "--host", "127.0.0.1",
               "--port", str(self.porta), "-c", str(contesto), *argomenti]
        t0 = time.perf_counter()
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            self.proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        while True:
            if self.proc.poll() is not None:
                coda = log.read_text(encoding="utf-8", errors="replace")[-3000:]
                self.proc = None
                return {"esito": "FAIL", "motivo": "llama-server terminato durante il caricamento", "log": str(log),
                        "coda_log": coda, "comando": cmd}
            try:
                with urllib.request.urlopen(self.url("/health"), timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            if time.perf_counter() - t0 > self.attesa:
                self.ferma()
                return {"esito": "FAIL", "motivo": f"modello non pronto entro {self.attesa:.0f} s", "log": str(log),
                        "comando": cmd}
            time.sleep(1)
        self.modello, self.kwargs = nome, dict(kwargs or {})
        try:
            with urllib.request.urlopen(self.url("/props"), timeout=10) as r:
                p = json.loads(r.read().decode("utf-8"))
            gen = p.get("default_generation_settings") or {}
            self.props = {"model_path": p.get("model_path"), "build_info": p.get("build_info"),
                          "total_slots": p.get("total_slots"), "n_ctx": gen.get("n_ctx") or p.get("n_ctx"),
                          "default_generation_settings": gen}
        except (urllib.error.URLError, ValueError, OSError) as exc:
            self.props = {"errore": str(exc)}
        return {"esito": "OK", "modello": nome, "gguf": str(gguf), "secondi_caricamento": round(time.perf_counter() - t0, 1),
                "log": str(log), "comando": cmd, "props": self.props, "chat_template_kwargs": self.kwargs}

    def get_json(self, percorso: str, timeout: float = 30):
        with urllib.request.urlopen(self.url(percorso), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def post_json(self, percorso: str, corpo: dict, timeout: float = 60) -> dict:
        req = urllib.request.Request(self.url(percorso), data=json.dumps(corpo).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))


CAMPIONAMENTO_AMMESSO = {"temperature", "top_p", "top_k", "min_p", "seed"}


def contesto_superato(testo: str) -> bool:
    t = (testo or "").lower()
    return "exceed_context" in t or ("exceed" in t and "context" in t)


# ---------------------------------------------------------------------------
# Modelli ausiliari (S5, S6, S7)
# ---------------------------------------------------------------------------

class Ausiliari:
    """LLMLingua-2, Selective Context ed embedding. Si caricano e si scaldano all'avvio del servizio, così il
    caricamento non finisce mai nella misura di una richiesta."""

    def __init__(self, cfg_aus: dict):
        self.c = cfg_aus
        self.llmlingua = None
        self.sc = None
        self.sc_kwargs: dict = {}
        self.st = None
        self.info: dict = {}

    def precarica(self) -> None:
        from llmlingua import PromptCompressor
        self.llmlingua = PromptCompressor(model_name=self.c["llmlingua2"], use_llmlingua2=True,
                                          device_map=self.c["dispositivo"])
        from selective_context import SelectiveContext
        self.sc = SelectiveContext(model_type=self.c["selective_context_modello"], lang=self.c["selective_context_lingua"])
        parametri = inspect.signature(self.sc.__call__).parameters
        if "reduce_level" in parametri:
            self.sc_kwargs = {"reduce_level": self.c["selective_context_unita"]}
            self.info["selective_context_unita"] = self.c["selective_context_unita"]
        else:
            self.info["selective_context_unita"] = "parametro reduce_level assente: unità predefinita della libreria"
        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer(self.c["embedding"], device=self.c["dispositivo"])
        self.st.max_seq_length = int(self.c["embedding_max_token"])
        prova = "This is a short warm-up text used before any measurement."
        self.comprimi("llmlingua2", [prova], 0.5)
        self.comprimi("selective_context", [prova], 0.5)
        self.embedding([{"testo": prova, "tipo": "documento"}, {"testo": prova, "tipo": "query"}])
        self.info["caricati"] = True

    def comprimi(self, metodo: str, testi: list[str], quota: float) -> tuple[list[str], list[str]]:
        out, errori = [], []
        for t in testi:
            try:
                if metodo == "llmlingua2":
                    r = self.llmlingua.compress_prompt(t, rate=quota)
                    out.append(r["compressed_prompt"])
                elif metodo == "selective_context":
                    contesto, _ = self.sc(t, reduce_ratio=1 - quota, **self.sc_kwargs)
                    out.append(contesto)
                else:
                    raise Errore(f"metodo di compressione sconosciuto: {metodo}")
            except Errore:
                raise
            except Exception as exc:  # un testo che il compressore non gestisce resta intero, e si registra
                out.append(t)
                errori.append(f"{type(exc).__name__}: {str(exc)[:200]}")
        return out, errori

    def embedding(self, voci: list[dict]) -> list[list[list[float]]]:
        """Per ogni voce, la lista dei vettori normalizzati: un documento lungo si divide in pezzi della
        lunghezza massima del modello; la query riceve il prefisso indicato dagli autori."""
        tok = self.st.tokenizer
        quanti = int(self.c["embedding_max_token"]) - 2  # posto per i token speciali
        pezzi, appartenenza = [], []
        for i, v in enumerate(voci):
            if v.get("tipo") == "query":
                pezzi.append(self.c["embedding_prefisso_query"] + v["testo"])
                appartenenza.append(i)
                continue
            ids = tok(v["testo"], add_special_tokens=False)["input_ids"]
            parti = [tok.decode(ids[a:a + quanti]) for a in range(0, len(ids), quanti)] or [v["testo"]]
            pezzi += parti
            appartenenza += [i] * len(parti)
        vettori = self.st.encode(pezzi, normalize_embeddings=True, convert_to_numpy=True)
        out: list[list[list[float]]] = [[] for _ in voci]
        for i, vec in zip(appartenenza, vettori):
            out[i].append([round(float(x), 6) for x in vec])
        return out


# ---------------------------------------------------------------------------
# Servizio
# ---------------------------------------------------------------------------

class Servizio:
    def __init__(self, cfg: configurazione.Configurazione, misuratore: Misuratore, ausiliari: Ausiliari):
        self.cfg = cfg
        self.mis = misuratore
        self.aus = ausiliari
        self.llama = LlamaServer(cfg)
        self.avvio = adesso()

    # --- stato e modello ---
    def stato(self, _corpo=None) -> dict:
        stampa = self.cfg.percorso(self.cfg["llama"]["sorgenti"]) / "build" / ".commit_benchmark"
        return {
            "servizio": "misura", "versione": VERSIONE, "avvio": self.avvio, "ora_utc": adesso(),
            "host": socket.gethostname(), "python": platform.python_version(),
            "llama_commit_atteso": self.cfg["llama"]["commit"],
            "llama_commit_compilato": stampa.read_text().strip() if stampa.exists() else None,
            "contesto": self.llama.contesto, "argomenti": self.llama.argomenti,
            "cache_prompt": self.cfg["llama"]["cache_prompt"], "generazione": self.cfg["generazione"],
            "modello": self.llama.modello, "chat_template_kwargs": self.llama.kwargs,
            "codecarbon": self.mis.versione, "modalita": self.mis.modalita, "metodo_finestre": self.mis.metodo,
            "hardware_codecarbon": self.mis.hardware, "rapl": rapl_leggibile(), "gpu": gpu_nvidia(),
            "ausiliari": self.aus.info, "config_sha256": self.cfg["_sha256"],
            "giudice": {k: self.cfg["giudice"][k] for k in ("nome", "repository", "file", "commit", "contesto")},
        }

    def carica(self, corpo: dict) -> dict:
        if corpo.get("ruolo") == "giudice":
            g = self.cfg["giudice"]
            gguf = self.llama.cartella_modelli / g["nome"] / g["file"]
            return self.llama.avvia(g["nome"], dict(g["chat_template_kwargs"]), gguf=gguf, contesto=int(g["contesto"]),
                                    argomenti=list(g["argomenti"]))
        nome = corpo.get("nome")
        if not nome:
            raise Errore("manca il nome del modello")
        return self.llama.avvia(nome, corpo.get("chat_template_kwargs") or {})

    def scarica(self, _corpo=None) -> dict:
        self.llama.ferma()
        return {"esito": "OK"}

    def _serve_modello(self) -> None:
        if not self.llama.modello or not self.llama.proc or self.llama.proc.poll() is not None:
            raise Errore("nessun modello caricato (o llama-server terminato)")

    # --- richiesta al modello ---
    def chat(self, corpo: dict) -> dict:
        self._serve_modello()
        misura = bool(corpo.get("misura", True))
        g = self.cfg["generazione"]
        payload = {"model": self.llama.modello, "messages": corpo["messaggi"], "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": g["temperatura"], "seed": g["seed"],
                   "cache_prompt": bool(self.cfg["llama"]["cache_prompt"])}
        # Il chiamante (es. il giudice) può dare i suoi parametri di campionamento e uno schema JSON per l'uscita
        campionamento = corpo.get("campionamento") or {}
        ignoti = set(campionamento) - CAMPIONAMENTO_AMMESSO
        if ignoti:
            raise Errore(f"parametri di campionamento non ammessi: {sorted(ignoti)}")
        payload.update(campionamento)
        if corpo.get("schema"):
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "giudizio",
                                                                                 "schema": corpo["schema"]}}
        if self.llama.kwargs:
            payload["chat_template_kwargs"] = self.llama.kwargs
        # nessun max_tokens: la risposta finisce quando il modello si ferma o quando la finestra è piena
        f_pre = self.mis.finestra("prefill") if misura else None
        f_dec = self.mis.finestra("decode") if misura else None
        ris = {"esito": "ok", "errore": None, "testo": "", "ragionamento": "", "finish_reason": None,
               "usage": None, "timings": None, "ttft_ms": None, "wall_ms": None, "inizio_utc": adesso(),
               "energia": {"prefill": None, "decode": None}, "metodo_finestre": self.mis.metodo if misura else None}
        testo, ragionamento = [], []
        primo = None
        t0 = time.perf_counter()
        if f_pre:
            f_pre.avvia()
        try:
            req = urllib.request.Request(self.llama.url("/v1/chat/completions"),
                                         data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=None) as resp:
                for grezza in resp:
                    riga = grezza.decode("utf-8", "replace").strip()
                    if not riga.startswith("data:"):
                        continue
                    corpo_sse = riga[5:].strip()
                    if corpo_sse == "[DONE]":
                        break
                    try:
                        obj = json.loads(corpo_sse)
                    except ValueError:
                        continue
                    if obj.get("error"):
                        msg = json.dumps(obj["error"], ensure_ascii=False)
                        ris["esito"] = "contesto_superato" if contesto_superato(msg) else "errore"
                        ris["errore"] = msg[:1000]
                        break
                    scelta = (obj.get("choices") or [None])[0] or {}
                    delta = scelta.get("delta") or {}
                    c, r = delta.get("content"), delta.get("reasoning_content")
                    if (c or r) and primo is None:
                        primo = time.perf_counter()
                        if f_pre:
                            ris["energia"]["prefill"] = f_pre.ferma()
                            f_dec.avvia()
                    if c:
                        testo.append(c)
                    if r:
                        ragionamento.append(r)
                    if scelta.get("finish_reason"):
                        ris["finish_reason"] = scelta["finish_reason"]
                    if obj.get("usage"):
                        ris["usage"] = obj["usage"]
                    if obj.get("timings"):
                        ris["timings"] = obj["timings"]
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode("utf-8", "replace")
            ris["esito"] = "contesto_superato" if contesto_superato(msg) else "errore"
            ris["errore"] = f"HTTP {exc.code}: {msg[:1000]}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            ris["esito"], ris["errore"] = "errore", f"{type(exc).__name__}: {exc}"
        fine = time.perf_counter()
        if f_pre:
            if primo is None:
                ris["energia"]["prefill"] = f_pre.ferma()
            else:
                ris["energia"]["decode"] = f_dec.ferma()
        ris["wall_ms"] = round((fine - t0) * 1000, 1)
        ris["ttft_ms"] = None if primo is None else round((primo - t0) * 1000, 1)
        ris["testo"], ris["ragionamento"] = "".join(testo), "".join(ragionamento)
        # senza max_tokens, "length" vuol dire che la risposta ha riempito la finestra di contesto
        ris["fine_finestra"] = ris["finish_reason"] == "length"
        if corpo.get("leggi_slot"):
            ris["parametri_slot"] = self.parametri_slot()
        return ris

    def parametri_slot(self) -> dict | None:
        """I parametri che llama-server dichiara di aver usato nell'ultima richiesta (endpoint /slots)."""
        try:
            slot = self.llama.get_json("/slots")
            params = (slot[0] if isinstance(slot, list) and slot else {}).get("params") or {}
            return {k: params.get(k) for k in ("temperature", "top_k", "top_p", "min_p", "seed", "samplers")}
        except (urllib.error.URLError, ValueError, OSError, AttributeError) as exc:
            return {"errore": str(exc)[:300]}

    # --- conteggio dei token (misura spenta) ---
    def conta_token(self, corpo: dict) -> dict:
        self._serve_modello()
        messaggi = corpo["messaggi"]
        try:
            corpo_t = {"messages": messaggi}
            if self.llama.kwargs:
                corpo_t["chat_template_kwargs"] = self.llama.kwargs
            prompt = self.llama.post_json("/apply-template", corpo_t)["prompt"]
            token = self.llama.post_json("/tokenize", {"content": prompt, "add_special": False,
                                                       "parse_special": True})["tokens"]
            return {"token": len(token), "metodo": "apply-template+tokenize"}
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise Errore(f"conteggio dei token non riuscito: HTTP {exc.code}") from None
        testo = "\n".join(str(m.get("content") or "") for m in messaggi)
        token = self.llama.post_json("/tokenize", {"content": testo})["tokens"]
        return {"token": len(token), "metodo": "solo-testo (apply-template non disponibile)"}

    # --- overhead: compressione ed embedding ---
    def _con_misura(self, corpo: dict, nome: str, funzione):
        f = self.mis.finestra(nome) if corpo.get("misura", True) else None
        inizio = adesso()
        t0 = time.perf_counter()
        if f:
            f.avvia()
        try:
            risultato = funzione()
        finally:
            energia = f.ferma() if f else None
        return risultato, {"overhead": energia}, round((time.perf_counter() - t0) * 1000, 1), inizio

    def comprimi(self, corpo: dict) -> dict:
        if not self.aus.info.get("caricati"):
            raise Errore("modelli ausiliari non caricati")
        (testi, errori), energia, ms, inizio = self._con_misura(
            corpo, "compressione", lambda: self.aus.comprimi(corpo["metodo"], corpo["testi"], float(corpo["quota_tenuta"])))
        return {"testi": testi, "errori": errori, "energia": energia, "wall_ms": ms, "inizio_utc": inizio,
                "metodo_finestre": self.mis.metodo}

    def embedding(self, corpo: dict) -> dict:
        if not self.aus.info.get("caricati"):
            raise Errore("modelli ausiliari non caricati")
        vettori, energia, ms, inizio = self._con_misura(corpo, "embedding", lambda: self.aus.embedding(corpo["voci"]))
        return {"vettori": vettori, "energia": energia, "wall_ms": ms, "inizio_utc": inizio,
                "metodo_finestre": self.mis.metodo}

    # --- consumo a riposo ---
    def riposo(self, corpo: dict) -> dict:
        secondi = float(corpo.get("secondi") or 60)
        f = self.mis.finestra("riposo")
        inizio = adesso()
        f.avvia()
        time.sleep(secondi)
        e = f.ferma()
        durata = e["durata_s"] or secondi
        potenze = {k.replace("_wh", "_w"): (None if e[k] is None else e[k] * 3600 / durata)
                   for k in ("gpu_wh", "cpu_wh", "ram_wh", "totale_wh")}
        return {"energia": e, "potenza_media": potenze, "inizio_utc": inizio, "modello": self.llama.modello}


ROTTE = {
    ("GET", "/stato"): "stato",
    ("POST", "/modello/carica"): "carica",
    ("POST", "/modello/scarica"): "scarica",
    ("POST", "/chat"): "chat",
    ("POST", "/conta_token"): "conta_token",
    ("POST", "/comprimi"): "comprimi",
    ("POST", "/embedding"): "embedding",
    ("POST", "/riposo"): "riposo",
}


def crea_gestore(servizio: Servizio, chiave: str):
    class Gestore(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _rispondi(self, codice: int, dati: dict) -> None:
            corpo = json.dumps(dati, ensure_ascii=False).encode("utf-8")
            self.send_response(codice)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def _gestisci(self, metodo: str) -> None:
            nome = ROTTE.get((metodo, self.path))
            if nome is None:
                return self._rispondi(404, {"errore": f"operazione sconosciuta: {metodo} {self.path}"})
            if chiave and self.headers.get("X-Chiave") != chiave:
                return self._rispondi(403, {"errore": "chiave mancante o errata"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                corpo = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                return self._rispondi(200, getattr(servizio, nome)(corpo))
            except Errore as exc:
                return self._rispondi(400, {"errore": str(exc)})
            except Exception as exc:  # errore inatteso: il servizio resta in piedi e lo segnala
                return self._rispondi(500, {"errore": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            self._gestisci("GET")

        def do_POST(self):
            self._gestisci("POST")

        def log_message(self, formato, *args):
            sys.stderr.write(f"{adesso()} {self.address_string()} {formato % args}\n")

    return Gestore


def esegui(cfg: configurazione.Configurazione) -> int:
    if cfg["ausiliari"]["offline"]:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    r = rapl_leggibile()
    if not r["ok"]:
        print(f"ATTENZIONE: contatori RAPL leggibili {r['leggibili']}/{r['trovati']}: CodeCarbon stimerà la CPU "
              "invece di misurarla (vedi README).")
    misuratore = Misuratore(cfg["misura"])
    print(f"CodeCarbon {misuratore.versione}, modalità {misuratore.modalita}, finestre '{misuratore.metodo}', "
          f"hardware {misuratore.hardware}")
    ausiliari = Ausiliari(cfg["ausiliari"])
    print("Caricamento e riscaldamento dei modelli ausiliari (LLMLingua-2, Selective Context, embedding)...")
    ausiliari.precarica()
    servizio = Servizio(cfg, misuratore, ausiliari)
    rete = cfg["rete"]
    server = HTTPServer((rete["host"], int(rete["porta"])), crea_gestore(servizio, rete.get("chiave", "")))
    print(f"Servizio di misura in ascolto su {rete['host']}:{rete['porta']} (una richiesta alla volta). Ctrl+C per fermarlo.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servizio.llama.ferma()
        server.server_close()
    return 0


# ---------------------------------------------------------------------------
# Preparazione (una volta, con internet)
# ---------------------------------------------------------------------------

def _esegui_comando(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _cmake() -> str:
    """Il comando cmake: quello di sistema, o quello del venv; se manca lo installa nel venv con pip (non servono
    permessi di amministratore)."""
    trovato = shutil.which("cmake")
    nel_venv = Path(sys.executable).parent / "cmake"
    if trovato:
        return trovato
    if not nel_venv.exists():
        print("  cmake non trovato: lo installo nell'ambiente Python (pip)")
        _esegui_comando([sys.executable, "-m", "pip", "install", "cmake>=3.28,<4"])
    return str(nel_venv)


def prepara_llama(cfg: configurazione.Configurazione) -> dict:
    c = cfg["llama"]
    sorgenti = cfg.percorso(c["sorgenti"])
    if not (sorgenti / ".git").exists():
        _esegui_comando(["git", "clone", c["repository"], str(sorgenti)])
    stampa = sorgenti / "build" / ".commit_benchmark"
    attuale = subprocess.run(["git", "rev-parse", "HEAD"], cwd=sorgenti, capture_output=True, text=True).stdout.strip()
    if attuale != c["commit"]:
        _esegui_comando(["git", "fetch", "--all", "--tags"], cwd=sorgenti)
        _esegui_comando(["git", "checkout", "--detach", c["commit"]], cwd=sorgenti)
    attuale = subprocess.run(["git", "rev-parse", "HEAD"], cwd=sorgenti, capture_output=True, text=True).stdout.strip()
    sporco = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=sorgenti,
                            capture_output=True, text=True).stdout.strip()
    if attuale != c["commit"] or sporco:
        raise SystemExit(f"llama.cpp non è al commit congelato senza modifiche (HEAD {attuale}, modifiche: {bool(sporco)})")
    eseguibile = cfg.percorso(c["eseguibile"])
    if stampa.exists() and stampa.read_text().strip() == c["commit"] and eseguibile.exists():
        print("  llama.cpp già compilato a questo commit.")
    else:
        cmake = _cmake()
        _esegui_comando([cmake, "-B", "build", *c.get("opzioni_cmake", [])], cwd=sorgenti)
        _esegui_comando([cmake, "--build", "build", "--config", "Release", "-j"], cwd=sorgenti)
        if not eseguibile.exists():
            raise SystemExit(f"compilazione finita ma {eseguibile} non esiste: controlla le opzioni di CMake")
        stampa.write_text(c["commit"] + "\n")
    return {"commit": attuale, "eseguibile": str(eseguibile)}


def prepara_giudice(cfg: configurazione.Configurazione) -> dict:
    """Il GGUF del giudice, al commit fissato nella configurazione."""
    from huggingface_hub import hf_hub_download
    g = cfg["giudice"]
    cartella = cfg.percorso(cfg["llama"]["cartella_modelli"]) / g["nome"]
    print(f"  {g['repository']} / {g['file']} @ {g['commit'][:12]}")
    percorso = hf_hub_download(repo_id=g["repository"], filename=g["file"], revision=g["commit"],
                               local_dir=str(cartella))
    return {"repository": g["repository"], "file": g["file"], "commit": g["commit"], "percorso": percorso,
            "byte": Path(percorso).stat().st_size}


def prepara(cfg: configurazione.Configurazione, passi: set[str], modelli: list[str] | None = None) -> int:
    esito: dict = {"data_utc": adesso(), "host": socket.gethostname(), "python": platform.python_version(),
                   "config_sha256": cfg["_sha256"]}
    p = cfg.get("preparazione", {})
    print("1. Controlli")
    esito["gpu"] = gpu_nvidia()
    esito["rapl"] = rapl_leggibile()
    print(f"  GPU: {esito['gpu'] or 'nessuna trovata con nvidia-smi'}")
    print(f"  RAPL: {esito['rapl']}")
    if not esito["rapl"]["ok"]:
        print("  ATTENZIONE: i contatori RAPL non sono tutti leggibili: CodeCarbon stimerà la CPU invece di misurarla.\n"
              "  Serve che l'utente che lancia il servizio possa leggere /sys/class/powercap/intel-rapl*/energy_uj\n"
              "  (va deciso con chi amministra la macchina; lo script non cambia i permessi da solo).")
    if "llama" in passi:
        print("2. llama.cpp al commit congelato")
        esito["llama"] = prepara_llama(cfg)
    if "librerie" in passi:
        print("3. Librerie Python")
        req = cfg.percorso(p.get("requisiti", "requirements_misura.txt"))
        _esegui_comando([sys.executable, "-m", "pip", "install", "-r", str(req)])
        senza_dip = cfg.percorso(p.get("requisiti_senza_dipendenze", "requirements_misura_nodeps.txt"))
        if senza_dip.exists():
            _esegui_comando([sys.executable, "-m", "pip", "install", "--no-deps", "-r", str(senza_dip)])
        bloccato = cfg.cartella / "requirements_misura.lock.txt"
        if bloccato.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            bloccato = cfg.cartella / f"requirements_misura_{timestamp}.lock.txt"
        bloccato.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                           text=True).stdout, encoding="utf-8")
        esito["librerie_bloccate"] = str(bloccato)
    if "modelli" in passi:
        print("4. Modelli (lista congelata, con download_models.py)" + (f": solo {', '.join(modelli)}" if modelli else ""))
        script = cfg.percorso(p.get("script_download", "../download_models.py"))
        cmd = [sys.executable, str(script), "--da-lista", str(cfg.percorso(cfg["input"]["modelli"])),
               "--out", str(cfg.percorso(cfg["llama"]["cartella_modelli"]))]
        if modelli:
            cmd += ["--only", ",".join(modelli)]
        _esegui_comando(cmd, cwd=script.parent)
        esito["modelli"] = modelli or "tutti"
    if "giudice" in passi:
        print("4b. Giudice")
        esito["giudice"] = prepara_giudice(cfg)
    if "ausiliari" in passi:
        print("5. Modelli ausiliari")
        from huggingface_hub import snapshot_download
        a = cfg["ausiliari"]
        for repo in (a["llmlingua2"], a["embedding"]):
            print(f"  {repo}")
            snapshot_download(repo_id=repo)
        # Selective Context carica il modello per nome (es. "gpt2"): si scaricano solo configurazione, tokenizer e
        # pesi safetensors (gpt2 completo è 5,6 GB di formati inutili). Il nome "gpt2" è un alias di
        # openai-community/gpt2 e con il trasferimento Xet di Hugging Face risponde 404: lo si scarica senza Xet,
        # in un processo a parte perché l'impostazione vale solo all'avvio della libreria.
        repo = a["selective_context_modello"]
        print(f"  {repo} (solo i file necessari, senza Xet)")
        codice = ("import sys; from huggingface_hub import snapshot_download; "
                  "print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.txt', '*.safetensors']))")
        _esegui_comando([sys.executable, "-c", codice, repo], env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
        _esegui_comando([sys.executable, "-m", "spacy", "download", p.get("spacy_modello", "en_core_web_sm")])
    destinazione = cfg.cartella / "preparazione.json"
    destinazione.write_text(json.dumps(esito, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Preparazione conclusa: {destinazione}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Macchina di misura del benchmark: preparazione e servizio di misura.")
    ap.add_argument("comando", choices=["prepara", "esegui"])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    ap.add_argument("--solo", help="prepara: solo questi passi, separati da virgola "
                                   "(llama,librerie,modelli,giudice,ausiliari)")
    ap.add_argument("--modelli", help="prepara: scarica solo questi modelli della lista (nomi brevi, separati da "
                                      "virgola), per esempio per il pilota")
    args = ap.parse_args()
    cfg = configurazione.carica(args.config)
    if args.comando == "prepara":
        tutti = {"llama", "librerie", "modelli", "giudice", "ausiliari"}
        passi = {x.strip() for x in args.solo.split(",")} if args.solo else tutti
        if passi - tutti:
            sys.exit(f"passi sconosciuti: {sorted(passi - tutti)}")
        modelli = [x.strip() for x in args.modelli.split(",") if x.strip()] if args.modelli else None
        if modelli:
            import dati
            noti = {m["nome"] for m in dati.carica_modelli(cfg.percorso(cfg["input"]["modelli"]),
                                                          cfg.percorso(cfg["input"]["selezione"]))}
            ignoti = [m for m in modelli if m not in noti]
            if ignoti:
                sys.exit(f"modelli sconosciuti: {ignoti} (nomi possibili: {', '.join(sorted(noti))})")
        return prepara(cfg, passi, modelli)
    return esegui(cfg)


if __name__ == "__main__":
    sys.exit(main())
