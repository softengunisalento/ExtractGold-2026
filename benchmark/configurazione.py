"""Lettura e controllo di config.toml, comune a tutti gli script del benchmark.

I percorsi scritti nel file sono relativi alla cartella del file stesso. Tre impronte SHA-256:
  _sha256            tutto il file: deve essere identico sulla macchina di misura e sul PC di controllo;
  _sha256_benchmark  tutto tranne [giudice] ed [elaborazione]: identifica i risultati del benchmark, così
                     cambiare il giudice dopo il pilota non invalida le misure già fatte;
  _sha256_giudice    solo [giudice]: i giudizi si salvano in una cartella per ogni impronta.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("serve Python 3.11 o superiore (tomllib)")
import tomllib

# Chiavi obbligatorie, per sezione: un errore di battitura si scopre subito, non dopo ore di benchmark.
OBBLIGATORIE = {
    "esecuzione": ["nome", "seed", "cartella_risultati", "fasce_turni"],
    "input": ["conversazioni", "selezione", "modelli"],
    "rete": ["host", "porta", "indirizzo_misura"],
    "llama": ["sorgenti", "repository", "commit", "eseguibile", "porta", "contesto", "argomenti",
              "cache_prompt", "attesa_caricamento_s", "cartella_modelli"],
    "generazione": ["temperatura", "seed"],
    "misura": ["modalita", "metodo_finestre", "riposo_secondi", "riscaldamento_richieste",
               "riscaldamento_messaggio"],
    "strategie": ["elenco", "soglia_token", "turni_recenti", "s3_ogni_turni", "s7_recuperati",
                  "quota_token_tenuti", "prefisso_riassunto", "prompt_ge", "prompt_s3_primo",
                  "prompt_s3_aggiornamento", "formato_s3_aggiornamento", "prompt_s4"],
    "ausiliari": ["llmlingua2", "selective_context_modello", "selective_context_lingua",
                  "selective_context_unita", "embedding", "embedding_max_token",
                  "embedding_prefisso_query", "dispositivo", "offline"],
    "giudice": ["nome", "repository", "file", "commit", "contesto", "argomenti", "chat_template_kwargs",
                "temperatura", "top_p", "top_k", "min_p", "seed", "schema_json", "tentativi_extra",
                "campione_stabilita", "campione_identita", "campione_degrado", "peso_relativi", "peso_assoluti",
                "prompt_relativo", "prompt_assoluto", "storia_vuota"],
    "elaborazione": ["bootstrap_ripetizioni", "livello_confidenza", "seed", "min_conversazioni_per_punto"],
}

# Sezioni che non toccano le misure del benchmark
SEZIONI_NON_BENCHMARK = ("giudice", "elaborazione")
SEGNAPOSTO = re.compile(r"\{(storia|messaggio|riferimento|risposta)\}")
SOGLIE_NOTE = {"non_validi_max", "quota_dieci_max", "stabilita_differenza_media_max", "identita_media_min",
               "degrado_media_max", "bias_lunghezza_max", "coerenza_interna_min"}


def _impronta(dati) -> str:
    return hashlib.sha256(json.dumps(dati, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()

STRATEGIE_NOTE = {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}


class Configurazione(dict):
    """Il contenuto di config.toml, più il percorso e l'impronta del file."""

    @property
    def cartella(self) -> Path:
        return Path(self["_percorso"]).parent

    def percorso(self, valore: str) -> Path:
        """Un percorso del file di configurazione, risolto rispetto alla sua cartella."""
        p = Path(valore)
        return p if p.is_absolute() else (self.cartella / p).resolve()


def carica(percorso: str | Path) -> Configurazione:
    p = Path(percorso)
    try:
        grezzo = p.read_bytes()
    except OSError as exc:
        sys.exit(f"configurazione non leggibile ({p}): {exc}")
    try:
        dati = tomllib.loads(grezzo.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        sys.exit(f"configurazione non valida ({p}): {exc}")
    cfg = Configurazione(dati)
    cfg["_percorso"] = str(p.resolve())
    # Impronta calcolata con i fine riga uniformati a LF: su Windows Git può riscrivere il file con CRLF, e
    # lo stesso config.toml deve dare la stessa impronta sulla macchina di misura e sul PC di controllo.
    cfg["_sha256"] = hashlib.sha256(grezzo.replace(b"\r\n", b"\n")).hexdigest()
    # Impronte per sezione, calcolate sul contenuto letto (commenti e fine riga non contano)
    cfg["_sha256_benchmark"] = _impronta({k: v for k, v in dati.items() if k not in SEZIONI_NON_BENCHMARK})
    cfg["_sha256_giudice"] = _impronta(dati.get("giudice"))
    controlla(cfg)
    return cfg


def controlla(cfg: dict) -> None:
    mancanti = []
    for sezione, chiavi in OBBLIGATORIE.items():
        blocco = cfg.get(sezione)
        if not isinstance(blocco, dict):
            mancanti.append(f"[{sezione}]")
            continue
        mancanti += [f"{sezione}.{k}" for k in chiavi if k not in blocco]
    if mancanti:
        sys.exit("configurazione incompleta, mancano: " + ", ".join(mancanti))
    ignote = [s for s in cfg["strategie"]["elenco"] if s not in STRATEGIE_NOTE]
    if ignote:
        sys.exit(f"strategie sconosciute in strategie.elenco: {ignote}")
    if "S0" not in cfg["strategie"]["elenco"]:
        sys.exit("strategie.elenco deve contenere S0: i turni prima del trigger si misurano con S0")
    q = cfg["strategie"]["quota_token_tenuti"]
    if not (0 < q < 1):
        sys.exit("strategie.quota_token_tenuti deve essere tra 0 e 1 (es. 0.5 = si tiene il 50% dei token)")
    if cfg["misura"]["metodo_finestre"] not in ("auto", "task", "tracker"):
        sys.exit("misura.metodo_finestre deve essere auto, task o tracker")
    f = cfg["esecuzione"]["fasce_turni"]
    if not f or not all(isinstance(x, int) and x > 0 for x in f) or list(f) != sorted(set(f)):
        sys.exit("esecuzione.fasce_turni: servono interi positivi in ordine crescente, senza ripetizioni")
    g = cfg["giudice"]
    attesi = {"prompt_relativo": {"storia", "messaggio", "riferimento", "risposta"},
              "prompt_assoluto": {"storia", "messaggio", "risposta"}}
    for chiave, voluti in attesi.items():
        trovati = SEGNAPOSTO.findall(g[chiave])
        if set(trovati) != voluti or len(trovati) != len(voluti):
            sys.exit(f"giudice.{chiave}: servono una volta sola i segnaposto {sorted(voluti)}, trovati {trovati}")
    for chiave in ("campione_stabilita", "campione_identita", "campione_degrado"):
        if not (0 <= g[chiave] <= 1):
            sys.exit(f"giudice.{chiave} deve essere tra 0 e 1")
    if abs(g["peso_relativi"] + g["peso_assoluti"] - 1) > 1e-9:
        sys.exit("giudice.peso_relativi + giudice.peso_assoluti deve fare 1")
    ignote = set(g.get("soglie") or {}) - SOGLIE_NOTE
    if ignote:
        sys.exit(f"soglie sconosciute in [giudice.soglie]: {sorted(ignote)}")
