"""Lettura e controllo di config.toml, comune a tutti gli script del benchmark.

I percorsi scritti nel file sono relativi alla cartella del file stesso. L'impronta SHA-256 del file
finisce nei risultati, così ogni esecuzione dice con quale configurazione è stata fatta.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("serve Python 3.11 o superiore (tomllib)")
import tomllib

# Chiavi obbligatorie, per sezione: un errore di battitura si scopre subito, non dopo ore di benchmark.
OBBLIGATORIE = {
    "esecuzione": ["nome", "seed", "cartella_risultati"],
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
}

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
    cfg["_sha256"] = hashlib.sha256(grezzo).hexdigest()
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
