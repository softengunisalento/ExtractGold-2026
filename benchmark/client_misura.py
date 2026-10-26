"""Client del servizio di misura (misura.py esegui) sulla macchina di misura.

Usa solo la libreria standard, così il PC di controllo non ha dipendenze da installare. Il servizio esegue una
richiesta alla volta e non sa chi lo chiama: lo usano il runner e il giudice (e, nella seconda versione,
l'utente simulato).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class ErroreServizio(Exception):
    """Il servizio non risponde o risponde con un errore."""


class ClientMisura:
    def __init__(self, indirizzo: str, chiave: str = "", timeout_breve: float = 120):
        self.base = indirizzo.rstrip("/")
        self.chiave = chiave
        self.timeout_breve = timeout_breve

    def _chiama(self, metodo: str, percorso: str, corpo: dict | None = None, timeout: float | None = None) -> dict:
        dati = None if corpo is None else json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base + percorso, data=dati, method=metodo,
                                     headers={"Content-Type": "application/json"})
        if self.chiave:
            req.add_header("X-Chiave", self.chiave)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            testo = exc.read().decode("utf-8", "replace")
            raise ErroreServizio(f"{percorso}: HTTP {exc.code}: {testo[:500]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ErroreServizio(f"{percorso}: servizio di misura non raggiungibile ({exc})") from None

    # --- operazioni del servizio ---
    def stato(self) -> dict:
        return self._chiama("GET", "/stato", timeout=self.timeout_breve)

    def carica_modello(self, nome: str, chat_template_kwargs: dict | None = None) -> dict:
        """Avvia llama-server con il modello; i parametri del template (es. reasoning spento) valgono per
        tutte le richieste successive a quel modello."""
        return self._chiama("POST", "/modello/carica", {"nome": nome, "chat_template_kwargs": chat_template_kwargs or {}})

    def carica_giudice(self) -> dict:
        """Avvia llama-server con il giudice: file, contesto, argomenti e parametri del template da [giudice]."""
        return self._chiama("POST", "/modello/carica", {"ruolo": "giudice"})

    def scarica_modello(self) -> dict:
        return self._chiama("POST", "/modello/scarica", {}, timeout=self.timeout_breve)

    def chat(self, messaggi: list[dict], misura: bool, etichetta: str = "", campionamento: dict | None = None,
             schema: dict | None = None, leggi_slot: bool = False) -> dict:
        corpo = {"messaggi": messaggi, "misura": misura, "etichetta": etichetta}
        if campionamento:
            corpo["campionamento"] = campionamento
        if schema:
            corpo["schema"] = schema
        if leggi_slot:
            corpo["leggi_slot"] = True
        return self._chiama("POST", "/chat", corpo)

    def conta_token(self, messaggi: list[dict]) -> dict:
        return self._chiama("POST", "/conta_token", {"messaggi": messaggi})

    def comprimi(self, metodo: str, testi: list[str], quota: float, misura: bool = True, etichetta: str = "") -> dict:
        return self._chiama("POST", "/comprimi", {"metodo": metodo, "testi": testi, "quota_tenuta": quota,
                                                  "misura": misura, "etichetta": etichetta})

    def embedding(self, voci: list[dict], misura: bool = True, etichetta: str = "") -> dict:
        return self._chiama("POST", "/embedding", {"voci": voci, "misura": misura, "etichetta": etichetta})

    def riposo(self, secondi: float) -> dict:
        return self._chiama("POST", "/riposo", {"secondi": secondi})
