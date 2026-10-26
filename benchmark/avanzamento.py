"""Riga di avanzamento nella console (runner e giudice), solo libreria standard.

In una console (PowerShell, terminale di M1) la riga si aggiorna sul posto, al massimo due volte al secondo e
tagliata alla larghezza della finestra. Se l'uscita va in un file o in un'altra pipe, si scrive una riga nuova a
ogni punto percentuale, così il log resta leggibile.

La percentuale conta anche il lavoro fatto in sessioni precedenti (ripresa); la stima residua usa solo il ritmo
di questa sessione: tempo trascorso / elementi fatti ora x elementi mancanti. È indicativa.
"""
from __future__ import annotations

import shutil
import sys
import time


def numero(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def durata(secondi: float | None) -> str:
    if secondi is None:
        return "–"
    s = int(round(secondi))
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


class Avanzamento:
    def __init__(self, etichetta: str, totale: int, fatti_prima: int = 0, unita: str = "", flusso=None):
        self.etichetta, self.totale, self.unita = etichetta, max(0, int(totale)), unita
        self.fatti = min(int(fatti_prima), self.totale)
        self.flusso = flusso or sys.stdout
        try:
            self.console = self.flusso.isatty()
        except (AttributeError, ValueError):
            self.console = False
        self.fatti_sessione = 0
        self.t0 = time.monotonic()
        self.ultimo = 0.0
        self.ultimo_punto = -1
        self.stampati = -1          # elementi fatti all'ultima riga scritta (uscita non su console)
        self.dettaglio = ""
        self.mostra(forza=True)

    def percentuale(self) -> float:
        return 100.0 if self.totale == 0 else 100.0 * self.fatti / self.totale

    def riga(self) -> str:
        trascorso = time.monotonic() - self.t0
        residuo = (trascorso / self.fatti_sessione * (self.totale - self.fatti)) if self.fatti_sessione else None
        pct = f"{self.percentuale():.1f}".replace(".", ",")
        parti = [self.etichetta, f"{pct}%", f"({numero(self.fatti)}/{numero(self.totale)}{' ' + self.unita if self.unita else ''})"]
        if self.dettaglio:
            parti.append(self.dettaglio)
        parti += [f"trascorso {durata(trascorso)}", f"stima residua {durata(residuo)}"]
        return "  ".join(parti)

    def mostra(self, forza: bool = False) -> None:
        if self.console:
            ora = time.monotonic()
            if not forza and ora - self.ultimo < 0.5:
                return
            self.ultimo = ora
            larghezza = max(20, shutil.get_terminal_size((120, 20)).columns - 1)
            testo = self.riga()[:larghezza]
            self.flusso.write("\r" + testo.ljust(larghezza))
            self.flusso.flush()
        else:
            punto = int(self.percentuale())
            if forza or punto > self.ultimo_punto:
                self.ultimo_punto, self.stampati = punto, self.fatti
                self.flusso.write(self.riga() + "\n")
                self.flusso.flush()

    def avanza(self, n: int = 1, dettaglio: str | None = None) -> None:
        self.fatti = min(self.totale, self.fatti + n)
        self.fatti_sessione += n
        if dettaglio is not None:
            self.dettaglio = dettaglio
        self.mostra(forza=self.fatti == self.totale)

    def imposta_dettaglio(self, dettaglio: str) -> None:
        self.dettaglio = dettaglio

    def chiudi(self) -> None:
        """Ultima riga, poi a capo: i messaggi successivi non si sovrappongono alla riga di avanzamento."""
        if self.console:
            self.mostra(forza=True)
            self.flusso.write("\n")
            self.flusso.flush()
        elif self.fatti != self.stampati:
            self.mostra(forza=True)
