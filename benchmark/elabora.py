#!/usr/bin/env python3
"""Elaborazione dei risultati (PC di controllo): tabelle CSV, grafici PNG e rapporto Word.

    python elabora.py [--config config.toml]

Legge i risultati del benchmark (risultati/<nome>/) e i giudizi del giudice configurato
(risultati/<nome>/giudizi/<impronta>/) e scrive tutto in risultati/<nome>/elaborazione/, rifacendola da capo.
Confronti sempre appaiati: ogni strategia contro S0 sulla stessa conversazione e sugli stessi turni, solo dal
trigger in poi. Unità di analisi: la conversazione.

  A  quadro dei dati          B  energia            C  qualità
  D  compromesso energia-qualità, risultati per fascia di turni, medie semplici e pesate
  E  statistica               F  affidabilità del giudice e robustezza   G  confronto tra modelli (se più di uno)

Medie pesate: il run usa le prime N conversazioni nell'ordine a giri (una per fascia di turni a ogni giro), quindi
le fasce hanno nel run un peso diverso da quello del dataset. La media pesata dà a ogni fascia il peso che ha
nell'intero dataset congelato (rinormalizzato sulle fasce presenti nel run).

Richiede le librerie di requirements_runner.txt.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from docx import Document
    from docx.enum.section import WD_ORIENT  # noqa: F401  (verifica che python-docx sia completo)
    from docx.shared import Cm, Pt
    from scipy import stats
except ImportError as exc:  # messaggio chiaro invece di un errore a metà
    sys.exit(f"manca una libreria per l'elaborazione ({exc.name}): pip install -r requirements_runner.txt")

import configurazione
import dati
from giudice import VOTI, cartella_giudizi, leggi_jsonl
from risultati import cartella_esecuzione

COMPONENTI_E = ("prefill", "decode", "overhead")
ASS, REL = VOTI["B"], VOTI["A"]


# ---------------------------------------------------------------------------
# Lettura
# ---------------------------------------------------------------------------

def _energia(r: dict, chiave: str) -> dict:
    """Energia totale (Wh) per finestra: prefill e decode per le risposte ai turni, overhead per il resto."""
    e = r.get(chiave) or {}

    def tot(f):
        v = (e.get(f) or {}).get("totale_wh")
        return np.nan if v is None else float(v)

    if r.get("tipo") == "turno":
        return {"prefill": tot("prefill"), "decode": tot("decode"), "overhead": 0.0}
    valori = [tot(f) for f in e if e.get(f) is not None]
    return {"prefill": 0.0, "decode": 0.0, "overhead": float(np.nansum(valori)) if valori else np.nan}


def leggi_richieste(res: Path) -> pd.DataFrame:
    righe = []
    for r in leggi_jsonl(res / "richieste.jsonl"):
        netta, lorda = _energia(r, "energia_netta"), _energia(r, "energia_lorda")
        righe.append({k: r.get(k) for k in ("modello", "conversazione", "fonte", "strategia", "turno", "turno_trigger",
                                             "pre_trigger", "tipo", "esito", "fine_finestra", "token_prompt",
                                             "token_valutati", "token_in_cache", "token_generati")}
                     | {f"netta_{c}": netta[c] for c in COMPONENTI_E} | {f"lorda_{c}": lorda[c] for c in COMPONENTI_E})
    df = pd.DataFrame(righe)
    if df.empty:
        return df
    df["netta"] = df[[f"netta_{c}" for c in COMPONENTI_E]].sum(axis=1, min_count=1)
    df["lorda"] = df[[f"lorda_{c}" for c in COMPONENTI_E]].sum(axis=1, min_count=1)
    df["dopo_trigger"] = df["turno_trigger"].notna() & (df["turno"] >= df["turno_trigger"])
    return df


def leggi_giudizi(cartella: Path) -> pd.DataFrame:
    righe = []
    for g in leggi_jsonl(cartella / "giudizi.jsonl"):
        r = {k: g.get(k) for k in ("id", "modello", "conversazione", "fonte", "turno", "turno_trigger", "distanza",
                                   "strategia", "tipo", "controllo", "originale", "esito", "motivo",
                                   "risposta_fine_finestra", "riferimento_fine_finestra", "lunghezza_risposta",
                                   "token_risposta", "tentativi")}
        for v in ASS + REL:
            r[v] = (g.get("voti") or {}).get(v, np.nan)
        righe.append(r)
    return pd.DataFrame(righe)


# ---------------------------------------------------------------------------
# Statistica
# ---------------------------------------------------------------------------

class Stat:
    def __init__(self, cfg_el: dict):
        self.B = int(cfg_el["bootstrap_ripetizioni"])
        self.livello = float(cfg_el["livello_confidenza"])
        self.seed = int(cfg_el["seed"])

    def ic(self, x) -> tuple[float, float]:
        x = np.asarray(pd.Series(x).dropna(), dtype=float)
        if len(x) < 2:
            return np.nan, np.nan
        rng = np.random.default_rng(self.seed)
        medie = x[rng.integers(0, len(x), (self.B, len(x)))].mean(axis=1)
        a = (1 - self.livello) / 2
        return float(np.quantile(medie, a)), float(np.quantile(medie, 1 - a))


def wilcoxon_appaiato(a, b) -> dict:
    """Test di Wilcoxon sulle differenze a-b (coppie complete) e correlazione rango-biseriale."""
    d = (pd.Series(a, dtype=float) - pd.Series(b, dtype=float)).dropna()
    d = d[d != 0]
    out = {"n": int(len(d)), "mediana_differenza": float(d.median()) if len(d) else np.nan, "p": np.nan, "r_rb": np.nan}
    if len(d) >= 2:
        try:
            out["p"] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            pass
        ranghi = stats.rankdata(d.abs())
        wp, wm = ranghi[d.values > 0].sum(), ranghi[d.values < 0].sum()
        out["r_rb"] = float((wp - wm) / (wp + wm)) if wp + wm else np.nan
    return out


def holm(p: list[float]) -> list[float]:
    idx = [i for i, v in enumerate(p) if not np.isnan(v)]
    ordinati = sorted(idx, key=lambda i: p[i])
    out, massimo = [np.nan] * len(p), 0.0
    for rango, i in enumerate(ordinati):
        massimo = max(massimo, min(1.0, (len(ordinati) - rango) * p[i]))
        out[i] = massimo
    return out


def spearman(x, y) -> float:
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


# ---------------------------------------------------------------------------
# Analisi di un modello
# ---------------------------------------------------------------------------

class Analisi:
    def __init__(self, cfg, modello: str, rich: pd.DataFrame, giud: pd.DataFrame, trig: dict, st: Stat, out: Path,
                 fasce: dict | None = None):
        self.cfg, self.m, self.st, self.out = cfg, modello, st, out
        # fasce: {"di": {id conversazione: fascia}, "ordine": [fasce], "pesi": {fascia: quota nel dataset},
        #         "dataset": {fascia: conversazioni nel dataset}}
        self.fasce = fasce or {"di": {}, "ordine": [], "pesi": {}, "dataset": {}}
        self.elenco = list(cfg["strategie"]["elenco"])
        self.tecniche = [s for s in self.elenco if s != "S0"]
        self.rich = rich[rich["modello"] == modello].copy() if not rich.empty else rich
        self.giud = giud[giud["modello"] == modello].copy() if not giud.empty else giud
        self.trig = trig
        self.min_conv = int(cfg["elaborazione"]["min_conversazioni_per_punto"])
        self.tabelle: dict[str, pd.DataFrame] = {}
        self.grafici: dict[str, Path] = {}
        self.note: dict[str, list[str]] = {}
        self.colori = {s: plt.get_cmap("tab10")(i % 10) for i, s in enumerate(self.elenco)}

    # --- utilità ---
    def salva(self, nome: str, df: pd.DataFrame) -> pd.DataFrame:
        self.tabelle[nome] = df
        df.to_csv(self.out / "tabelle" / f"{self.m}__{nome}.csv", index=False)
        return df

    def figura(self, nome: str, fig) -> None:
        p = self.out / "grafici" / f"{self.m}__{nome}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        self.grafici[nome] = p

    def nota(self, sezione: str, testo: str) -> None:
        self.note.setdefault(sezione, []).append(testo)

    # --- A: quadro dei dati ---
    def quadro(self, modelli_jsonl: list[dict]) -> None:
        conv = self.trig.get("conversazioni") or {}
        if not self.rich.empty:  # solo le conversazioni del run (il file del trigger può contenerne di più)
            nel_run = set(self.rich["conversazione"])
            conv = {k: v for k, v in conv.items() if k in nel_run}
        con = [v for v in conv.values() if v.get("turno_trigger")]
        dopo = [v["turni"] - v["turno_trigger"] + 1 for v in con]
        tt = [v["turno_trigger"] for v in con]
        caric = [m for m in modelli_jsonl if m.get("modello") == self.m and m.get("evento") == "caricamento"]
        riposo = [m for m in modelli_jsonl if m.get("modello") == self.m and m.get("evento") == "riposo"]
        r = self.rich
        voci = [
            ("Conversazioni", len(conv)), ("Con trigger", len(con)), ("Senza trigger", len(conv) - len(con)),
            ("Turno di trigger (min / mediana / max)", f"{min(tt)} / {np.median(tt):g} / {max(tt)}" if tt else "–"),
            ("Turni dal trigger in poi (min / mediana / max)", f"{min(dopo)} / {np.median(dopo):g} / {max(dopo)}" if dopo else "–"),
            ("Turni giudicati per strategia", sum(dopo)),
            ("Caricamenti (esiti)", ", ".join(m.get("esito", "?") for m in caric) or "–"),
            ("Consumo a riposo misurato (W totali, per sessione)",
             ", ".join(f"{(m.get('potenza_media') or {}).get('totale_w', float('nan')):.1f}" for m in riposo) or "–"),
        ]
        if not r.empty:
            voci += [("Richieste registrate", len(r)),
                     ("Richieste con contesto superato", int((r["esito"] == "contesto_superato").sum())),
                     ("Richieste con errore", int((r["esito"] == "errore").sum())),
                     ("Risposte troncate (fine finestra)", int(r["fine_finestra"].fillna(False).astype(bool).sum()))]
        self.salva("A_quadro", pd.DataFrame(voci, columns=["voce", "valore"]).astype({"valore": str}))
        if not r.empty:
            t = r[r["tipo"] == "turno"].groupby("strategia").agg(
                richieste=("esito", "size"), contesto_superato=("esito", lambda x: int((x == "contesto_superato").sum())),
                errori=("esito", lambda x: int((x == "errore").sum())),
                troncate=("fine_finestra", lambda x: int(x.fillna(False).astype(bool).sum()))).reindex(self.elenco)
            self.salva("A_richieste_per_strategia", t.reset_index())

    # --- B: energia ---
    def per_conversazione(self) -> pd.DataFrame:
        """Energia e token dal trigger in poi, per conversazione e strategia (energia netta, in Wh)."""
        r = self.rich[self.rich["dopo_trigger"]]
        if r.empty:
            return pd.DataFrame()
        turni = r[r["tipo"] == "turno"]
        g = r.groupby(["conversazione", "fonte", "strategia"], as_index=False).agg(
            netta=("netta", "sum"), lorda=("lorda", "sum"), **{f"netta_{c}": (f"netta_{c}", "sum") for c in COMPONENTI_E})
        n = turni.groupby(["conversazione", "strategia"], as_index=False).agg(
            turni=("turno", "nunique"), token_prompt=("token_prompt", "sum"), token_in_cache=("token_in_cache", "sum"),
            token_valutati=("token_valutati", "sum"), token_generati=("token_generati", "sum"))
        g = g.merge(n, on=["conversazione", "strategia"], how="left")
        s0 = g[g["strategia"] == "S0"][["conversazione", "netta"]].rename(columns={"netta": "netta_S0"})
        g = g.merge(s0, on="conversazione", how="left")
        g["risparmio"] = np.where(g["netta_S0"] > 0, 1 - g["netta"] / g["netta_S0"], np.nan)
        s0l = g[g["strategia"] == "S0"].set_index("conversazione")["lorda"]
        g["risparmio_lordo"] = 1 - g["lorda"] / g["conversazione"].map(s0l)
        g["netta_per_turno"] = g["netta"] / g["turni"]
        return g

    def energia(self) -> None:
        pc = self.per_conversazione()
        if pc.empty:
            self.nota("B", "Nessuna richiesta dopo il trigger: analisi dell'energia non disponibile.")
            return
        self.pc = pc
        righe = []
        for s in self.elenco:
            x = pc[pc["strategia"] == s]
            lo, hi = self.st.ic(x["risparmio"]) if s != "S0" else (np.nan, np.nan)
            righe.append({"strategia": s, "conversazioni": len(x),
                          "Wh_per_turno": x["netta_per_turno"].mean(),
                          **{f"Wh_{c}_per_turno": (x[f"netta_{c}"] / x["turni"]).mean() for c in COMPONENTI_E},
                          "risparmio_medio": x["risparmio"].mean() if s != "S0" else 0.0,
                          "risparmio_ic_basso": lo, "risparmio_ic_alto": hi,
                          "risparmio_mediano": x["risparmio"].median() if s != "S0" else 0.0,
                          "risparmio_lordo_medio": x["risparmio_lordo"].mean() if s != "S0" else 0.0,
                          "risparmio_sul_totale": 1 - x["netta"].sum() / pc[pc["strategia"] == "S0"]["netta"].sum()
                          if s != "S0" else 0.0})
        self.salva("B_energia_per_strategia", pd.DataFrame(righe))
        tok = []
        turni = self.rich[self.rich["dopo_trigger"] & (self.rich["tipo"] == "turno")]
        for s in self.elenco:
            x = turni[turni["strategia"] == s]
            tok.append({"strategia": s, "token_prompt_per_turno": x["token_prompt"].mean(),
                        "token_in_cache_per_turno": x["token_in_cache"].mean(),
                        "token_valutati_per_turno": x["token_valutati"].mean(),
                        "token_generati_per_turno": x["token_generati"].mean(),
                        "mWh_per_token_valutato": 1000 * x["netta_prefill"].sum() / x["token_valutati"].sum()
                        if x["token_valutati"].sum() else np.nan,
                        "mWh_per_token_generato": 1000 * x["netta_decode"].sum() / x["token_generati"].sum()
                        if x["token_generati"].sum() else np.nan})
        self.salva("B_token_per_strategia", pd.DataFrame(tok))
        # grafico: energia per turno, divisa in prefill, decode e overhead
        t = self.tabelle["B_energia_per_strategia"]
        fig, ax = plt.subplots(figsize=(8, 4))
        base = np.zeros(len(t))
        for c, col in zip(COMPONENTI_E, ("#4C72B0", "#DD8452", "#55A868")):
            v = t[f"Wh_{c}_per_turno"].fillna(0).values
            ax.bar(t["strategia"], v, bottom=base, label=c, color=col)
            base += v
        ax.set_ylabel("Wh per turno (netti)")
        ax.set_title(f"{self.m}: energia per turno dal trigger in poi")
        ax.legend()
        self.figura("B_energia_per_turno", fig)
        # grafico: risparmio rispetto a S0 con intervallo di confidenza
        tt = t[t["strategia"] != "S0"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(tt["strategia"], 100 * tt["risparmio_medio"], color=[self.colori[s] for s in tt["strategia"]],
               yerr=[100 * (tt["risparmio_medio"] - tt["risparmio_ic_basso"]).fillna(0),
                     100 * (tt["risparmio_ic_alto"] - tt["risparmio_medio"]).fillna(0)], capsize=4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("risparmio rispetto a S0 (%)")
        ax.set_title(f"{self.m}: risparmio medio per conversazione (IC {self.st.livello:.0%})")
        self.figura("B_risparmio", fig)
        self.andamento_e_pareggio()

    def andamento_e_pareggio(self) -> None:
        r = self.rich[self.rich["dopo_trigger"]].copy()
        r["distanza"] = r["turno"] - r["turno_trigger"]
        pt = r.groupby(["conversazione", "strategia", "distanza"], as_index=False)["netta"].sum()
        # andamento: energia media per turno in funzione della distanza dal trigger
        fig, ax = plt.subplots(figsize=(8, 4))
        righe = []
        for s in self.elenco:
            x = pt[pt["strategia"] == s].groupby("distanza")["netta"].agg(["mean", "count"])
            x = x[x["count"] >= self.min_conv]
            ax.plot(x.index, x["mean"], marker="o", ms=3, label=s, color=self.colori[s])
            righe += [{"strategia": s, "distanza": int(d), "Wh_medi": v, "conversazioni": int(n)}
                      for d, (v, n) in x.iterrows()]
        ax.set_xlabel("turni dopo il trigger")
        ax.set_ylabel("Wh per turno (netti)")
        ax.set_title(f"{self.m}: energia per turno (punti con almeno {self.min_conv} conversazioni)")
        ax.legend(ncol=4, fontsize=8)
        self.figura("B_andamento", fig)
        self.salva("B_andamento", pd.DataFrame(righe))
        # pareggio: dopo quanti turni il costo una tantum (riassunti, gold prompt) è ripagato
        s0 = pt[pt["strategia"] == "S0"].set_index(["conversazione", "distanza"])["netta"]
        righe, curve = [], {}
        for s in self.tecniche:
            x = pt[pt["strategia"] == s].set_index(["conversazione", "distanza"])["netta"]
            diff = (x - s0.reindex(x.index)).dropna().reset_index()
            diff = diff.sort_values(["conversazione", "distanza"])
            diff["cumulata"] = diff.groupby("conversazione")["netta"].cumsum()
            pareggi = diff[diff["cumulata"] <= 0].groupby("conversazione")["distanza"].min()
            n = diff["conversazione"].nunique()
            righe.append({"strategia": s, "conversazioni": n, "quota_che_pareggia": len(pareggi) / n if n else np.nan,
                          "distanza_mediana_di_pareggio": pareggi.median() if len(pareggi) else np.nan,
                          "pareggia_al_trigger": float((pareggi == 0).mean()) if len(pareggi) else np.nan})
            cm = diff.groupby("distanza")["cumulata"].agg(["mean", "count"])
            curve[s] = cm[cm["count"] >= self.min_conv]["mean"]
        self.salva("B_pareggio", pd.DataFrame(righe))
        fig, ax = plt.subplots(figsize=(8, 4))
        for s, c in curve.items():
            ax.plot(c.index, c.values, marker="o", ms=3, label=s, color=self.colori[s])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("turni dopo il trigger")
        ax.set_ylabel("differenza cumulata con S0 (Wh)")
        ax.set_title(f"{self.m}: costo cumulato rispetto a S0 (sotto lo zero = la tecnica ha già pareggiato)")
        ax.legend(ncol=4, fontsize=8)
        self.figura("B_pareggio", fig)

    # --- C: qualità ---
    def voti_per_turno(self, escludi_troncate: bool = False) -> pd.DataFrame:
        """Una riga per (conversazione, turno, strategia) con i 4 voti e il punteggio combinato."""
        g = self.giud
        if g.empty:
            return pd.DataFrame()
        base = g[g["controllo"].isna() & (g["esito"] == "ok")]
        if escludi_troncate:
            tronc = base["risposta_fine_finestra"].fillna(False).astype(bool) | \
                base["riferimento_fine_finestra"].fillna(False).astype(bool)
            base = base[~tronc]
        chiavi = ["conversazione", "fonte", "turno", "distanza", "strategia"]
        a = base[base["tipo"] == "B"][chiavi + list(ASS)]
        r = base[base["tipo"] == "A"][chiavi + list(REL)]
        v = a.merge(r, on=chiavi, how="outer")
        g_ = self.cfg["giudice"]
        v["media_assoluti"] = v[list(ASS)].mean(axis=1, skipna=False)
        v["media_relativi"] = v[list(REL)].mean(axis=1, skipna=False)
        v["punteggio"] = g_["peso_relativi"] * v["media_relativi"] + g_["peso_assoluti"] * v["media_assoluti"]
        if escludi_troncate and not self.rich.empty:
            # si tolgono anche i turni in cui è troncata la risposta S0 (il riferimento)
            t = self.rich[(self.rich["tipo"] == "turno") & self.rich["fine_finestra"].fillna(False).astype(bool)]
            tr = set(zip(t["conversazione"], t["turno"], t["strategia"]))
            v = v[[(c, k, s) not in tr and (c, k, "S0") not in tr
                   for c, k, s in zip(v["conversazione"], v["turno"], v["strategia"])]]
        return v

    def qualita(self) -> None:
        v = self.voti_per_turno()
        if v.empty:
            self.nota("C", "Nessun giudizio disponibile per questo modello: analisi della qualità non disponibile.")
            return
        self.v = v
        pc = v.groupby(["conversazione", "fonte", "strategia"], as_index=False)[
            list(ASS) + list(REL) + ["media_assoluti", "media_relativi", "punteggio"]].mean()
        self.vc = pc
        righe = []
        for s in self.elenco:
            x, xt = pc[pc["strategia"] == s], v[v["strategia"] == s]
            lo, hi = self.st.ic(x["punteggio"])
            righe.append({"strategia": s, "conversazioni": len(x), "turni": len(xt),
                          **{c: x[c].mean() for c in ASS + REL},
                          "punteggio_medio": x["punteggio"].mean(), "punteggio_ic_basso": lo, "punteggio_ic_alto": hi,
                          "punteggio_mediano": x["punteggio"].median()})
        t = self.salva("C_qualita_per_strategia", pd.DataFrame(righe))
        self.nota("C", "S0 riceve solo i voti assoluti (è il riferimento dei relativi): il suo punteggio combinato "
                       "non è definito.")
        # grafico: voti medi
        fig, ax = plt.subplots(figsize=(8, 4))
        larg = 0.2
        xs = np.arange(len(t))
        for i, c in enumerate(ASS + REL):
            ax.bar(xs + (i - 1.5) * larg, t[c], larg, label=c)
        ax.set_xticks(xs, t["strategia"])
        ax.set_ylim(0, 10.5)
        ax.set_ylabel("voto medio (1–10)")
        ax.set_title(f"{self.m}: voti medi per strategia")
        ax.legend(fontsize=8, ncol=2)
        self.figura("C_voti", fig)
        # distribuzione del punteggio
        fig, ax = plt.subplots(figsize=(8, 4))
        dati_box = [v[v["strategia"] == s]["punteggio"].dropna() for s in self.tecniche]
        ax.boxplot(dati_box, tick_labels=self.tecniche)
        ax.set_ylabel("punteggio combinato (per turno)")
        ax.set_title(f"{self.m}: distribuzione del punteggio")
        self.figura("C_distribuzione", fig)
        # andamento con la distanza dal trigger
        fig, ax = plt.subplots(figsize=(8, 4))
        righe = []
        for s in self.elenco:
            col = "media_assoluti" if s == "S0" else "punteggio"
            x = v[v["strategia"] == s].groupby("distanza")[col].agg(["mean", "count"])
            x = x[x["count"] >= self.min_conv]
            ax.plot(x.index, x["mean"], marker="o", ms=3, label=s + (" (assoluti)" if s == "S0" else ""),
                    color=self.colori[s], ls="--" if s == "S0" else "-")
            righe += [{"strategia": s, "misura": col, "distanza": int(d), "media": m_, "turni": int(n)}
                      for d, (m_, n) in x.iterrows()]
        ax.set_xlabel("turni dopo il trigger")
        ax.set_ylabel("punteggio")
        ax.set_title(f"{self.m}: qualità con la distanza dal trigger")
        ax.legend(ncol=4, fontsize=8)
        self.figura("C_andamento", fig)
        self.salva("C_andamento", pd.DataFrame(righe))
        # per fonte del dataset
        fonti = pc.groupby(["fonte", "strategia"], as_index=False).agg(
            conversazioni=("conversazione", "nunique"), media_assoluti=("media_assoluti", "mean"),
            punteggio=("punteggio", "mean"))
        if hasattr(self, "pc"):
            en = self.pc.groupby(["fonte", "strategia"], as_index=False)["risparmio"].mean()
            fonti = fonti.merge(en, on=["fonte", "strategia"], how="left")
        self.salva("C_per_fonte", fonti)

    # --- D: compromesso ---
    def compromesso(self) -> None:
        if not hasattr(self, "pc") or not hasattr(self, "vc"):
            self.nota("D", "Servono sia l'energia sia i giudizi.")
            return
        e = self.tabelle["B_energia_per_strategia"].set_index("strategia")
        q = self.tabelle["C_qualita_per_strategia"].set_index("strategia")
        righe = [{"strategia": s, "risparmio_medio": e.loc[s, "risparmio_medio"], "punteggio_medio": q.loc[s, "punteggio_medio"],
                  "media_assoluti": q.loc[s, ["absolute_correctness", "absolute_coherence"]].mean()} for s in self.tecniche]
        t = pd.DataFrame(righe)
        pareto = []
        for _, a in t.iterrows():
            dominata = any((b["risparmio_medio"] >= a["risparmio_medio"]) and (b["punteggio_medio"] >= a["punteggio_medio"])
                           and ((b["risparmio_medio"] > a["risparmio_medio"]) or (b["punteggio_medio"] > a["punteggio_medio"]))
                           for _, b in t.iterrows() if b["strategia"] != a["strategia"])
            pareto.append(not dominata and not np.isnan(a["risparmio_medio"]) and not np.isnan(a["punteggio_medio"]))
        t["frontiera_pareto"] = pareto
        t = t.sort_values("punteggio_medio", ascending=False)
        t.insert(0, "posizione_per_qualita", range(1, len(t) + 1))
        t["posizione_per_risparmio"] = t["risparmio_medio"].rank(ascending=False, method="min")
        self.salva("D_compromesso", t)
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, a in t.iterrows():
            ax.scatter(100 * a["risparmio_medio"], a["punteggio_medio"], s=80 if a["frontiera_pareto"] else 40,
                       color=self.colori[a["strategia"]], edgecolor="black" if a["frontiera_pareto"] else "none")
            ax.annotate(a["strategia"], (100 * a["risparmio_medio"], a["punteggio_medio"]), textcoords="offset points",
                        xytext=(5, 5))
        f = t[t["frontiera_pareto"]].sort_values("risparmio_medio")
        ax.plot(100 * f["risparmio_medio"], f["punteggio_medio"], color="gray", ls=":", lw=1)
        ax.set_xlabel("risparmio di energia rispetto a S0 (%)")
        ax.set_ylabel("punteggio combinato medio")
        ax.set_title(f"{self.m}: energia contro qualità (bordo nero = frontiera di Pareto)")
        self.figura("D_compromesso", fig)

    # --- D (seconda parte): fasce di turni e medie pesate ---
    def per_fascia(self) -> None:
        f = self.fasce
        if not f["ordine"]:
            return
        # composizione: fasce nel dataset e nel run
        nel_run = pd.Series([f["di"].get(c) for c in self.rich["conversazione"].unique()]).value_counts() \
            if not self.rich.empty else pd.Series(dtype=int)
        tot_ds, tot_run = sum(f["dataset"].values()), int(nel_run.sum())
        self.salva("A_fasce", pd.DataFrame([{"fascia_turni": x, "conversazioni_dataset": f["dataset"].get(x, 0),
                                             "quota_dataset": f["dataset"].get(x, 0) / tot_ds if tot_ds else np.nan,
                                             "conversazioni_run": int(nel_run.get(x, 0)),
                                             "quota_run": nel_run.get(x, 0) / tot_run if tot_run else np.nan}
                                            for x in f["ordine"]]))
        pezzi = []
        if hasattr(self, "pc"):
            pezzi.append(self.pc[["conversazione", "strategia", "risparmio"]])
        if hasattr(self, "vc"):
            pezzi.append(self.vc[["conversazione", "strategia", "punteggio", "media_assoluti"]])
        if not pezzi:
            return
        d = pezzi[0]
        for x in pezzi[1:]:
            d = d.merge(x, on=["conversazione", "strategia"], how="outer")
        d["fascia"] = d["conversazione"].map(f["di"])
        misure = [c for c in ("risparmio", "punteggio", "media_assoluti") if c in d]
        pf = d.groupby(["strategia", "fascia"], as_index=False).agg(
            conversazioni=("conversazione", "nunique"), **{m: (m, "mean") for m in misure})
        pf["fascia"] = pd.Categorical(pf["fascia"], categories=f["ordine"], ordered=True)
        pf = pf.sort_values(["strategia", "fascia"])
        self.salva("D_per_fascia", pf.assign(fascia=pf["fascia"].astype(str)))
        righe = []
        for s in self.elenco:
            x = pf[pf["strategia"] == s]
            r = {"strategia": s, "fasce_presenti": int(x["fascia"].nunique())}
            for m in misure:
                ok = x[x[m].notna()]
                w = ok["fascia"].astype(str).map(f["pesi"]).astype(float)
                r[f"{m}_semplice"] = d[(d["strategia"] == s)][m].mean()
                r[f"{m}_pesata"] = float((w * ok[m]).sum() / w.sum()) if len(ok) and w.sum() > 0 else np.nan
            righe.append(r)
        self.salva("D_medie_pesate", pd.DataFrame(righe))
        self.nota("D", "Medie pesate: ogni fascia di turni conta in proporzione alle sue conversazioni nell'intero "
                       "dataset (rinormalizzato sulle fasce presenti); medie semplici: ogni conversazione del run conta "
                       "uguale. I risultati per fascia mostrano come cambia l'effetto con la lunghezza della conversazione.")
        for m, etich, scala in (("risparmio", "risparmio rispetto a S0 (%)", 100), ("punteggio", "punteggio combinato", 1)):
            if m not in pf:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            for s in self.tecniche:
                x = pf[pf["strategia"] == s]
                ax.plot(x["fascia"].astype(str), scala * x[m], marker="o", ms=4, label=s, color=self.colori[s])
            ax.set_xlabel("fascia di turni della conversazione")
            ax.set_ylabel(etich)
            ax.set_title(f"{self.m}: {etich} per fascia di turni")
            ax.legend(ncol=4, fontsize=8)
            self.figura(f"D_fasce_{m}", fig)

    # --- E: statistica ---
    def statistica(self) -> None:
        righe = []
        for famiglia in ("energia", "voti_assoluti"):
            parz = []
            for s in self.tecniche:
                if famiglia == "energia" and hasattr(self, "pc"):
                    x = self.pc.pivot_table(index="conversazione", columns="strategia", values="netta_per_turno")
                elif famiglia == "voti_assoluti" and hasattr(self, "vc"):
                    x = self.vc.pivot_table(index="conversazione", columns="strategia", values="media_assoluti")
                else:
                    continue
                if s not in x or "S0" not in x:
                    continue
                parz.append({"famiglia": famiglia, "strategia": s, **wilcoxon_appaiato(x[s], x["S0"])})
            for r, ph in zip(parz, holm([r["p"] for r in parz])):
                r["p_holm"] = ph
            righe += parz
        if righe:
            self.salva("E_test_contro_S0", pd.DataFrame(righe))
        self.nota("E", "Test di Wilcoxon appaiato per conversazione (strategia contro S0), correzione di Holm sui 7 "
                       "confronti di ogni famiglia; r_rb = correlazione rango-biseriale (dimensione dell'effetto). "
                       "Energia: Wh netti per turno dal trigger in poi; differenza negativa = la tecnica consuma meno. "
                       "Voti assoluti: media di correttezza e coerenza; differenza negativa = la tecnica è giudicata peggio.")

    # --- F: affidabilità del giudice e robustezza ---
    def referto(self, sessioni_giudice: list[dict]) -> None:
        g = self.giud
        if g.empty:
            self.nota("F", "Nessun giudizio disponibile.")
            return
        soglie = self.cfg["giudice"].get("soglie") or {}
        righe = []

        def voce(n, nome, valore, chiave_soglia=None, alto_male=True, dettaglio=""):
            s = soglie.get(chiave_soglia) if chiave_soglia else None
            if valore is None or (isinstance(valore, float) and np.isnan(valore)):
                stato = "dati insufficienti"
            elif s is None:
                stato = "soglia da fissare" if chiave_soglia else "–"
            else:
                stato = "avviso" if (valore > s if alto_male else valore < s) else "ok"
            righe.append({"indicatore": n, "voce": nome, "valore": valore, "soglia": s, "stato": stato,
                          "dettaglio": dettaglio})

        chiamati = g[g["esito"].isin(["ok", "non_valido", "contesto_superato_giudice"])]
        base = g[g["controllo"].isna()]
        ok = base[base["esito"] == "ok"]
        # 1. uscite non valide
        q_nv = float((chiamati["esito"] == "non_valido").mean()) if len(chiamati) else np.nan
        voce(1, "quota di uscite non valide (dopo i nuovi tentativi)", q_nv, "non_validi_max",
             dettaglio=f"{int((chiamati['esito'] == 'non_valido').sum())} su {len(chiamati)}; "
                       f"giudizi riusciti solo con un nuovo tentativo: {int((chiamati['tentativi'].fillna(1) > 1).sum())}")
        voce(1, "giudizi con contesto del giudice superato", int((chiamati["esito"] == "contesto_superato_giudice").sum()))
        # 2. distribuzione dei voti
        distr = []
        for c in ASS + REL:
            x = ok[c].dropna()
            if len(x):
                distr.append({"voto": c, "n": len(x), "media": x.mean(), "dev_std": x.std(), "quota_10": (x == 10).mean(),
                              **{str(i): int((x == i).sum()) for i in range(1, 11)}})
        dd = pd.DataFrame(distr)
        if not dd.empty:
            self.salva("F_distribuzione_voti", dd)
            voce(2, "quota massima di voti 10 (effetto soffitto)", float(dd["quota_10"].max()), "quota_dieci_max",
                 dettaglio="; ".join(f"{r.voto}: media {r.media:.2f}, dev. std {r.dev_std:.2f}" for r in dd.itertuples()))
            fig, ax = plt.subplots(figsize=(8, 4))
            for c in dd["voto"]:
                ax.plot(range(1, 11), [dd.set_index("voto").loc[c, str(i)] for i in range(1, 11)], marker="o", label=c)
            ax.set_xlabel("voto")
            ax.set_ylabel("numero di giudizi")
            ax.set_title(f"{self.m}: distribuzione dei voti (giudizi base)")
            ax.legend(fontsize=8)
            self.figura("F_distribuzione_voti", fig)
        # 3. stabilità
        orig = g.set_index("id")
        diff = []
        for _, r in g[(g["controllo"] == "stabilita") & (g["esito"] == "ok")].iterrows():
            o = orig.loc[r["originale"]] if r["originale"] in orig.index else None
            if o is not None and o["esito"] == "ok":
                diff += [abs(r[c] - o[c]) for c in VOTI[r["tipo"]]]
        diff = np.array(diff, dtype=float)
        voce(3, "stabilità: differenza media tra due giudizi uguali (seed diverso)",
             float(diff.mean()) if len(diff) else np.nan, "stabilita_differenza_media_max",
             dettaglio=f"{len(diff)} voti confrontati; entro ±1: {(diff <= 1).mean():.1%}" if len(diff) else "")
        # 4. identità
        idn = g[(g["controllo"] == "identita") & (g["esito"] == "ok")]
        vi = idn[list(REL)].stack() if not idn.empty else pd.Series(dtype=float)
        voce(4, "identità: voto relativo medio quando la risposta è S0 stessa (atteso 10)",
             float(vi.mean()) if len(vi) else np.nan, "identita_media_min", alto_male=False,
             dettaglio=f"{len(idn)} giudizi; voti 10: {(vi == 10).mean():.1%}" if len(vi) else "")
        # 5. degrado
        deg = g[(g["controllo"] == "degrado") & (g["esito"] == "ok")]
        vd = pd.concat([deg[deg["tipo"] == "A"][list(REL)].stack(), deg[deg["tipo"] == "B"][list(ASS)].stack()]) \
            if not deg.empty else pd.Series(dtype=float)
        coer = pd.concat([deg[deg["tipo"] == "A"]["relative_coherence"], deg[deg["tipo"] == "B"]["absolute_coherence"]]) \
            if not deg.empty else pd.Series(dtype=float)
        rif = pd.concat([ok[ok["tipo"] == "A"][list(REL)].stack(), ok[ok["tipo"] == "B"][list(ASS)].stack()])
        voce(5, "degrado: voto medio quando la risposta viene da un'altra conversazione (atteso basso)",
             float(vd.mean()) if len(vd) else np.nan, "degrado_media_max",
             dettaglio=(f"{len(deg)} giudizi; coerenza media {coer.mean():.2f}; voto medio dei giudizi normali "
                        f"{rif.mean():.2f}") if len(vd) else "")
        # 6. bias di lunghezza (a parità di turno)
        rho = []
        for tipo, voti in VOTI.items():
            x = ok[ok["tipo"] == tipo].copy()
            x["media"] = x[list(voti)].mean(axis=1)
            for _, gr in x.groupby(["conversazione", "turno"]):
                rho.append(spearman(gr["lunghezza_risposta"], gr["media"]))
        rho = np.array([r for r in rho if not np.isnan(r)])
        voce(6, "bias di lunghezza: correlazione media (Spearman) tra lunghezza e voto nello stesso turno",
             float(rho.mean()) if len(rho) else np.nan, "bias_lunghezza_max",
             dettaglio=f"{len(rho)} turni con almeno 3 risposte confrontabili" if len(rho) else "")
        # 7. coerenza interna
        v = getattr(self, "v", pd.DataFrame())
        tec = v[v["strategia"] != "S0"] if not v.empty else v
        r_ar = spearman(tec["media_assoluti"], tec["media_relativi"]) if not tec.empty else np.nan
        voce(7, "coerenza interna: correlazione tra voti assoluti e relativi", r_ar, "coerenza_interna_min",
             alto_male=False,
             dettaglio=(f"correttezza–coerenza: assoluti {spearman(ok['absolute_correctness'], ok['absolute_coherence']):.2f}, "
                        f"relativi {spearman(ok['relative_correctness'], ok['relative_coherence']):.2f}") if not tec.empty else "")
        # altri conteggi
        ng = g[g["esito"] == "non_giudicabile"]
        voce("–", "casi «tecnica senza S0» (voti relativi non giudicabili, punteggio non calcolato)",
             int((ng["motivo"] == "tecnica_senza_S0").sum()))
        voce("–", "giudizi non fatti per risposta mancante (contesto superato o errore nel benchmark)",
             int((ng["motivo"] == "risposta_mancante").sum()))
        tronc = base[(base["esito"] == "ok") & (base["risposta_fine_finestra"].fillna(False).astype(bool)
                                               | base["riferimento_fine_finestra"].fillna(False).astype(bool))]
        voce("–", "giudizi che coinvolgono risposte troncate", len(tronc))
        par = [s for s in sessioni_giudice if s.get("evento") == "parametri_dichiarati"]
        if par:
            diversi = [s.get("diversi") for s in par if not s.get("coerenti")]
            voce("–", "parametri di campionamento dichiarati da llama-server uguali a quelli richiesti",
                 "sì" if not diversi else "no",
                 dettaglio=f"{len(par)} controlli (uno per caricamento del giudice)"
                           + (f"; differenze: {json.dumps(diversi[-1])}" if diversi else ""))
        self.salva("F_referto_giudice", pd.DataFrame(righe).astype({"valore": object}))
        self.robustezza()

    def robustezza(self) -> None:
        if not hasattr(self, "pc") or not hasattr(self, "vc"):
            return
        v2 = self.voti_per_turno(escludi_troncate=True)
        q2 = v2.groupby(["conversazione", "strategia"], as_index=False)["punteggio"].mean() if not v2.empty else v2
        # energia senza i turni con risposte troncate (della tecnica o di S0)
        r = self.rich[self.rich["dopo_trigger"]]
        t = r[(r["tipo"] == "turno") & r["fine_finestra"].fillna(False).astype(bool)]
        tr = set(zip(t["conversazione"], t["turno"]))
        r2 = r[[(c, k) not in tr for c, k in zip(r["conversazione"], r["turno"])]]
        e2 = r2.groupby(["conversazione", "strategia"], as_index=False)["netta"].sum()
        s0 = e2[e2["strategia"] == "S0"].set_index("conversazione")["netta"]
        e2["risparmio"] = 1 - e2["netta"] / e2["conversazione"].map(s0)
        righe = []
        for s in self.tecniche:
            righe.append({"strategia": s,
                          "risparmio_tutti": self.pc[self.pc["strategia"] == s]["risparmio"].mean(),
                          "risparmio_senza_troncate": e2[e2["strategia"] == s]["risparmio"].mean(),
                          "punteggio_tutti": self.vc[self.vc["strategia"] == s]["punteggio"].mean(),
                          "punteggio_senza_troncate": q2[q2["strategia"] == s]["punteggio"].mean() if not q2.empty else np.nan,
                          "turni_giudicati_tutti": int((self.v["strategia"] == s).sum()),
                          "turni_giudicati_senza_troncate": int((v2["strategia"] == s).sum()) if not v2.empty else 0})
        self.salva("F_robustezza", pd.DataFrame(righe))
        self.nota("F", "Robustezza: i giudizi non validi non entrano mai nelle medie; la tabella confronta i risultati "
                       "con e senza i turni in cui la risposta della tecnica o quella di S0 è troncata (fine finestra).")


# ---------------------------------------------------------------------------
# Rapporto Word
# ---------------------------------------------------------------------------

def formato(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "sì" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def tabella_word(doc, df: pd.DataFrame, max_righe: int = 60) -> None:
    if df is None or df.empty:
        doc.add_paragraph("(nessun dato)")
        return
    df = df.head(max_righe)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c).replace("_", " ")
    for _, r in df.iterrows():
        celle = t.add_row().cells
        for i, c in enumerate(df.columns):
            celle[i].text = formato(r[c])
    for riga in t.rows:
        for cella in riga.cells:
            for p in cella.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(7.5)
    doc.add_paragraph()


SEZIONI = [
    ("A. Quadro dei dati", [("A_quadro", "Riepilogo"), ("A_richieste_per_strategia", "Richieste per strategia (tutti i turni)")], []),
    ("B. Energia", [("B_energia_per_strategia", "Energia netta dal trigger in poi (Wh) e risparmio rispetto a S0"),
                    ("B_token_per_strategia", "Token per turno ed energia per token"),
                    ("B_pareggio", "Punto di pareggio (turni dopo il trigger)")],
     ["B_energia_per_turno", "B_risparmio", "B_andamento", "B_pareggio"]),
    ("C. Qualità", [("C_qualita_per_strategia", "Voti medi per strategia (media per conversazione)"),
                    ("C_per_fonte", "Per fonte del dataset")],
     ["C_voti", "C_distribuzione", "C_andamento"]),
    ("D. Compromesso energia-qualità, fasce di turni e medie pesate",
     [("D_compromesso", "Classifica e frontiera di Pareto (medie semplici)"),
      ("A_fasce", "Composizione per fascia di turni: dataset e run"),
      ("D_per_fascia", "Risultati per fascia di turni"),
      ("D_medie_pesate", "Medie semplici e pesate per strategia")],
     ["D_compromesso", "D_fasce_risparmio", "D_fasce_punteggio"]),
    ("E. Statistica", [("E_test_contro_S0", "Test contro S0")], []),
    ("F. Affidabilità del giudice e robustezza", [("F_referto_giudice", "Referto del giudice"),
                                                  ("F_distribuzione_voti", "Distribuzione dei voti"),
                                                  ("F_robustezza", "Con e senza risposte troncate")],
     ["F_distribuzione_voti"]),
]


def rapporto(cfg, meta: dict, analisi: list[Analisi], confronto: dict, out: Path, info: dict) -> Path:
    doc = Document()
    sez = doc.sections[0]
    sez.page_width, sez.page_height = Cm(21), Cm(29.7)
    for lato in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sez, lato, Cm(2))
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Benchmark delle strategie di gestione del contesto: risultati", 0)
    p = doc.add_paragraph()
    p.add_run(f"Esecuzione «{meta.get('nome')}», elaborata il {info['data']}. ").italic = True
    p.add_run(f"Impronta della configurazione del benchmark {str(meta.get('config_benchmark_sha256'))[:12]}…, "
              f"del giudice {cfg['_sha256_giudice'][:12]}…; conversazioni sha256 "
              f"{str((meta.get('input') or {}).get('conversazioni'))[:12]}…").italic = True
    doc.add_paragraph("Confronti appaiati: ogni strategia contro S0 sulla stessa conversazione e sugli stessi turni, "
                      "solo dal trigger in poi; unità di analisi la conversazione. Energia netta = lorda meno la stima "
                      f"del consumo a riposo. Punteggio combinato = {cfg['giudice']['peso_relativi']} × media dei voti "
                      f"relativi + {cfg['giudice']['peso_assoluti']} × media dei voti assoluti. Intervalli di confidenza "
                      f"al {cfg['elaborazione']['livello_confidenza']:.0%} con bootstrap sulle conversazioni.")
    if info.get("ambiente"):
        doc.add_heading("Ambiente di misura", 1)
        tabella_word(doc, pd.DataFrame(info["ambiente"], columns=["voce", "valore"]))
    for a in analisi:
        doc.add_heading(f"Modello: {a.m}", 1)
        for titolo, tabs, figs in SEZIONI:
            doc.add_heading(titolo, 2)
            for nome, didascalia in tabs:
                if nome in a.tabelle:
                    doc.add_paragraph(didascalia).runs[0].bold = True
                    tabella_word(doc, a.tabelle[nome])
            for f in figs:
                if f in a.grafici:
                    doc.add_picture(str(a.grafici[f]), width=Cm(16))
            for n in a.note.get(titolo[0], []):
                doc.add_paragraph(n).runs[0].italic = True
    if confronto:
        doc.add_heading("G. Confronto tra modelli", 1)
        for nome, df in confronto.items():
            doc.add_paragraph(nome.replace("_", " ")).runs[0].bold = True
            tabella_word(doc, df)
    p = out / "rapporto.docx"
    doc.save(p)
    return p


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Elaborazione dei risultati del benchmark e dei giudizi.")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.toml")))
    ap.add_argument("--modelli", help="solo questi modelli (nomi brevi, separati da virgola)")
    args = ap.parse_args(argv)
    cfg = configurazione.carica(args.config)
    res = cartella_esecuzione(cfg)
    p_meta = res / "esecuzione.json"
    if not p_meta.exists():
        sys.exit(f"nessun risultato in {res}")
    meta = json.loads(p_meta.read_text(encoding="utf-8"))
    if meta.get("config_benchmark_sha256") != cfg["_sha256_benchmark"]:
        sys.exit("i risultati del benchmark sono stati prodotti con una configurazione diversa da questa")
    out = res / "elaborazione"
    if out.exists():
        shutil.rmtree(out)
    (out / "tabelle").mkdir(parents=True)
    (out / "grafici").mkdir()

    rich = leggi_richieste(res)
    cg = cartella_giudizi(cfg, res)
    giud = leggi_giudizi(cg)
    sess_g = list(leggi_jsonl(cg / "sessioni.jsonl"))
    modelli_jsonl = list(leggi_jsonl(res / "modelli.jsonl"))
    ordine = [m["nome"] for m in dati.carica_modelli(cfg.percorso(cfg["input"]["modelli"]),
                                                      cfg.percorso(cfg["input"]["selezione"]))]
    presenti = set(rich["modello"]) if not rich.empty else set()
    modelli = [m for m in ordine if m in presenti]
    if args.modelli:
        voluti = {x.strip() for x in args.modelli.split(",") if x.strip()}
        modelli = [m for m in modelli if m in voluti]
    st = Stat(cfg["elaborazione"])
    # fasce di turni: dell'intero dataset (per i pesi) e di ogni conversazione del run
    tutte = dati.conversazioni_ordinate(cfg)
    limiti = list(cfg["esecuzione"]["fasce_turni"])
    ordine_fasce = [dati.nome_fascia(i, limiti) for i in range(len(limiti))]
    nel_dataset = {x: sum(1 for c in tutte if c.meta["fascia"] == x) for x in ordine_fasce}
    fasce = {"di": {c.id: c.meta["fascia"] for c in tutte}, "ordine": ordine_fasce, "dataset": nel_dataset,
             "pesi": {x: n / len(tutte) for x, n in nel_dataset.items()}}
    analisi = []
    for m in modelli:
        print(f"[{m}] elaborazione...")
        p_trig = res / "trigger" / f"{m}.json"
        trig = json.loads(p_trig.read_text(encoding="utf-8")) if p_trig.exists() else {}
        a = Analisi(cfg, m, rich, giud, trig, st, out, fasce)
        a.quadro(modelli_jsonl)
        a.energia()
        a.qualita()
        a.compromesso()
        a.per_fascia()
        a.statistica()
        a.referto(sess_g)
        analisi.append(a)
    confronto = {}
    if len(analisi) > 1:
        for nome, tab, col in (("risparmio_medio_per_modello", "B_energia_per_strategia", "risparmio_medio"),
                               ("punteggio_medio_per_modello", "C_qualita_per_strategia", "punteggio_medio")):
            righe = {a.m: a.tabelle[tab].set_index("strategia")[col] for a in analisi if tab in a.tabelle}
            if righe:
                df = pd.DataFrame(righe).T.reset_index().rename(columns={"index": "modello"})
                df.to_csv(out / "tabelle" / f"G_{nome}.csv", index=False)
                confronto[nome] = df
    # ambiente di misura (dall'ultima sessione del benchmark)
    sess = list(leggi_jsonl(res / "sessioni.jsonl"))
    ambiente = []
    if sess:
        s = sess[-1].get("stato_servizio") or {}
        ambiente = [("host", s.get("host")), ("llama.cpp (commit compilato)", s.get("llama_commit_compilato")),
                    ("CodeCarbon", f"{s.get('codecarbon')}, modalità {s.get('modalita')}, finestre {s.get('metodo_finestre')}"),
                    ("Metodo di misura per componente", json.dumps(s.get("hardware_codecarbon"), ensure_ascii=False)),
                    ("Contatori RAPL leggibili", json.dumps(s.get("rapl"))), ("GPU", "; ".join(s.get("gpu") or []))]
        ambiente = [(k, str(v)) for k, v in ambiente]
    info = {"data": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "ambiente": ambiente}
    p = rapporto(cfg, meta, analisi, confronto, out, info)
    print(f"Elaborazione in {out} (rapporto: {p.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
