#!/usr/bin/env python3
"""Costruisce il replay set congelato a partire dai tre dataset scelti.

Pipeline in quattro fasi, piu' una diagnosi opzionale, ognuna ripartibile:

    prepare   normalizza i dataset, elimina vuoti/duplicati/non validi, filtra
              la lingua e applica il filtro sui turni (economico, nessun tokenizzatore);
    tokenize  conta i token di prompt con i tokenizer dei modelli del benchmark;
    sweep     varia --min-tokeni e mostra copertura e strati: serve a scegliere
              la soglia prima di congelare il set, senza scrivere nulla;
    select    applica le soglie, calcola il campione comune, stratifica e
              scrive il set congelato con il manifest;
    verify    ricontrolla hash, soglie, finestre e coerenza del set congelato.

La tokenizzazione e' separata dalle soglie: cambiare --min-turni o
--min-tokeni in `select` non richiede di rieseguire `tokenize`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSIONE = "1.3"

# ---------------------------------------------------------------------------
# Configurazione fissa: fonti, revisioni, modelli
# ---------------------------------------------------------------------------
# Le revisioni sono quelle viste al momento della selezione dei dataset
# (2026-10-20). Vanno congelate nel manifest e citate nella metodologia.
FONTI = {
    "wildchat": {
        "repo": "allenai/WildChat-1M",
        "revision": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
        "split": "train",
        "data_files": None,
        "licenza": "ODC-BY",
        "id_colonna": "conversation_hash",
    },
    "sharegpt_gpt4": {
        "repo": "shibing624/sharegpt_gpt4",
        "revision": "3fb53354e02a931777556fb1da37e931d73af48a",
        "split": "train",
        # Il repo contiene anche sharegpt_zh_38K_format.jsonl e
        # sharegpt_V3_format.jsonl: il file va scelto esplicitamente.
        "data_files": "sharegpt_gpt4.jsonl",
        "licenza": "CC-BY-4.0",
        "id_colonna": None,
    },
    "codechat": {
        "repo": "Suzhen/CodeChat",
        "revision": "1681793a6cfb27b68c1f285bd1e78588132bc1c1",
        "split": "train",
        "data_files": None,
        "licenza": "ODC-BY",
        "id_colonna": "conversation_id",
    },
}
ORDINE_FONTI = ("wildchat", "sharegpt_gpt4", "codechat")

# I 10 modelli della selezione congelata (selezioni/candidati_2026-10-19).
# - etichetta: deve coincidere con models/<cartella> e con --alias di llama-server;
# - contesto: finestra dichiarata, usata solo se --contesto-runtime 0;
# - template_kwargs: interruttore del reasoning usato negli esperimenti, per
#   contare lo stesso prompt che ricevera' llama-server.
MODELLI = [
    {
        "etichetta": "qwen2.5-coder-14b-instruct",
        "repo": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "revision": "aedcc2d42b622764e023cf882b6652e646b95671",
        "contesto": 32768,
        "template_kwargs": {},
    },
    {
        "etichetta": "qwen2.5-coder-32b-instruct",
        "repo": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "revision": "381fc969f78efac66bc87ff7ddeadb7e73c218a7",
        "contesto": 32768,
        "template_kwargs": {},
    },
    {
        "etichetta": "glm-4.7-flash",
        "repo": "zai-org/GLM-4.7-Flash",
        "revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        "contesto": 202752,
        "template_kwargs": {"enable_thinking": False},
    },
    {
        "etichetta": "gemma-4-31b-it-qat-q4-0-unquantized",
        "repo": "google/gemma-4-31B-it-qat-q4_0-unquantized",
        "revision": "1e4d8beecacb8b7590c1d8bedd7335f687bf311f",
        "contesto": 262144,
        "template_kwargs": {"enable_thinking": False},
    },
    {
        "etichetta": "gemma-4-26b-a4b-it-qat-q4-0-unquantized",
        "repo": "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized",
        "revision": "f1e06dc520982d9b9edd76859fdb7ab209449949",
        "contesto": 262144,
        "template_kwargs": {"enable_thinking": False},
    },
    {
        "etichetta": "llama-3.3-70b-instruct",
        "repo": "meta-llama/Llama-3.3-70B-Instruct",
        "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        # Il config e' gated: la model card dichiara 131.072.
        "contesto": 131072,
        "template_kwargs": {},
    },
    {
        "etichetta": "deepseek-coder-v2-lite-instruct",
        "repo": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "revision": "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11",
        "contesto": 163840,
        "template_kwargs": {},
    },
    {
        "etichetta": "llama-3.1-70b-instruct",
        "repo": "meta-llama/Llama-3.1-70B-Instruct",
        "revision": "1605565b47bb9346c5515c34102e054115b4f98b",
        "contesto": 131072,
        "template_kwargs": {},
    },
    {
        "etichetta": "kimi-linear-48b-a3b-instruct",
        "repo": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "revision": "e1df551a447157d4658b573f9a695d57658590e9",
        "contesto": 1048576,
        # Tokenizer TikToken con codice remoto: non ha tokenizer.json.
        "trust_remote_code": True,
        "template_kwargs": {},
    },
    {
        "etichetta": "seed-oss-36b-instruct",
        "repo": "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "revision": "497f1dca95ebdec98e41d517b9f060ee753c902f",
        "contesto": 524288,
        "template_kwargs": {"thinking_budget": 0},
    },
]
MODELLI_PER_ETICHETTA = {m["etichetta"]: m for m in MODELLI}

CHIAVI_MESSAGGI = ("conversation", "messages", "conversations", "dialogue")
RUOLI = {
    "user": "user", "human": "user",
    "assistant": "assistant", "gpt": "assistant", "model": "assistant", "bot": "assistant",
    "system": "system",
}

# Token di controllo che, se lasciati nel testo, possono essere interpretati
# come struttura del template da llama-server. Si rimuovono e si contano.
TOKEN_SPECIALI = [
    "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
    "<|eot_id|>", "<|eom_id|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>",
    "<|user|>", "<|assistant|>", "<|system|>", "<|observation|>", "<|file_separator|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|gh_stars|>", "<|endoftext|>",
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
]
RE_SPECIALI = re.compile("|".join(re.escape(t) for t in sorted(TOKEN_SPECIALI, key=len, reverse=True)))

# Rilevamento della lingua: fastText lid.176, lo stesso modello usato per
# etichettare molti corpus (incluso WildChat). Si applica a tutte le fonti,
# cosi' il set e' omogeneo; l'etichetta dei metadati resta come controprova.
LINGUA_MODELLO_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LINGUA_NOMI = {
    "english": "en", "italian": "it", "spanish": "es", "french": "fr",
    "german": "de", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "chinese": "zh", "japanese": "ja", "korean": "ko", "arabic": "ar",
    "hindi": "hi", "turkish": "tr", "polish": "pl", "swedish": "sv",
}


# ---------------------------------------------------------------------------
# Utilita' comuni
# ---------------------------------------------------------------------------
def ora_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blocco in iter(lambda: fh.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def sha256_script() -> str:
    return sha256_file(Path(__file__).resolve())


def leggi_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for riga in fh:
            riga = riga.strip()
            if riga:
                yield json.loads(riga)


def scrivi_jsonl(path: Path, righe) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for riga in righe:
            fh.write(json.dumps(riga, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def scrivi_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def controlla_sovrascrittura(path: Path, sovrascrivi: bool, cosa: str) -> None:
    if path.exists() and not sovrascrivi:
        sys.exit(f"{cosa} esiste gia': {path}\n"
                 "I documenti congelati non si sovrascrivono: usa --sovrascrivi solo per una prova.")


def lista_argomenti(valore: str, ammessi) -> list[str]:
    voci = [v.strip() for v in valore.split(",") if v.strip()]
    ignoti = [v for v in voci if v not in ammessi]
    if ignoti:
        sys.exit(f"valori non riconosciuti: {', '.join(ignoti)} (ammessi: {', '.join(ammessi)})")
    return voci


def risolvi_modelli(valore: str | None) -> list[dict]:
    if not valore:
        return list(MODELLI)
    etichette = lista_argomenti(valore, MODELLI_PER_ETICHETTA)
    return [MODELLI_PER_ETICHETTA[e] for e in etichette]


# ---------------------------------------------------------------------------
# Normalizzazione delle conversazioni
# ---------------------------------------------------------------------------
def flatten(node):
    """Appiattisce liste annidate di messaggi (CodeChat li raggruppa per turno)."""
    if isinstance(node, dict):
        return [node]
    if isinstance(node, (list, tuple)):
        out = []
        for item in node:
            out.extend(flatten(item))
        return out
    return []


def testo_da_contenuto(contenuto) -> str:
    if isinstance(contenuto, str):
        return contenuto
    if isinstance(contenuto, list):
        parti = []
        for parte in contenuto:
            if isinstance(parte, str):
                parti.append(parte)
            elif isinstance(parte, dict):
                testo = parte.get("text")
                if testo is None:
                    testo = parte.get("content")
                if isinstance(testo, str):
                    parti.append(testo)
        return "\n".join(parti)
    if isinstance(contenuto, dict):
        testo = contenuto.get("text")
        if testo is None:
            testo = contenuto.get("content")
        return testo if isinstance(testo, str) else ""
    return ""


def messaggi_grezzi(row: dict) -> list[tuple[str, str]] | None:
    """Estrae (ruolo, testo) da ShareGPT, WildChat e CodeChat."""
    for chiave in CHIAVI_MESSAGGI:
        if chiave not in row:
            continue
        nodi = flatten(row.get(chiave))
        if not nodi:
            continue
        messaggi = []
        for nodo in nodi:
            ruolo = str(nodo.get("role") or nodo.get("from") or "").strip().lower()
            contenuto = nodo.get("content")
            if contenuto is None:
                contenuto = nodo.get("value")
            messaggi.append((ruolo, testo_da_contenuto(contenuto)))
        return messaggi
    return None


def pulisci_testo(testo: str) -> tuple[str, int]:
    testo = testo.replace("\x00", "")
    testo, n_speciali = RE_SPECIALI.subn("", testo)
    # Non si usa strip() sul contenuto: toglierebbe l'indentazione iniziale
    # dei blocchi di codice. Si scartano solo i messaggi vuoti o di soli spazi.
    if not testo.strip():
        return "", n_speciali
    return testo, n_speciali


def normalizza(grezzi: list[tuple[str, str]], politica_system: str, stats: Counter):
    """Applica le regole di validita' strutturale e restituisce la conversazione."""
    messaggi = []
    for ruolo_grezzo, testo in grezzi:
        ruolo = RUOLI.get(ruolo_grezzo)
        if ruolo is None:
            stats["ruoli_sconosciuti"] += 1
            continue
        testo, n_speciali = pulisci_testo(testo)
        stats["token_speciali_rimossi"] += n_speciali
        if not testo:
            stats["messaggi_vuoti"] += 1
            continue
        messaggi.append({"role": ruolo, "content": testo})

    if politica_system == "drop":
        n = sum(1 for m in messaggi if m["role"] == "system")
        stats["system_rimossi"] += n
        messaggi = [m for m in messaggi if m["role"] != "system"]
    else:  # "first": tiene solo i system iniziali, uniti in uno
        iniziali = []
        while messaggi and messaggi[0]["role"] == "system":
            iniziali.append(messaggi.pop(0)["content"])
        n = sum(1 for m in messaggi if m["role"] == "system")
        stats["system_rimossi"] += n + max(0, len(iniziali) - 1)
        messaggi = [m for m in messaggi if m["role"] != "system"]
        if iniziali:
            messaggi.insert(0, {"role": "system", "content": "\n\n".join(iniziali)})

    # Un assistant iniziale puo' stare anche dopo un system: va tolto comunque,
    # altrimenti la sequenza non inizia con l'utente.
    while True:
        indice = next((i for i, m in enumerate(messaggi) if m["role"] != "system"), None)
        if indice is None or messaggi[indice]["role"] != "assistant":
            break
        messaggi.pop(indice)
        stats["assistenti_iniziali"] += 1

    uniti = []
    for m in messaggi:
        if uniti and uniti[-1]["role"] == m["role"]:
            uniti[-1]["content"] += "\n\n" + m["content"]
            stats["messaggi_uniti"] += 1
        else:
            uniti.append(dict(m))
    messaggi = uniti

    sequenza = [m["role"] for m in messaggi if m["role"] != "system"]
    if not sequenza:
        stats["senza_messaggi"] += 1
        return None
    if sequenza[0] != "user":
        stats["non_inizia_con_utente"] += 1
        return None
    if any(sequenza[i] == sequenza[i + 1] for i in range(len(sequenza) - 1)):
        stats["non_alternata"] += 1
        return None
    if "user" not in sequenza:
        stats["senza_utente"] += 1
        return None
    if "assistant" not in sequenza:
        stats["senza_assistente"] += 1
        return None

    indice_ultimo_utente = max(i for i, m in enumerate(messaggi) if m["role"] == "user")
    return {
        "messages": messaggi,
        "turni": sequenza.count("user"),
        "indice_ultimo_utente": indice_ultimo_utente,
    }


def hash_contenuto(messaggi: list[dict]) -> str:
    h = hashlib.sha256()
    for m in messaggi:
        h.update(m["role"].encode("utf-8"))
        h.update(b"\x1e")
        h.update(m["content"].encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Rilevamento della lingua
# ---------------------------------------------------------------------------
def codice_lingua(valore: str) -> str:
    """Normalizza 'English' o 'en' in un codice ISO 639-1/2."""
    v = valore.strip().lower()
    v = LINGUA_NOMI.get(v, v)
    if not re.fullmatch(r"[a-z]{2,3}", v):
        sys.exit(f"lingua non riconosciuta: {valore} (usa un codice ISO, es. en)")
    return v


def norma_metadati_lingua(valore: str) -> str | None:
    """Come codice_lingua ma senza uscire: i metadati possono contenere
    etichette inattese e servono solo per il controllo di accordo."""
    v = (valore or "").strip().lower()
    if not v:
        return None
    v = LINGUA_NOMI.get(v, v)
    return v if re.fullmatch(r"[a-z]{2,3}", v) else None


def scarica_file(url: str, destinazione: Path) -> None:
    import urllib.request

    destinazione.parent.mkdir(parents=True, exist_ok=True)
    tmp = destinazione.with_suffix(destinazione.suffix + ".tmp")
    print(f"  scarico {url}")
    richiesta = urllib.request.Request(url, headers={"User-Agent": f"build_replay_set/{VERSIONE}"})
    with urllib.request.urlopen(richiesta) as risposta, open(tmp, "wb") as fh:
        while True:
            blocco = risposta.read(1 << 20)
            if not blocco:
                break
            fh.write(blocco)
    os.replace(tmp, destinazione)


def carica_rilevatore_lingua(percorso: Path, url: str) -> tuple[object, dict]:
    try:
        import fasttext
    except ImportError:
        sys.exit("fasttext non installato: pip install -r requirements.txt (fasttext-wheel)")
    if not percorso.exists():
        scarica_file(url, percorso)
    info = {
        "tipo": "fastText lid.176",
        "file": str(percorso),
        "sha256": sha256_file(percorso),
        "byte": percorso.stat().st_size,
    }
    try:
        modello = fasttext.load_model(str(percorso))
    except Exception as exc:
        sys.exit(f"modello lingua non caricabile ({percorso}): {exc}. "
                 f"Cancellalo per riscaricarlo.")
    return modello, info


def testo_per_lingua(messaggi: list[dict], caratteri: int) -> str:
    """Testo su cui si decide la lingua: i soli messaggi utente, che sono
    quelli effettivamente replicati nell'esperimento."""
    testo = "\n".join(m["content"] for m in messaggi if m["role"] == "user")
    return " ".join(testo.split())[:caratteri]


def rileva_lingua(modello, testo: str) -> tuple[str | None, float]:
    if len(testo) < 20:
        return None, 0.0
    # fasttext-wheel 0.9.2 chiama np.array(..., copy=False) e rompe con NumPy 2:
    # si usa il binding C++ sottostante, che restituisce [(probabilita, etichetta)]
    # e vale sia con NumPy 1 sia con NumPy 2.
    interno = getattr(modello, "f", None)
    if interno is not None:
        risultati = interno.predict(testo, 1, 0.0, "strict")
        if not risultati:
            return None, 0.0
        probabilita, etichetta = risultati[0]
        return etichetta.replace("__label__", ""), float(probabilita)
    etichette, probabilita = modello.predict(testo, k=1)
    if not len(etichette):
        return None, 0.0
    return etichette[0].replace("__label__", ""), float(probabilita[0])



# ---------------------------------------------------------------------------
# Filtro coding
# ---------------------------------------------------------------------------
# Criterio riproducibile e volutamente conservativo: una conversazione e'
# "coding" se contiene un blocco di codice con un linguaggio riconoscibile (o
# con dentro righe di codice), oppure se contiene una riga di codice vera.
# Guardare anche i messaggi dell'assistente e' importante: molte domande di
# programmazione non incollano codice, ma la risposta si'.
# La validazione e' fatta sulle conversazioni WildChat che compaiono in
# CodeChat (derivato coding di WildChat): il criterio ne recupera l'82-83% e
# segnala lo 0,5% delle altre. Vedi README per i numeri.
RE_BLOCCO = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
LINGUAGGI_BLOCCO = {
    "python", "py", "python3", "javascript", "js", "jsx", "typescript", "ts", "tsx",
    "java", "c", "cpp", "c++", "csharp", "cs", "c#", "go", "golang", "rust", "rs",
    "php", "ruby", "rb", "sql", "html", "xml", "css", "scss", "sass", "less",
    "bash", "sh", "shell", "zsh", "powershell", "ps1", "json", "yaml", "yml",
    "toml", "ini", "kotlin", "kt", "swift", "scala", "r", "matlab", "perl", "lua",
    "dockerfile", "makefile", "diff", "patch", "dart", "elixir", "haskell",
    "julia", "ocaml", "clojure", "groovy", "solidity", "asm", "nasm", "vb",
    "visualbasic", "objective-c", "objc", "fortran", "cobol", "lisp", "scheme",
}
SEGNALI_FORTI = {
    "riga": re.compile(
        r"(?m)^\s*(?:def\s+\w+\s*\(|class\s+\w+\s*[\(:]|import\s+[\w.]+|"
        r"from\s+[\w.]+\s+import\s+|#include\s*[<\"]|using\s+namespace\s+|"
        r"package\s+[\w.]+\s*;|"
        r"public\s+(?:static\s+)?(?:void|int|String|class)\b|"
        r"function\s+\w*\s*\(|(?:const|let|var)\s+\w+\s*=|"
        r"for\s*\([^;\n]*;[^;\n]*;|while\s*\([^)\n]*\)\s*\{|"
        r"if\s*\([^)\n]*\)\s*\{|return\s+[^;\n]+;|"
        r"(?:printf|print|console\.log|System\.out\.println)\s*\(|"
        r"(?:SELECT|INSERT|UPDATE|DELETE|CREATE)\s+(?:[\w*]+\s+)?FROM\b|"
        r"</?(?:html|div|span|body|head|script|style|p|a|ul|li|table|form|"
        r"input|button|h[1-6])\b[^>]*>)"),
}


def ha_codice(messaggi: list[dict]) -> bool:
    """True se c'e' un blocco di codice o una riga di codice, in qualsiasi messaggio."""
    for msg in messaggi:
        contenuto = msg["content"]
        for info, corpo in RE_BLOCCO.findall(contenuto):
            tag = info.strip().lower().split()[0] if info.strip() else ""
            if tag in LINGUAGGI_BLOCCO:
                return True
            if SEGNALI_FORTI["riga"].search(corpo):
                return True
            if sum(1 for riga in corpo.splitlines()
                   if riga.strip().endswith((";", "{", "}"))) >= 2:
                return True
        if SEGNALI_FORTI["riga"].search(contenuto):
            return True
    return False

# ---------------------------------------------------------------------------
# Filtro tema informatico (CS/IT)
# ---------------------------------------------------------------------------
# `--solo-coding` resta il filtro stretto (codice). `--solo-informatica`
# aggiunge le conversazioni che, anche senza codice, trattano in modo
# riconoscibile un tema di informatica/IT: algoritmi e strutture dati,
# sistemi operativi, reti, basi di dati, sicurezza, ingegneria del software,
# AI/ML, web/mobile, hardware, ecc.
# Il criterio e' volutamente conservativo: guarda solo il primo messaggio
# dell'utente (il tema della richiesta), richiede almeno due segnali tecnici
# distinti oppure un segnale tecnico con una richiesta esplicita di
# spiegazione/implementazione, ed esclude i domini non informatici e i task
# creativi, commerciali, legali, medici, matematici o di traduzione.
# Con 3.081 conversazioni >=10 turni e >=4.000 token, il filtro stretto
# seleziona 523 conversazioni; quello informatico 565 (+42). Vedi README.
_BLOCCO_CS = {
    "gpt", "https", "monitoring", "logging", "backup", "authorization",
    "scrum", "transistor", "interpreter", "chatbot", "risc", "cisc", "iam",
}
_TEMI_CS_GREZZI = ['data structure', 'linked list', 'doubly linked list', 'hash table', 'hash map', 'binary tree',
 'binary search tree', 'avl tree', 'red-black tree', 'b-tree', 'graph algorithm', 'graph theory',
 'depth-first search', 'breadth-first search', 'dynamic programming', 'greedy algorithm',
 'divide and conquer', 'sorting algorithm', 'quicksort', 'mergesort', 'heapsort', 'binary search',
 'time complexity', 'space complexity', 'computational complexity', 'big o notation', 'np-complete',
 'np-hard', 'turing machine', 'formal language', 'recursion', 'memoization', 'backtracking',
 'monte carlo tree search', 'bloom filter', 'union find', 'priority queue', 'segment tree',
 'fenwick tree', 'finite state machine', 'state machine', 'regular grammar', 'context-free grammar',
 'lambda calculus', 'type theory', 'source code', 'codebase', 'software development',
 'software engineering', 'computer science', 'computer engineering', 'information technology',
 'development environment', 'version control', 'pull request', 'code review', 'design pattern',
 'unit test', 'integration test', 'test suite', 'test-driven development', 'continuous integration',
 'continuous deployment', 'technical debt', 'refactoring', 'debugger', 'compiler', 'linker',
 'loader', 'bytecode', 'virtual machine', 'garbage collection', 'memory leak', 'stack overflow',
 'null pointer', 'segmentation fault', 'syntax error', 'type error', 'compile error', 'stack trace',
 'runtime error', 'exception handling', 'object-oriented programming', 'functional programming',
 'imperative programming', 'declarative programming', 'polymorphism', 'encapsulation',
 'inheritance', 'abstraction', 'type system', 'generic programming', 'metaprogramming',
 'dependency injection', 'inversion of control', 'regular expression', 'shell script',
 'bash script', 'command line', 'package manager', 'virtual environment', 'merge conflict',
 'docker container', 'docker image', 'kubernetes cluster', 'microservice architecture',
 'serverless computing', 'devops', 'agile development', 'kanban', 'user story', 'sprint planning',
 'product backlog', 'operating system', 'linux kernel', 'system call', 'syscall', 'sysctl',
 'file system', 'inode', 'process scheduling', 'context switch', 'interrupt handler',
 'device driver', 'bootloader', 'virtual memory', 'paging', 'page table', 'page replacement',
 'segmentation fault', 'fragmentation', 'thrashing', 'deadlock', 'mutex', 'semaphore',
 'race condition', 'concurrency', 'parallelism', 'producer consumer', 'reader writer',
 'dining philosophers', "banker's algorithm", 'memory management', 'cpu cache', 'cache coherence',
 'memory hierarchy', 'instruction set', 'assembly language', 'machine code', 'cpu register',
 'microprocessor', 'microcontroller', 'motherboard', 'bios', 'uefi', 'solid state drive',
 'hard disk drive', 'logic gate', 'asic', 'fpga', 'verilog', 'vhdl', 'embedded system',
 'real-time system', 'rtos', 'firmware', 'internet of things', 'arduino', 'raspberry pi',
 'arm architecture', 'x86 architecture', 'simd', 'superscalar', 'branch prediction',
 'little endian', 'big endian', "two's complement", 'floating point', 'quadruple precision',
 'von neumann architecture', 'harvard architecture', 'computer network', 'network protocol',
 'ip address', 'dns server', 'dhcp server', 'http request', 'http response', 'ftp server', 'ssh',
 'smtp server', 'imap', 'pop3', 'websocket', 'network socket', 'packet loss', 'router',
 'network switch', 'subnet mask', 'vlan', 'vpn', 'proxy server', 'reverse proxy', 'load balancer',
 'firewall', 'network address translation', 'latency', 'bandwidth', 'mac address', 'osi model',
 'tls handshake', 'ssl certificate', 'network topology', 'network interface', 'network congestion',
 'quality of service', 'bgp', 'ospf', 'arp', 'icmp', 'tcp window', 'congestion control', 'mtu',
 'database', 'dbms', 'relational database', 'sql query', 'nosql', 'mongodb', 'postgresql', 'mysql',
 'sqlite', 'redis', 'elasticsearch', 'object relational mapping', 'data modeling',
 'database schema', 'database normalization', 'primary key', 'foreign key', 'stored procedure',
 'database transaction', 'acid properties', 'data warehouse', 'data pipeline', 'etl', 'big data',
 'hadoop', 'apache spark', 'apache kafka', 'mapreduce', 'data lake', 'query optimization',
 'database design', 'entity relationship', 'er diagram', 'cybersecurity', 'information security',
 'encryption', 'decryption', 'cryptography', 'cipher', 'aes encryption', 'rsa encryption',
 'public key', 'private key', 'hash function', 'digital signature', 'authentication', 'oauth',
 'jwt', 'access control', 'vulnerability', 'exploit', 'malware', 'ransomware', 'computer virus',
 'phishing', 'sql injection', 'cross-site scripting', 'csrf', 'penetration test', 'ethical hacking',
 'capture the flag', 'reverse engineering', 'digital forensics', 'zero-day', 'cve',
 'security audit', 'threat model', 'sandbox', 'intrusion detection', 'artificial intelligence',
 'machine learning', 'deep learning', 'neural network', 'convolutional neural network',
 'recurrent neural network', 'transformer model', 'large language model', 'llm', 'bert',
 'feature engineering', 'overfitting', 'regularization', 'dropout', 'gradient descent',
 'backpropagation', 'supervised learning', 'unsupervised learning', 'reinforcement learning',
 'support vector machine', 'decision tree', 'random forest', 'xgboost',
 'natural language processing', 'nlp', 'computer vision', 'speech recognition',
 'recommendation system', 'prompt engineering', 'fine-tuning', 'embedding', 'vector database',
 'retrieval augmented generation', 'generative ai', 'diffusion model', 'model parallelism',
 'data parallelism', 'hyperparameter', 'cross-validation', 'confusion matrix',
 'precision and recall', 'f1 score', 'attention mechanism', 'tokenization', 'web development',
 'frontend', 'backend', 'full stack', 'html', 'css', 'javascript', 'typescript', 'react.js',
 'react component', 'angular', 'vue.js', 'svelte', 'node.js', 'express.js', 'django', 'flask',
 'fastapi', 'spring boot', 'laravel', 'ruby on rails', 'asp.net', '.net framework', 'jquery',
 'bootstrap', 'tailwind', 'responsive design', 'document object model', 'web browser', 'web cookie',
 'web session', 'web hosting', 'domain name', 'content delivery network', 'web server',
 'web application', 'mobile app', 'android', 'ios', 'flutter', 'react native', 'xamarin',
 'game development', 'game engine', 'unity engine', 'unreal engine', 'godot', 'webgl', 'opengl',
 'vulkan', 'directx', 'shader', 'computer graphics', 'image processing', 'opencv', 'linux',
 'ubuntu', 'debian', 'fedora', 'centos', 'unix', 'posix', 'macos', 'windows server', 'systemd',
 'cron', 'chmod', 'sudo', 'apt-get', 'yum', 'dnf', 'homebrew', 'windows registry',
 'active directory', 'group policy', 'virtualization', 'vmware', 'virtualbox', 'hypervisor', 'raid',
 'network attached storage', 'aws', 'azure', 'gcp', 'google cloud', 'amazon s3', 'amazon ec2',
 'aws lambda', 'cloud computing', 'terraform', 'ansible', 'jenkins', 'github actions', 'grafana',
 'prometheus', 'nginx', 'apache server', 'sysadmin', 'system administrator', 'it support',
 'help desk', 'infrastructure as code', 'information retrieval', 'tf-idf', 'inverted index',
 'search engine', 'recommendation algorithm', 'collaborative filtering', 'latent semantic analysis',
 'vector space model', 'boolean retrieval', 'text mining', 'data mining', 'blockchain',
 'smart contract', 'solidity', 'distributed ledger', 'consensus algorithm', 'paxos',
 'raft consensus', 'byzantine fault', 'cap theorem', 'eventual consistency', 'merkle tree']

_ACRONIMI_CS_GREZZI = ['TCP', 'UDP', 'IPv4', 'IPv6', 'DNS', 'DHCP', 'HTTP', 'FTP', 'SSH', 'SMTP', 'IMAP', 'POP3', 'VPN',
 'VLAN', 'NAT', 'TLS', 'SSL', 'AES', 'RSA', 'JWT', 'XSS', 'CSRF', 'CVE', 'LLM', 'BERT', 'NLP',
 'SVM', 'CPU', 'GPU', 'RAM', 'ROM', 'SSD', 'HDD', 'BIOS', 'UEFI', 'ASIC', 'FPGA', 'VHDL', 'IoT',
 'AWS', 'GCP', 'S3', 'EC2', 'CDN', 'DOM', 'SDK', 'ORM', 'ETL', 'ACID', 'CAP', 'RAG', 'IDS', 'IPS',
 'MAC', 'OSI', 'ARP', 'ICMP', 'BGP', 'OSPF', 'MTU', 'SACK', 'TLB', 'RTOS', 'SIMD', 'AVL', 'DFS',
 'BFS', 'NP', 'SQL', 'API', 'HTML', 'CSS', 'JSON', 'YAML', 'XML', 'URL', 'URI', 'UUID', 'IDE',
 'CLI', 'GUI', 'VPS', 'VPC', 'SQS', 'SNS', 'RDS', 'ECS', 'EKS', 'GKE', 'AKS', 'Python', 'Java',
 'JavaScript', 'TypeScript', 'C++', 'C#', 'Golang', 'Rust', 'Kotlin', 'Scala', 'Haskell', 'Elixir',
 'Clojure', 'Dart', 'React', 'Angular', 'Vue', 'Svelte', 'Django', 'Flask', 'FastAPI', 'Spring',
 'Laravel', 'Rails', 'TensorFlow', 'PyTorch', 'Keras', 'NumPy', 'OpenCV', 'Unity', 'Unreal',
 'Godot', 'WebGL', 'OpenGL', 'Vulkan', 'DirectX', 'CUDA', 'OpenCL']

TEMI_CS = [t for t in _TEMI_CS_GREZZI if t.lower() not in _BLOCCO_CS]
ACRONIMI_CS = [t for t in _ACRONIMI_CS_GREZZI if t.lower() not in _BLOCCO_CS]


def _compila_termini(termini: list[str], flags: int = 0) -> re.Pattern:
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(t) for t in sorted(termini, key=len, reverse=True))
        + r")(?!\w)", flags)


RE_TEMI_CS = _compila_termini(TEMI_CS, re.IGNORECASE)
RE_ACRONIMI_CS = _compila_termini(ACRONIMI_CS)
RE_INTENTO_CS = re.compile(
    r"(?i)\b(?:what is|what are|what's|how does|how do|how to|why does|why do|"
    r"explain|difference between|compare|implement|build|design|debug|fix|error|"
    r"install|configure|setup|optimize|performance|architecture|algorithm|code|"
    r"program|write a function|write a script|example of|guide|understand|learn|"
    r"study|question|calculate|compute|which is|which one|can you|could you|help me)\b")
RE_ESCLUSIONI_CS = re.compile(r"""(?ix)
 (?:
   write\ a\ story|tell\ me\ a\ story|short\ story|story\ about|write\ a\ poem|
   poem\ about|write\ an\ essay|write\ an\ article|write\ a\ blog|blog\ post|
   write\ a\ script|screenplay|write\ a\ song|lyrics|write\ a\ letter|cover\ letter|
   write\ a\ resume|personal\ statements?|write\ a\ thesis|dissertation|marketing|
   sales|business\ plan|startup|pitch\ deck|brand|social\ media|seo|advertising|
   real\ estate|investment|finance|stock\ trading|crypto\ trading|accounting|
   our\ services|we\ offer|why\ choose\ us|contact\ us|request\ a\ quote|homework|
   exam|school|university|coursework|assignment|job\ application|job\ interview|
   physics|chemistry|biology|calculus|algebra|geometry|statistics|economics|
   microeconomics|macroeconomics|protein|molecular\ docking|bioinformatics|translate|
   translation|paraphrase|proofread|proofreading|rewrite|summarize|synonym|
   replacing\ these\ words|replacing\ the\ bracketed|翻译|dịch|traduci|traduire|
   übersetzen|перевести|dan\ mode|jailbreak|scam|scammer|scam-baiting|midjourney|dall-e|
   stable\ diffusion|image\ generation|generate\ images|artwork|art\ style|titles?|
   compliance|nft|non-fungible|firm|price|pricing|profit|revenue|consumer|demand|supply|
   market\ share|sentient|zorg|no\ allegiance|forget\ all\ prior\ prompts|
   two\ chat\ gpt\ agents|roleplay|role-playing|hypothetical\ episode|episode|sitcom|
   fanfic|fan\ fiction|itcrowd|it\ crowd|storytelling|absurdist|comedy|sketch|film|
   fashion\ design|damage|equps|rpg|tabletop|blog|article|toc|heading|intro|gaim|
   game\ ideas|game\ publisher|grossing|profitable|architectural\ teacher|thesis\ statements?|
   innovation\ mastermind|gas\ giants|astronomy|astrophysics|divine\ feminine|spiritual
 )\b
""")


def ha_tema_informatico(messaggi: list[dict]) -> bool:
    """True se il primo messaggio utente tratta un tema informatico/IT."""
    prima = next((m["content"] for m in messaggi if m["role"] == "user"), "")
    testo = prima[:20000]
    if not testo or RE_ESCLUSIONI_CS.search(testo):
        return False
    hit = {x.lower() for x in RE_TEMI_CS.findall(testo)}
    hit |= {x.lower() for x in RE_ACRONIMI_CS.findall(testo)}
    if len(hit) >= 2:
        return True
    return bool(hit) and bool(RE_INTENTO_CS.search(testo))


def passa_filtro_coding(messaggi: list[dict], solo_coding: bool,
                        solo_informatica: bool) -> bool:
    if solo_informatica:
        return ha_codice(messaggi) or ha_tema_informatico(messaggi)
    if solo_coding:
        return ha_codice(messaggi)
    return True


def apri_fonte(nome: str, args) -> "object":
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets non installato: usa il venv del README (pip install -r requirements.txt)")

    cfg = FONTI[nome]
    kwargs = {
        "split": cfg["split"],
        "streaming": True,
        "revision": cfg["revision"],
    }
    if cfg["data_files"]:
        kwargs["data_files"] = cfg["data_files"]
    if args.hf_token:
        kwargs["token"] = args.hf_token
    return load_dataset(cfg["repo"], **kwargs)


def cmd_prepare(args) -> int:
    fonti = lista_argomenti(args.fonti, FONTI)
    if "codechat" in fonti and "wildchat" in fonti and fonti.index("codechat") < fonti.index("wildchat"):
        sys.exit("CodeChat deriva da WildChat: elabora wildchat prima di codechat, "
                 "altrimenti le conversazioni duplicate vengono attribuite alla fonte sbagliata.")

    out_dir = Path(args.cache) / "normalizzato"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.cache) / "prepare.json"
    controlla_sovrascrittura(manifest_path, args.sovrascrivi, "La cache normalizzata")

    visti_id: dict[str, str] = {}
    visti_hash: set[str] = set()
    imbuto: dict[str, dict] = {}
    file_prodotti: dict[str, dict] = {}

    codice = None
    rilevatore = None
    info_lingua = None
    if args.tutte_le_lingue:
        print("\nFiltro lingua disattivato (--tutte-le-lingue): le fonti restano multilingua")
    elif args.lingua:
        codice = codice_lingua(args.lingua)
        if not args.modello_lingua:
            args.modello_lingua = str(Path(args.cache) / "models" / "lid.176.bin")
        rilevatore, info_lingua = carica_rilevatore_lingua(
            Path(args.modello_lingua), LINGUA_MODELLO_URL)
        print(f"\nRilevamento lingua: fastText lid.176 -> '{codice}', "
              f"soglia {args.lingua_soglia:.2f}, primi {args.lingua_caratteri} caratteri "
              f"dei messaggi utente\n  modello: {info_lingua['file']} "
              f"({info_lingua['byte'] / 1e6:.0f} MB, sha256 {info_lingua['sha256'][:16]}...)")

    for fonte in fonti:
        cfg = FONTI[fonte]
        stats = Counter()
        lingue = Counter()
        lingue_rilevate = Counter()
        modelli_origine = Counter()
        print(f"\n=== {fonte} ({cfg['repo']}@{cfg['revision'][:10]}) ===")
        percorso = out_dir / f"{fonte}.jsonl"
        tmp = percorso.with_suffix(percorso.suffix + ".tmp")
        fh = open(tmp, "w", encoding="utf-8")
        scritte = 0
        ds = apri_fonte(fonte, args)
        for i, row in enumerate(ds):
            if args.limite and i >= args.limite:
                break
            stats["lette"] += 1
            if stats["lette"] % 10000 == 0:
                print(f"  ...{stats['lette']:,} lette, {scritte:,} scritte")

            grezzi = messaggi_grezzi(row)
            if grezzi is None:
                stats["formato_non_riconosciuto"] += 1
                continue

            norm = normalizza(grezzi, args.politica_system, stats)
            if norm is None:
                continue

            lingua = row.get("language") if isinstance(row.get("language"), str) else ""
            lingue[lingua or "(n/d)"] += 1

            # L'id di fonte va registrato prima dei filtri: se una conversazione
            # WildChat viene esclusa (toxic, turni, lingua), la sua copia in
            # CodeChat non deve rientrare dalla finestra.
            h = hash_contenuto(norm["messages"])
            id_fonte = ""
            if cfg["id_colonna"]:
                id_fonte = str(row.get(cfg["id_colonna"]) or "").strip()
            if not id_fonte:
                id_fonte = h[:16]
            usa_id_condiviso = fonte in ("wildchat", "codechat")
            if usa_id_condiviso and id_fonte in visti_id:
                stats["duplicati_id"] += 1
                continue
            if usa_id_condiviso:
                visti_id[id_fonte] = fonte

            if fonte == "wildchat":
                if not args.includi_tossici and bool(row.get("toxic")):
                    stats["tossiche_escluse"] += 1
                    continue
                if args.escludi_redatti and bool(row.get("redacted")):
                    stats["redatte_escluse"] += 1
                    continue

            if norm["turni"] < args.min_turni:
                stats["turni_sotto_soglia"] += 1
                continue

            prompt = norm["messages"][: norm["indice_ultimo_utente"] + 1]
            byte_prompt = sum(len(m["content"].encode("utf-8")) for m in prompt)
            # Prefiltro conservativo: token_prompt <= byte + margine; se non
            # arriva alla soglia nemmeno con il margine, e' escluso con certezza.
            if args.prefiltro_byte and byte_prompt + args.margine_prefiltro < args.prefiltro_byte:
                stats["byte_sotto_soglia"] += 1
                continue

            if h in visti_hash:
                stats["duplicati_contenuto"] += 1
                continue

            rilevata, confidenza = None, None
            if codice:
                rilevata, confidenza = rileva_lingua(
                    rilevatore, testo_per_lingua(norm["messages"], args.lingua_caratteri))
                lingue_rilevate[rilevata or "(non valutabile)"] += 1
                metadati = norma_metadati_lingua(lingua)
                if rilevata and metadati:
                    # L'etichetta dei metadati non decide: serve solo a misurare
                    # l'accordo con il rilevatore (per WildChat).
                    if rilevata == metadati:
                        stats["accordo_metadati"] += 1
                    else:
                        stats["disaccordo_metadati"] += 1
                if rilevata is None:
                    stats["lingua_non_valutabile"] += 1
                elif rilevata != codice:
                    stats["lingua_diversa"] += 1
                elif confidenza < args.lingua_soglia:
                    stats["lingua_sotto_soglia"] += 1
                else:
                    stats["lingua_ok"] += 1
                if rilevata != codice or confidenza < args.lingua_soglia:
                    stats["lingua_esclusa"] += 1
                    continue
            visti_hash.add(h)

            rec = {
                "id": f"{fonte}:{id_fonte}",
                "fonte": fonte,
                "id_fonte": id_fonte,
                "modello_origine": str(row.get("model") or ""),
                "lingua": rilevata if codice else lingua,
                "turni": norm["turni"],
                "hash_contenuto": h,
                "indice_ultimo_utente": norm["indice_ultimo_utente"],
                "byte_prompt": byte_prompt,
                "messages": norm["messages"],
            }
            if codice:
                rec["lingua_rilevata"] = rilevata
                rec["lingua_confidenza"] = round(confidenza or 0.0, 4)
                rec["lingua_metadati"] = lingua
            if fonte == "wildchat":
                rec["meta"] = {"toxic": bool(row.get("toxic")), "redacted": bool(row.get("redacted"))}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            scritte += 1
            modelli_origine[str(row.get("model") or "(n/d)")] += 1

        fh.close()
        os.replace(tmp, percorso)
        if not scritte:
            print(f"  nessuna conversazione idonea per {fonte}: controlla --min-turni, --lingua, --limite")
        imbuto[fonte] = dict(stats)
        imbuto[fonte]["modelli_origine_top5"] = modelli_origine.most_common(5)
        file_prodotti[fonte] = {
            "file": str(percorso),
            "conversazioni": scritte,
            "sha256": sha256_file(percorso),
        }
        print(f"  scritte {scritte:,} conversazioni in {percorso}")
        for chiave in ("formato_non_riconosciuto", "ruoli_sconosciuti", "messaggi_vuoti",
                       "token_speciali_rimossi", "system_rimossi", "assistenti_iniziali",
                       "messaggi_uniti", "senza_messaggi", "non_inizia_con_utente",
                       "non_alternata", "senza_assistente", "senza_utente",
                       "lingua_esclusa", "lingua_non_valutabile", "lingua_diversa",
                       "lingua_sotto_soglia", "tossiche_escluse",
                       "redatte_escluse", "turni_sotto_soglia", "byte_sotto_soglia",
                       "duplicati_id", "duplicati_contenuto"):
            if stats.get(chiave):
                print(f"    {chiave:24} {stats[chiave]:,}")
        print(f"    lingue top5              {lingue.most_common(5)}")
        if codice:
            print(f"    lingue rilevate top5     {lingue_rilevate.most_common(5)}")
            print(f"    lingua ok ({codice})       {stats.get('lingua_ok', 0):,}  "
                  f"(escluse per lingua: {stats.get('lingua_diversa', 0):,}, "
                  f"sotto soglia: {stats.get('lingua_sotto_soglia', 0):,}, "
                  f"non valutabili: {stats.get('lingua_non_valutabile', 0):,})")
            valutabili = stats.get("accordo_metadati", 0) + stats.get("disaccordo_metadati", 0)
            if valutabili:
                print(f"    accordo con i metadati   {stats.get('accordo_metadati', 0):,}/{valutabili:,} "
                      f"({stats.get('accordo_metadati', 0) / valutabili:.1%})")
        print(f"    modelli origine top5     {modelli_origine.most_common(5)}")

    manifest = {
        "fase": "prepare",
        "versione": VERSIONE,
        "script": {"nome": Path(__file__).name, "sha256": sha256_script()},
        "data_utc": ora_utc(),
        "parametri": {
            "fonti": fonti,
            "min_turni": args.min_turni,
            "politica_system": args.politica_system,
            "lingua": codice,
            "includi_tossici": args.includi_tossici,
            "escludi_redatti": args.escludi_redatti,
            "prefiltro_byte": args.prefiltro_byte,
            "margine_prefiltro": args.margine_prefiltro,
            "limite_per_fonte": args.limite,
            "hf_token_usato": bool(args.hf_token),
        },
        "rilevamento_lingua": {
            "codice": codice,
            "modo": "rilevatore su tutte le fonti (l'etichetta dei metadati e' solo di controllo)",
            "soglia_confidenza": args.lingua_soglia,
            "caratteri_valutati": args.lingua_caratteri,
            "modello": info_lingua,
        } if codice else None,
        "dataset": {
            f: {"repo": FONTI[f]["repo"], "revision": FONTI[f]["revision"],
                "data_files": FONTI[f]["data_files"], "licenza": FONTI[f]["licenza"]}
            for f in fonti
        },
        "imbuto": imbuto,
        "file": file_prodotti,
    }
    scrivi_json(manifest_path, manifest)
    print(f"\nCache normalizzata: {out_dir}\nManifest: {manifest_path}")
    return 0


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------
def carica_tokenizer(modello: dict, hf_token: str | None = None):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers non installato: pip install -r requirements.txt")
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs = {
        "revision": modello["revision"],
        "trust_remote_code": bool(modello.get("trust_remote_code")),
    }
    if token:
        kwargs["token"] = token
    try:
        tok = AutoTokenizer.from_pretrained(modello["repo"], **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"tokenizer non caricabile per {modello['etichetta']} ({modello['repo']}): {exc}. "
            "Per i repo gated (meta-llama) serve HF_TOKEN con la licenza accettata."
        ) from exc

    if getattr(tok, "chat_template", None) is None:
        # Alcuni repo (GLM, Gemma, Seed, Kimi) tengono il template in un file
        # separato che le versioni recenti di transformers caricano da sole;
        # questo fallback copre le versioni che non lo fanno.
        try:
            from huggingface_hub import hf_hub_download
            percorso = hf_hub_download(
                modello["repo"], "chat_template.jinja",
                revision=modello["revision"], token=token or None,
            )
            tok.chat_template = Path(percorso).read_text(encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"chat template assente per {modello['etichetta']}: {exc}") from exc
    return tok


def rileva_piegatura_system(tok, kwargs_template: dict) -> bool:
    """True se il template non accetta il ruolo system e va piegato nel primo user."""
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=False, add_generation_prompt=True, **kwargs_template,
        )
        return False
    except Exception:
        return True


def adatta_system(messaggi: list[dict], piega: bool) -> list[dict]:
    if not piega:
        return [dict(m) for m in messaggi]
    sistema = [m["content"] for m in messaggi if m["role"] == "system"]
    resto = [dict(m) for m in messaggi if m["role"] != "system"]
    if sistema:
        testo = "\n\n".join(sistema)
        for m in resto:
            if m["role"] == "user":
                m["content"] = testo + "\n\n" + m["content"]
                break
    return resto


def conta_token(tok, messaggi: list[dict], piega_system: bool, kwargs_template: dict,
                generazione: bool) -> int:
    messaggi = adatta_system(messaggi, piega_system)
    ids = tok.apply_chat_template(
        messaggi, tokenize=True, add_generation_prompt=generazione, **kwargs_template,
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def cmd_tokenize(args) -> int:
    modelli = risolvi_modelli(args.modelli)
    norm_dir = Path(args.cache) / "normalizzato"
    fonti = [f for f in ORDINE_FONTI if (norm_dir / f"{f}.jsonl").exists()]
    if not fonti:
        sys.exit(f"nessuna cache normalizzata in {norm_dir}: esegui prima `prepare`")
    out_dir = Path(args.cache) / "token"
    out_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    for modello in modelli:
        etichetta = modello["etichetta"]
        percorso = out_dir / f"{etichetta}.jsonl"
        meta_path = out_dir / f"{etichetta}.json"
        if percorso.exists() and not args.sovrascrivi:
            print(f"salto {etichetta}: {percorso} esiste (usa --sovrascrivi per rifare)")
            continue

        print(f"\n=== {etichetta} ({modello['repo']}@{modello['revision'][:10]}) ===")
        try:
            tok = carica_tokenizer(modello, args.hf_token)
        except RuntimeError as exc:
            sys.exit(str(exc))
        kwargs_template = dict(modello.get("template_kwargs") or {})
        piega = rileva_piegatura_system(tok, kwargs_template)
        commit = getattr(tok, "init_kwargs", {}).get("_commit_hash") or modello["revision"]
        print(f"  commit tokenizer: {commit} | system piegato: {piega}")

        righe = []
        errori = []
        per_fonte = Counter()
        for fonte in fonti:
            lette = 0
            for rec in leggi_jsonl(norm_dir / f"{fonte}.jsonl"):
                if args.limite and lette >= args.limite:
                    break
                lette += 1
                prompt = rec["messages"][: rec["indice_ultimo_utente"] + 1]
                try:
                    n = conta_token(tok, prompt, piega, kwargs_template, generazione=True)
                    riga = {"id": rec["id"], "token_prompt": n}
                    if args.conteggio_totale:
                        riga["token_totali"] = conta_token(
                            tok, rec["messages"], piega, kwargs_template, generazione=False)
                except Exception as exc:
                    errori.append({"id": rec["id"], "errore": str(exc)})
                    riga = {"id": rec["id"], "token_prompt": -1}
                righe.append(riga)
                per_fonte[fonte] += 1
                if len(righe) % 1000 == 0:
                    print(f"  ...{len(righe):,} conversazioni, {len(errori):,} errori")

        if righe and len(errori) / len(righe) > args.max_errori:
            scrivi_json(meta_path.with_suffix(".errori.json"), {"errori": errori[:200]})
            sys.exit(f"troppi errori di tokenizzazione per {etichetta}: "
                     f"{len(errori)}/{len(righe)} (limite {args.max_errori:.0%}). "
                     f"Primi errori in {meta_path.with_suffix('.errori.json')}")

        scrivi_jsonl(percorso, righe)
        meta = {
            "fase": "tokenize",
            "versione": VERSIONE,
            "script": {"nome": Path(__file__).name, "sha256": sha256_script()},
            "data_utc": ora_utc(),
            "modello": {
                "etichetta": etichetta,
                "repo": modello["repo"],
                "revision": modello["revision"],
                "commit_risolto": commit,
                "contesto_dichiarato": modello["contesto"],
                "template_kwargs": kwargs_template,
                "trust_remote_code": bool(modello.get("trust_remote_code")),
                "hf_token_usato": bool(token),
            },
            "metodo": "apply_chat_template(tokenize=True, add_generation_prompt=True) "
                      "sui messaggi fino all'ultimo user",
            "system_piegato": piega,
            "conteggio_totale": args.conteggio_totale,
            "limite_per_fonte": args.limite,
            "conversazioni": len(righe),
            "per_fonte": dict(per_fonte),
            "errori": len(errori),
            "errori_campione": errori[:20],
            "file": {"percorso": str(percorso), "sha256": sha256_file(percorso)},
        }
        scrivi_json(meta_path, meta)
        print(f"  scritte {len(righe):,} righe in {percorso}, {len(errori):,} errori")

    return 0


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
def contesto_effettivo(modello: dict, runtime: int) -> int:
    dichiarato = int(modello.get("contesto") or 0)
    if runtime > 0:
        return min(dichiarato, runtime) if dichiarato else runtime
    if not dichiarato:
        sys.exit(f"contesto non dichiarato per {modello['etichetta']}: usa --contesto-runtime")
    return dichiarato


def costruisci_edges(minimo: int, valori) -> list[int]:
    if isinstance(valori, str):
        valori = [v.strip() for v in valori.split(",") if v.strip()]
    edges = {int(minimo)}
    edges.update(int(v) for v in valori if int(v) > int(minimo))
    return sorted(edges)


def etichetta_fascia(valore: int, edges: list[int]) -> str:
    for a, b in zip(edges, edges[1:]):
        if a <= valore < b:
            return f"{a}-{b - 1}"
    return f">={edges[-1]}"


def cmd_select(args) -> int:
    modelli = risolvi_modelli(args.modelli)
    etichette = [m["etichetta"] for m in modelli]
    # Il riferimento serve sia alla soglia (modo "riferimento") sia alle fasce
    # (misura "riferimento"): va rispettato anche in modo "tutti".
    if args.tokenizer_riferimento is None:
        args.tokenizer_riferimento = etichette[0]
    if args.tokenizer_riferimento not in etichette:
        sys.exit("--tokenizer-riferimento deve essere uno dei modelli selezionati")

    token_dir = Path(args.cache) / "token"
    mappe: dict[str, dict[str, int]] = {}
    for etichetta in etichette:
        percorso = token_dir / f"{etichetta}.jsonl"
        if not percorso.exists():
            sys.exit(f"manca {percorso}: esegui `tokenize --modelli {etichetta}` prima di `select`")
        mappe[etichetta] = {
            r["id"]: int(r["token_prompt"]) for r in leggi_jsonl(percorso)
            if int(r.get("token_prompt", -1)) >= 0
        }

    contesti = {m["etichetta"]: contesto_effettivo(m, args.contesto_runtime) for m in modelli}
    meta_tokenizer = {}
    for etichetta in etichette:
        percorso_meta = token_dir / f"{etichetta}.json"
        if percorso_meta.exists():
            meta_tokenizer[etichetta] = json.loads(percorso_meta.read_text(encoding="utf-8"))
    percorso_prepare = Path(args.cache) / "prepare.json"
    info_prepare = (json.loads(percorso_prepare.read_text(encoding="utf-8"))
                    if percorso_prepare.exists() else {})
    info_lingua = info_prepare.get("rilevamento_lingua") or {}
    norm_dir = Path(args.cache) / "normalizzato"
    fonti = [f for f in ORDINE_FONTI if (norm_dir / f"{f}.jsonl").exists()]
    if not fonti:
        sys.exit(f"nessuna cache normalizzata in {norm_dir}: esegui prima `prepare`")

    fasce_token_edges = costruisci_edges(args.min_tokeni, args.fasce_token)
    fasce_turni_edges = costruisci_edges(args.min_turni, args.fasce_turni)

    imbuto_fonte = defaultdict(Counter)
    imbuto_modello = defaultdict(Counter)
    senza_token = 0
    candidati = []

    for fonte in fonti:
        for rec in leggi_jsonl(norm_dir / f"{fonte}.jsonl"):
            imbuto_fonte[fonte]["lette"] += 1
            if rec["turni"] < args.min_turni:
                continue
            imbuto_fonte[fonte]["turni_ok"] += 1
            if not passa_filtro_coding(rec["messages"], args.solo_coding,
                                       args.solo_informatica):
                continue
            imbuto_fonte[fonte]["coding_ok"] += 1
            token = {}
            for etichetta in etichette:
                valore = mappe[etichetta].get(rec["id"])
                if valore is None:
                    break
                token[etichetta] = valore
            if len(token) != len(etichette):
                senza_token += 1
                continue
            imbuto_fonte[fonte]["con_token"] += 1

            if args.soglia_token_modo == "riferimento":
                soglia_ok = token[args.tokenizer_riferimento] >= args.min_tokeni
            else:
                soglia_ok = min(token.values()) >= args.min_tokeni
            if not soglia_ok:
                continue
            imbuto_fonte[fonte]["soglia_token_ok"] += 1

            finestra_ok = True
            for etichetta, valore in token.items():
                if valore + args.riserva_output > contesti[etichetta]:
                    finestra_ok = False
                    imbuto_modello[etichetta]["fuori_finestra"] += 1
                else:
                    imbuto_modello[etichetta]["in_finestra"] += 1
            if not finestra_ok:
                continue
            imbuto_fonte[fonte]["finestra_ok"] += 1

            misura = max(token.values()) if args.misura_strato == "max" else token[args.tokenizer_riferimento]
            fascia_t = etichetta_fascia(rec["turni"], fasce_turni_edges)
            fascia_k = etichetta_fascia(misura, fasce_token_edges)
            rango = hashlib.sha256(
                f"{args.seed}|{fonte}|{rec['id_fonte']}".encode("utf-8")).hexdigest()
            candidati.append({
                "id": rec["id"], "fonte": fonte, "id_fonte": rec["id_fonte"],
                "modello_origine": rec["modello_origine"], "lingua": rec["lingua"],
                "lingua_confidenza": rec.get("lingua_confidenza"),
                "turni": rec["turni"], "hash_contenuto": rec["hash_contenuto"],
                "indice_ultimo_utente": rec["indice_ultimo_utente"],
                "byte_prompt": rec["byte_prompt"],
                "token_prompt": token, "fascia_turni": fascia_t,
                "fascia_token": fascia_k, "_rango": rango,
            })

    gruppi: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in candidati:
        gruppi[(rec["fonte"], rec["fascia_turni"], rec["fascia_token"])].append(rec)

    scelti = []
    strati = []
    for chiave in sorted(gruppi):
        righe = sorted(gruppi[chiave], key=lambda r: r["_rango"])
        quante = len(righe) if args.n_per_strato == 0 else min(args.n_per_strato, len(righe))
        scelti.extend(righe[:quante])
        strati.append({
            "fonte": chiave[0], "fascia_turni": chiave[1], "fascia_token": chiave[2],
            "idonei": len(righe), "selezionati": quante,
        })

    scelti.sort(key=lambda r: (r["fonte"], r["fascia_turni"], r["fascia_token"], r["_rango"]))
    for rec in scelti:
        rec.pop("_rango", None)

    print("\n=== Imbuto per fonte ===")
    for fonte in fonti:
        c = imbuto_fonte[fonte]
        print(f"  {fonte:16} lette={c['lette']:>8,}  turni={c['turni_ok']:>7,}  "
              f"coding={c['coding_ok']:>7,}  token={c['soglia_token_ok']:>7,}  "
              f"finestra={c['finestra_ok']:>7,}")
    print(f"  conversazioni senza token per qualche modello: {senza_token:,}")
    print("\n=== Idonee per modello (dentro la finestra) ===")
    for etichetta in etichette:
        c = imbuto_modello[etichetta]
        print(f"  {etichetta:42} in={c['in_finestra']:>7,}  fuori={c['fuori_finestra']:>7,}  "
              f"contesto={contesti[etichetta]:,}  riserva={args.riserva_output:,}")
    print("\n=== Strati ===")
    for s in strati:
        print(f"  {s['fonte']:16} turni {s['fascia_turni']:>8} token {s['fascia_token']:>12} "
              f"idonei={s['idonei']:>6,} selezionati={s['selezionati']:>5,}")
    print(f"\nSelezionate {len(scelti):,} conversazioni su {len(candidati):,} idonee")

    if args.dry_run:
        print("--dry-run: nessun file scritto")
        return 0

    out_dir = Path(args.out)
    controlla_sovrascrittura(out_dir / "conversazioni.jsonl", args.sovrascrivi, "Il replay set")
    controlla_sovrascrittura(out_dir / "selezione.json", args.sovrascrivi, "Il replay set")
    controlla_sovrascrittura(out_dir / "strati.csv", args.sovrascrivi, "Il replay set")

    # Seconda passata: recupera i messaggi solo delle selezionate.
    voluti = {r["id"]: r for r in scelti}
    messaggi = {}
    for fonte in fonti:
        for rec in leggi_jsonl(norm_dir / f"{fonte}.jsonl"):
            if rec["id"] in voluti:
                messaggi[rec["id"]] = rec["messages"]
    mancanti = [i for i in voluti if i not in messaggi]
    if mancanti:
        sys.exit(f"messaggi mancanti per {len(mancanti)} conversazioni selezionate "
                 f"(primo: {mancanti[0]}): la cache normalizzata e' cambiata dopo la tokenizzazione")

    uscita = []
    for rec in scelti:
        uscita.append({
            "id": rec["id"],
            "fonte": rec["fonte"],
            "id_fonte": rec["id_fonte"],
            "modello_origine": rec["modello_origine"],
            "lingua": rec["lingua"],
            "lingua_confidenza": rec["lingua_confidenza"],
            "turni": rec["turni"],
            "hash_contenuto": rec["hash_contenuto"],
            "indice_ultimo_utente": rec["indice_ultimo_utente"],
            "byte_prompt": rec["byte_prompt"],
            "token_prompt": rec["token_prompt"],
            "fascia_turni": rec["fascia_turni"],
            "fascia_token": rec["fascia_token"],
            "messages": messaggi[rec["id"]],
        })
    scrivi_jsonl(out_dir / "conversazioni.jsonl", uscita)

    with open(out_dir / "strati.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(strati[0].keys()) if strati else
                           ["fonte", "fascia_turni", "fascia_token", "idonei", "selezionati"])
        w.writeheader()
        w.writerows(strati)

    manifest = {
        "fase": "select",
        "versione": VERSIONE,
        "script": {"nome": Path(__file__).name, "sha256": sha256_script()},
        "data_utc": ora_utc(),
        "parametri": {
            "min_turni": args.min_turni,
            "solo_coding": args.solo_coding,
            "solo_informatica": args.solo_informatica,
            "criterio_coding": (
                "codice oppure tema informatico/IT riconosciuto nel primo "
                "messaggio utente (almeno due segnali tecnici distinti, oppure "
                "un segnale con richiesta esplicita di spiegazione/implementazione; "
                "esclusi domini non informatici e task creativi/commerciali)"
                if args.solo_informatica else
                "blocco di codice (linguaggio riconosciuto o righe di codice) "
                "oppure riga di codice, in qualsiasi messaggio"
                if args.solo_coding else None),
            "min_tokeni": args.min_tokeni,
            "soglia_token_modo": args.soglia_token_modo,
            "tokenizer_riferimento": args.tokenizer_riferimento,
            "misura_strato": args.misura_strato,
            "fasce_token": args.fasce_token,
            "fasce_turni": args.fasce_turni,
            "n_per_strato": args.n_per_strato,
            "seed": args.seed,
            "contesto_runtime": args.contesto_runtime,
            "riserva_output": args.riserva_output,
        },
        "confini_fasce": {
            "turni": fasce_turni_edges,
            "token": fasce_token_edges,
        },
        "modelli": {
            m["etichetta"]: {
                "repo": m["repo"],
                "revision": m["revision"],
                "contesto_dichiarato": m["contesto"],
                "contesto_effettivo": contesti[m["etichetta"]],
                "template_kwargs": m.get("template_kwargs") or {},
                "commit_risolto": (meta_tokenizer.get(m["etichetta"], {})
                                   .get("modello", {}).get("commit_risolto")),
                "system_piegato": meta_tokenizer.get(m["etichetta"], {}).get("system_piegato"),
            } for m in modelli
        },
        "dataset": {
            f: {"repo": FONTI[f]["repo"], "revision": FONTI[f]["revision"],
                "licenza": FONTI[f]["licenza"]} for f in fonti
        },
        "lingua": {
            "codice": info_lingua.get("codice"),
            "soglia_confidenza": info_lingua.get("soglia_confidenza"),
            "rilevatore": (info_lingua.get("modello") or {}).get("tipo"),
            "modello_sha256": (info_lingua.get("modello") or {}).get("sha256"),
        } if info_lingua.get("codice") else None,
        "imbuto": {
            "per_fonte": {f: dict(imbuto_fonte[f]) for f in fonti},
            "per_modello": {e: dict(imbuto_modello[e]) for e in etichette},
            "senza_token": senza_token,
            "candidate_idonee": len(candidati),
        },
        "strati": strati,
        "selezionate": len(uscita),
        "file": {
            "conversazioni": "conversazioni.jsonl",
            "sha256_conversazioni": sha256_file(out_dir / "conversazioni.jsonl"),
            "strati": "strati.csv",
        },
        "note": "Campione comune: ogni conversazione rispetta la soglia minima e la finestra "
                "di tutti i modelli elencati. CodeChat deriva da WildChat: la deduplica usa "
                "l'id di fonte, non solo il contenuto.",
        "attribuzione": "Data extracted from allenai/WildChat-1M (ODC-BY), "
                        "Suzhen/CodeChat (ODC-BY) and shibing624/sharegpt_gpt4 (CC-BY-4.0). "
                        "You are responsible for complying with the dataset-specific licenses "
                        "and attribution terms.",
    }
    scrivi_json(out_dir / "selezione.json", manifest)
    print(f"\nSet congelato: {out_dir / 'conversazioni.jsonl'}")
    print(f"Manifest:      {out_dir / 'selezione.json'}")
    return 0


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def quantili(valori: list[int], punti) -> list[float]:
    """Quantili con interpolazione lineare tra ranghi (nessuna dipendenza)."""
    if not valori:
        return [float("nan")] * len(punti)
    ordinati = sorted(valori)
    ultimo = len(ordinati) - 1
    out = []
    for p in punti:
        k = ultimo * p
        i = int(k)
        j = min(i + 1, ultimo)
        out.append(ordinati[i] + (ordinati[j] - ordinati[i]) * (k - i))
    return out


def cmd_sweep(args) -> int:
    """Varia --min-tokeni e misura copertura, strati e dimensione del campione.

    Non scrive nel replay set: serve a scegliere la soglia dei token prima di
    `select`, con lo stesso criterio di stratificazione e la stessa finestra.
    """
    modelli = risolvi_modelli(args.modelli)
    etichette = [m["etichetta"] for m in modelli]
    if args.tokenizer_riferimento is None:
        args.tokenizer_riferimento = etichette[0]
    if args.tokenizer_riferimento not in etichette:
        sys.exit("--tokenizer-riferimento deve essere uno dei modelli selezionati")

    token_dir = Path(args.cache) / "token"
    mappe: dict[str, dict[str, int]] = {}
    for etichetta in etichette:
        percorso = token_dir / f"{etichetta}.jsonl"
        if not percorso.exists():
            sys.exit(f"manca {percorso}: esegui `tokenize --modelli {etichetta}` prima di `sweep`")
        mappe[etichetta] = {
            r["id"]: int(r["token_prompt"]) for r in leggi_jsonl(percorso)
            if int(r.get("token_prompt", -1)) >= 0
        }

    contesti = {m["etichetta"]: contesto_effettivo(m, args.contesto_runtime) for m in modelli}
    norm_dir = Path(args.cache) / "normalizzato"
    fonti = [f for f in ORDINE_FONTI if (norm_dir / f"{f}.jsonl").exists()]
    if not fonti:
        sys.exit(f"nessuna cache normalizzata in {norm_dir}: esegui prima `prepare`")

    fasce_turni_edges = costruisci_edges(args.min_turni, args.fasce_turni)

    corpus = []
    lette = turni_ok = coding_ok = 0
    for fonte in fonti:
        for rec in leggi_jsonl(norm_dir / f"{fonte}.jsonl"):
            lette += 1
            if rec["turni"] < args.min_turni:
                continue
            turni_ok += 1
            if not passa_filtro_coding(rec["messages"], args.solo_coding,
                                       args.solo_informatica):
                continue
            coding_ok += 1
            token = {}
            for etichetta in etichette:
                valore = mappe[etichetta].get(rec["id"])
                if valore is None:
                    break
                token[etichetta] = valore
            if len(token) != len(etichette):
                continue
            finestra_ok = all(valore + args.riserva_output <= contesti[e]
                              for e, valore in token.items())
            corpus.append({"fonte": fonte, "turni": rec["turni"], "token": token,
                           "finestra_ok": finestra_ok})

    print(f"\n{len(corpus):,} conversazioni con {args.min_turni}+ turni e token per "
          f"tutti i {len(etichette)} modelli")
    if args.solo_informatica:
        print(f"Filtro informatico: {coding_ok:,} conversazioni su {turni_ok:,} con "
              f"{args.min_turni}+ turni ({lette:,} lette)")
    elif args.solo_coding:
        print(f"Filtro coding: {coding_ok:,} conversazioni su {turni_ok:,} con "
              f"{args.min_turni}+ turni ({lette:,} lette)")
    if not corpus:
        sys.exit("nessuna conversazione tokenizzata: controlla --modelli e la cache")

    print("\nDistribuzione dei token di prompt per modello:")
    print(f"  {'modello':42} {'p10':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}")
    for etichetta in etichette:
        valori = [c["token"][etichetta] for c in corpus]
        p10, p50, p90, p99 = quantili(valori, (0.1, 0.5, 0.9, 0.99))
        print(f"  {etichetta:42} {p10:8,.0f} {p50:8,.0f} {p90:8,.0f} {p99:8,.0f} {max(valori):8,}")

    soglie = sorted({int(v) for v in re.split(r"[,\s]+", str(args.soglie)) if v.strip()})
    righe = []
    dettaglio_gruppi: dict[int, Counter] = {}
    for soglia in soglie:
        idonei = []
        for c in corpus:
            if not c["finestra_ok"]:
                continue
            if args.soglia_token_modo == "riferimento":
                ok = c["token"][args.tokenizer_riferimento] >= soglia
            else:
                ok = min(c["token"].values()) >= soglia
            if ok:
                idonei.append(c)

        per_fonte = Counter(c["fonte"] for c in idonei)
        per_modello = {e: sum(1 for c in idonei if c["token"][e] >= soglia)
                       for e in etichette}
        # Le fasce partono dalla soglia, come in `select`: cosi' le etichette
        # (es. "6000-7999") descrivono davvero il contenuto dello strato.
        fasce_token_edges = costruisci_edges(soglia, args.fasce_token)
        gruppi = Counter()
        for c in idonei:
            misura = (max(c["token"].values()) if args.misura_strato == "max"
                      else c["token"][args.tokenizer_riferimento])
            gruppi[(c["fonte"], etichetta_fascia(c["turni"], fasce_turni_edges),
                    etichetta_fascia(misura, fasce_token_edges))] += 1
        if args.n_per_strato == 0:
            pieni = len(gruppi)
            selezionati = sum(gruppi.values())
        else:
            pieni = sum(1 for n in gruppi.values() if n >= args.n_per_strato)
            selezionati = sum(min(args.n_per_strato, n) for n in gruppi.values())
        dettaglio_gruppi[soglia] = gruppi
        misure = [max(c["token"].values()) for c in idonei]
        p50, p90 = quantili(misure, (0.5, 0.9))
        riga = {
            "soglia": soglia,
            "conversazioni": len(idonei),
            "strati": len(gruppi),
            "strati_pieni": pieni,
            "selezionati": selezionati,
            "copertura_min": min(gruppi.values()) if gruppi else 0,
            "mediana_token_max": round(p50, 1),
            "p90_token_max": round(p90, 1),
            "max_token": max(misure) if misure else 0,
        }
        for fonte in fonti:
            riga[f"fonte_{fonte}"] = per_fonte.get(fonte, 0)
        for etichetta in etichette:
            riga[f"token_ok_{etichetta}"] = per_modello[etichetta]
        righe.append(riga)

    print(f"\n  {'soglia':>7} {'idonee':>7} " +
          " ".join(f"{f[:9]:>9}" for f in fonti) +
          f" {'strati':>6} {'pieni':>6} {'selez':>6} {'cop.min':>7} {'p50':>7} {'p90':>7}")
    for r in righe:
        print(f"  {r['soglia']:7,} {r['conversazioni']:7,} " +
              " ".join(f"{r[f'fonte_{f}']:9,}" for f in fonti) +
              f" {r['strati']:6,} {r['strati_pieni']:6,} {r['selezionati']:6,} "
              f"{r['copertura_min']:7,} {r['mediana_token_max']:7,.0f} {r['p90_token_max']:7,.0f}")

    out = Path(args.out) if args.out else Path(args.cache) / "sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(righe[0].keys()) if righe else ["soglia"])
        w.writeheader()
        w.writerows(righe)
    print(f"\nCSV: {out}")

    vuote = [f for f in fonti if all(r[f"fonte_{f}"] == 0 for r in righe)]
    if vuote:
        print(f"Attenzione: fonti senza conversazioni in nessuna soglia: {', '.join(vuote)} "
              f"(per CodeChat e' atteso: deriva da WildChat e gli id sono gia' visti)")

    massimo = max((r["selezionati"] for r in righe), default=0)
    if massimo:
        plateau = [r["soglia"] for r in righe
                   if r["selezionati"] >= 0.9 * massimo
                   and all(r[f"fonte_{f}"] > 0 for f in fonti)]
        if plateau:
            print(f"Plateau (>= 90% del campione massimo, tutte le fonti presenti): "
                  f"{min(plateau):,}-{max(plateau):,} token")
        massime = [r["soglia"] for r in righe
                   if all(r[f"fonte_{f}"] > 0 for f in fonti) and r["conversazioni"]]
        if massime:
            print(f"Soglia massima con tutte le fonti presenti: {max(massime):,} token")
        complete = [r for r in righe if r["strati"] and r["strati_pieni"] == r["strati"]]
        if complete and args.n_per_strato:
            migliore = max(complete, key=lambda r: r["soglia"])
            print(f"Soglia massima con ogni strato >= {args.n_per_strato}: "
                  f"{migliore['soglia']:,} token ({migliore['selezionati']:,} conversazioni)")

    if args.dettaglio:
        gruppi = dettaglio_gruppi.get(args.dettaglio)
        if gruppi is None:
            print(f"\n{args.dettaglio:,} non e' tra le soglie provate")
        else:
            target = (f"n target {args.n_per_strato}" if args.n_per_strato
                      else "n=0: tutte le conversazioni")
            print(f"\nStrati a {args.dettaglio:,} token ({target}):")
            for chiave, n in sorted(gruppi.items(), key=lambda kv: (kv[1], kv[0])):
                if args.n_per_strato:
                    stato = "ok" if n >= args.n_per_strato else "sotto"
                else:
                    stato = ""
                print(f"  {chiave[0]:16} turni {chiave[1]:>8} token {chiave[2]:>12} "
                      f"{n:6,}  {stato}")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def cmd_verify(args) -> int:
    out_dir = Path(args.out)
    manifest_path = out_dir / "selezione.json"
    conversazioni_path = out_dir / "conversazioni.jsonl"
    if not manifest_path.exists() or not conversazioni_path.exists():
        sys.exit(f"manca {manifest_path} o {conversazioni_path}: esegui prima `select`")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    errori = []
    atteso = manifest.get("file", {}).get("sha256_conversazioni")
    if atteso and sha256_file(conversazioni_path) != atteso:
        errori.append("sha256 di conversazioni.jsonl diverso dal manifest")

    parametri = manifest["parametri"]
    contesti = {e: m["contesto_effettivo"] for e, m in manifest["modelli"].items()}
    righe = list(leggi_jsonl(conversazioni_path))
    visti = set()
    per_strato = Counter()
    for riga in righe:
        if riga["id"] in visti:
            errori.append(f"id duplicato: {riga['id']}")
        visti.add(riga["id"])
        if riga["id"] != f"{riga['fonte']}:{riga['id_fonte']}":
            errori.append(f"{riga['id']}: id incoerente con fonte/id_fonte")
        if riga["turni"] < parametri["min_turni"]:
            errori.append(f"{riga['id']}: turni {riga['turni']} < {parametri['min_turni']}")
        if hash_contenuto(riga["messages"]) != riga["hash_contenuto"]:
            errori.append(f"{riga['id']}: hash_contenuto non corrisponde ai messaggi")
        messaggi = riga["messages"]
        if not messaggi:
            errori.append(f"{riga['id']}: conversazione vuota")
            continue
        if parametri.get("solo_informatica"):
            if not (ha_codice(messaggi) or ha_tema_informatico(messaggi)):
                errori.append(f"{riga['id']}: non informatica ma selezionata "
                              f"con --solo-informatica")
        elif parametri.get("solo_coding") and not ha_codice(messaggi):
            errori.append(f"{riga['id']}: non coding ma selezionata con --solo-coding")
        ruoli = [m["role"] for m in messaggi]
        turni_veri = sum(1 for r in ruoli if r == "user")
        if turni_veri != riga["turni"]:
            errori.append(f"{riga['id']}: turni dichiarati {riga['turni']}, trovati {turni_veri}")
        sequenza = [r for r in ruoli if r != "system"]
        if any(sequenza[i] == sequenza[i + 1] for i in range(len(sequenza) - 1)):
            errori.append(f"{riga['id']}: ruoli non alternati")
        if sequenza[:1] != ["user"]:
            errori.append(f"{riga['id']}: la sequenza non inizia con l'utente")
        indice = riga.get("indice_ultimo_utente", -1)
        if not (0 <= indice < len(messaggi)) or messaggi[indice]["role"] != "user":
            errori.append(f"{riga['id']}: indice_ultimo_utente non valido")
        else:
            if any(m["role"] == "user" for m in messaggi[indice + 1:]):
                errori.append(f"{riga['id']}: esistono messaggi user dopo l'indice dichiarato")
            byte_prompt = sum(len(m["content"].encode("utf-8")) for m in messaggi[: indice + 1])
            if byte_prompt != riga.get("byte_prompt"):
                errori.append(f"{riga['id']}: byte_prompt non corrisponde ai messaggi")
        if set(riga["token_prompt"]) != set(contesti):
            errori.append(f"{riga['id']}: token mancanti per qualche modello")
        for etichetta, valore in riga["token_prompt"].items():
            if valore < 0:
                errori.append(f"{riga['id']}: token negativi per {etichetta}")
            if valore + parametri["riserva_output"] > contesti[etichetta]:
                errori.append(f"{riga['id']}: fuori finestra per {etichetta} "
                              f"({valore}+{parametri['riserva_output']}>{contesti[etichetta]})")
        if parametri["soglia_token_modo"] == "riferimento":
            if riga["token_prompt"][parametri["tokenizer_riferimento"]] < parametri["min_tokeni"]:
                errori.append(f"{riga['id']}: sotto soglia token (riferimento)")
        elif min(riga["token_prompt"].values()) < parametri["min_tokeni"]:
            errori.append(f"{riga['id']}: sotto soglia token (tutti)")
        codice_atteso = (manifest.get("lingua") or {}).get("codice")
        if codice_atteso and riga.get("lingua") != codice_atteso:
            errori.append(f"{riga['id']}: lingua {riga.get('lingua')!r} != {codice_atteso!r}")
        confini = manifest.get("confini_fasce", {})
        if confini:
            attesa_t = etichetta_fascia(riga["turni"], confini["turni"])
            if attesa_t != riga["fascia_turni"]:
                errori.append(f"{riga['id']}: fascia_turni {riga['fascia_turni']} != {attesa_t}")
            if parametri["misura_strato"] == "max":
                misura = max(riga["token_prompt"].values())
            else:
                misura = riga["token_prompt"][parametri["tokenizer_riferimento"]]
            attesa_k = etichetta_fascia(misura, confini["token"])
            if attesa_k != riga["fascia_token"]:
                errori.append(f"{riga['id']}: fascia_token {riga['fascia_token']} != {attesa_k}")
        per_strato[(riga["fonte"], riga["fascia_turni"], riga["fascia_token"])] += 1

    for strato in manifest.get("strati", []):
        chiave = (strato["fonte"], strato["fascia_turni"], strato["fascia_token"])
        if per_strato.get(chiave, 0) != strato["selezionati"]:
            errori.append(f"strato {chiave}: attesi {strato['selezionati']}, "
                          f"trovati {per_strato.get(chiave, 0)}")
    if len(righe) != manifest.get("selezionate"):
        errori.append(f"selezionate: attese {manifest.get('selezionate')}, trovate {len(righe)}")

    if args.ricalcola_token:
        errori.extend(ricalcola_token(manifest, righe, args))

    if errori:
        print(f"VERIFICA FALLITA: {len(errori)} problemi")
        for e in errori[:50]:
            print(f"  - {e}")
        return 1
    print(f"VERIFICA OK: {len(righe):,} conversazioni, {len(manifest.get('strati', []))} strati, "
          f"sha256 e soglie coerenti")
    return 0


def ricalcola_token(manifest: dict, righe: list[dict], args) -> list[str]:
    errori = []
    campione = righe[: args.campione] if args.campione else righe
    for etichetta in manifest["modelli"]:
        modello = MODELLI_PER_ETICHETTA.get(etichetta)
        if modello is None:
            errori.append(f"modello {etichetta} non piu' in MODELLI: impossibile ricalcolare")
            continue
        try:
            tok = carica_tokenizer(modello, getattr(args, "hf_token", None))
        except RuntimeError as exc:
            errori.append(f"{etichetta}: {exc}")
            continue
        info = manifest["modelli"][etichetta]
        kwargs_template = info.get("template_kwargs") or {}
        piega = info.get("system_piegato")
        if piega is None:
            piega = rileva_piegatura_system(tok, kwargs_template)
        for riga in campione:
            indice = riga.get("indice_ultimo_utente")
            if isinstance(indice, int) and 0 <= indice < len(riga["messages"]):
                prompt = riga["messages"][: indice + 1]
            else:  # selezione di una versione precedente
                prompt = riga["messages"]
                while prompt and prompt[-1]["role"] != "user":
                    prompt = prompt[:-1]
            n = conta_token(tok, prompt, piega, kwargs_template, generazione=True)
            if n != riga["token_prompt"][etichetta]:
                errori.append(f"{riga['id']}/{etichetta}: token ricalcolati {n} != "
                              f"{riga['token_prompt'][etichetta]}")
    return errori


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_modelli(args) -> int:
    print(f"{'etichetta':42} {'contesto':>10}  repo")
    for m in MODELLI:
        print(f"{m['etichetta']:42} {m['contesto']:>10,}  {m['repo']}")
    return 0


def aggiungi_argomenti_comuni(p) -> None:
    p.add_argument("--cache", default="dataset_cache",
                   help="cartella della cache (default: dataset_cache)")
    p.add_argument("--hf-token", default=None,
                   help="token Hugging Face; se omesso usa HF_TOKEN / HUGGING_FACE_HUB_TOKEN")
    p.add_argument("--sovrascrivi", action="store_true",
                   help="sovrascrive i file esistenti (solo per prove)")
    p.add_argument("--limite", type=int, default=0,
                   help="legge solo le prime N righe per fonte (0 = tutte; utile per prove)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Costruisce il replay set congelato dai dataset WildChat, ShareGPT-GPT4 e CodeChat.")
    ap.add_argument("--versione", action="version", version=f"%(prog)s {VERSIONE}")
    sub = ap.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("prepare", help="normalizza, elimina duplicati/non validi, filtra i turni")
    aggiungi_argomenti_comuni(p)
    p.add_argument("--fonti", default=",".join(ORDINE_FONTI),
                   help="fonti in ordine di precedenza (default: wildchat,sharegpt_gpt4,codechat)")
    p.add_argument("--min-turni", type=int, default=10,
                   help="turni minimi (default: 10, da confermare)")
    p.add_argument("--politica-system", choices=("first", "drop"), default="first",
                   help="tiene il system iniziale o lo elimina (default: first)")
    p.add_argument("--lingua", default="en",
                   help="tiene solo questa lingua (codice ISO, default: en; accetta anche 'English'). "
                        "Il rilevamento usa fastText su tutte le fonti")
    p.add_argument("--tutte-le-lingue", action="store_true",
                   help="disattiva il filtro lingua e tiene anche le conversazioni non inglesi "
                        "(solo per prove: l'esperimento richiede l'inglese)")
    p.add_argument("--lingua-soglia", type=float, default=0.80,
                   help="confidenza minima del rilevatore (default: 0.80: a 0.90 si perde "
                        "circa il 20%% delle conversazioni inglesi lunghe)")
    p.add_argument("--lingua-caratteri", type=int, default=2000,
                   help="caratteri dei messaggi utente valutati dal rilevatore (default: 2000)")
    p.add_argument("--modello-lingua", default=None,
                   help="percorso del modello fastText lid.176 "
                        "(default: <cache>/models/lid.176.bin, scaricato se assente)")
    p.add_argument("--includi-tossici", action="store_true",
                   help="include le conversazioni WildChat con toxic=True (default: escluse)")
    p.add_argument("--escludi-redatti", action="store_true",
                   help="esclude le conversazioni WildChat con redacted=True")
    p.add_argument("--prefiltro-byte", type=int, default=0,
                   help="soglia preliminare in byte UTF-8: scarta se byte_prompt + margine < soglia "
                        "(0 = nessun prefiltro). I byte sono un limite inferiore sicuro dei token, "
                        "i caratteri no (codice e CJK)")
    p.add_argument("--margine-prefiltro", type=int, default=1024,
                   help="margine in token del prefiltro conservativo (default: 1024)")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("tokenize", help="conta i token di prompt con i tokenizer dei modelli")
    aggiungi_argomenti_comuni(p)
    p.add_argument("--modelli", default=None,
                   help="etichette separate da virgola (default: tutti i 10 modelli)")
    p.add_argument("--conteggio-totale", action="store_true",
                   help="salva anche i token dell'intera conversazione (piu' lento)")
    p.add_argument("--max-errori", type=float, default=0.02,
                   help="frazione massima di errori di template (default: 0.02)")
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("select", help="applica le soglie, stratifica e congela il campione")
    p.add_argument("--cache", default="dataset_cache")
    p.add_argument("--out", default="replay_set", help="cartella del set congelato (default: replay_set)")
    p.add_argument("--modelli", default=None, help="etichette separate da virgola (default: tutti)")
    p.add_argument("--min-turni", type=int, default=10, help="turni minimi (default: 10, da confermare)")
    p.add_argument("--solo-coding", action="store_true",
                   help="tiene solo le conversazioni con codice (blocco o riga di codice)")
    p.add_argument("--solo-informatica", action="store_true",
                   help="tiene il codice e le conversazioni su temi informatici/IT "
                        "(piu' ampio di --solo-coding)")
    p.add_argument("--min-tokeni", type=int, default=4000,
                   help="token minimi di prompt (default: 4000, da confermare)")
    p.add_argument("--soglia-token-modo", choices=("tutti", "riferimento"), default="tutti",
                   help="la soglia vale per tutti i tokenizer o solo per quello di riferimento (default: tutti)")
    p.add_argument("--tokenizer-riferimento", default=None,
                   help="etichetta del tokenizer di riferimento (default: primo modello)")
    p.add_argument("--misura-strato", choices=("max", "riferimento"), default="max",
                   help="token usati per la fascia: massimo tra i modelli o riferimento (default: max)")
    p.add_argument("--fasce-token", default="4000,8000,16000",
                   help="confini delle fasce di token (default: 4000,8000,16000)")
    p.add_argument("--fasce-turni", default="10,15,25",
                   help="confini delle fasce di turni (default: 10,15,25)")
    p.add_argument("--n-per-strato", type=int, required=True,
                   help="conversazioni per strato (0 = tutte; richiesto: dipende dal disegno sperimentale)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--contesto-runtime", type=int, default=32768,
                   help="context size di llama-server (-c), uguale per tutti (default: 32768; 0 = usa il dichiarato)")
    p.add_argument("--riserva-output", type=int, default=2048,
                   help="token riservati alla generazione (default: 2048)")
    p.add_argument("--dry-run", action="store_true", help="stampa l'imbuto senza scrivere")
    p.add_argument("--sovrascrivi", action="store_true")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("sweep",
                       help="varia --min-tokeni e misura copertura, strati e campione (non scrive il set)")
    p.add_argument("--cache", default="dataset_cache")
    p.add_argument("--out", default=None,
                   help="CSV di uscita (default: <cache>/sweep.csv)")
    p.add_argument("--modelli", default=None, help="etichette separate da virgola (default: tutti)")
    p.add_argument("--min-turni", type=int, default=10,
                   help="turni minimi (default: 10)")
    p.add_argument("--solo-coding", action="store_true",
                   help="tiene solo le conversazioni con codice (blocco o riga di codice)")
    p.add_argument("--solo-informatica", action="store_true",
                   help="tiene il codice e le conversazioni su temi informatici/IT "
                        "(piu' ampio di --solo-coding)")
    p.add_argument("--soglie", default="2000,3000,4000,5000,6000,8000,10000,12000,16000,20000,24000,28000",
                   help="soglie di token da confrontare (default: 2000..28000)")
    p.add_argument("--soglia-token-modo", choices=("tutti", "riferimento"), default="tutti",
                   help="la soglia vale per tutti i tokenizer o solo per quello di riferimento")
    p.add_argument("--tokenizer-riferimento", default=None)
    p.add_argument("--misura-strato", choices=("max", "riferimento"), default="max",
                   help="token usati per la fascia: massimo tra i modelli o riferimento (default: max)")
    p.add_argument("--fasce-token", default="4000,8000,16000",
                   help="confini delle fasce di token (default: 4000,8000,16000)")
    p.add_argument("--fasce-turni", default="10,15,25",
                   help="confini delle fasce di turni (default: 10,15,25)")
    p.add_argument("--n-per-strato", type=int, default=30,
                   help="conversazioni per strato per stimare la copertura (default: 30; 0 = tutte)")
    p.add_argument("--contesto-runtime", type=int, default=32768,
                   help="context size di llama-server (-c), uguale per tutti (default: 32768)")
    p.add_argument("--riserva-output", type=int, default=2048,
                   help="token riservati alla generazione (default: 2048)")
    p.add_argument("--dettaglio", type=int, default=0,
                   help="stampa gli strati di una singola soglia (0 = non stampare)")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("verify", help="ricontrolla hash, soglie, finestre e coerenza")
    p.add_argument("--out", default="replay_set")
    p.add_argument("--cache", default="dataset_cache")
    p.add_argument("--hf-token", default=None,
                   help="token Hugging Face per ricalcolare i token dei repo gated")
    p.add_argument("--ricalcola-token", action="store_true",
                   help="ricarica i tokenizer e ricontrolla un campione (lento, richiede rete)")
    p.add_argument("--campione", type=int, default=5,
                   help="conversazioni da ricalcolare (default: 5; 0 = tutte)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("modelli", help="elenca i 10 modelli e i loro contesti dichiarati")
    p.set_defaults(func=cmd_modelli)

    args = ap.parse_args()
    if getattr(args, "limite", None) is None:
        args.limite = 0
    return args.func(args)


if __name__ == "__main__":
    # datasets/huggingface_hub lasciano thread di streaming attivi: in alcuni
    # run la finalizzazione dell'interprete aborta con
    # "PyGILState_Release: thread state ... must be current". I file sono gia'
    # chiusi e sincronizzati, quindi si esce senza eseguire i finalizer.
    try:
        codice = main()
    except SystemExit as exc:
        codice = exc.code
        if isinstance(codice, str):
            print(codice, file=sys.stderr)
            codice = 1
        elif codice is None:
            codice = 0
        elif not isinstance(codice, int):
            codice = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(codice)
