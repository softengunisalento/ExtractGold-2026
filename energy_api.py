import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import codecarbon
import requests
from codecarbon import EmissionsTracker
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

# =========================
# CONFIG
# =========================

LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "dummy-key")  # llama.cpp accetta qualsiasi valore
EMISSIONS_DIR = os.getenv("EMISSIONS_DIR", "./emissions")

LOG_DIR = os.getenv("LOG_DIR", "./logs")
TRACKING_MODE = os.getenv("TRACKING_MODE", "machine")
GE_MARKER = os.getenv("GE_MARKER", "riepilogo conciso del contesto").lower()
LOG_FULL_REQUEST = os.getenv("LOG_FULL_REQUEST", "1") == "1"
REQUEST_USAGE = os.getenv("REQUEST_USAGE", "1") == "1"
# richieste senza identificativo di conversazione (es. il titolo generato da Chat UI): ausiliarie
AUX_NO_CONV_ID = os.getenv("AUX_NO_CONV_ID", "1") == "1"
# nome del modello mostrato in Chat UI (e inviato come campo "model"): volutamente neutro;
# il modello vero è in model_label (ricavato da llama-server)
MODEL_ID = os.getenv("MODEL_ID", "assistente-locale")

os.makedirs(EMISSIONS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# riepilogo all'avvio: serve a vedere subito dove finiscono i log (i percorsi relativi dipendono dalla cartella di avvio)
print("=== energy_api: configurazione ===")
print("LOG_DIR       :", Path(LOG_DIR).resolve())
print("EMISSIONS_DIR :", Path(EMISSIONS_DIR).resolve())
print("LLAMA_BASE_URL:", LLAMA_BASE_URL)
print("TRACKING_MODE :", TRACKING_MODE, "| MODEL_ID:", MODEL_ID)

app = FastAPI(title="Energy API with CodeCarbon")


def llama_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLAMA_API_KEY}",
    }

def build_clean_messages(payload):
    messages = payload.get("messages", [])

    last_user = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m
            break

    user_content = ""
    if last_user:
        user_content = normalize_content(last_user.get("content", "")).strip()

    return [
        {
            "role": "system",
            "content": "Rispondi in modo conciso e diretto."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

def normalize_content(content):
    #"""
    #Normalizza il contenuto dei messaggi per llama.cpp:
    #- garantisce una stringa
    #- rimuove wrapper tipo 'User message: "..."`
    #"""
    if content is None:
        return ""

    if not isinstance(content, str):
        return str(content)

    # Rimuove wrapper legacy di Chat UI
    if content.startswith("User message:"):
        content = content.replace("User message:", "", 1).strip()
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]

    return content.strip()

# =========================
# LOG E SESSIONE SPERIMENTALE
# =========================

_state_lock = threading.RLock()
_log_lock = threading.Lock()
SESSION: dict = {}  # sessione corrente (una alla volta: c'è una sola workstation)

SESSION_FIELDS = ("session_id", "participant_id", "task_id", "condition", "model_label")


_active_lock = threading.Lock()
_active: dict = {}  # richieste in corso: id -> {"aux": bool, "overlap": bool}


def enter_request(rid: str, aux: bool) -> None:
    """Registra l'inizio di una richiesta; se ce ne sono altre in corso, le segna tutte come sovrapposte."""
    with _active_lock:
        overlap = bool(_active)
        for entry in _active.values():
            entry["overlap"] = True
        _active[rid] = {"aux": aux, "overlap": overlap}


def leave_request(rid: str) -> bool:
    """Registra la fine di una richiesta e dice se si è sovrapposta a un'altra."""
    with _active_lock:
        return bool(_active.pop(rid, {}).get("overlap"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append_jsonl(name: str, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _log_lock:
        with open(Path(LOG_DIR) / name, "a", encoding="utf-8") as fh:
            fh.write(line)


def fetch_server_props() -> dict:
    """Proprietà di llama-server (modello caricato, contesto, slot) da salvare a inizio sessione."""
    try:
        r = requests.get(f"{LLAMA_BASE_URL}/props", headers=llama_headers(), timeout=5)
        r.raise_for_status()
        props = r.json()
    except Exception as e:
        return {"error": str(e)}
    gen = props.get("default_generation_settings") or {}
    return {
        "model_path": props.get("model_path"),
        "model_alias": props.get("model_alias"),
        "build_info": props.get("build_info"),
        "total_slots": props.get("total_slots"),
        "n_ctx": gen.get("n_ctx"),
    }


def derive_model_label(props: dict) -> str | None:
    """Etichetta del modello ricavata da llama-server.

    Ordine: alias impostato con --alias; altrimenti la cartella models/<nome>/<file>.gguf (come la crea
    download_models.py, quindi coincide con la voce di manifest.json); altrimenti il nome del file.
    """
    path, alias = props.get("model_path"), props.get("model_alias")
    if alias and alias != path:
        return str(alias)
    if not path:
        return None
    parts = [p for p in re.split(r"[\\/]+", str(path)) if p and p not in (".", "..")]
    if not parts:
        return None
    if len(parts) >= 2 and parts[-2].lower() != "models":
        return parts[-2]
    stem = re.sub(r"\.gguf$", "", parts[-1], flags=re.IGNORECASE)
    return re.sub(r"-\d{5}-of-\d{5}$", "", stem)


def labels_match(declared: str | None, derived: str | None) -> bool | None:
    """True se l'etichetta scritta a mano è compatibile con quella del server (None se ne manca una)."""
    if not declared or not derived:
        return None
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    a, b = norm(declared), norm(derived)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def end_checks() -> dict:
    """Rilegge le proprietà di llama-server e le confronta con quelle di inizio sessione.

    Va chiamata fuori da _state_lock: è una chiamata di rete.
    """
    start = SESSION.get("server_props_start") or {}
    now = fetch_server_props()
    out = {"server_props_end": now, "model_changed": None, "config_changed": None, "warnings": []}
    if "error" in now or "error" in start or not start:
        out["warnings"].append("controllo di fine sessione non possibile: proprietà di llama-server non disponibili")
        return out
    out["model_changed"] = any(start.get(k) != now.get(k) for k in ("model_path", "model_alias"))
    out["config_changed"] = [k for k in ("build_info", "total_slots", "n_ctx") if start.get(k) != now.get(k)]
    if out["model_changed"]:
        out["warnings"].append("il modello caricato in llama-server è cambiato durante la sessione")
    if out["config_changed"]:
        out["warnings"].append("configurazione di llama-server cambiata: " + ", ".join(out["config_changed"]))
    return out


def public_session() -> dict:
    return {k: SESSION.get(k) for k in SESSION_FIELDS} | {
        "started_at": SESSION.get("started_at"),
        "turns_so_far": SESSION.get("turn_counter"),
        "segment": SESSION.get("segment"),
        "model_declared": SESSION.get("model_declared"),
        "model_label_source": SESSION.get("model_label_source"),
        "warnings": SESSION.get("warnings", []),
        "log_dir": str(Path(LOG_DIR).resolve()),
    }


def open_session(info: dict, event: str) -> None:
    """Apre una nuova sessione (sostituisce quella corrente) e la registra in sessions.jsonl."""
    props = fetch_server_props()  # prima del lock: è una chiamata di rete
    derived = None if "error" in props else derive_model_label(props)
    declared = info.get("model")
    warnings = []
    if "error" in props:
        warnings.append("llama-server non raggiungibile: etichetta del modello non verificata")
    elif derived and declared and labels_match(declared, derived) is False:
        warnings.append(f'etichetta dichiarata "{declared}" diversa da quella del server "{derived}": nei log vale quella del server')
    if not derived and not declared:
        warnings.append("etichetta del modello assente")
    for w in warnings:
        print("ATTENZIONE:", w)
    with _state_lock:
        SESSION.clear()
        SESSION.update({
            "session_id": info.get("session_id") or "S" + time.strftime("%Y%m%d-%H%M%S"),
            "participant_id": info.get("participant_id"),
            "task_id": info.get("task_id"),
            "condition": info.get("condition"),
            "model_label": derived or declared,
            "model_declared": declared,
            "model_label_source": "server" if derived else ("declared" if declared else None),
            "server_props_start": props,
            "warnings": warnings,
            "started_at": now_iso(),
            "turn_counter": 0,
            "segment": 1,
            "turn_in_segment": 0,
            "pending_new_segment": False,
            "ge_requests": 0,
            "aux_requests": 0,
        })
        extra = {k: v for k, v in info.items()
                 if k not in ("session_id", "participant_id", "task_id", "condition", "model")}
        append_jsonl("sessions.jsonl", {
            "event": event, **{k: SESSION[k] for k in SESSION_FIELDS if k in SESSION},
            "started_at": SESSION["started_at"], "extra": extra, "server_props": props,
            "model_declared": declared, "model_label_source": SESSION["model_label_source"], "warnings": warnings,
            "exposed_model_id": MODEL_ID, "log_dir": str(Path(LOG_DIR).resolve()),
            "codecarbon_version": getattr(codecarbon, "__version__", None),
            "tracking_mode": TRACKING_MODE, "ge_marker": GE_MARKER,
        })


def register_turn(is_ge_request: bool, auxiliary: bool = False) -> dict:
    """Assegna indice di turn e di segmento. Un nuovo segmento inizia dopo la richiesta di riepilogo GE.

    Una richiesta ausiliaria non è un turn: non incrementa i contatori né apre segmenti.
    """
    if not SESSION:
        open_session({"session_id": "unassigned-" + time.strftime("%Y%m%d-%H%M%S")}, "auto_start")
    with _state_lock:
        if auxiliary:
            SESSION["aux_requests"] = SESSION.get("aux_requests", 0) + 1
            return {
                **{k: SESSION.get(k) for k in SESSION_FIELDS},
                "segment": SESSION["segment"],
                "turn_index": None,
                "turn_in_segment": None,
            }
        if SESSION["pending_new_segment"]:
            SESSION["segment"] += 1
            SESSION["turn_in_segment"] = 0
            SESSION["pending_new_segment"] = False
        SESSION["turn_counter"] += 1
        SESSION["turn_in_segment"] += 1
        if is_ge_request:
            SESSION["ge_requests"] += 1
            SESSION["pending_new_segment"] = True
        return {
            **{k: SESSION.get(k) for k in SESSION_FIELDS},
            "segment": SESSION["segment"],
            "turn_index": SESSION["turn_counter"],
            "turn_in_segment": SESSION["turn_in_segment"],
        }


def last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return normalize_content(m.get("content", ""))
    return ""


def energy_from(tracker) -> dict:
    """Energia dell'ultimo turn in Wh (CodeCarbon la fornisce in kWh)."""
    data = getattr(tracker, "final_emissions_data", None)
    if data is None:
        return {}

    def wh(value):
        return None if value is None else value * 1000

    return {
        "cpu": wh(getattr(data, "cpu_energy", None)),
        "gpu": wh(getattr(data, "gpu_energy", None)),
        "ram": wh(getattr(data, "ram_energy", None)),
        "total": wh(getattr(data, "energy_consumed", None)),
        "duration_s": getattr(data, "duration", None),
        "emissions_kg": getattr(data, "emissions", None),
    }


def absorb_chunk(acc: dict, obj: dict, t0: float) -> None:
    """Accumula risposta, ragionamento, motivo di stop, usage e timings da un chunk SSE di llama-server."""
    choice = (obj.get("choices") or [None])[0] or {}
    delta = choice.get("delta") or {}
    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    if content or reasoning:
        if acc["ttft_ms"] is None:
            acc["ttft_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        acc["chunks"] += 1
    if content:
        acc["content"].append(content)
    if reasoning:
        acc["reasoning"].append(reasoning)
    if choice.get("finish_reason"):
        acc["finish_reason"] = choice["finish_reason"]
    if obj.get("usage"):
        acc["usage"] = obj["usage"]
    if obj.get("timings"):
        acc["timings"] = obj["timings"]


# =========================
# ENDPOINT DELLA SESSIONE
# =========================

@app.post("/experiment/session")
async def experiment_start(request: Request):
    info = await request.json()
    missing = [k for k in ("participant_id", "task_id", "condition") if not info.get(k)]
    if missing:
        raise HTTPException(status_code=422, detail=f"campi mancanti: {', '.join(missing)}")
    condition = str(info["condition"]).strip().lower()
    if condition not in ("baseline", "ge"):
        raise HTTPException(status_code=422, detail='condition deve essere "baseline" o "GE"')
    info["condition"] = "GE" if condition == "ge" else "baseline"

    with _state_lock:
        if SESSION and not str(SESSION.get("session_id", "")).startswith("unassigned"):
            append_jsonl("sessions.jsonl", {"event": "end", "session_id": SESSION["session_id"],
                                            "ended_at": now_iso(), "turns": SESSION["turn_counter"],
                                            "ge_requests": SESSION["ge_requests"],
                                            "aux_requests": SESSION.get("aux_requests", 0),
                                            "model_label": SESSION.get("model_label"),
                                            "auto_closed": True, "model_check": "non eseguito (chiusura automatica)"})
    open_session(info, "start")
    return public_session()


@app.get("/experiment/session")
async def experiment_current():
    return public_session() if SESSION else {}


@app.post("/experiment/session/end")
async def experiment_end(request: Request):
    try:
        info = await request.json()
    except Exception:
        info = {}
    checks = end_checks() if SESSION else {}  # chiamata di rete: fuori dal lock
    with _state_lock:
        if not SESSION:
            raise HTTPException(status_code=409, detail="nessuna sessione aperta")
        for w in checks.get("warnings", []):
            print("ATTENZIONE:", w)
        record = {"event": "end", "model_label": SESSION.get("model_label"), **checks, "session_id": SESSION["session_id"], "ended_at": now_iso(),
                  "turns": SESSION["turn_counter"], "ge_requests": SESSION["ge_requests"],
                  "aux_requests": SESSION.get("aux_requests", 0),
                  "task_completed": info.get("task_completed"), "notes": info.get("notes")}
        append_jsonl("sessions.jsonl", record)
        SESSION.clear()
    return record


# =========================
# HEALTH / MODELS
# =========================

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


# =========================
# CHAT COMPLETIONS
# =========================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()

    # forza streaming
    payload["stream"] = True
    if REQUEST_USAGE:  # llama-server invia così sempre token usati e timings nell'ultimo chunk
        stream_opts = payload.get("stream_options") or {}
        stream_opts["include_usage"] = True
        payload["stream_options"] = stream_opts

    conv_id = request.headers.get("chatui-conversation-id") or ""
    print("CHATUI CONV ID:", conv_id)

    print("HEADERS:", dict(request.headers))
    print("BODY KEYS", list(payload.keys()))
    print("FINAL MESSAGE COUNT:", len(payload.get("messages", [])))

    # --- dati per il log (calcolati prima di avviare la misura) ---
    messages = payload.get("messages", [])
    forced_turn = request.headers.get("x-experiment-turn") == "1"  # es. un replay offline senza id di conversazione
    auxiliary = AUX_NO_CONV_ID and not conv_id and not forced_turn
    if auxiliary:
        print("AUXILIARY REQUEST (no conversation id): not counted as a turn, energy not measured")
    is_ge_request = (not auxiliary) and bool(GE_MARKER) and GE_MARKER in last_user_text(messages).lower()
    turn = register_turn(is_ge_request, auxiliary)
    request_params = {k: v for k, v in payload.items() if k != "messages"}
    messages_sha256 = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    rid = uuid.uuid4().hex[:8]
    tracker = None if auxiliary else EmissionsTracker(
        output_dir=EMISSIONS_DIR,
        tracking_mode=TRACKING_MODE,
        log_level="error",
        project_name=str(turn["session_id"]),  # lega le righe di emissions.csv alla sessione
    )

    def event_generator():
        tracker_started = False
        completed = False
        error = None
        t0 = time.perf_counter()
        acc = {"content": [], "reasoning": [], "chunks": 0, "ttft_ms": None,
               "finish_reason": None, "usage": None, "timings": None}
        enter_request(rid, auxiliary)
        started_at = now_iso()
        try:
            if tracker is not None:
                tracker.start()
                tracker_started = True

            with requests.post(
                f"{LLAMA_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=llama_headers(),
                stream=True,
                timeout=None,
            ) as r:
                r.raise_for_status()

                for line in r.iter_lines():
                    if not line:
                        continue

                    decoded = line.decode("utf-8")

                    if decoded.startswith("data:"):
                        yield decoded + "\n\n"

                        body = decoded[5:].strip()
                        if body == "[DONE]":
                            completed = True
                            break
                        try:
                            absorb_chunk(acc, json.loads(body), t0)
                        except Exception:
                            pass  # il log non deve mai interrompere lo streaming

        except Exception as e:
            error = str(e)
            err = {
                "error": {
                    "message": str(e),
                    "type": "proxy_error",
                }
            }
            yield f"data: {json.dumps(err)}\n\n"

        finally:
            wall_ms = round((time.perf_counter() - t0) * 1000, 1)
            overlapped = leave_request(rid)
            energy = {}
            if tracker_started:
                try:
                    tracker.stop()
                    energy = energy_from(tracker)
                except Exception as e:
                    energy = {"error": str(e)}

            # il log si scrive dopo tracker.stop(): fuori dalla misura dell'energia
            try:
                usage = acc["usage"] or {}
                timings = acc["timings"] or {}
                append_jsonl("turns.jsonl", {
                    "ts": now_iso(),
                    **turn,
                    "conversation_id": conv_id or None,
                    "is_ge_request": is_ge_request,
                    "auxiliary": auxiliary,
                    "n_messages": len(messages),
                    "request_params": request_params,
                    "messages_sha256": messages_sha256,
                    "request_messages": messages if LOG_FULL_REQUEST else messages[-1:],
                    "response": "".join(acc["content"]),
                    "reasoning": "".join(acc["reasoning"]) or None,
                    "finish_reason": acc["finish_reason"],
                    "completed": completed,
                    "error": error,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "prompt_tokens_evaluated": timings.get("prompt_n"),
                    "prompt_tokens_cached": timings.get("cache_n"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "usage": acc["usage"],
                    "timings": acc["timings"],
                    "stream_chunks": acc["chunks"],
                    "ttft_ms": acc["ttft_ms"],
                    "wall_ms": wall_ms,
                    "started_at": started_at,
                    "overlapped": overlapped,
                    "energy_wh": energy,
                    "tracking_mode": TRACKING_MODE,
                    "codecarbon_run_id": str(getattr(tracker, "run_id", "")) or None,
                })
            except Exception as e:
                print("LOG ERROR:", e)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
