#!/usr/bin/env python3
"""llama-server FINTO, solo per la prova simulata.

Accetta le stesse opzioni di avvio usate da misura.py, risponde a /health, /props, /apply-template, /tokenize e
/v1/chat/completions in streaming (con usage e timings), e scrive ogni richiesta di chat in FINTO_LOG.
Un token = una parola. Un modello il cui file contiene "fail" termina subito (come un modello che non entra).
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("-m")
ap.add_argument("--alias")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int)
ap.add_argument("-c", type=int, default=4096)
args, _ = ap.parse_known_args()

if "fail" in (args.m or ""):
    print("llama_model_load: error: out of memory (finto)")
    sys.exit(1)

AVVIO = time.time()
ULTIMO_PROMPT: list[str] = []


def template(messaggi):
    return " ".join(f"<{m['role']}> {m['content']}" for m in messaggi) + " <assistant>"


def risposta_per(messaggi):
    ultimo = messaggi[-1]["content"]
    if "Generate a prompt" in ultimo:
        return "GOLD PROMPT: the user works on a project and wants to continue solving the problem step by step."
    if "Update the summary" in ultimo:
        return "UPDATED SUMMARY: earlier summary plus the new messages about the same project."
    if "Generate a summary" in ultimo:
        return "SUMMARY: the user and the assistant discussed the project and its code in detail."
    parole = ultimo.split()[:6]
    return "Answer about " + " ".join(parole) + " with a few more words to make a reply of some length."


class Gestore(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, codice, dati):
        corpo = json.dumps(dati).encode()
        self.send_response(codice)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200 if time.time() - AVVIO > 0.3 else 503, {"status": "ok"})
        if self.path == "/props":
            return self._json(200, {"model_path": args.m, "build_info": "finto", "total_slots": 1,
                                    "default_generation_settings": {"n_ctx": args.c}})
        self._json(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        corpo = json.loads(self.rfile.read(n))
        if self.path == "/apply-template":
            return self._json(200, {"prompt": template(corpo["messages"])})
        if self.path == "/tokenize":
            return self._json(200, {"tokens": list(range(len(corpo["content"].split())))})
        if self.path != "/v1/chat/completions":
            return self._json(404, {})
        with open(os.environ["FINTO_LOG"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"alias": args.alias, **{k: v for k, v in corpo.items()}}) + "\n")
        prompt = template(corpo["messages"]).split()
        if len(prompt) > args.c:
            return self._json(400, {"error": {"code": 400, "type": "exceed_context_size_error",
                                              "message": "the request exceeds the available context size"}})
        comune = 0
        for a, b in zip(prompt, ULTIMO_PROMPT):
            if a != b:
                break
            comune += 1
        if not corpo.get("cache_prompt", True):
            comune = 0
        ULTIMO_PROMPT[:] = prompt
        parole = risposta_per(corpo["messages"]).split()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.002 * (len(prompt) - comune) / 100 + 0.002)
        for i, p in enumerate(parole):
            chunk = {"choices": [{"delta": {"content": (" " if i else "") + p}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            time.sleep(0.0005)
        fine = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(parole)},
                "timings": {"prompt_n": len(prompt) - comune, "cache_n": comune, "prompt_ms": 1.0, "predicted_ms": 2.0}}
        self.wfile.write(f"data: {json.dumps(fine)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True

    def log_message(self, *a):
        pass


HTTPServer((args.host, args.port), Gestore).serve_forever()
