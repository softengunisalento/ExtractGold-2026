# AGENTS.md

Research tooling for a controlled LLM energy-measurement experiment ("Gold Extraction"). Five standalone Python scripts plus Chat UI configuration; there is no application build in this repo.

## Architecture

```
Chat UI :5173  ->  energy_api.py :9000 (FastAPI + CodeCarbon)  ->  llama-server :8080
      |
  MongoDB :27017 (Chat UI conversation store)
```

`chat-ui/`, `llama.cpp/`, `models/`, `venv/`, `logs/`, `emissions/` are external/generated and gitignored. Setup is documented in `README.md` (English) and `README_IT.md` (Italian); keep both in sync when behavior changes. Full startup order is in `commands.txt`.

## Entry points

- `energy_api.py` — proxy that streams to llama-server, measures each turn with CodeCarbon, writes `logs/turns.jsonl` and `logs/sessions.jsonl`.
- `download_models.py` — selects HF models, freezes the selection in `selezioni/`, downloads GGUF files and maintains `models/manifest.json`. It excludes non-autoregressive (diffusion) models and admits reasoning models only when the chat template exposes a switch to turn reasoning off; both decisions are recorded in the frozen selection.
- `conversation_stats.py` / `export_conversations.py` — dataset sampling and replay-set export; `export` imports helpers from `conversation_stats.py`, so they must stay in the same directory.
- `dataset_discovery.py` — one-off dataset survey; produces the tracked `dataset_discovery_<date>.txt`, not part of the main pipeline.

## Commands and verification

- No test suite, CI, linter, formatter, or typecheck config exists for the Python tooling. `package.json`'s `test`/`lint`/`check` scripts belong to Chat UI (and need `chat-ui/` installed), not this repo.
- Minimal check after Python edits: `python3 -m py_compile energy_api.py download_models.py conversation_stats.py export_conversations.py dataset_discovery.py`.
- Manual end-to-end check: `curl http://127.0.0.1:8080/props`, `curl http://127.0.0.1:9000/v1/models` (must return `assistente-locale`), then a 2-message Chat UI session whose closing `sessions.jsonl` record shows `"turns": 2`, `"aux_requests": 0`.
- `download_models.py --solo-selezione` and `--dry-run` exercise selection without downloading. Reuse `--da-lista selezioni/candidati_<date>.json` to download an already-frozen list; re-running selection can change the set.

## Non-obvious gotchas

- `package.json` is Chat UI's pinned manifest copied into `chat-ui/` during setup, and `env.local` is copied there as `.env.local`. `npm` commands belong to `chat-ui/`, not this repo.
- `energy_api.py` requires Python 3.10+ because it uses `str | None` without `from __future__ import annotations`, despite the README claiming Python 3.9+.
- `requirements.txt` omits `datasets`, which both dataset scripts import; `matplotlib` is only needed for `--plot`.
- README examples for `conversation_stats.py` and `export_conversations.py` are stale: `--tokenizer`, `--min-turns`, `--min-tokens` (plus `--n` for export) are required argparse arguments, even with `--inspect`.
- Run `uvicorn energy_api:app ...` from the repo root: `LOG_DIR`/`EMISSIONS_DIR` are relative and resolved at import.
- `commands.txt` says to run from the repo root but passes `-m ../models/...`; models actually live at `./models/<name>/<file>.gguf`.
- The llama-server `--alias <folderName>` becomes `model_label` in the logs and must match the `models/<name>` folder / manifest entry.
- Requests without a `chatui-conversation-id` header are treated as auxiliary: forwarded and logged but not counted as turns or energy-measured. Keep `LLM_SUMMARIZATION=false` in Chat UI so title generation does not create extra requests.
- `energy_api.py` forces streaming and `stream_options.include_usage`; `build_clean_messages()` is dead code — do not use it to trim history, since the proxy must forward Chat UI messages unchanged.
- `env.local` is a template: its `MODELS` entry (`default_id`) and `DEFAULT_MODEL=default_model` do not match the proxy's exposed `MODEL_ID` (`assistente-locale`). The `OPENAI_BASE_URL`/`ENABLE_CONVERSATIONS`/`DISABLE_CONVERSATION_MEMORY`/`LLM_SUMMARIZATION` variables are the ones the experiment relies on.
- `download_models.py` excludes diffusion architectures (`model_type`/`architectures`/`auto_map`/tags) and reasoning models whose chat template has no off switch. Admitted reasoning models must be started with `--reasoning off` (templates using `enable_thinking`) or `--reasoning-budget 0` (templates using `thinking_budget`); the chosen switch is recorded per candidate as `reasoning.interruttori`. `--da-lista` bypasses these selection criteria, so a frozen list can still contain models selected before this rule.
- `selezioni/` is not present in this checkout. `download_models.py` creates it; `candidati_*.json/.txt` are meant to be committed, `esito_download_*.txt` is gitignored.
- The two READMEs are the setup source of truth; `commands.txt` is the condensed runbook for an experimental session. Update docs when changing ports, env vars, log fields, or CLI flags.

## Conventions

- Code, CLI flags, help text, log field names, and comments are in Italian (`--solo-selezione`, `--da-lista`, `--only`); do not rename them for consistency with English docs. `.run/*.xml` IntelliJ configs reference these flags.
- Output directories are fixed by default (`selezioni/`, `models/`, `logs/`, `emissions/`, `stats_out/`, `replay_set/`); prefer the existing overrides (`--out`, `--out-dir`, or `LOG_DIR`/`EMISSIONS_DIR` env vars) over new paths.
- Logs and frozen selections are experimental evidence: append-only, never overwrite `candidati_*`/`esito_download_*` or edit previously measured JSONL.
