# AGENTS.md

Research tooling for a controlled LLM energy-measurement experiment ("Gold Extraction"): root-level dataset/model scripts, the Chat UI proxy, and an independent `benchmark/` subsystem. There is no application build in this repo.

## Architecture

```
Chat UI :5173  ->  energy_api.py :9000 (FastAPI + CodeCarbon)  ->  llama-server :8080
      |
  MongoDB :27017 (Chat UI conversation store)
```

`chat-ui/`, `llama.cpp/`, `models/`, `venv/`, `logs/`, `emissions/`, `dataset_cache/`, and `replay_export/` are external/generated and gitignored. The frozen inputs in `replay_set/` and `selezioni/` are tracked evidence. Setup is documented in `README.md` (English) and `README_IT.md` (Italian); keep both in sync when behavior changes. Full startup order is in `commands.txt`.

## Entry points

- `energy_api.py` — proxy that streams to llama-server, measures each turn with CodeCarbon, writes `logs/turns.jsonl` and `logs/sessions.jsonl`.
- `download_models.py` — selects HF models, freezes the selection in `selezioni/`, downloads GGUF files and maintains `models/manifest.json`. It excludes non-autoregressive (diffusion) models and admits reasoning models only when the chat template exposes a switch to turn reasoning off; both decisions are recorded in the frozen selection.
- `conversation_stats.py` / `export_conversations.py` — dataset sampling and legacy export; `export` imports helpers from `conversation_stats.py`, so they must stay in the same directory. Export defaults to `replay_export/` and refuses to overwrite existing files without `--sovrascrivi`.
- `build_replay_set.py` — frozen replay set from WildChat-1M, sharegpt_gpt4 and CodeChat: `prepare` (normalization, dedup, language and turn filters), `tokenize` (per-model prompt tokens via chat template), `sweep` (grid of `--min-tokeni` with per-source counts, strata fill and retained-token distribution; writes only `dataset_cache/sweep.csv`), `select` (thresholds, context windows, optional `--solo-coding`/`--solo-informatica`, stratification, freeze), `verify`. Token counts are cached in `dataset_cache/`, so threshold or coding-filter changes only re-run `select`.
- `dataset_discovery.py` — one-off dataset survey; produces the tracked `dataset_discovery_<date>.txt`, not part of the main pipeline.
- `benchmark/` — standalone S0–S7 strategy benchmark; keep its path-based configuration and shared M1/runner inputs intact.

## Commands and verification

- No test suite, CI, linter, formatter, or typecheck config exists for the Python tooling. `package.json`'s `test`/`lint`/`check` scripts belong to Chat UI (and need `chat-ui/` installed), not this repo.
- Minimal check after Python edits: `python3 -m py_compile energy_api.py download_models.py conversation_stats.py export_conversations.py dataset_discovery.py build_replay_set.py`.
- Replay-set pipeline smoke test: `python build_replay_set.py prepare --limite 300 --min-turni 1 --cache /tmp/cache`, then `tokenize --modelli qwen2.5-coder-14b-instruct --limite 300`, `select --min-turni 1 --min-tokeni 100 --n-per-strato 2 --dry-run`, and `verify` on a written set. The full run needs network, `HF_TOKEN` for the gated Llama tokenizers, and about 1M conversations of streaming.
- Manual end-to-end check: `curl http://127.0.0.1:8080/props`, `curl http://127.0.0.1:9000/v1/models` (must return `assistente-locale`), then a 2-message Chat UI session whose closing `sessions.jsonl` record shows `"turns": 2`, `"aux_requests": 0`.
- `download_models.py --solo-selezione` and `--dry-run` exercise selection without downloading. Reuse `--da-lista selezioni/candidati_<date>.json` to download an already-frozen list; re-running selection can change the set.

## Non-obvious gotchas

- `package.json` is Chat UI's pinned manifest copied into `chat-ui/` during setup, and `env.local` is copied there as `.env.local`. `npm` commands belong to `chat-ui/`, not this repo.
- `energy_api.py` requires Python 3.10+ because it uses `str | None` without `from __future__ import annotations`; the READMEs state Python 3.10+.
- `requirements.txt` is the dependency input; install the pinned Linux/Python 3.11 environment from `requirements.lock.txt`. `requirements-dev.txt` pins `pip-tools` for refreshing that lock. `matplotlib` is only needed for `--plot`.
- `conversation_stats.py` requires `--tokenizer`, `--min-turns`, and `--min-tokens`, including with `--inspect`; `export_conversations.py` additionally requires `--n`.
- Run `uvicorn energy_api:app ...` from the repo root: `LOG_DIR`/`EMISSIONS_DIR` are relative and resolved at import.
- `commands.txt` runs from the repo root; model files are addressed as `./models/<name>/<file>.gguf`.
- `energy_api.py` defaults to `TRACKING_MODE=machine`: process mode measures the proxy rather than the separate llama-server inference process.
- The llama-server `--alias <folderName>` becomes `model_label` in the logs and must match the `models/<name>` folder / manifest entry.
- Requests without a `chatui-conversation-id` header are treated as auxiliary: forwarded and logged but not counted as turns or energy-measured. Keep `LLM_SUMMARIZATION=false` in Chat UI so title generation does not create extra requests.
- `energy_api.py` forces streaming and `stream_options.include_usage`; `build_clean_messages()` is dead code — do not use it to trim history, since the proxy must forward Chat UI messages unchanged.
- `env.local` is a template: its `MODELS` entry (`default_id`) and `DEFAULT_MODEL=default_model` do not match the proxy's exposed `MODEL_ID` (`assistente-locale`). The `OPENAI_BASE_URL`/`ENABLE_CONVERSATIONS`/`DISABLE_CONVERSATION_MEMORY`/`LLM_SUMMARIZATION` variables are the ones the experiment relies on.
- `download_models.py` excludes diffusion architectures (`model_type`/`architectures`/`auto_map`/tags) and reasoning models whose chat template has no off switch. Admitted reasoning models must be started with `--reasoning off` (templates using `enable_thinking`) or `--reasoning-budget 0` (templates using `thinking_budget`); the chosen switch is recorded per candidate as `reasoning.interruttori`. `--da-lista` bypasses these selection criteria, so a frozen list can still contain models selected before this rule.
- `build_replay_set.py` counts tokens with the tokenizer of the original HF repo, not with the GGUF: Kimi-Linear has no `tokenizer.json` (custom TikToken tokenizer, needs `trust_remote_code=True` and `tiktoken`), and the two Llama tokenizers are gated. It applies the chat template with the reasoning-off switch from the frozen selection (`enable_thinking=false`, `thinking_budget=0`). Validate a sample against llama-server (`/apply-template` + `/tokenize`) before freezing; if a template does not accept the system role, `system_piegato` records the fold into the first user and the replay must do the same.
- CodeChat is derived from WildChat and its `conversation_id` equals WildChat's `conversation_hash`: deduplicate on the source id (WildChat wins), not only on content. `prepare` registers the ids of filtered WildChat rows too, so a toxic or short conversation cannot re-enter through CodeChat.
- `select` must use `--contesto-runtime` equal to llama-server's `-c` (32768 in `commands.txt`), not the declared model context, and the value must be identical on the four hardware setups. The declared contexts in `MODELLI` were checked against each repo's `config.json` on 2026-10-20 (`max_position_embeddings`; Kimi-Linear uses `model_max_length: 1048576`): qwen 32768, glm 202752, gemma 262144, llama 131072, deepseek 163840, kimi 1048576, seed 524288. With `-c 32768` the effective window is therefore 32768 for every model and the usable prompt is 32768 − 2048 = 30720; 1.4–1.6% of the eligible conversations per model exceed it and are dropped by the window check.
- `prepare` is English-only by default: `--lingua` defaults to `en` and fastText `lid.176` runs on the user messages of all three sources (model in `<cache>/models/lid.176.bin`, ~131 MB, downloaded on first use); `--tutte-le-lingue` disables the filter and is for trials only. The WildChat `language` column is only used to measure agreement with the detector. The token threshold applies to every model by default and strata use the maximum count across models.
- The 10-turns / 4000-token thresholds are provisional: 4000 is about the p50 of prompt tokens, so it discards half of the eligible English corpus. Use `sweep --n-per-strato N` to choose the threshold from the data (per-source survival, strata reaching N, retained p50/p90) and `--dettaglio <soglia>` to see the thin strata; the CSV goes to `dataset_cache/sweep.csv`. Re-running `select` is cheap; `prepare` only needs re-running if `--min-turni` changes.
- `--solo-coding` (in `sweep` and `select`) keeps conversations with a fenced code block (known language tag or code-like body) or a code line in any message, assistant included. At 4000 it leaves 523 of the 3,081 eligible conversations (333 WildChat + 190 sharegpt_gpt4) on all 18 strata; at 5000, 445; at 6000, 372; at 8000, 254. CodeChat is a validation label, not a source: of the 387 WildChat conversations that also appear in CodeChat, the filter recovers 82–83% and flags 0.5% of the others. The comparison uses `dataset_cache/codechat_ids.txt`, a derived id list; CodeChat itself still contributes 0 because of the WildChat-first dedup.
- `--solo-informatica` extends `--solo-coding` with non-code conversations that discuss a CS/IT topic. It reads only the first user message, requires two distinct technical signals (algorithms/data structures, OS, networks, databases, security, software engineering, AI/ML, web/mobile, hardware) or one signal plus an explicit explain/implement/debug request, and applies a negative list for creative, business, legal, medical, math/physics, translation and career tasks. The lexicon, intent regex and exclusions are in `TEMI_CS`/`ACRONIMI_CS`/`RE_ESCLUSIONI_CS`; the filter is recorded as `parametri.solo_informatica`. At 4000 it leaves 563 conversations (vs 523 code-only); at 3000, 1,042 (vs 925); at 5000, 479 (vs 445); at 6000/8000, 404/273. At 4000, `--n-per-strato 5` selects 80 conversations / 1,669 turns (3 without code) and `--n-per-strato 6` selects 94 / 1,928. `verify` checks the same criterion from the manifest.
- In the full run CodeChat contributes 0 conversations: its 82,834 rows all carry a WildChat `conversation_hash`, already registered (and mostly filtered) during the WildChat pass, so the replay set effectively comes from WildChat + sharegpt_gpt4. Do not treat the empty `codechat.jsonl` as a bug.
- `dataset_cache/` is derived data and is gitignored; `replay_set/` is frozen evidence and `select` refuses to overwrite it without `--sovrascrivi` (use only for trials).
- `selezioni/` contains the frozen selection (`candidati_2026-10-19.json` and `.txt`); `download_models.py` creates new dated documents; `candidati_*.json/.txt` are meant to be committed, `esito_download_*.txt` is gitignored.
- The two READMEs are the setup source of truth; `commands.txt` is the condensed runbook for an experimental session. Update docs when changing ports, env vars, log fields, or CLI flags.

## Conventions

- Code, CLI flags, help text, log field names, and comments are in Italian (`--solo-selezione`, `--da-lista`, `--only`); do not rename them for consistency with English docs. `.run/*.xml` IntelliJ configs reference these flags.
- Output directories are fixed by default (`selezioni/`, `models/`, `logs/`, `emissions/`, `stats_out/`, `dataset_cache/`, `replay_set/`); prefer the existing overrides (`--out`, `--out-dir`, `--cache`, or `LOG_DIR`/`EMISSIONS_DIR` env vars) over new paths.
- Logs and frozen selections are experimental evidence: append-only, never overwrite `candidati_*`/`esito_download_*` or edit previously measured JSONL.
- Do not push to GitHub. The experiment artefacts (`replay_set/`, `dataset_cache/`, `models/`, `logs/`) stay on the machine; any commit remains local.
