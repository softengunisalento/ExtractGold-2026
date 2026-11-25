# Gold Extraction – experiment tooling

*Italian version: [README_IT.md](README_IT.md)*

## Architecture

```
Hugging Face Chat UI  ──►  energy_api.py (proxy)  ──►  llama-server  ──►  GGUF model
      :5173                 FastAPI + CodeCarbon         :8080
        │                        :9000
  MongoDB :27017
  (conversation context of Chat UI)
```

- Hugging Face Chat UI runs on port 5173 and is the interface used by the participants.
- `energy_api.py` is the proxy server on port 9000. It forwards each request to the model, measures the energy of every turn with CodeCarbon and logs the conversations. It forwards the messages exactly as Chat UI sends them (Chat UI already sends the whole conversation history); `logs/turns.jsonl` records the text actually sent to the model. It does not access MongoDB.
- `llama-server` (llama.cpp) runs the model on port 8080.
- MongoDB runs on port 27017 and stores the conversations of Chat UI.
- The whole architecture runs inside a Python virtual environment (`venv`).

Configuration files of Chat UI:

- `.env.local` contains the local configuration of Chat UI. It must be placed in the `chat-ui` directory as `.env.local`. Three settings are essential: the base URL must point to the proxy (port 9000); conversation memory must be enabled (ENABLE_CONVERSATIONS=true, DISABLE_CONVERSATION_MEMORY=false), because by default, with local models, Chat UI runs in stateless mode and does not keep the context; and the generation of conversation titles must be disabled (LLM_SUMMARIZATION=false), otherwise Chat UI sends an extra request to the model at the first message of every conversation.
- `commands.txt` lists the commands that start the whole architecture.

## Contents

| File | Purpose |
|---|---|
| `download_models.py` | Selects models from Hugging Face metadata, freezes the selection in dated documents, downloads GGUF files and records a manifest. |
| `conversation_stats.py` / `export_conversations.py` | Analyze conversation datasets and export a legacy replay sample. The exporter writes to `replay_export/` by default and protects existing outputs. |
| `build_replay_set.py` | Builds and verifies the frozen replay set from the selected datasets. |
| `energy_api.py` | Proxy between Chat UI and llama-server: measures energy (CodeCarbon) and logs conversations. |
| [`benchmark/README.md`](benchmark/README.md) | Documentation for the independent S0–S7 context-strategy benchmark, measurement service and simulated test. |

## Setup from scratch

The repository contains the code, configuration and frozen inputs for this work. Chat UI, llama.cpp, the models and the virtual environment are not included: they are installed once, following these steps. The commands target the Linux workstation; Windows requires different commands. Run everything from the repository root unless stated otherwise.

Layout after setup (runtime caches and results are ignored; frozen experimental inputs are versioned):

```
<repository>/
├── chat-ui/             step 4
├── llama.cpp/           step 5, with the build in llama.cpp/build/
├── models/              step 6, one folder per model: models/<name>/<file>.gguf
├── selezioni/           step 6, frozen model-selection documents (kept in Git)
├── venv/                step 2
├── logs/, emissions/    created at run time
├── energy_api.py, download_models.py, conversation_stats.py, export_conversations.py
├── build_replay_set.py
├── dataset_cache/       created by build_replay_set.py prepare (ignored by Git)
├── replay_set/          frozen replay evidence (versioned in Git)
├── replay_export/       legacy exports (ignored by Git)
├── benchmark/           independent strategy benchmark
└── commands.txt, requirements*.txt, README.md, README_IT.md
```

The Python entry points stay at the repository root intentionally: the legacy exporter imports
`conversation_stats.py`, and runbooks and configurations resolve paths relative to these locations.
`benchmark/` is already a separate subsystem; moving these files would require migrating those contracts.

**1. Prerequisites.** Git; Python 3.10 or later; Node.js (tested with v24.21.0); CMake and a C++ compiler (`build-essential`); the NVIDIA driver and the CUDA toolkit (`nvidia-smi` must list the GPUs and `nvcc --version` must work); MongoDB Community Edition, installed following the official instructions for the operating system; a Hugging Face account with a token of type Read.

**2. Python environment.**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.lock.txt
```

`requirements.txt` is the human-maintained dependency input; `requirements.lock.txt` pins the resolved
Linux/Python 3.11 setup environment. To update dependencies, install the pinned lock-generation tool
with `pip install -r requirements-dev.txt`, then regenerate and review the lock with
`pip-compile requirements.txt --output-file requirements.lock.txt`. Experimental runs should install
from the lock.

**3. MongoDB.** Chat UI keeps the conversations.

```bash
sudo systemctl enable --now mongod
systemctl status mongod
```

**4. Chat UI.** Only the version used in this work is supported: the configuration was tested only with this commit.

```bash
git clone https://github.com/huggingface/chat-ui
cd chat-ui
git checkout -f 9159eaa4
cp ../env.local .env.local  # template provided with this repository
cp ../package.json .
npm install
cd ..
```

The repository template is named `env.local`; copy it to Chat UI as `.env.local`. The variables that matter are:

```
OPENAI_BASE_URL=http://127.0.0.1:9000/v1
OPENAI_API_KEY=none
ENABLE_CONVERSATIONS=true
DISABLE_CONVERSATION_MEMORY=false
LLM_SUMMARIZATION=false
MONGODB_URL=mongodb://127.0.0.1:27017/chatui
```

**5. llama.cpp.** Build it with CUDA support:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout <commit-or-tag>          # pin the llama.cpp revision used in the experiment
nvcc --version
# If CUDA is installed in a non-standard location, add it to PATH:
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
cd ..
```

Check `docs/build.md` at the pinned revision for its build options. Without CUDA support the build runs on the CPU and `-ngl 99` has no effect: when the server starts, its log must list the GPUs.

**6. Models.** Check the free disk space first.

```bash
read -rs HF_TOKEN && export HF_TOKEN
#After running this command, paste the HF token and press Enter; to verify, run `echo $HF_TOKEN`
python download_models.py --dry-run                         # selects, writes the frozen documents, shows the download plan
python download_models.py                                   # selects, writes the frozen documents and downloads
```

The selection is done once: to download the same list again use `python download_models.py --da-lista selezioni/candidati_<date>.json`.

**7. Check the whole chain.** Start the services as in `commands.txt`, with a test log folder and, for the model, `--alias <folderName>`. Then:

```bash
curl http://127.0.0.1:8080/props                # the model and the context size
curl http://127.0.0.1:9000/v1/models            # must show the id assistente-locale
```

Open a test session (see Experimental session), write two messages in a new conversation of Chat UI and close the session. In the test log folder `turns.jsonl` must contain two records with `auxiliary: false`, and the closing record of `sessions.jsonl` must show `"turns": 2` and `"aux_requests": 0`. Delete the test folder before the real sessions.

## `download_models.py`

A script that selects the models, freezes the selection and downloads them. Run without arguments it does, in sequence:

1. **Selection** from the metadata of the Hugging Face Hub (parameters, chat template, languages, context, relations with other models, downloads). The organisations that publish the "official" models are listed in `ORG_PREDEFINITE` (or in a file passed with `--orgs-file`). A model is a candidate only if a GGUF with a 4-bit label exists from a trusted quantiser. Non-autoregressive (diffusion) models are excluded, because their energy per turn is not comparable with autoregressive models. Reasoning models are admitted only if their chat template exposes a switch that turns reasoning off (for example `enable_thinking=false` or `thinking_budget=0`); a model that only offers `reasoning_effort` is excluded. The switch found is recorded in the frozen document (`reasoning.interruttori`) and must be applied to llama-server at run time (see `commands.txt`).
2. **Freezing** in `selezioni/`, with the date in the name and without ever overwriting: `candidati_<date>.json` (criteria, versions, counters, candidates with the download plan, excluded models with the reason) and `candidati_<date>.txt` (the same, readable).
3. **Download** of the `.gguf` files into `models/<name>/`, ready for `llama-server -m <file>`. An interrupted download resumes from where it stopped. Files are downloaded from the exact commit of the repository and, once finished, their SHA-256 is checked against the one declared by the Hub.

```bash
python download_models.py                  # selection + frozen documents + download
python download_models.py --solo-selezione # selection and frozen documents only
python download_models.py --dry-run        # selection and documents, then the download plan, without downloading
python download_models.py --da-lista selezioni/candidati_<date>.json            # no selection: download a list already frozen
python download_models.py --da-lista selezioni/candidati_<date>.json --only <name>
python download_models.py --list           # show the download plan and exit
python download_models.py --max-modelli 10 --max-coding 5 --quant Q4_K_M,UD-Q4_K_XL
python download_models.py --no-verify      # skip the local SHA-256 computation
python download_models.py --update         # use the latest commit even if the manifest or the list pin another one (the weights change)
python download_models.py --force          # proceed even if the free space is not enough for all the selected models
```

The criteria and their thresholds are options of the script (`--help`) and are written in the frozen document. The token in `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) is used in both phases; without it the requests are anonymous, with tighter limits.

Output:

- `selezioni/candidati_<date>.json` and `.txt`: the frozen selection, to be kept together with the results and cited in the methodology;
- `selezioni/esito_download_<date>.txt`: the outcome of the download, written after downloading;
- `models/manifest.json`: repository, quantisation, commit, download date, the model it derives from and, for each file, name, size, SHA-256 and the type of tensors read from the GGUF header. It should be kept together with the results and cited in the methodology to state exactly which weights were used. The commit is pinned: the one in the manifest applies, otherwise the one of the frozen list, otherwise the latest (`--update` moves to the latest).

## `conversation_stats.py`

Reads a sample of conversations from Hugging Face in streaming mode and computes the number of turns and tokens of each conversation.

```bash
python conversation_stats.py --inspect --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --min-turns 10 --min-tokens 4000
python conversation_stats.py --sample 5000 --plot --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --min-turns 10 --min-tokens 4000
python conversation_stats.py --dataset allenai/WildChat-1M --lang English --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --min-turns 10 --min-tokens 4000
```
Output in `stats_out/` (configurable with `--out`):

- `conversazioni.csv`: one row per conversation;
- `summary.json`: statistics and metadata for reproducibility (commit, seed, date);
- `distribuzioni.png`: plot of the distributions (only with `--plot`).

## `export_conversations.py`

Selects long conversations (by number of turns, tokens and model) and extracts a reproducible random sample, saved as JSONL (one conversation per line) with the messages as `{"role": ..., "content": ...}`. It requires `conversation_stats.py` in the same folder.

```bash
python export_conversations.py --min-turns 10 --min-tokens 4000 --n 50 --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --max-tokens 24000
python export_conversations.py --dataset allenai/WildChat-1M --model-regex "^gpt-4" --min-turns 10 --min-tokens 4000 --n 100 --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --out replay_export/wildchat
```
Output in `replay_export/` by default (configurable with `--out`). Existing files are protected; pass
`--sovrascrivi` only when intentionally replacing them. Do not target the frozen `replay_set/` for legacy exports.

- `conversazioni.jsonl`: the selected conversations;
- `selezione.json`: parameters, dataset commit and counts, to be reported in the methodology.

## `build_replay_set.py`

Builds the frozen replay set from the three chosen datasets (`allenai/WildChat-1M`, `shibing624/sharegpt_gpt4`, `Suzhen/CodeChat`) in five commands: `prepare` and `tokenize` build the cache, `sweep` chooses the token threshold, `select` freezes the set and `verify` rechecks it:

- `prepare`: normalizes the three formats, removes empty, structurally invalid and duplicate conversations, keeps only English conversations (fastText lid.176 on the user messages), applies the minimum-turn filter and writes a normalized cache;
- `tokenize`: counts the prompt tokens with the tokenizer of every benchmark model, applying the chat template and the same reasoning-off switch used in the experiments;
- `sweep`: varies `--min-tokeni` over a grid and reports the surviving conversations per source, the strata that reach `--n-per-strato`, the size of the resulting sample and the prompt-token distribution (p10/p50/p90/p99) used to justify the threshold; the table is written to `dataset_cache/sweep.csv` (`--out` overrides the path);
- `select`: applies the minimum-token and context-window thresholds, builds the common sample, stratifies by turn and token bands and freezes the set; `--n-per-strato 0` keeps all eligible conversations, with no per-stratum cap;
- `verify`: rechecks hashes, thresholds, windows and the manifest.

Tokenization is cached separately from the thresholds: changing `--min-turni`, `--min-tokeni`, `--solo-coding` or `--solo-informatica` in `select` does not re-tokenize. `sweep` is the tool for choosing `--min-tokeni`: it prints the prompt-token percentiles of the eligible corpus and, for each threshold, the surviving conversations per source, the strata that reach `--n-per-strato`, the size of the resulting sample and the median/p90 of the retained prompts; `--dettaglio <threshold>` lists the strata below target. The threshold must keep the planned design feasible (every source and the main turn/token bands with at least `--n-per-strato`) while pulling the retained conversations away from short prompts; `4000` was chosen from the sweep, not as a measured optimum. The tokenizers and pinned revisions of the ten models are in `MODELLI`, copied from `selezioni/candidati_2026-10-19.json`: update that list and the `FONTI` revisions if the frozen selection or the datasets change. The two Llama repositories are gated, so `HF_TOKEN` with the accepted licence is required. The frozen set uses `10` turns and `4000` tokens; re-run `select` to test other thresholds.

`--solo-coding` keeps only coding conversations: a conversation qualifies if any message contains a fenced code block whose info string is a programming language (or whose body contains code lines), or a line of code in one of the patterns in `SEGNALI_FORTI["riga"]`. Assistant messages count, because many programming questions do not paste code but the answer does. CodeChat is used as an external check rather than as a source (its ids are already seen in WildChat): of the 387 WildChat conversations at 4000 that also appear in CodeChat, the filter recovers 82-83% and flags 0.5% of the others. Manual sampling shows mostly coding conversations; a few non-coding ones slip in when the assistant uses a fenced XML/JSON block as data. At 4000 the coding filter leaves 523 conversations (333 WildChat + 190 sharegpt_gpt4) out of 3,081, across all 18 strata; at 5000 it leaves 445, at 6000 372 and at 8000 254. The filter is recorded in `selezione.json` under `parametri.solo_coding` and in the per-source funnel as `coding_ok`.

`--solo-informatica` broadens the criterion to conversations that discuss a computer-science/IT topic even without pasting code. It looks only at the first user message, requires either two distinct technical signals (algorithms and data structures, operating systems, networks, databases, security, software engineering, AI/ML, web/mobile, hardware) or one signal plus an explicit request to explain, implement, debug or compare, and excludes creative, business, legal, medical, mathematical/physics, translation and career tasks. It includes the code filter. At 4000 it leaves 563 conversations versus 523 with `--solo-coding`; at 3000, 1,042 versus 925; at 5000, 479 versus 445; at 6000 and 8000, 404 and 273. At 4000 the stratified sample with `--n-per-strato 5` has 80 conversations (1,669 turns), 3 of them without code (software engineering and AI); with `--n-per-strato 6` it has 94 conversations (1,928 turns). The frozen set uses `--n-per-strato 0`: all 563 eligible conversations (9,804 turns; 367 WildChat + 196 sharegpt_gpt4, 523 with code and 40 without), in 18 strata. The SHA-256 of `replay_set/conversazioni.jsonl` is `44b790a91ac4fc903da8282fafd444d13835ec3a8a707f16df3bacde386a276d`. The filter is recorded in `selezione.json` under `parametri.solo_informatica` and `parametri.criterio_coding`; `--solo-coding` remains the stricter code-only option.

```bash
# 1. normalise once (re-run only if --min-turni changes)
python build_replay_set.py prepare --min-turni 10 --lingua en
# 2. count tokens once per model (resumable, one JSONL per model)
python build_replay_set.py tokenize
# 3. choose --min-tokeni: coverage of sources and strata at several thresholds
#    (6 is an example per-stratum target; sweep writes dataset_cache/sweep.csv)
python build_replay_set.py sweep --min-turni 10 --solo-informatica --n-per-strato 6 --dettaglio 4000
# 4. freeze the set: 0 keeps all eligible conversations (563 in the current set)
python build_replay_set.py select --min-turni 10 --min-tokeni 4000 --solo-informatica --n-per-strato 0
# 5. verify the frozen set
python build_replay_set.py verify
```

Output:

- `dataset_cache/` (ignored by Git): normalized corpus, token counts, the `prepare.json` manifest and one manifest per model written by `tokenize`;
- `replay_set/conversazioni.jsonl`: the frozen conversations with `messages`, per-model `token_prompt`, turn/token band and content hash;
- `replay_set/selezione.json`: dataset and tokenizer revisions, script SHA-256, thresholds, per-source and per-model funnel, strata, output SHA-256;
- `replay_set/strati.csv`: counts per stratum.

`--contesto-runtime` defaults to `32768`, the `-c` used by llama-server, so the window check uses the real experimental window, not the declared one. CodeChat is derived from WildChat: `prepare` deduplicates on the source id, giving precedence to the full WildChat conversation. Use `--limite` for a small trial run and `--dry-run` in `select` to inspect the funnel without writing.

The token count uses the tokenizer and chat template of the original Hugging Face repository, while the GGUF files come from third-party quantizers. Before freezing, validate a sample against the running llama-server (`POST /apply-template` and `POST /tokenize`) and record the difference in the methodology. If a template does not accept the system role, `tokenize` folds the system prompt into the first user message: the replay must apply the same rule.

The context-window guarantee refers to the recorded conversation, i.e. the assistant messages stored in the dataset. If the replay lets the model generate new answers, the context at later turns depends on the generated lengths: keep the same `-c` and a safety reserve, and check the runtime logs. WildChat rows with `toxic=true` are excluded by default (`--includi-tossici` keeps them); `redacted` rows are kept unless `--escludi-redatti` is given. For a faster first pass, `prepare --prefiltro-byte N` drops conversations whose raw text plus a safety margin cannot reach `N` tokens. The threshold is in UTF-8 bytes, not characters: for the byte-level tokenizers used here `tokens <= bytes`, while code and CJK can approach one token per character, so a character threshold would discard valid conversations. `--lingua` (default `en`) makes the replay set English-only: `prepare` detects the language of the user messages with fastText `lid.176` on all three sources (confidence `--lingua-soglia 0.80`, first `--lingua-caratteri 2000` characters) and drops the rest. WildChat's top-level `language` column is only used to measure agreement with the detector; `sharegpt_gpt4` has no language column and CodeChat's per-message labels are unreliable (the repository tag says `en`, but the content is multilingual), so the detector is the only criterion common to the three sources. A conversation whose user text is shorter than 20 characters is not assessable and is dropped. `--tutte-le-lingue` disables the filter and must only be used for trials. By default the minimum-token threshold must hold for every model (`--soglia-token-modo tutti`) and the strata use the maximum count across models (`--misura-strato max`). `replay_set/` contains tracked frozen evidence; select refuses to overwrite it unless `--sovrascrivi` is passed. Keep it with the experimental results and cite `selezione.json` in the methodology.

## `energy_api.py`

Proxy between Hugging Face Chat UI and llama-server. Every chat request is forwarded to llama-server; when the response ends, one line is written to `logs/turns.jsonl` with: messages sent to the model, response, tokens (total, evaluated and reused from the prompt cache, if llama-server provides them), timings, energy of the turn in Wh and session metadata. The write happens after `tracker.stop()`, hence outside the energy measurement window.

Start:
```bash
uvicorn energy_api:app --host 127.0.0.1 --port 9000
```

### Experimental session

Session metadata are set by the experimenter before each session:

```bash
curl -X POST http://127.0.0.1:9000/experiment/session -H "Content-Type: application/json" \
     -d '{"participant_id": "P01", "task_id": "T1", "condition": "GE", "model": "gemma-4-31b"}'
curl http://127.0.0.1:9000/experiment/session
curl -X POST http://127.0.0.1:9000/experiment/session/end -H "Content-Type: application/json" \
     -d '{"task_completed": false, "notes": "..."}'
```
### Environment variables (defaults in brackets)

| Variable | Meaning |
|---|---|
| `LLAMA_BASE_URL` | address of llama-server (`http://127.0.0.1:8080`) |
| `LLAMA_API_KEY` | key, ignored by llama.cpp (`dummy-key`) |
| `EMISSIONS_DIR` | folder of CodeCarbon's `emissions.csv` (`./emissions`) |
| `LOG_DIR` | folder of the logs (`./logs`) |
| `TRACKING_MODE` | CodeCarbon mode, `machine` or `process` (`machine`). Use `machine` for the experiment because inference runs in the separate llama-server process. |
| `GE_MARKER` | phrase that identifies the GE summary request, case-insensitive (`riepilogo conciso del contesto`) |
| `LOG_FULL_REQUEST` | `1` = save all messages sent to the model, `0` = only the last one (`1`) |
| `REQUEST_USAGE` | `1` = add `stream_options.include_usage` to the request, so that llama-server always returns tokens and timings (`1`) |
| `AUX_NO_CONV_ID` | `1` = requests without a conversation identifier (e.g. the title generated by Chat UI) are auxiliary: forwarded, logged with `auxiliary: true`, not counted as turns and not measured (`1`) |
| `MODEL_ID` | name of the model exposed to Chat UI in `/v1/models` and shown to the participants; neutral on purpose, the real model is in `model_label` (`assistente-locale`) |

### Log files

- `turns.jsonl`, one record per turn: `session_id`, `participant_id`, `task_id`, `condition`, `model_label`, `segment`, `turn_index`, `turn_in_segment`, `conversation_id`, `is_ge_request`, `auxiliary`, `n_messages`, `request_params`, `messages_sha256`, `request_messages`, `response`, `reasoning`, `finish_reason`, `completed`, `error`, `prompt_tokens`, `prompt_tokens_evaluated`, `prompt_tokens_cached`, `completion_tokens`, `usage`, `timings`, `ttft_ms`, `wall_ms`, `started_at`, `overlapped`, `energy_wh` (`cpu`, `gpu`, `ram`, `total`, `duration_s`, `emissions_kg`), `tracking_mode`, `codecarbon_run_id`.
- `sessions.jsonl`, session events (`start`, `end`, `auto_start`): metadata, model label (declared and taken from the server) and warnings, llama-server properties (loaded model, context, build), CodeCarbon version, task outcome (`task_completed`) and number of GE and auxiliary requests.

### Warnings

- Measurement scope. The default is `machine`, so CodeCarbon measures the whole workstation, including background activity; keep the machine idle and measure/subtract idle consumption. `process` measures the proxy process, not the separate llama-server inference process, and is unsuitable for the primary experiment.
- GE request detection. It relies on `GE_MARKER`: check that the standardised prompt given to the participants contains it.
