# Gold Extraction – experiment tooling

*Italian version: [README(IT).md](README%28IT%29.md)*

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
| `download_models.py` | Selects the models from the metadata of Hugging Face, freezes the selection in dated documents and downloads the GGUF files, recording a manifest. |
| `conversation_stats.py` | Computes statistics on turns and tokens of datasets of conversations with LLMs. |
| `export_conversations.py` | Extracts the selected conversations from a dataset for offline replay. |
| `energy_api.py` | Proxy between Chat UI and llama-server: measures energy (CodeCarbon) and logs the conversations. |

## Setup from scratch

The repository contains only the code of this work. Chat UI, llama.cpp, the models and the virtual environment are not included: they are installed once, following these steps. The commands are for the Linux workstation; on Windows the steps are the same, apart from the paths and the activation of the virtual environment. Run everything from the root of the repository unless stated otherwise.

Layout after the setup (everything that is not code is ignored by Git, except `selezioni/`):

```
<repository>/
├── chat-ui/             step 4
├── llama.cpp/           step 5, with the build in llama.cpp/build/
├── models/              step 6, one folder per model: models/<name>/<file>.gguf
├── selezioni/           step 6, frozen documents of the selection (kept in Git)
├── venv/                step 2
├── logs/, emissions/    created at run time
├── energy_api.py, download_models.py, conversation_stats.py, export_conversations.py
└── commands.txt, requirements.txt, README.md
```

**1. Prerequisites.** Git; Python 3.9 or later; Node.js (tested with v24.21.0); CMake and a C++ compiler (`build-essential`); the NVIDIA driver and the CUDA toolkit (`nvidia-smi` must list the GPUs and `nvcc --version` must work); MongoDB Community Edition, installed following the official instructions for the operating system; a Hugging Face account with a token of type Read.

**2. Python environment.**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
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
cp ../.env.local .          # configuration provided with this repository
cp ../package.json .
npm install
cd ..
```

If the two files are elsewhere in the repository, adapt the paths; if the provided file is named `env.local` (without the dot), it must be renamed `.env.local`. The variables that matter are:

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
git checkout <commit-or-tag>          # write it down: it is part of the experimental setup
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
cd ..
```

The option names change between versions: check `docs/build.md` of the version in use. Without CUDA support the build runs on the CPU and `-ngl 99` has no effect: when the server starts, its log must list the GPUs.

**6. Models.** Check the free disk space first.

```bash
read -rs HF_TOKEN && export HF_TOKEN
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

1. **Selection** from the metadata of the Hugging Face Hub (parameters, chat template, languages, context, relations with other models, downloads). The organisations that publish the "official" models are listed in `ORG_PREDEFINITE` (or in a file passed with `--orgs-file`). A model is a candidate only if a GGUF with a 4-bit label exists from a trusted quantiser.
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
python conversation_stats.py --inspect
python conversation_stats.py --sample 5000 --plot
python conversation_stats.py --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --min-turns 10 --min-tokens 8000
python conversation_stats.py --dataset allenai/WildChat-1M --lang English
```
Output in `stats_out/` (configurable with `--out`):

- `conversazioni.csv`: one row per conversation;
- `summary.json`: statistics and metadata for reproducibility (commit, seed, date);
- `distribuzioni.png`: plot of the distributions (only with `--plot`).

## `export_conversations.py`

Selects long conversations (by number of turns, tokens and model) and extracts a reproducible random sample, saved as JSONL (one conversation per line) with the messages as `{"role": ..., "content": ...}`. It requires `conversation_stats.py` in the same folder.

```bash
python export_conversations.py
python export_conversations.py --n 50 --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --max-tokens 24000
python export_conversations.py --dataset allenai/WildChat-1M --model-regex "" --n 100
```
Output in `replay_set/` (configurable with `--out`):

- `conversazioni.jsonl`: the selected conversations;
- `selezione.json`: parameters, dataset commit and counts, to be reported in the methodology.

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
| `TRACKING_MODE` | CodeCarbon mode, `process` or `machine` (`process`); see Warnings |
| `GE_MARKER` | phrase that identifies the GE summary request, case-insensitive (`riepilogo conciso del contesto`) |
| `LOG_FULL_REQUEST` | `1` = save all messages sent to the model, `0` = only the last one (`1`) |
| `REQUEST_USAGE` | `1` = add `stream_options.include_usage` to the request, so that llama-server always returns tokens and timings (`1`) |
| `AUX_NO_CONV_ID` | `1` = requests without a conversation identifier (e.g. the title generated by Chat UI) are auxiliary: forwarded, logged with `auxiliary: true`, not counted as turns and not measured (`1`) |
| `MODEL_ID` | name of the model exposed to Chat UI in `/v1/models` and shown to the participants; neutral on purpose, the real model is in `model_label` (`assistente-locale`) |

### Log files

- `turns.jsonl`, one record per turn: `session_id`, `participant_id`, `task_id`, `condition`, `model_label`, `segment`, `turn_index`, `turn_in_segment`, `conversation_id`, `is_ge_request`, `auxiliary`, `n_messages`, `request_params`, `messages_sha256`, `request_messages`, `response`, `reasoning`, `finish_reason`, `completed`, `error`, `prompt_tokens`, `prompt_tokens_evaluated`, `prompt_tokens_cached`, `completion_tokens`, `usage`, `timings`, `ttft_ms`, `wall_ms`, `started_at`, `overlapped`, `energy_wh` (`cpu`, `gpu`, `ram`, `total`, `duration_s`, `emissions_kg`), `tracking_mode`, `codecarbon_run_id`.
- `sessions.jsonl`, session events (`start`, `end`, `auto_start`): metadata, model label (declared and taken from the server) and warnings, llama-server properties (loaded model, context, build), CodeCarbon version, task outcome (`task_completed`) and number of GE and auxiliary requests.

### Warnings

- Measurement scope. With `TRACKING_MODE=process` CodeCarbon tries to isolate the consumption of the current process, i.e. the proxy, whereas inference runs in the llama-server process; with `machine` (CodeCarbon's default) the whole machine is measured, including background activity, whose idle consumption must be measured and subtracted.
- GE request detection. It relies on `GE_MARKER`: check that the standardised prompt given to the participants contains it.
