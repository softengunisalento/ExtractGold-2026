# Gold Extraction – experiment tooling

This repository collects the tools used in the thesis on Gold Extraction (GE), a prompt-engineering technique in which the model is asked to summarise the conversation into a concise prompt from which a new session is started, with the aim of reducing the energy consumption of locally executed large language models (LLMs).

## Architecture

```
Hugging Face Chat UI  ──►  energy_api.py (proxy)  ──►  llama-server  ──►  GGUF model
      :5173                 FastAPI + CodeCarbon         :8080
                                 :9000
                                   │
                              MongoDB :27017
                       (conversation context of Chat UI)
```

- Hugging Face Chat UI runs on port 5173 and is the interface used by the participants.
- `energy_api.py` is the proxy server on port 9000. It forwards each request to the model, measures the energy of every turn with CodeCarbon and logs the conversations. It rebuilds the conversation history from the MongoDB server, using the conversation identifier sent by Chat UI.
- `llama-server` (llama.cpp) runs the model on port 8080.
- MongoDB runs on port 27017 and stores the conversations of Chat UI.
- The whole architecture runs inside a Python virtual environment (`venv`).

Configuration files of Chat UI (inherited from the previous work):

- `.env.local` contains the local configuration of Chat UI. It must be placed in the `chat-ui` directory as `.env.local`. Two settings are essential: the base URL must point to the proxy (port 9000), and conversation memory must be enabled (`ENABLE_CONVERSATIONS=true`, `DISABLE_CONVERSATION_MEMORY=false`); by default, with local models, Chat UI runs in stateless mode and does not keep the context.
- package.json is located in the chat-ui directory. To ensure the application correctly tracks the context and saves it to the MongoDB server (making it accessible to the proxy), you may need to check the scripts section. Ensure the dev startup script looks similar to "dev": "vite dev --host 127.0.0.1" and does not contain flags that mock or disable the database in development mode.
(Note: depending on the exact version of the cloned Chat UI, a .env.local file with ENABLE_CONVERSATIONS=true might be sufficient.)
- `commands.txt` lists the commands that start the whole architecture.

Example of how to serve a model (the context size must be the same for all models):

```bash
llama-server -m models/<name>/<file>.gguf --port 8080 -c 32768 -ngl 99
```

## Contents

| File | Purpose |
|---|---|
| `download_models.py` | Downloads the GGUF models from Hugging Face and records a manifest of the files obtained. |
| `conversation_stats.py` | Computes statistics on turns and tokens of datasets of conversations with LLMs. |
| `export_conversations.py` | Extracts the selected conversations from a dataset for offline replay. |
| `energy_api.py` | Proxy between Chat UI and llama-server: measures energy (CodeCarbon) and logs the conversations. |

## Requirements

- Python 3.9 or later.
- `download_models.py`: `pip install -U huggingface_hub`
- `conversation_stats.py` and `export_conversations.py`: `pip install -U datasets` (optional: `tokenizers` for `--tokenizer`, `matplotlib` for `--plot`)
- `energy_api.py`: `pip install fastapi uvicorn requests codecarbon pymongo`; at run time also llama-server, MongoDB and Hugging Face Chat UI.
- A Hugging Face token of type Read, set in the environment variable `HF_TOKEN`:

  ```bash
  export HF_TOKEN="hf_..."
  ```

  The token must never appear in scripts, commits or output files.

## Workflow

1. `download_models.py` downloads the models and produces `models/manifest.json`.
2. `conversation_stats.py` analyses the length of real conversations, to choose the context threshold that triggers GE.
3. `export_conversations.py` extracts the sample of conversations for the offline replay.
4. `energy_api.py` measures energy and logs turns and sessions during the experimental sessions with the participants.

## `download_models.py`

Downloads the `.gguf` files into `models/<name>/`, ready for `llama-server -m <file>`. An interrupted download resumes from where it stopped. Files are downloaded from the exact commit of the repository and, once finished, their SHA-256 is checked against the one declared by the Hub.

```bash
python download_models.py --list
python download_models.py --dry-run
python download_models.py --only gpt-oss-20b,qwen3-coder-30b-a3b
python download_models.py --only laguna-xs-2.1 --quant I-Balanced
python download_models.py --no-verify      # skip the local SHA-256 computation
```

Output: `models/manifest.json`, with repository, quantisation, commit, download date and, for each file, name, size and SHA-256. It should be kept together with the results and cited in the methodology to state exactly which weights were used. The list of models is edited in the `MODELS` variable at the top of the script.

## `conversation_stats.py`

Reads a sample of conversations from Hugging Face in streaming mode and computes the number of turns and tokens of each conversation.

Definitions:

- turn: a user message (prompt) together with the corresponding response.
- final context: tokens of all messages up to and including the last user message, i.e. the amount of text the model must process at the last turn.
- total prefill: sum, over all turns, of the tokens processed in the prefill phase, assuming that the whole history is reprocessed at every turn (no prompt cache).

Without `--tokenizer` the number of tokens is an estimate (1 token per 4 characters), which tends to underestimate it for source code. For values comparable with the models of the thesis, the tokenizer of one of them should be given (`--tokenizer`).

The sample is pseudo-random: in streaming mode shuffling only happens within a buffer (`--shuffle-buffer`). With `--no-streaming` the whole dataset is downloaded and the sample is random.

```bash
python conversation_stats.py --inspect
python conversation_stats.py --sample 5000 --plot
python conversation_stats.py --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --min-turns 10 --min-tokens 8000
python conversation_stats.py --dataset allenai/WildChat-1M --lang English
```

It is advisable to run `--inspect` first: it shows the columns of the dataset and checks that the format is recognised. Gated datasets require accepting the specific terms on their Hugging Face repository page and ensuring the HF_TOKEN environment variable is set."

Output in `stats_out/` (configurable with `--out`):

- `conversazioni.csv`: one row per conversation;
- `summary.json`: statistics and metadata for reproducibility (commit, seed, date);
- `distribuzioni.png`: plot of the distributions (only with `--plot`).

## `export_conversations.py`

Selects long conversations (by number of turns, tokens and model) and extracts a reproducible random sample, saved as JSONL (one conversation per line) with the messages as `{"role": ..., "content": ...}`. It requires `conversation_stats.py` in the same folder.

The dataset is read entirely in streaming mode; the sample is chosen by reservoir sampling and is therefore uniform among all eligible conversations. Given the same dataset (commit), parameters, tokenizer and `--seed`, the result is identical.

```bash
python export_conversations.py
python export_conversations.py --n 50 --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --max-tokens 24000
python export_conversations.py --dataset allenai/WildChat-1M --model-regex "" --n 100
```

For datasets without a `model` column use `--model-regex ""` (no filter).

Output in `replay_set/` (configurable with `--out`):

- `conversazioni.jsonl`: the selected conversations;
- `selezione.json`: parameters, dataset commit and counts, to be reported in the methodology.

## `energy_api.py`

Proxy between Hugging Face Chat UI and llama-server. Every chat request is forwarded to llama-server; when the response ends, one line is written to `logs/turns.jsonl` with: messages sent to the model, response, tokens (total, evaluated and reused from the prompt cache, if llama-server provides them), timings, energy of the turn in Wh and session metadata. The write happens after `tracker.stop()`, hence outside the energy measurement window.

Start (port 9000, as in the Chat UI configuration):

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

Required fields: `participant_id`, `task_id`, `condition` (`"baseline"` or `"GE"`) and `model` (label of the model). Other fields (e.g. `notes`) are saved in `sessions.jsonl`. Without a session, turns are logged anyway, with `session_id` `unassigned-...`.

### Environment variables (defaults in brackets)

| Variable | Meaning |
|---|---|
| `LLAMA_BASE_URL` | address of llama-server (`http://127.0.0.1:8080`) |
| `LLAMA_API_KEY` | key, ignored by llama.cpp (`dummy-key`) |
| `MONGODB_URL` | database of Chat UI (`mongodb://127.0.0.1:27017/chatui`) |
| `MONGO_DB_NAME` | database name (`chatui`) |
| `EMISSIONS_DIR` | folder of CodeCarbon's `emissions.csv` (`./emissions`) |
| `LOG_DIR` | folder of the logs (`./logs`) |
| `TRACKING_MODE` | CodeCarbon mode, `process` or `machine` (`process`); see Warnings |
| `GE_MARKER` | phrase that identifies the GE summary request, case-insensitive (`riepilogo conciso del contesto`) |
| `LOG_FULL_REQUEST` | `1` = save all messages sent to the model, `0` = only the last one (`1`) |
| `REQUEST_USAGE` | `1` = add `stream_options.include_usage` to the request, so that llama-server always returns tokens and timings (`1`) |

### Log files

- `turns.jsonl`, one record per turn: `session_id`, `participant_id`, `task_id`, `condition`, `model_label`, `segment`, `turn_index`, `turn_in_segment`, `conversation_id`, `is_ge_request`, `n_messages`, `request_params`, `messages_sha256`, `request_messages`, `response`, `reasoning`, `finish_reason`, `completed`, `error`, `prompt_tokens`, `prompt_tokens_evaluated`, `prompt_tokens_cached`, `completion_tokens`, `usage`, `timings`, `ttft_ms`, `wall_ms`, `energy_wh` (`cpu`, `gpu`, `ram`, `total`, `duration_s`, `emissions_kg`), `tracking_mode`, `codecarbon_run_id`.
- `sessions.jsonl`, session events (`start`, `end`, `auto_start`): metadata, llama-server properties (loaded model, context, build), CodeCarbon version, task outcome (`task_completed`) and number of GE requests.

A new *segment* begins with the request that follows the GE summary request, i.e. when the participant starts the new conversation. A `finish_reason` of `length` means that the response was cut by the token limit, which is typical of repetitive loops (cycling).

### Warnings

- Measurement scope. With `TRACKING_MODE=process` CodeCarbon tries to isolate the consumption of the current process, i.e. the proxy, whereas inference runs in the llama-server process; with `machine` (CodeCarbon's default) the whole machine is measured, including background activity, whose idle consumption must be measured and subtracted. Before the final measurements, the values should be compared with an independent measurement (NVML for GPUs, RAPL for the CPU, or a power meter).
- Auxiliary requests. If Chat UI sends requests that are not user turns (e.g. title generation), they go through the proxy, consume energy and are logged; they can be recognised from `n_messages` and `request_params`, and it is advisable to disable them.
- GE detection relies on `GE_MARKER`: check that the standardised prompt given to the participants contains it.

## Reproducibility

- `models/manifest.json`: which weights (repository, commit, SHA-256).
- `summary.json` and `selezione.json`: which data (dataset, commit, seed, parameters, date).
- `sessions.jsonl`: which configuration (llama.cpp build, context, CodeCarbon version, tracking mode, GE marker).

These files should be kept together with the results.

## Personal data and licences

- Session logs contain texts written by the participants: pseudonymous identifiers are used (P01, P02, …), the consent form must be updated to include the recording of the conversations, and identifying data (e.g. user name and machine name in terminal output) must be removed before the logs are published.
- CodeChat and WildChat are distributed under the ODC-BY licence: they must be cited and no attempt may be made to identify individuals. The licences of the individual models must be checked on their model cards.