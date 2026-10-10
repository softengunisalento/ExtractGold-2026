# Gold Extraction – strumenti per l'esperimento

*English version: [README.md](README.md)*

## Architettura

```
Hugging Face Chat UI  ──►  energy_api.py (proxy)  ──►  llama-server  ──►  modello GGUF
      :5173                 FastAPI + CodeCarbon         :8080
                                 :9000
                                   │
                              MongoDB :27017
                      (contesto delle conversazioni di Chat UI)
```

- Hugging Face Chat UI è in esecuzione sulla porta 5173 ed è l'interfaccia usata dai partecipanti.
- `energy_api.py` è il proxy sulla porta 9000. Inoltra ogni richiesta al modello, misura l'energia di ciascun turn con CodeCarbon e registra le conversazioni. Inoltra i messaggi esattamente come li invia Chat UI (che manda già l'intera cronologia); `logs/turns.jsonl` registra il testo effettivamente inviato al modello. Non accede a MongoDB.
- `llama-server` (llama.cpp) esegue il modello sulla porta 8080.
- MongoDB è in esecuzione sulla porta 27017 e conserva le conversazioni di Chat UI.
- L'intera architettura viene eseguita in un ambiente virtuale Python (`venv`).

File di configurazione di Chat UI:

- `.env.local` contiene la configurazione locale di Chat UI. Va collocato nella cartella `chat-ui` con il nome `.env.local`. Tre impostazioni sono essenziali: l'URL di base deve puntare al proxy (porta 9000); la memoria della conversazione deve essere attiva (ENABLE_CONVERSATIONS=true, DISABLE_CONVERSATION_MEMORY=false), perché con i modelli locali Chat UI funziona per impostazione predefinita in modalità stateless e non conserva il contesto; e la generazione dei titoli delle conversazioni deve essere disattivata (LLM_SUMMARIZATION=false), altrimenti Chat UI invia una richiesta in più al modello al primo messaggio di ogni conversazione (vedi Avvertenze).
- `commands.txt` elenca i comandi che avviano l'intera architettura.

## Contenuto

| File | Scopo |
|---|---|
| `download_models.py` | Scarica da Hugging Face i modelli in formato GGUF e registra un manifest dei file ottenuti. |
| `conversation_stats.py` | Calcola statistiche su turn e token di dataset di conversazioni con LLM. |
| `export_conversations.py` | Estrae da un dataset le conversazioni selezionate per il replay offline. |
| `energy_api.py` | Proxy tra Chat UI e llama-server: misura l'energia (CodeCarbon) e registra le conversazioni. |

## Setup da zero

Il repository contiene solo il codice di questo lavoro. Chat UI, llama.cpp, i modelli e l'ambiente virtuale non sono inclusi: si installano una volta, seguendo questi passi. I comandi sono per la workstation Linux; su Windows i passi sono gli stessi, tranne i percorsi e l'attivazione dell'ambiente virtuale. Si esegue tutto dalla radice del repository, salvo diversa indicazione.

Struttura dopo il setup (tutto ciò che non è codice è ignorato da Git):

```
<repository>/
├── chat-ui/             passo 4
├── llama.cpp/           passo 5, con la compilazione in llama.cpp/build/
├── models/              passo 6, una cartella per modello: models/<nome>/<file>.gguf
├── venv/                passo 2
├── logs/, emissions/    creati durante l'esecuzione
├── energy_api.py, download_models.py, conversation_stats.py, export_conversations.py
└── commands.txt, requirements.txt, README.md
```

**1. Prerequisiti.** Git; Python 3.9 o superiore; Node.js (provato con v24.21.0); CMake e un compilatore C++ (`build-essential`); il driver NVIDIA e il CUDA toolkit (`nvidia-smi` deve elencare le GPU e `nvcc --version` deve funzionare); MongoDB Community Edition, installato seguendo le istruzioni ufficiali per il sistema operativo; un account Hugging Face con un token di tipo Read.

**2. Ambiente Python.**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
**3. MongoDB.** Chat UI conserva le conversazioni.

```bash
sudo systemctl enable --now mongod
systemctl status mongod
```

**4. Chat UI.** È supportata solo la versione usata in questo lavoro: la configurazione è stata provata solo con questo commit.

```bash
git clone https://github.com/huggingface/chat-ui
cd chat-ui
git checkout -f 9159eaa4
cp ../.env.local .          # configurazione fornita con questo repository
cp ../package.json .
npm install
cd ..
```

Se i due file sono altrove nel repository, si adattano i percorsi; se il file fornito si chiama `env.local` (senza punto), va rinominato `.env.local`. Le variabili che contano sono:

```
OPENAI_BASE_URL=http://127.0.0.1:9000/v1
OPENAI_API_KEY=none
ENABLE_CONVERSATIONS=true
DISABLE_CONVERSATION_MEMORY=false
LLM_SUMMARIZATION=false
MONGODB_URL=mongodb://127.0.0.1:27017/chatui
```
**5. llama.cpp.** Si compila con il supporto CUDA:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout <commit-o-tag>          # va annotato: fa parte dell'assetto sperimentale
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
cd ..
```

I nomi delle opzioni cambiano tra le versioni: si controlla `docs/build.md` della versione in uso. Senza il supporto CUDA la compilazione gira su CPU e `-ngl 99` non ha effetto: all'avvio il log del server deve elencare le GPU.

**6. Modelli.** Prima si controlla lo spazio libero: ogni modello pesa decine di GB.

```bash
read -rs HF_TOKEN && export HF_TOKEN
python download_models.py --list
python download_models.py --dry-run
python download_models.py                                   # scarica tutti i modelli
```
**7. Verifica dell'intera catena.** Si avviano i servizi come in `commands.txt`, con una cartella di log di prova (vedi Avvertenze) e, per il modello, `--alias <folderName>`. Poi:

```bash
curl http://127.0.0.1:8080/props                # il modello e la dimensione del contesto
curl http://127.0.0.1:9000/v1/models            # deve mostrare l'id assistente-locale
```

Si apre una sessione di prova (vedi Sessione sperimentale), si scrivono due messaggi in una conversazione nuova di Chat UI e si chiude la sessione. Nella cartella di log di prova `turns.jsonl` deve contenere due record con `auxiliary: false`, e il record di chiusura di `sessions.jsonl` deve mostrare `"turns": 2` e `"aux_requests": 0`. Si elimina la cartella di prova prima delle sessioni vere.

## `download_models.py`

Scarica i file `.gguf` in `models/<nome>/`, pronti per `llama-server -m <file>`. Un download interrotto riprende da dove si era fermato. I file sono scaricati dal commit esatto del repository e, al termine, se ne verifica lo SHA-256 rispetto a quello dichiarato dal Hub.

```bash
python download_models.py --list
python download_models.py --dry-run
python download_models.py --only gpt-oss-20b,qwen3-coder-30b-a3b
python download_models.py --only laguna-xs-2.1 --quant I-Balanced
python download_models.py --no-verify      # salta il calcolo dello SHA-256 locale
python download_models.py --update         # usa l'ultimo commit anche se il manifest ne fissa un altro (i pesi cambiano)
python download_models.py --force          # prosegue anche se lo spazio libero non basta per tutti i modelli selezionati
```

Output: `models/manifest.json`, con repository, quantizzazione, commit, data del download e, per ciascun file, nome, dimensione e SHA-256. Va conservato insieme ai risultati e riportato nella metodologia per indicare esattamente quali pesi sono stati usati. La lista dei modelli si modifica nella variabile `MODELS` all'inizio dello script.

## `conversation_stats.py`

Legge in streaming un campione di conversazioni da Hugging Face e calcola, per ciascuna, il numero di turn e di token.

```bash
python conversation_stats.py --inspect
python conversation_stats.py --sample 5000 --plot
python conversation_stats.py --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --min-turns 10 --min-tokens 8000
python conversation_stats.py --dataset allenai/WildChat-1M --lang English
```
Output in `stats_out/` (modificabile con `--out`):

- `conversazioni.csv`: una riga per conversazione;
- `summary.json`: statistiche e metadati per la riproducibilità (commit, seed, data);
- `distribuzioni.png`: grafico delle distribuzioni (solo con `--plot`).

## `export_conversations.py`

Seleziona le conversazioni lunghe (per numero di turn, token e modello) ed estrae un campione casuale riproducibile, salvato in formato JSONL (una conversazione per riga), con i messaggi nel formato `{"role": ..., "content": ...}`. Richiede `conversation_stats.py` nella stessa cartella.

```bash
python export_conversations.py
python export_conversations.py --n 50 --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --max-tokens 24000
python export_conversations.py --dataset allenai/WildChat-1M --model-regex "" --n 100
```
Output in `replay_set/` (modificabile con `--out`):

- `conversazioni.jsonl`: le conversazioni selezionate;
- `selezione.json`: parametri, commit del dataset e conteggi, da riportare nella metodologia.

## `energy_api.py`

Proxy tra Hugging Face Chat UI e llama-server. Ogni richiesta di chat viene inoltrata a llama-server; al termine della risposta viene scritta una riga in `logs/turns.jsonl` con: messaggi inviati al modello, risposta, token (totali, valutati e riusati dalla prompt cache, se llama-server li fornisce), tempi, energia del turn in Wh e metadati della sessione. La scrittura avviene dopo `tracker.stop()`, quindi fuori dalla finestra di misura dell'energia.

Avvio:
```bash
uvicorn energy_api:app --host 127.0.0.1 --port 9000
```

### Sessione sperimentale

I metadati della sessione sono impostati dallo sperimentatore prima di ciascuna sessione:

```bash
curl -X POST http://127.0.0.1:9000/experiment/session -H "Content-Type: application/json" \
     -d '{"participant_id": "P01", "task_id": "T1", "condition": "GE", "model": "gemma-4-31b"}'
curl http://127.0.0.1:9000/experiment/session
curl -X POST http://127.0.0.1:9000/experiment/session/end -H "Content-Type: application/json" \
     -d '{"task_completed": false, "notes": "..."}'
```
### Variabili d'ambiente (valori predefiniti tra parentesi)

| Variabile | Significato |
|---|---|
| `LLAMA_BASE_URL` | indirizzo di llama-server (`http://127.0.0.1:8080`) |
| `LLAMA_API_KEY` | chiave, ignorata da llama.cpp (`dummy-key`) |
| `MONGODB_URL` | database di Chat UI (`mongodb://127.0.0.1:27017/chatui`) |
| `MONGO_DB_NAME` | nome del database (`chatui`) |
| `EMISSIONS_DIR` | cartella di `emissions.csv` di CodeCarbon (`./emissions`) |
| `LOG_DIR` | cartella dei log (`./logs`) |
| `TRACKING_MODE` | modalità di CodeCarbon, `process` o `machine` (`process`); vedi Avvertenze |
| `GE_MARKER` | frase che identifica la richiesta di riepilogo GE, senza distinzione tra maiuscole e minuscole (`riepilogo conciso del contesto`) |
| `LOG_FULL_REQUEST` | `1` = salva tutti i messaggi inviati al modello, `0` = solo l'ultimo (`1`) |
| `REQUEST_USAGE` | `1` = aggiunge `stream_options.include_usage` alla richiesta, così llama-server restituisce sempre token e tempi (`1`) |
| `AUX_NO_CONV_ID` | `1` = le richieste senza identificativo di conversazione (ad es. il titolo generato da Chat UI) sono ausiliarie: inoltrate, registrate con `auxiliary: true`, non contate come turn e non misurate (`1`) |
| `MODEL_ID` | nome del modello esposto a Chat UI in `/v1/models` e mostrato ai partecipanti; volutamente neutro, il modello vero è in `model_label` (`assistente-locale`) |

### File di log

- `turns.jsonl`, un record per turn: `session_id`, `participant_id`, `task_id`, `condition`, `model_label`, `segment`, `turn_index`, `turn_in_segment`, `conversation_id`, `is_ge_request`, `n_messages`, `request_params`, `messages_sha256`, `request_messages`, `response`, `reasoning`, `finish_reason`, `completed`, `error`, `prompt_tokens`, `prompt_tokens_evaluated`, `prompt_tokens_cached`, `completion_tokens`, `usage`, `timings`, `ttft_ms`, `wall_ms`, `energy_wh` (`cpu`, `gpu`, `ram`, `total`, `duration_s`, `emissions_kg`), `tracking_mode`, `codecarbon_run_id`.
- `sessions.jsonl`, eventi di sessione (`start`, `end`, `auto_start`): metadati, proprietà di llama-server (modello caricato, contesto, build), versione di CodeCarbon, esito del task (`task_completed`) e numero di richieste GE.

### Avvertenze

- Perimetro della misura.Con `TRACKING_MODE=process` CodeCarbon cerca di isolare il consumo del processo corrente, cioè del proxy, mentre l'inferenza avviene nel processo di llama-server; con `machine` (impostazione predefinita di CodeCarbon) si misura l'intera macchina, comprese le attività di fondo, il cui consumo a riposo va misurato e sottratto. Prima delle misure definitive si consiglia di confrontare i valori con una misura indipendente (NVML per le GPU, RAPL per la CPU o un wattmetro).
- Riconoscimento della richiesta GE. Si basa su `GE_MARKER`: va verificato che il prompt standardizzato fornito ai partecipanti lo contenga.
