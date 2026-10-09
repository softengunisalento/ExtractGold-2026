# Gold Extraction – strumenti per l'esperimento

*English version: [README.md](README.md)*

Questo repository raccoglie gli strumenti usati nella tesi sulla Gold Extraction (GE), una tecnica di prompt engineering in cui si chiede al modello di riassumere la conversazione in un prompt conciso, da cui avviare una nuova sessione, allo scopo di ridurre il consumo energetico dell'inferenza di modelli linguistici (LLM) eseguiti in locale.

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
- `energy_api.py` è il proxy sulla porta 9000. Inoltra ogni richiesta al modello, misura l'energia di ciascun turn con CodeCarbon e registra le conversazioni. Ricostruisce la cronologia della conversazione dal server MongoDB, usando l'identificativo di conversazione inviato da Chat UI.
- `llama-server` (llama.cpp) esegue il modello sulla porta 8080.
- MongoDB è in esecuzione sulla porta 27017 e conserva le conversazioni di Chat UI.
- L'intera architettura viene eseguita in un ambiente virtuale Python (`venv`).

File di configurazione di Chat UI (ereditati dal lavoro precedente):

- `.env.local` contiene la configurazione locale di Chat UI. Va collocato nella cartella `chat-ui` con il nome `.env.local`. Due impostazioni sono essenziali: l'URL di base deve puntare al proxy (porta 9000) e la memoria della conversazione deve essere attiva (`ENABLE_CONVERSATIONS=true`, `DISABLE_CONVERSATION_MEMORY=false`); con i modelli locali, infatti, Chat UI funziona per impostazione predefinita in modalità stateless e non conserva il contesto.
- - package.json è uno dei file della cartella chat-ui. Affinché l'applicazione tracci correttamente il contesto salvandolo sul server MongoDB (rendendolo accessibile al proxy), devi aprire il file e aggiungere \"--experimental-modules\" oppure accertarti che il database sia forzato all'avvio. Aggiungi/modifica la riga di avvio in scripts:
"dev": "vite dev --host 127.0.0.1" (assicurati che non ci siano parametri che disabilitano il DB in sviluppo).
(Nota: a seconda della versione esatta di Chat UI clonata, potrebbe essere sufficiente il .env.local con ENABLE_CONVERSATIONS=true)
- `commands.txt` elenca i comandi che avviano l'intera architettura.

Esempio di avvio di un modello (la dimensione del contesto deve essere la stessa per tutti i modelli):

```bash
llama-server -m models/<nome>/<file>.gguf --port 8080 -c 32768 -ngl 99
```

## Contenuto

| File | Scopo |
|---|---|
| `download_models.py` | Scarica da Hugging Face i modelli in formato GGUF e registra un manifest dei file ottenuti. |
| `conversation_stats.py` | Calcola statistiche su turn e token di dataset di conversazioni con LLM. |
| `export_conversations.py` | Estrae da un dataset le conversazioni selezionate per il replay offline. |
| `energy_api.py` | Proxy tra Chat UI e llama-server: misura l'energia (CodeCarbon) e registra le conversazioni. |

## Requisiti

- Python 3.9 o superiore.
- `download_models.py`: `pip install -U huggingface_hub`
- `conversation_stats.py` e `export_conversations.py`: `pip install -U datasets` (opzionali: `tokenizers` per `--tokenizer`, `matplotlib` per `--plot`)
- `energy_api.py`: `pip install fastapi uvicorn requests codecarbon pymongo`; in esecuzione servono anche llama-server, MongoDB e Hugging Face Chat UI.
- Un token Hugging Face di tipo **Read**, impostato nella variabile d'ambiente `HF_TOKEN`:

  ```bash
  export HF_TOKEN="hf_..."
  ```

  Il token non deve comparire negli script, nei commit o nei file di output.

## Flusso di lavoro

1. `download_models.py` scarica i modelli e produce `models/manifest.json`.
2. `conversation_stats.py` analizza la lunghezza delle conversazioni reali, per scegliere la soglia di contesto che attiva la GE.
3. `export_conversations.py` estrae il campione di conversazioni per il replay offline.
4. `energy_api.py` misura l'energia e registra turn e sessioni durante le sessioni sperimentali con i partecipanti.

## `download_models.py`

Scarica i file `.gguf` in `models/<nome>/`, pronti per `llama-server -m <file>`. Un download interrotto riprende da dove si era fermato. I file sono scaricati dal commit esatto del repository e, al termine, se ne verifica lo SHA-256 rispetto a quello dichiarato dal Hub.

```bash
python download_models.py --list
python download_models.py --dry-run
python download_models.py --only gpt-oss-20b,qwen3-coder-30b-a3b
python download_models.py --only laguna-xs-2.1 --quant I-Balanced
python download_models.py --no-verify      # salta il calcolo dello SHA-256 locale
```

Output: `models/manifest.json`, con repository, quantizzazione, commit, data del download e, per ciascun file, nome, dimensione e SHA-256. Va conservato insieme ai risultati e riportato nella metodologia per indicare esattamente quali pesi sono stati usati. La lista dei modelli si modifica nella variabile `MODELS` all'inizio dello script.

## `conversation_stats.py`

Legge in streaming un campione di conversazioni da Hugging Face e calcola, per ciascuna, il numero di turn e di token.

Definizioni:

- turn: un messaggio dell'utente (prompt) con la relativa risposta.
- contesto finale: token di tutti i messaggi fino all'ultimo messaggio dell'utente incluso, cioè la quantità di testo che il modello deve elaborare all'ultimo turn.
- prefill totale: somma, su tutti i turn, dei token elaborati nella fase di prefill, nell'ipotesi che la cronologia sia rielaborata per intero a ogni turn (assenza di prompt cache).

Senza `--tokenizer` il numero di token è una stima (1 token ogni 4 caratteri), che per il codice sorgente tende a sottostimare. Per valori confrontabili con i modelli della tesi si indica il tokenizer di uno di essi (`--tokenizer`).

Il campione è pseudo-casuale: in streaming il rimescolamento avviene solo all'interno di un buffer (`--shuffle-buffer`). Con `--no-streaming` l'intero dataset viene scaricato e il campione è casuale.

```bash
python conversation_stats.py --inspect
python conversation_stats.py --sample 5000 --plot
python conversation_stats.py --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --min-turns 10 --min-tokens 8000
python conversation_stats.py --dataset allenai/WildChat-1M --lang English
```

Si consiglia di eseguire prima `--inspect`, che mostra le colonne del dataset e verifica che il formato sia riconosciuto. Per eventuali dataset ad accesso condizionato (Gated Datasets) occorre accettare le condizioni sulla rispettiva pagina di Hugging Face e assicurarsi che la variabile d'ambiente HF_TOKEN sia impostata correttamente.

Output in `stats_out/` (modificabile con `--out`):

- `conversazioni.csv`: una riga per conversazione;
- `summary.json`: statistiche e metadati per la riproducibilità (commit, seed, data);
- `distribuzioni.png`: grafico delle distribuzioni (solo con `--plot`).

## `export_conversations.py`

Seleziona le conversazioni lunghe (per numero di turn, token e modello) ed estrae un campione casuale riproducibile, salvato in formato JSONL (una conversazione per riga), con i messaggi nel formato `{"role": ..., "content": ...}`. Richiede `conversation_stats.py` nella stessa cartella.

Il dataset è letto per intero in streaming; il campione è scelto con reservoir sampling ed è quindi uniforme tra tutte le conversazioni idonee. A parità di dataset (commit), parametri, tokenizer e `--seed` il risultato è identico.

```bash
python export_conversations.py
python export_conversations.py --n 50 --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --max-tokens 24000
python export_conversations.py --dataset allenai/WildChat-1M --model-regex "" --n 100
```

Con dataset privi della colonna `model` si usa `--model-regex ""` (nessun filtro).

Output in `replay_set/` (modificabile con `--out`):

- `conversazioni.jsonl`: le conversazioni selezionate;
- `selezione.json`: parametri, commit del dataset e conteggi, da riportare nella metodologia.

## `energy_api.py`

Proxy tra Hugging Face Chat UI e llama-server. Ogni richiesta di chat viene inoltrata a llama-server; al termine della risposta viene scritta una riga in `logs/turns.jsonl` con: messaggi inviati al modello, risposta, token (totali, valutati e riusati dalla prompt cache, se llama-server li fornisce), tempi, energia del turn in Wh e metadati della sessione. La scrittura avviene dopo `tracker.stop()`, quindi fuori dalla finestra di misura dell'energia.

Avvio (porta 9000, come nella configurazione di Chat UI):

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

Campi obbligatori: `participant_id`, `task_id`, `condition` (`"baseline"` o `"GE"`) e `model` (etichetta del modello). Gli altri campi (ad es. `notes`) sono salvati in `sessions.jsonl`. Senza una sessione impostata i turn sono registrati comunque, con `session_id` `unassigned-...`.

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

### File di log

- `turns.jsonl`, un record per turn: `session_id`, `participant_id`, `task_id`, `condition`, `model_label`, `segment`, `turn_index`, `turn_in_segment`, `conversation_id`, `is_ge_request`, `n_messages`, `request_params`, `messages_sha256`, `request_messages`, `response`, `reasoning`, `finish_reason`, `completed`, `error`, `prompt_tokens`, `prompt_tokens_evaluated`, `prompt_tokens_cached`, `completion_tokens`, `usage`, `timings`, `ttft_ms`, `wall_ms`, `energy_wh` (`cpu`, `gpu`, `ram`, `total`, `duration_s`, `emissions_kg`), `tracking_mode`, `codecarbon_run_id`.
- `sessions.jsonl`, eventi di sessione (`start`, `end`, `auto_start`): metadati, proprietà di llama-server (modello caricato, contesto, build), versione di CodeCarbon, esito del task (`task_completed`) e numero di richieste GE.

Un nuovo *segmento* inizia con la richiesta successiva a quella di riepilogo GE, cioè quando il partecipante avvia la nuova conversazione. Un `finish_reason` pari a `length` indica che la risposta è stata interrotta dal limite di token, tipico dei cicli ripetitivi (cycling).

### Avvertenze

- Perimetro della misura.Con `TRACKING_MODE=process` CodeCarbon cerca di isolare il consumo del processo corrente, cioè del proxy, mentre l'inferenza avviene nel processo di llama-server; con `machine` (impostazione predefinita di CodeCarbon) si misura l'intera macchina, comprese le attività di fondo, il cui consumo a riposo va misurato e sottratto. Prima delle misure definitive si consiglia di confrontare i valori con una misura indipendente (NVML per le GPU, RAPL per la CPU o un wattmetro).
- Richieste ausiliarie. Se Chat UI invia richieste che non sono turn dell'utente (ad es. la generazione del titolo), esse passano dal proxy, consumano energia e vengono registrate: si riconoscono da `n_messages` e `request_params`, e conviene disattivarle.
- Riconoscimento della richiesta GE. Si basa su `GE_MARKER`: va verificato che il prompt standardizzato fornito ai partecipanti lo contenga.

## Riproducibilità

- `models/manifest.json`: quali pesi (repository, commit, SHA-256).
- `summary.json` e `selezione.json`: quali dati (dataset, commit, seed, parametri, data).
- `sessions.jsonl`: quale configurazione (build di llama.cpp, contesto, versione di CodeCarbon, modalità di tracking, GE marker).

Questi file vanno conservati insieme ai risultati.

## Dati personali e licenze

- I log delle sessioni contengono testi scritti dai partecipanti: si usano identificativi pseudonimi (P01, P02, …), il modulo di consenso va aggiornato in modo da includere la registrazione delle conversazioni, e prima di pubblicare i log vanno rimossi i dati identificativi (ad es. nome utente e nome della macchina nell'output del terminale).
- CodeChat e WildChat sono distribuiti con licenza ODC-BY: vanno citati e non si deve tentare di identificare le persone. 