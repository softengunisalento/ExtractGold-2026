# Gold Extraction – strumenti per l'esperimento

*English version: [README.md](README.md)*

## Architettura

```
Hugging Face Chat UI  ──►  energy_api.py (proxy)  ──►  llama-server  ──►  modello GGUF
      :5173                 FastAPI + CodeCarbon         :8080
        │                        :9000
  MongoDB :27017
  (contesto delle conversazioni di Chat UI)
```

- Hugging Face Chat UI è in esecuzione sulla porta 5173 ed è l'interfaccia usata dai partecipanti.
- `energy_api.py` è il proxy sulla porta 9000. Inoltra ogni richiesta al modello, misura l'energia di ciascun turn con CodeCarbon e registra le conversazioni. Inoltra i messaggi esattamente come li invia Chat UI (che manda già l'intera cronologia); `logs/turns.jsonl` registra il testo effettivamente inviato al modello. Non accede a MongoDB.
- `llama-server` (llama.cpp) esegue il modello sulla porta 8080.
- MongoDB è in esecuzione sulla porta 27017 e conserva le conversazioni di Chat UI.
- L'intera architettura viene eseguita in un ambiente virtuale Python (`venv`).

File di configurazione di Chat UI:

- `.env.local` contiene la configurazione locale di Chat UI. Va collocato nella cartella `chat-ui` con il nome `.env.local`. Tre impostazioni sono essenziali: l'URL di base deve puntare al proxy (porta 9000); la memoria della conversazione deve essere attiva (ENABLE_CONVERSATIONS=true, DISABLE_CONVERSATION_MEMORY=false), perché con i modelli locali Chat UI funziona per impostazione predefinita in modalità stateless e non conserva il contesto; e la generazione dei titoli delle conversazioni deve essere disattivata (LLM_SUMMARIZATION=false), altrimenti Chat UI invia una richiesta in più al modello al primo messaggio di ogni conversazione.
- `commands.txt` elenca i comandi che avviano l'intera architettura.

## Contenuto

| File | Scopo |
|---|---|
| `download_models.py` | Seleziona i modelli dai metadati di Hugging Face, congela la selezione in documenti datati e scarica i file GGUF, registrando un manifest. |
| `conversation_stats.py` | Calcola statistiche su turn e token di dataset di conversazioni con LLM. |
| `export_conversations.py` | Estrae da un dataset le conversazioni selezionate per il replay offline. |
| `build_replay_set.py` | Costruisce il replay set congelato dai tre dataset scelti, con token per modello, controllo delle finestre di contesto e stratificazione. |
| `energy_api.py` | Proxy tra Chat UI e llama-server: misura l'energia (CodeCarbon) e registra le conversazioni. |

## Setup da zero

Il repository contiene solo il codice di questo lavoro. Chat UI, llama.cpp, i modelli e l'ambiente virtuale non sono inclusi: si installano una volta, seguendo questi passi. I comandi sono per la workstation Linux; su Windows i passi sono gli stessi, tranne i percorsi e l'attivazione dell'ambiente virtuale. Si esegue tutto dalla radice del repository, salvo diversa indicazione.

Struttura dopo il setup (i dati generati sono ignorati da Git; documentazione e configurazione sono versionate):

```
<repository>/
├── chat-ui/             passo 4
├── llama.cpp/           passo 5, con la compilazione in llama.cpp/build/
├── models/              passo 6, una cartella per modello: models/<nome>/<file>.gguf
├── selezioni/           passo 6, documenti congelati della selezione (versionati in Git)
├── venv/                passo 2
├── logs/, emissions/    creati durante l'esecuzione
├── energy_api.py, download_models.py, conversation_stats.py, export_conversations.py
├── build_replay_set.py
├── dataset_cache/       creato da build_replay_set.py prepare (ignorato da Git)
├── replay_set/          creato da build_replay_set.py select (ignorato da Git)
└── commands.txt, requirements.txt, README.md
```

**1. Prerequisiti.** Git; Python 3.10 o superiore; Node.js (provato con v24.21.0); CMake e un compilatore C++ (`build-essential`); il driver NVIDIA e il CUDA toolkit (`nvidia-smi` deve elencare le GPU e `nvcc --version` deve funzionare); MongoDB Community Edition, installato seguendo le istruzioni ufficiali per il sistema operativo; un account Hugging Face con un token di tipo Read.

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
git checkout <commit-o-tag>    #per verificare il numero di build da sostituire: git -C llama.cpp describe --tags
`#----------------------------------------------------------------------*`
#Verifica se CUDA-toolkit è  installato:
nvcc --version
#Se è installato indica il percorso a CMake:
export PATH=/usr/local/cuda/bin:$PATH
#Se non è installato:
sudo apt update
sudo apt install nvidia-cuda-toolkit #il nome potrebbe cambiare a seconda della versione di linux in uso
rm -rf build
#---------------------------------------------------------------------------
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
cd ..
```

I nomi delle opzioni cambiano tra le versioni: si controlla `docs/build.md` della versione in uso. Senza il supporto CUDA la compilazione gira su CPU e `-ngl 99` non ha effetto: all'avvio il log del server deve elencare le GPU.

**6. Modelli.** Prima si controlla lo spazio libero.

```bash
read -rs HF_TOKEN && export HF_TOKEN
#dopo aver lanciato questo comando incolla il token HF e premi invio, per verificare lancia echo $HF_TOKEN
python download_models.py --dry-run                         # seleziona, scrive i documenti congelati, mostra il piano di download
python download_models.py                                   # seleziona, scrive i documenti congelati e scarica
```

La selezione si fa una volta sola: per scaricare di nuovo la stessa lista si usa `python download_models.py --da-lista selezioni/candidati_<data>.json`.

**7. Verifica dell'intera catena.** Si avviano i servizi come in `commands.txt`, con una cartella di log di prova e, per il modello, `--alias <folderName>`. Poi:

```bash
curl http://127.0.0.1:8080/props                # il modello e la dimensione del contesto
curl http://127.0.0.1:9000/v1/models            # deve mostrare l'id assistente-locale
```

Si apre una sessione di prova (vedi Sessione sperimentale), si scrivono due messaggi in una conversazione nuova di Chat UI e si chiude la sessione. Nella cartella di log di prova `turns.jsonl` deve contenere due record con `auxiliary: false`, e il record di chiusura di `sessions.jsonl` deve mostrare `"turns": 2` e `"aux_requests": 0`. Si elimina la cartella di prova prima delle sessioni vere.

## `download_models.py`

Uno script che seleziona i modelli, congela la selezione e li scarica. Lanciato senza argomenti fa, in sequenza:

1. **Selezione** dai metadati del Hugging Face Hub (parametri, chat template, lingue, contesto, relazioni con altri modelli, download). Le organizzazioni che pubblicano i modelli «ufficiali» sono elencate in `ORG_PREDEFINITE` (o in un file passato con `--orgs-file`). Un modello è candidato solo se esiste un GGUF con un'etichetta a 4 bit pubblicato da un quantizzatore fidato. Sono esclusi i modelli non autoregressivi (diffusione), perché l'energia per turn non sarebbe confrontabile con quella dei modelli autoregressivi. I modelli reasoning sono ammessi solo se il chat template espone un interruttore che disattiva il ragionamento (ad esempio `enable_thinking=false` o `thinking_budget=0`); un modello che offre solo `reasoning_effort` è escluso. L'interruttore trovato è registrato nel documento congelato (`reasoning.interruttori`) e va applicato a llama-server in fase di avvio (vedi `commands.txt`).
2. **Congelamento** in `selezioni/`, con la data nel nome e senza mai sovrascrivere: `candidati_<data>.json` (criteri, versioni, contatori, candidati con il piano di download, esclusi con il motivo) e `candidati_<data>.txt` (lo stesso, leggibile).
3. **Download** dei file `.gguf` in `models/<nome>/`, pronti per `llama-server -m <file>`. Un download interrotto riprende da dove si era fermato. I file sono scaricati dal commit esatto del repository e, al termine, se ne verifica lo SHA-256 rispetto a quello dichiarato dal Hub.

```bash
python download_models.py                  # selezione + documenti congelati + download
python download_models.py --solo-selezione # solo selezione e documenti congelati
python download_models.py --dry-run        # selezione e documenti, poi il piano di download, senza scaricare
python download_models.py --da-lista selezioni/candidati_<data>.json            # nessuna selezione: scarica una lista già congelata
python download_models.py --da-lista selezioni/candidati_<data>.json --only <nome>
python download_models.py --list           # mostra il piano di download ed esce
python download_models.py --max-modelli 10 --max-coding 5 --quant Q4_K_M,UD-Q4_K_XL
python download_models.py --no-verify      # salta il calcolo dello SHA-256 locale
python download_models.py --update         # usa l'ultimo commit anche se il manifest o la lista ne fissano un altro (i pesi cambiano)
python download_models.py --force          # prosegue anche se lo spazio libero non basta per tutti i modelli selezionati
```

I criteri e le soglie sono opzioni dello script (`--help`) e sono scritti nel documento congelato. Il token in `HF_TOKEN` (o `HUGGING_FACE_HUB_TOKEN`) è usato in entrambe le fasi; senza, le richieste sono anonime, con limiti più stretti.

Output:

- `selezioni/candidati_<data>.json` e `.txt`: la selezione congelata, da conservare insieme ai risultati e riportare nella metodologia;
- `selezioni/esito_download_<data>.txt`: l'esito del download, scritto dopo aver scaricato;
- `models/manifest.json`: repository, quantizzazione, commit, data del download, modello da cui deriva e, per ciascun file, nome, dimensione, SHA-256 e tipo dei tensori letto dall'intestazione GGUF. Va conservato insieme ai risultati e riportato nella metodologia per indicare esattamente quali pesi sono stati usati. Il commit è fissato: vale quello del manifest, in mancanza quello della lista congelata, altrimenti l'ultimo (`--update` passa all'ultimo).

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

## `build_replay_set.py`

Costruisce il replay set congelato dai tre dataset scelti (`allenai/WildChat-1M`, `shibing624/sharegpt_gpt4`, `Suzhen/CodeChat`) in cinque comandi: `prepare` e `tokenize` costruiscono la cache, `sweep` sceglie la soglia dei token, `select` congela il set e `verify` lo ricontrolla:

- `prepare`: normalizza i tre formati, elimina conversazioni vuote, strutturalmente non valide e duplicate, tiene solo le conversazioni in inglese (fastText lid.176 sui messaggi utente), applica il filtro sui turni e scrive la cache normalizzata;
- `tokenize`: conta i token di prompt con il tokenizer di ogni modello del benchmark, applicando il chat template e lo stesso interruttore del reasoning usato negli esperimenti;
- `sweep`: varia `--min-tokeni` su una griglia e riporta le conversazioni che sopravvivono per fonte, gli strati che raggiungono `--n-per-strato`, la dimensione del campione risultante e la distribuzione dei token di prompt (p10/p50/p90/p99) con cui giustificare la soglia; la tabella è scritta in `dataset_cache/sweep.csv` (`--out` cambia il percorso);
- `select`: applica la soglia minima di token e la finestra di contesto, costruisce il campione comune, stratifica per fasce di turni e token e congela il set; `--n-per-strato 0` mantiene tutte le conversazioni idonee, senza limite per strato;
- `verify`: ricontrolla hash, soglie, finestre e manifest.

La tokenizzazione è in cache separata dalle soglie: cambiare `--min-turni`, `--min-tokeni`, `--solo-coding` o `--solo-informatica` in `select` non richiede di ritokenizzare. `sweep` è lo strumento per scegliere `--min-tokeni`: stampa i percentili dei token di prompt del corpus idoneo e, per ogni soglia, le conversazioni che sopravvivono per fonte, gli strati che raggiungono `--n-per-strato`, la dimensione del campione risultante e mediana/p90 dei prompt trattenuti; `--dettaglio <soglia>` elenca gli strati sotto target. La soglia deve mantenere fattibile il disegno previsto (ogni fonte e le principali fasce di turni/token con almeno `--n-per-strato`) e al tempo stesso allontanare le conversazioni dai prompt brevi; `4000` è stato scelto dal `sweep`, non è un ottimo misurato. I tokenizer e le revisioni fissate dei dieci modelli sono in `MODELLI`, copiati da `selezioni/candidati_2026-10-19.json`: aggiornare quella lista e le revisioni di `FONTI` se cambiano la selezione congelata o i dataset. I due repository Llama sono gated, quindi serve `HF_TOKEN` con la licenza accettata. Il set congelato usa `10` turni e `4000` token; rieseguire `select` per provare altre soglie.

`--solo-coding` tiene solo le conversazioni di programmazione: una conversazione è coding se un qualsiasi messaggio contiene un blocco di codice con un linguaggio di programmazione riconoscibile (o con righe di codice nel corpo), oppure una riga di codice che corrisponde a uno dei pattern di `SEGNALI_FORTI["riga"]`. Contano anche i messaggi dell'assistente, perché molte domande di programmazione non incollano codice ma la risposta sì. CodeChat serve da controllo esterno, non da fonte (i suoi id sono già visti in WildChat): delle 387 conversazioni WildChat a 4000 che compaiono anche in CodeChat, il filtro ne recupera l'82-83% e segnala lo 0,5% delle altre. Il campione manuale è in prevalenza coding; qualche conversazione non coding entra quando l'assistente usa un blocco XML/JSON come dati. A 4000 il filtro coding lascia 523 conversazioni (333 WildChat + 190 sharegpt_gpt4) su 3.081, su tutti i 18 strati; a 5000 ne lascia 445, a 6000 372 e a 8000 254. Il filtro è registrato in `selezione.json` sotto `parametri.solo_coding` e nell'imbuto per fonte come `coding_ok`.

`--solo-informatica` allarga il criterio alle conversazioni che trattano un tema di informatica/IT anche senza incollare codice. Guarda solo il primo messaggio utente, richiede almeno due segnali tecnici distinti (algoritmi e strutture dati, sistemi operativi, reti, basi di dati, sicurezza, ingegneria del software, AI/ML, web/mobile, hardware) oppure un segnale con una richiesta esplicita di spiegazione, implementazione, debug o confronto, ed esclude i task creativi, commerciali, legali, medici, matematici/fisici, di traduzione e di carriera. Include il filtro codice. A 4000 lascia 563 conversazioni contro 523 di `--solo-coding`; a 3000, 1.042 contro 925; a 5000, 479 contro 445; a 6000 e 8000, 404 e 273. A 4000 il campione stratificato con `--n-per-strato 5` ha 80 conversazioni (1.669 turni), 3 senza codice (ingegneria del software e AI); con `--n-per-strato 6` ne ha 94 (1.928 turni). Il set congelato usa `--n-per-strato 0`: tutte le 563 conversazioni idonee (9.804 turni; 367 WildChat + 196 sharegpt_gpt4, 523 con codice e 40 senza), su 18 strati. Lo SHA-256 di `replay_set/conversazioni.jsonl` è `44b790a91ac4fc903da8282fafd444d13835ec3a8a707f16df3bacde386a276d`. Il filtro è registrato in `selezione.json` sotto `parametri.solo_informatica` e `parametri.criterio_coding`; `--solo-coding` resta l'opzione più stretta, solo codice.

```bash
# 1. normalizza una volta (rieseguire solo se cambia --min-turni)
python build_replay_set.py prepare --min-turni 10 --lingua en
# 2. conta i token una volta per modello (ripartibile, un JSONL per modello)
python build_replay_set.py tokenize
# 3. scegli --min-tokeni: copertura di fonti e strati a più soglie
#    (6 è un esempio di target per strato; sweep scrive dataset_cache/sweep.csv)
python build_replay_set.py sweep --min-turni 10 --solo-informatica --n-per-strato 6 --dettaglio 4000
# 4. congela il set: 0 mantiene tutte le conversazioni idonee (563 nel set corrente)
python build_replay_set.py select --min-turni 10 --min-tokeni 4000 --solo-informatica --n-per-strato 0
# 5. verifica il set congelato
python build_replay_set.py verify
```

Output:

- `dataset_cache/` (ignorata da Git): corpus normalizzato, conteggi dei token e manifest di `prepare`/`tokenize`;
- `replay_set/conversazioni.jsonl`: le conversazioni congelate con `messages`, `token_prompt` per modello, fascia turni/token e hash del contenuto;
- `replay_set/selezione.json`: revisioni di dataset e tokenizer, SHA-256 dello script, soglie, imbuto per fonte e per modello, strati, SHA-256 dell'output;
- `replay_set/strati.csv`: conteggi per strato.

`--contesto-runtime` vale `32768` per impostazione predefinita, lo stesso `-c` usato da llama-server, quindi il controllo della finestra usa la finestra reale dell'esperimento, non quella dichiarata. CodeChat deriva da WildChat: `prepare` deduplica sull'id di fonte, dando precedenza alla conversazione completa di WildChat. Usa `--limite` per una prova ridotta e `--dry-run` in `select` per vedere l'imbuto senza scrivere.

Il conteggio dei token usa il tokenizer e il chat template del repository Hugging Face originale, mentre i file GGUF provengono da quantizzatori terzi. Prima di congelare, validare un campione contro il llama-server in esecuzione (`POST /apply-template` e `POST /tokenize`) e riportare lo scarto nella metodologia. Se un template non accetta il ruolo system, `tokenize` piega il system nel primo messaggio utente: il replay deve applicare la stessa regola.

La garanzia sulla finestra di contesto riguarda la conversazione registrata, cioè i messaggi dell'assistente presenti nel dataset. Se il replay lascia generare nuove risposte, il contesto ai turni successivi dipende dalle lunghezze generate: mantenere lo stesso `-c` e una riserva di sicurezza, e controllare i log di esecuzione. Le conversazioni WildChat con `toxic=true` sono escluse per impostazione predefinita (`--includi-tossici` le mantiene); quelle con `redacted` restano, salvo `--escludi-redatti`. Per accelerare la prima passata, `prepare --prefiltro-byte N` scarta le conversazioni il cui testo grezzo, più un margine di sicurezza, non può raggiungere `N` token. La soglia è in byte UTF-8, non in caratteri: per i tokenizer byte-level usati qui vale `token <= byte`, mentre codice e CJK possono avvicinarsi a un token per carattere, quindi una soglia in caratteri scarterebbe conversazioni valide. `--lingua` (default `en`) rende il replay set interamente in inglese: `prepare` rileva la lingua dei messaggi utente con fastText `lid.176` su tutte e tre le fonti (confidenza `--lingua-soglia 0.80`, primi `--lingua-caratteri 2000` caratteri) e scarta le altre. La colonna `language` di WildChat serve solo a misurare l'accordo con il rilevatore; `sharegpt_gpt4` non ha una colonna lingua e le etichette per messaggio di CodeChat non sono affidabili (il tag del repository dice `en`, ma il contenuto è multilingua), quindi il rilevatore è l'unico criterio comune alle tre fonti. Le conversazioni con meno di 20 caratteri di testo utente non sono valutabili e vengono scartate. `--tutte-le-lingue` disattiva il filtro e va usato solo per prove. Per impostazione predefinita la soglia minima di token deve valere per tutti i modelli (`--soglia-token-modo tutti`) e le fasce usano il massimo tra i modelli (`--misura-strato max`). `replay_set/` è ignorata da Git come gli altri output: va conservata insieme ai risultati sperimentali e `selezione.json` va citato nella metodologia.

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
| `EMISSIONS_DIR` | cartella di `emissions.csv` di CodeCarbon (`./emissions`) |
| `LOG_DIR` | cartella dei log (`./logs`) |
| `TRACKING_MODE` | modalità di CodeCarbon, `process` o `machine` (`process`); vedi Avvertenze |
| `GE_MARKER` | frase che identifica la richiesta di riepilogo GE, senza distinzione tra maiuscole e minuscole (`riepilogo conciso del contesto`) |
| `LOG_FULL_REQUEST` | `1` = salva tutti i messaggi inviati al modello, `0` = solo l'ultimo (`1`) |
| `REQUEST_USAGE` | `1` = aggiunge `stream_options.include_usage` alla richiesta, così llama-server restituisce sempre token e tempi (`1`) |
| `AUX_NO_CONV_ID` | `1` = le richieste senza identificativo di conversazione (ad es. il titolo generato da Chat UI) sono ausiliarie: inoltrate, registrate con `auxiliary: true`, non contate come turn e non misurate (`1`) |
| `MODEL_ID` | nome del modello esposto a Chat UI in `/v1/models` e mostrato ai partecipanti; volutamente neutro, il modello vero è in `model_label` (`assistente-locale`) |

### File di log

- `turns.jsonl`, un record per turn: `session_id`, `participant_id`, `task_id`, `condition`, `model_label`, `segment`, `turn_index`, `turn_in_segment`, `conversation_id`, `is_ge_request`, `auxiliary`, `n_messages`, `request_params`, `messages_sha256`, `request_messages`, `response`, `reasoning`, `finish_reason`, `completed`, `error`, `prompt_tokens`, `prompt_tokens_evaluated`, `prompt_tokens_cached`, `completion_tokens`, `usage`, `timings`, `ttft_ms`, `wall_ms`, `started_at`, `overlapped`, `energy_wh` (`cpu`, `gpu`, `ram`, `total`, `duration_s`, `emissions_kg`), `tracking_mode`, `codecarbon_run_id`.
- `sessions.jsonl`, eventi di sessione (`start`, `end`, `auto_start`): metadati, etichetta del modello (dichiarata e ricavata dal server) e avvisi, proprietà di llama-server (modello caricato, contesto, build), versione di CodeCarbon, esito del task (`task_completed`) e numero di richieste GE e ausiliarie.

### Avvertenze

- Perimetro della misura. Con `TRACKING_MODE=process` CodeCarbon cerca di isolare il consumo del processo corrente, cioè del proxy, mentre l'inferenza avviene nel processo di llama-server; con `machine` (impostazione predefinita di CodeCarbon) si misura l'intera macchina, comprese le attività di fondo, il cui consumo a riposo va misurato e sottratto.
- Riconoscimento della richiesta GE. Si basa su `GE_MARKER`: va verificato che il prompt standardizzato fornito ai partecipanti lo contenga.
s