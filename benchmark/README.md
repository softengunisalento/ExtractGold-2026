# Benchmark delle strategie di gestione del contesto (prima versione)

Replay a storia fissa delle conversazioni congelate con le strategie S0–S7, su un modello alla volta, misurando
l'energia di ogni richiesta con CodeCarbon; poi giudizio della qualità con un giudice LLM (gpt-oss-120b) ed
elaborazione dei risultati in tabelle, grafici e un rapporto Word. Le scelte sono descritte nei documenti
«Strategie di gestione del contesto: scelte di utilizzo» e «Benchmark: decisioni su infrastruttura e misure».

## File

| File | Dove | Cosa fa |
|---|---|---|
| `config.toml` | entrambe le macchine | tutti i parametri; **lo stesso file** su entrambe (il runner controlla l'impronta) |
| `misura.py` | macchina di misura (M1) | `prepara`: llama.cpp al commit congelato, librerie, modelli, giudice, modelli ausiliari. `esegui`: servizio di misura |
| `catena.py` | PC di controllo | **comando unico**: benchmark → giudice → elaborazione |
| `runner.py` | PC di controllo | guida il benchmark e salva i risultati; riprende da dove si era fermato |
| `giudice.py` | PC di controllo | giudica le risposte con il giudice su M1 (misura spenta); riprende da dove si era fermato |
| `elabora.py` | PC di controllo | tabelle CSV, grafici PNG e rapporto Word |
| `strategie.py` | PC di controllo | le strategie S0–S7 |
| `configurazione.py`, `dati.py`, `client_misura.py`, `risultati.py`, `avanzamento.py` | moduli comuni | configurazione, file di input, comunicazione con M1, risultati, riga di avanzamento |
| `requirements_misura.txt` | M1 | librerie della macchina di misura |
| `requirements_runner.txt` | PC di controllo | librerie dell'elaborazione |
| `test/` | qualsiasi | prova simulata (llama-server, CodeCarbon e modelli ausiliari finti) |

Runner e giudice usano solo la libreria standard; l'elaborazione richiede le librerie di
`requirements_runner.txt`. Sul PC di controllo serve Python 3.11 o superiore.

## Macchina di misura (M1)

```bash
source venv/bin/activate
cd benchmark
python misura.py prepara      # una volta, con internet (HF_TOKEN impostato per i modelli ad accesso condizionato)
python misura.py prepara --modelli qwen2.5-coder-32b-instruct   # solo il modello del pilota (più il giudice)
python misura.py esegui       # servizio di misura; funziona anche senza internet
```

`prepara` si può ripetere a pezzi: `--solo llama`, `--solo librerie`, `--solo modelli`, `--solo giudice`,
`--solo ausiliari`. Il giudice è il file `giudice.file` del repository `giudice.repository`, al commit
`giudice.commit`, salvato in `<cartella_modelli>/<giudice.nome>/`.
Scrive `preparazione.json` e `requirements_misura.lock.txt` (le versioni effettivamente installate).
Il lock non è ignorato da Git: conservarlo con la configurazione e i risultati del benchmark.
Se esiste già, una nuova preparazione salva un lock con timestamp invece di sovrascrivere quello
precedente; `preparazione.json` indica il percorso prodotto.

**Contatori della CPU (RAPL).** CodeCarbon misura la CPU con i contatori RAPL; se l'utente non può leggerli,
la stima invece di misurarla. `prepara` ed `esegui` controllano i permessi e lo segnalano. I file sono
`/sys/class/powercap/intel-rapl*/energy_uj`; come renderli leggibili va deciso con chi amministra la macchina.

**Durante il benchmark** M1 deve restare libera da altri programmi e aggiornamenti automatici: CodeCarbon in
modalità machine misura l'intera macchina.

## PC di controllo

```bash
cd benchmark
pip install -r requirements_runner.txt        # una volta (librerie dell'elaborazione)
python catena.py --modelli qwen2.5-coder-32b-instruct --max-conversazioni 10   # pilota: primi 2 giri
python catena.py --modelli qwen2.5-coder-32b-instruct --max-conversazioni 20   # allunga: giri 3 e 4
python catena.py                              # tutti i modelli della lista, tutte le conversazioni
python catena.py --misura http://<ip-di-M1>:9100 --modelli ...
```

`catena.py` esegue in ordine `runner.py`, `giudice.py` ed `elabora.py`; prima di partire controlla che le
librerie dell'elaborazione ci siano. Le tre fasi si possono anche lanciare da sole:

```bash
python runner.py [--modelli ...] [--solo-verifiche]   # benchmark (--solo-verifiche: solo trigger e verifiche)
python giudice.py [--modelli ...]                     # giudizi
python elabora.py [--modelli ...]                     # elaborazione
```

**Avanzamento.** Durante il benchmark e durante il giudizio la console (per esempio PowerShell) mostra una riga
con la percentuale, i turni o i giudizi fatti sul totale, il tempo trascorso e una stima del tempo residuo; alla
ripresa il lavoro già concluso conta come fatto. Il run vive nella finestra in cui è stato lanciato: se si chiude
la finestra o il PC va in sospensione si ferma, e si riprende rilanciando lo stesso comando.

**Ordine a giri e allungamento.** Le conversazioni si dividono in fasce di turni (`esecuzione.fasce_turni`:
10–11, 12–14, 15–19, 20–29, ≥30); ogni fascia si mescola con `esecuzione.seed` e ogni giro prende una
conversazione per fascia, dalla più corta alla più lunga (una fascia esaurita si salta). `--max-conversazioni N`
prende le prime N conversazioni di questa sequenza: con N = 5 un giro, con N = 10 due giri. Rilanciando con un N
più grande il run si allunga: le conversazioni già fatte non si rifanno, giudice ed elaborazione si aggiornano.
Diminuire N è rifiutato. Senza `--max-conversazioni`, un'esecuzione nuova usa tutte le conversazioni e una già
avviata mantiene il suo N. Se cambiano fasce o seed (cioè l'ordine), i risultati vanno in una cartella nuova
`<nome>_<impronta>`. Il rapporto riporta i risultati per fascia, le medie semplici e le medie pesate secondo la
quota di ogni fascia nell'intero dataset.

**Ripresa.** Se qualcosa si interrompe, si rilancia lo stesso comando. Benchmark: le unità (modello,
conversazione, strategia) concluse si saltano, quella interrotta si rifà da capo. Giudice: i giudizi già
salvati si saltano. Elaborazione: si rifà sempre da capo. Un modello si giudica solo a benchmark concluso.

**Impronte.** Il servizio di misura e il PC di controllo devono avere lo stesso `config.toml`. I risultati del
benchmark sono legati alla configurazione esclusi `[giudice]` ed `[elaborazione]`: si possono cambiare il
giudice o l'elaborazione (per esempio dopo il pilota) senza invalidare le misure. I giudizi si salvano in una
cartella per ogni configurazione del giudice, così giudizi diversi non si mescolano.

## Rete

Funziona sia sulla rete dell'università sia con uno switch dedicato: cambia solo `rete.indirizzo_misura`.
Con lo switch nessuno assegna gli indirizzi: vanno impostati a mano nel sistema operativo delle due macchine
(per esempio 192.168.50.1 su M1 e 192.168.50.2 sul PC di controllo, maschera 255.255.255.0), e `prepara` va
fatto prima, quando M1 ha ancora internet. Sulla rete dell'università conviene impostare `rete.chiave`.

## Risultati

In `risultati/<esecuzione.nome>/`: `richieste.jsonl` (una riga per richiesta: chiavi stabili, token, tempi,
energia lorda e netta per prefill e decode o per l'overhead), `testi.jsonl` (risposte), `intermedi.jsonl`
(gold prompt, riassunti, messaggi compressi, turni recuperati), `modelli.jsonl` (caricamenti e consumo a
riposo), `trigger/` e `verifiche/` (turno di attivazione per modello e conteggi), `completate.jsonl`,
`sessioni.jsonl`, `esecuzione.json`; `giudizi/<impronta del giudice>/giudizi.jsonl` (un giudizio per riga:
voti, motivazione, ragionamento, tentativi, controlli e casi non giudicabili); `elaborazione/` (tabelle CSV,
grafici PNG e `rapporto.docx`, con le sezioni A quadro dei dati, B energia, C qualità, D compromesso
energia-qualità, E statistica, F affidabilità del giudice e robustezza, G confronto tra modelli).

**Giudice.** Turni dal trigger in poi. Variante A (voti relativi, per S1–S7): storia originale fino al turno
precedente, messaggio dell'utente, risposta S0 come «Reference answer» e risposta della tecnica. Variante B
(voti assoluti, per S0–S7): la stessa storia e il messaggio con una sola risposta. Uscita JSON vincolata da
schema, fino a 2 nuovi tentativi; controlli di stabilità, identità e degrado su campioni del 5%. Le soglie
del referto (`[giudice.soglie]`) si fissano dopo il pilota: finché mancano, il referto riporta i valori senza
avvisi.

## Prova simulata

```bash
python test/prova_simulata.py
```

Controlla le strategie sui contesti esatti dell'esempio del documento e fa girare servizio, runner, giudice,
elaborazione e `catena.py` veri con llama-server (che fa anche da giudice), CodeCarbon e modelli ausiliari
finti. Non verifica l'hardware, CodeCarbon vero, le librerie dei modelli ausiliari, llama-server vero né il
giudice vero (schema JSON, parametri, memoria): quelle si provano su M1. Con `CONSERVA_PROVA=1` la cartella
della prova resta su disco.

La prova gira solo su Linux o macOS (per esempio su M1): al posto di llama-server crea un piccolo eseguibile di
shell, che Windows non sa avviare.
