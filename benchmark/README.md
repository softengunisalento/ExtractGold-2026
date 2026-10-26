# Benchmark delle strategie di gestione del contesto (prima versione)

Replay a storia fissa delle conversazioni congelate con le strategie S0–S7, su un modello alla volta, misurando
l'energia di ogni richiesta con CodeCarbon. Le scelte sono descritte nei documenti «Strategie di gestione del
contesto: scelte di utilizzo» e «Benchmark: decisioni su infrastruttura e misure».

## File

| File | Dove | Cosa fa |
|---|---|---|
| `config.toml` | entrambe le macchine | tutti i parametri; **lo stesso file** su entrambe (il runner controlla l'impronta) |
| `misura.py` | macchina di misura (M1) | `prepara`: llama.cpp al commit congelato, librerie, modelli, modelli ausiliari. `esegui`: servizio di misura |
| `runner.py` | PC di controllo | guida il benchmark e salva i risultati; riprende da dove si era fermato |
| `strategie.py` | PC di controllo | le strategie S0–S7 |
| `configurazione.py`, `dati.py`, `client_misura.py`, `risultati.py` | moduli comuni | configurazione, file di input, comunicazione con M1, risultati |
| `requirements_misura.txt` | M1 | librerie della macchina di misura |
| `test/` | qualsiasi | prova simulata (llama-server, CodeCarbon e modelli ausiliari finti) |

Il runner usa solo la libreria standard: sul PC di controllo basta Python 3.11 o superiore.

## Macchina di misura (M1)

```bash
source venv/bin/activate
cd benchmark
python misura.py prepara      # una volta, con internet (HF_TOKEN impostato per i modelli ad accesso condizionato)
python misura.py esegui       # servizio di misura; funziona anche senza internet
```

`prepara` si può ripetere a pezzi: `--solo llama`, `--solo librerie`, `--solo modelli`, `--solo ausiliari`.
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
python runner.py                              # indirizzo di M1 da rete.indirizzo_misura
python runner.py --misura http://<ip-di-M1>:9100
python runner.py --solo-verifiche             # solo trigger e verifiche, senza misure
```

Se si interrompe, si rilancia lo stesso comando: le unità (modello, conversazione, strategia) concluse si
saltano, quella interrotta si rifà da capo.

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
`sessioni.jsonl`, `esecuzione.json`.

## Prova simulata

```bash
python test/prova_simulata.py
```

Controlla le strategie sui contesti esatti dell'esempio del documento e fa girare servizio e runner veri con
llama-server, CodeCarbon e modelli ausiliari finti. Non verifica l'hardware, CodeCarbon vero, le librerie dei
modelli ausiliari né llama-server vero: quelle si provano su M1.

La prova gira solo su Linux o macOS (per esempio su M1): al posto di llama-server crea un piccolo eseguibile di
shell, che Windows non sa avviare.
