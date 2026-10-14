#!/usr/bin/env python3
import os
import sys
from datetime import datetime
from huggingface_hub import HfApi

def main():
    api = HfApi()
    oggi = datetime.now().strftime("%Y-%m-%d")
    out_file = f"dataset_discovery_{oggi}.txt"
    
    print("Ricerca di dataset conversazionali open-source in corso...")
    
    # Interroga l'API per i dataset più popolari.
    try:
        datasets = api.list_datasets(sort="downloads", limit=2000)
    except Exception as e:
        sys.exit(f"Errore di connessione a Hugging Face: {e}")

    # Criteri di ricerca
    open_licenses = ["license:mit", "license:apache-2.0", "license:cc-by-4.0", "license:odc-by", "license:openrail"]
    keywords = ["chat", "conversational", "dialogue", "sharegpt", "multiturn", "rlhf"]
    
    candidati = []
    
    for ds in datasets:
        tags = [t.lower() for t in getattr(ds, 'tags', [])]
        ds_id = ds.id.lower()
        downloads = getattr(ds, 'downloads', 0)
        
        # 1. Filtro rumore: ignora dataset con meno di 10.000 download (poco affidabili)
        if downloads < 10000:
            continue
            
        # 2. Filtro licenza: deve avere una licenza permissiva esplicita
        has_open_license = any(l in tags for l in open_licenses)
        if not has_open_license:
            continue
            
        # 3. Filtro dominio: deve indicare natura conversazionale
        is_chat = any(kw in tags or kw in ds_id for kw in keywords)
        if not is_chat:
            continue
            
        # 4. Filtro manuale preventivo
        if "lmsys" in ds_id:
            continue
            
        # Raccoglie i metadati utili
        licenze_trovate = [t.replace("license:", "") for t in tags if t.startswith("license:")]
        
        candidati.append({
            "id": ds.id,
            "downloads": downloads,
            "license": licenze_trovate[0] if licenze_trovate else "Sconosciuta"
        })
        
        # Limita ai top 25 per avere un pool gestibile manualmente
        if len(candidati) >= 25:
            break

    # Generazione del report con i comandi pronti
    with open(out_file, "w", encoding="utf-8") as fh:
        fh.write("=== REPORT DISCOVERY DATASET ===\n")
        fh.write(f"Data: {oggi}\n")
        fh.write("Filtri applicati: Download >= 10k, Licenza Open (MIT, Apache, CC-BY, ecc.), Tag Conversazionali.\n\n")
        fh.write("Istruzioni: Copia e incolla il comando di ispezione nel terminale.\n")
        fh.write("Se lo script rileva correttamente le colonne 'role' e 'content'/'text', puoi procedere al campionamento.\n")
        fh.write("========================================================================================\n\n")
        
        for i, c in enumerate(candidati, 1):
            fh.write(f"{i}. {c['id']}\n")
            fh.write(f"   Downloads: {c['downloads']:,} | Licenza: {c['license']}\n")
            fh.write(f"   CMD Ispezione: python conversation_stats.py --dataset {c['id']} --inspect --tokenizer Qwen/Qwen2.5-32B-Instruct --min-turns 10 --min-tokens 4000\n")
            fh.write(f"   CMD Test 1k:   python conversation_stats.py --dataset {c['id']} --tokenizer Qwen/Qwen2.5-32B-Instruct --sample 1000 --min-turns 10 --min-tokens 4000\n")
            fh.write("-" * 90 + "\n")

    print(f"Completato! Trovati {len(candidati)} dataset idonei.")
    print(f"Apri il file '{out_file}' per vedere la lista e i comandi di ispezione.")

if __name__ == "__main__":
    main()