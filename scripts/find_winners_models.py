# Questo script legge i tre file CSV che hai generato per i tre dataset, 
# unisce i dati, calcola la media matematica dell'SRCC per 
# ogni modello e stampa una classifica chiara.

import pandas as pd

def main():
    print("=== CALCOLO VINCITORI FASE 1 ===")
    
    try:
        # Carica i CSV della Fase 1
        # (Assicurati che i nomi corrispondano ai file generati dalla tua collega)
        df_live = pd.read_csv("phase2_live_results.csv")
        df_tid = pd.read_csv("phase2_tid2013_results.csv")
        df_pipal = pd.read_csv("phase2_pipal_results.csv")
    except FileNotFoundError as e:
        print(f"Errore: impossibile trovare i file CSV. Dettagli: {e}")
        return

    # Uniamo tutti i DataFrame in uno solo
    df_all = pd.concat([df_live, df_tid, df_pipal])

    # Mappa i valori del registry alle loro chiavi corte
    name_mapping = {
        "facebook/dinov2-small": "dinov2_small",
        "facebook/dinov2-base": "dinov2_base",
        "facebook/dinov2-large": "dinov2_large",
        "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_small",
        "facebook/dinov3-vitb16-pretrain-lvd1689m": "dinov3_base",
        "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3_large",
        "google/siglip2-base-patch16-224": "siglip2_base",
        "google/siglip2-large-patch16-256": "siglip2_large"
    }

    # Applica la correzione: se il nome è lungo, lo fa diventare corto
    df_all['model'] = df_all['model'].replace(name_mapping)
    
    # Il paper richiede di selezionare in base alla media SRCC sui 3 dataset
    # Raggruppiamo per modello e calcoliamo la media di SRCC e PLCC
    results = df_all.groupby('model').agg({
        'srcc': 'mean',
        'plcc': 'mean'
    }).reset_index()

    # Ordiniamo in modo decrescente basandoci sull'SRCC (dal migliore al peggiore)
    results = results.sort_values(by='srcc', ascending=False)

    print("\nClassifica Finale (Media su LIVE, TID2013, PIPAL):")
    print(results.to_string(index=False))
    
    print("\nI due modelli vincitori da portare alla Fase 2 e 3 sono:")
    print(f"1. {results.iloc[0]['model']} (SRCC Medio: {results.iloc[0]['srcc']:.4f})")
    print(f"2. {results.iloc[1]['model']} (SRCC Medio: {results.iloc[1]['srcc']:.4f})")

if __name__ == "__main__":
    main()