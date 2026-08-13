import pandas as pd

datasets = ['live', 'tid2013', 'pipal']
for ds in datasets:
    df = pd.read_csv(f"phase2_{ds}_results.csv")
    
    # Prendi solo il layer "mean" (o trova il max)
    df_mean = df[df['layer'] == 'mean'].copy()
    
    # Se hai valori negativi, mettili in assoluto
    df_mean['srcc'] = df_mean['srcc'].abs()
    
    # Ordina per SRCC
    df_mean = df_mean.sort_values(by='srcc', ascending=False)
    
    print(f"\n=== TOP MODELLI SU {ds.upper()} (Layer: Mean) ===")
    print(df_mean[['model', 'srcc']].head(4).to_string(index=False))