"""
=============================================================================
Progetto: Image Quality Assessment (IQA) con Foundation Models
Fase: 5.5 - Confronto Encoder/Layer/Feature (Ablation Finale)
Ruolo: Model Engineer

Descrizione:
Questo script genera i grafici definitivi per la stesura del report finale.
Analizza i risultati estratti durante le fasi di training e validazione 
(salvati in formato CSV) per produrre due visualizzazioni fondamentali:

1. Trend dei Layer (layer_trend_comparison.png): 
   Mostra l'andamento della metrica SRCC strato per strato sul dataset TID2013.
   Evidenzia la differenza di comportamento tra l'estrazione tramite token [CLS] 
   (DINOv2) e il Global Average Pooling delle patch (SigLIP2).

2. Ablation Study (ablation_pipal.png):
   Confronta le performance di generalizzazione (Zero-Shot vs MLP vs Cross-Attention)
   sul dataset di test PIPAL, dimostrando la superiorità della fusione dinamica.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Imposta lo stile del grafico
sns.set_theme(style="whitegrid")

def plot_layer_trend():
    print("Generazione grafico andamento layer...")
    # Sostituisci col percorso corretto del file di Giuli
    df = pd.read_csv("phase2_tid2013_results.csv")
    
    # Rimuoviamo la riga "mean" per avere solo i layer numerici
    df = df[df['layer'] != 'mean'].copy()
    df['layer'] = pd.to_numeric(df['layer'])

    # Escludiamo il layer 0 per mostrare solo i veri blocchi Transformer (1-24)
    df = df[df['layer'] > 0]

    # Modelli che vogliamo confrontare
    models_to_plot = [
        "facebook/dinov2-large",
        "google/siglip2-large-patch16-256"
    ]
    
    plt.figure(figsize=(10, 6))
    
    for model in models_to_plot:
        model_data = df[df['model'] == model]
        # Pulisci i nomi per la legenda
        label_name = "DINOv2 Large (CLS)" if "dinov2" in model else "SigLIP2 Large (Patch Mean)"
        
        plt.plot(model_data['layer'], model_data['srcc'], marker='o', linewidth=2, label=label_name)

    plt.title("IQA Performance Trends Layer by Layer (TID2013)", fontsize=14, fontweight='bold')
    plt.xlabel("Layer index", fontsize=12)
    plt.xlim(left=1)
    plt.ylabel("SRCC", fontsize=12)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    plt.savefig("layer_trend_comparison.png", dpi=300)
    print("Salvato: layer_trend_comparison.png")

def plot_ablation_results():
    print("Generazione grafico ablation Phase 3...")
    # Dati presi direttamente dai tuoi summary
    architectures = ["Zero-Shot (Fase 2)", "MLP Medium", "Advanced Attention"]
    srcc_scores = [0.579, 0.721, 0.813]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(architectures, srcc_scores, color=['#95a5a6', '#3498db', '#e74c3c'])
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')

    plt.title("Generalizzazione su PIPAL (Test Set)", fontsize=14, fontweight='bold')
    plt.ylabel("SRCC (Più alto è meglio)", fontsize=12)
    plt.ylim(0.4, 0.9)
    plt.tight_layout()
    plt.savefig("ablation_pipal.png", dpi=300)
    print("Salvato: ablation_pipal.png")

if __name__ == "__main__":
    plot_layer_trend()
    plot_ablation_results()