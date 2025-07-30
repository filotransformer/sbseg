#!/usr/bin/env python3
"""
Script para gerar visualizações dos resultados experimentais.

Gera gráficos comparativos entre Baseline e Filo-Transformer,
análise de características filogenéticas e métricas por fold.

Autor: Acauan Cardoso Ribeiro, Eduardo Luzeiro Feitosa, André Carvalho
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Configuração de estilo
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def plot_metrics_comparison(baseline_results, filo_results, save_path="results/metrics_comparison.png"):
    """
    Gera gráfico comparativo das métricas entre Baseline e Filo-Transformer.
    """
    metrics = ['Accuracy', 'AUC', 'F1-Score', 'Recall']
    
    # Dados fictícios para demonstração - substituir pelos resultados reais
    baseline_means = [0.8287, 0.8900, 0.7202, 0.7580]
    baseline_stds = [0.0181, 0.0112, 0.0367, 0.0473]
    
    filo_means = [0.8331, 0.8957, 0.7287, 0.7778]
    filo_stds = [0.0153, 0.0114, 0.0315, 0.0384]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Barras com erro
    rects1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                     label='Baseline (Semântico)', capsize=5, alpha=0.8)
    rects2 = ax.bar(x + width/2, filo_means, width, yerr=filo_stds,
                     label='Filo-Transformer', capsize=5, alpha=0.8)
    
    # Customização
    ax.set_xlabel('Métricas', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparação de Desempenho: Baseline vs Filo-Transformer', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Adiciona valores nas barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {save_path}")
    
def plot_phylogenetic_features_importance(save_path="results/phylogenetic_features.png"):
    """
    Gera gráfico de importância das características filogenéticas.
    """
    features = [
        'Padrões de Casualidade',
        'Urgência',
        'Triggers Imediatos',
        'Amplificação',
        'Manipulação',
        'Centralidade Grau',
        'Centralidade Closeness',
        'PageRank',
        'Coeficiente Clustering',
        'Assortatividade',
        'Modularidade',
        'Densidade',
        'Diâmetro',
        'Componentes Conexas',
        'Caminho Médio',
        'Transitividade'
    ]
    
    # Valores de importância (exemplo)
    importance = [463.5, 237.8, 156.2, 98.7, 45.3, 38.2, 35.1, 32.4, 
                  28.9, 25.6, 22.3, 19.8, 17.2, 15.4, 12.8, 10.1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, alpha=0.8)
    
    # Colorir as barras mais importantes
    colors = ['#e74c3c' if imp > 100 else '#3498db' if imp > 50 else '#95a5a6' for imp in importance]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Aumento Percentual em Rumores (%)', fontsize=12)
    ax.set_title('Importância das Características Filogenéticas', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Adiciona valores
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'+{val:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {save_path}")

def plot_fold_performance(save_path="results/fold_performance.png"):
    """
    Gera gráfico de desempenho por fold.
    """
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    
    # Dados de exemplo - substituir pelos reais
    baseline_auc = [0.878, 0.892, 0.901, 0.885, 0.894]
    filo_auc = [0.885, 0.898, 0.908, 0.892, 0.902]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(folds))
    width = 0.35
    
    ax.plot(x, baseline_auc, 'o-', label='Baseline', linewidth=2, markersize=8)
    ax.plot(x, filo_auc, 's-', label='Filo-Transformer', linewidth=2, markersize=8)
    
    ax.set_xlabel('Folds', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('Desempenho por Fold - Validação Cruzada 5-Fold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 0.92)
    
    # Adiciona área sombreada mostrando melhoria
    ax.fill_between(x, baseline_auc, filo_auc, alpha=0.2, color='green', 
                    where=[f > b for f, b in zip(filo_auc, baseline_auc)])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {save_path}")

def generate_all_plots():
    """
    Gera todas as visualizações.
    """
    print("🎨 Gerando visualizações dos resultados...")
    
    # Cria diretório de resultados se não existir
    Path("results").mkdir(exist_ok=True)
    
    # Gera gráficos
    plot_metrics_comparison(None, None)
    plot_phylogenetic_features_importance()
    plot_fold_performance()
    
    print("\n✅ Todas as visualizações foram geradas!")
    print("📂 Verifique os gráficos em: results/")

if __name__ == "__main__":
    generate_all_plots()