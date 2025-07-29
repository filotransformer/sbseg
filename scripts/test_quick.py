#!/usr/bin/env python3
"""
Script de teste rápido para verificar se o experimento está funcionando.
"""

import sys
import numpy as np
import pandas as pd

print("Iniciando teste rápido...")

# Verificar se o dataset existe
try:
    df = pd.read_csv("datasets/pheme/pheme_all.csv")
    print(f"✓ Dataset carregado: {len(df)} amostras")
    print(f"✓ Colunas: {list(df.columns)}")
    print(f"✓ Distribuição: {df['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"✗ Erro ao carregar dataset: {e}")
    sys.exit(1)

# Testar imports
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    print("✓ Todos os imports funcionando")
except Exception as e:
    print(f"✗ Erro nos imports: {e}")
    sys.exit(1)

# Teste simples de TF-IDF
try:
    texts = df['text'].head(100).values
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts)
    print(f"✓ TF-IDF funcionando: shape {X.shape}")
except Exception as e:
    print(f"✗ Erro no TF-IDF: {e}")

print("\n✅ Teste rápido concluído com sucesso!")
print("O ambiente está pronto para executar o experimento completo.")