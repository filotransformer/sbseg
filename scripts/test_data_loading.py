#!/usr/bin/env python3
"""
Script para testar o carregamento dos dados com TAGs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from pheme_real_cascades_experiment_tags import load_pheme_tags_data

def test_data_loading():
    print("🔍 Testando carregamento de dados...")
    
    try:
        # Tenta carregar os dados
        X_semantic, X_phylo, y, phylo_cols = load_pheme_tags_data()
        
        if X_semantic is None:
            print("❌ Falha ao carregar dados!")
            return False
        
        print("✅ Dados carregados com sucesso!")
        print(f"\nInformações do dataset:")
        print(f"  - Features semânticas: {X_semantic.shape}")
        print(f"  - Features filogenéticas: {X_phylo.shape}")
        print(f"  - Labels: {y.shape}")
        print(f"  - Número de features filogenéticas: {len(phylo_cols)}")
        print(f"  - Distribuição de classes: {np.bincount(y)}")
        
        # Verifica integridade
        assert X_semantic.shape[0] == X_phylo.shape[0] == y.shape[0], "Número de amostras inconsistente!"
        assert X_semantic.shape[1] == 384, "Dimensão incorreta para embeddings semânticos!"
        assert X_phylo.shape[1] >= 70, "Número insuficiente de features filogenéticas!"
        
        print("\n✅ Todos os testes passaram!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loading()