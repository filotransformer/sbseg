#!/usr/bin/env python3
"""
Script para verificar a integridade completa do dataset processado.
"""

import os
import json
import pandas as pd
import pickle
from pathlib import Path

def check_file_integrity():
    """Verifica a integridade de todos os arquivos processados"""
    
    print("="*60)
    print("🔍 VERIFICAÇÃO DE INTEGRIDADE DO DATASET")
    print("="*60)
    print()
    
    processed_dir = Path("datasets/processed")
    
    # Lista de arquivos esperados
    expected_files = {
        "pheme_processed_cascades.csv": {
            "description": "Dataset básico (12 features)",
            "min_size_mb": 30,
            "check_func": lambda f: check_csv_file(f, expected_cols=12)
        },
        "pheme_processed_cascades_tags.csv": {
            "description": "Dataset com TAGs (70 features)",
            "min_size_mb": 4,
            "check_func": lambda f: check_csv_file(f, expected_cols=70)
        },
        "pheme_simplified.csv": {
            "description": "Dataset simplificado",
            "min_size_mb": 1,
            "check_func": lambda f: check_csv_file(f)
        },
        "pheme_metadata.json": {
            "description": "Metadados básicos",
            "min_size_mb": 0.001,
            "check_func": lambda f: check_json_file(f)
        },
        "pheme_metadata_tags.json": {
            "description": "Metadados TAGs",
            "min_size_mb": 0.001,
            "check_func": lambda f: check_json_file(f)
        },
        "pheme_semantic_embeddings.pkl": {
            "description": "Embeddings semânticos",
            "min_size_mb": 15,
            "check_func": lambda f: check_pickle_file(f)
        }
    }
    
    all_ok = True
    
    for filename, info in expected_files.items():
        filepath = processed_dir / filename
        print(f"\n📄 Verificando: {filename}")
        print(f"   Descrição: {info['description']}")
        
        if not filepath.exists():
            print(f"   ❌ ARQUIVO NÃO ENCONTRADO!")
            all_ok = False
            continue
        
        # Verifica tamanho
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"   Tamanho: {size_mb:.2f} MB", end="")
        
        if size_mb < info['min_size_mb']:
            print(f" ❌ (esperado >= {info['min_size_mb']} MB)")
            all_ok = False
        else:
            print(" ✅")
        
        # Verifica conteúdo
        try:
            if info['check_func'](filepath):
                print(f"   Conteúdo: ✅ Válido")
            else:
                print(f"   Conteúdo: ❌ Inválido")
                all_ok = False
        except Exception as e:
            print(f"   Conteúdo: ❌ Erro ao verificar: {str(e)}")
            all_ok = False
    
    # Verifica consistência entre arquivos
    print("\n" + "="*60)
    print("🔗 VERIFICANDO CONSISTÊNCIA ENTRE ARQUIVOS")
    print("="*60)
    
    try:
        # Carrega datasets
        df_basic = pd.read_csv(processed_dir / "pheme_processed_cascades.csv")
        df_tags = pd.read_csv(processed_dir / "pheme_processed_cascades_tags.csv")
        
        # Verifica número de linhas
        print(f"\nNúmero de cascatas:")
        print(f"   Dataset básico: {len(df_basic)}")
        print(f"   Dataset TAGs: {len(df_tags)}")
        
        if len(df_basic) == len(df_tags):
            print("   ✅ Consistente")
        else:
            print("   ❌ Inconsistente!")
            all_ok = False
        
        # Verifica labels
        if 'label' in df_basic.columns and 'label' in df_tags.columns:
            labels_match = (df_basic['label'] == df_tags['label']).all()
            print(f"\nLabels consistentes: {'✅' if labels_match else '❌'}")
            if not labels_match:
                all_ok = False
        
        # Verifica distribuição de classes
        if 'label' in df_tags.columns:
            class_dist = df_tags['label'].value_counts()
            print(f"\nDistribuição de classes:")
            print(f"   Rumores (1): {class_dist.get(1, 0)}")
            print(f"   Não-rumores (0): {class_dist.get(0, 0)}")
            print(f"   Proporção: {class_dist.get(1, 0) / len(df_tags):.1%} rumores")
        
    except Exception as e:
        print(f"\n❌ Erro ao verificar consistência: {str(e)}")
        all_ok = False
    
    # Resultado final
    print("\n" + "="*60)
    if all_ok:
        print("✅ DATASET ÍNTEGRO E PRONTO PARA USO!")
        print("\n🚀 Próximo passo: python scripts/main_experiment.py")
    else:
        print("❌ PROBLEMAS ENCONTRADOS NO DATASET!")
        print("\n⚠️  Execute: python scripts/prepare_dataset.py")
        print("   (após remover a pasta datasets/processed)")
    print("="*60)
    
    return all_ok

def check_csv_file(filepath, expected_cols=None):
    """Verifica integridade de arquivo CSV"""
    try:
        df = pd.read_csv(filepath, nrows=5)
        
        if expected_cols and len(df.columns) < expected_cols:
            print(f"   Colunas: {len(df.columns)} ❌ (esperado >= {expected_cols})")
            return False
        else:
            print(f"   Colunas: {len(df.columns)} ✅")
        
        return True
    except:
        return False

def check_json_file(filepath):
    """Verifica integridade de arquivo JSON"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return isinstance(data, dict)
    except:
        return False

def check_pickle_file(filepath):
    """Verifica integridade de arquivo pickle"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if hasattr(data, 'shape'):
            print(f"   Shape: {data.shape}")
        
        return True
    except:
        return False

if __name__ == "__main__":
    check_file_integrity()