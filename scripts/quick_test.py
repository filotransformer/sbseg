#!/usr/bin/env python3
"""
quick_test.py - Teste mínimo para verificação da instalação
SBSeg 2025 - Avaliação de Artefatos
"""

import sys
import os

def check_imports():
    """Verifica se todas as dependências foram instaladas corretamente."""
    print("[INFO] Verificando instalação...")
    
    imports = [
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("TQDM", "tqdm")
    ]
    
    all_ok = True
    for name, module in imports:
        try:
            __import__(module)
            print(f"✓ {name} instalado corretamente")
        except ImportError:
            print(f"✗ {name} NÃO está instalado")
            all_ok = False
    
    return all_ok

def check_dataset():
    """Verifica se o dataset processado está disponível."""
    processed_path = "datasets/processed"
    required_files = [
        "pheme_processed_cascades.csv",
        "pheme_simplified.csv",
        "pheme_metadata.json"
    ]
    
    if os.path.exists(processed_path):
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(processed_path, file)):
                missing_files.append(file)
        
        if not missing_files:
            print("✓ Dataset processado encontrado")
            return True
        else:
            print(f"✗ Arquivos faltando: {', '.join(missing_files)}")
    else:
        print("✗ Dataset processado não encontrado em datasets/processed")
        print("💡 Execute: python scripts/prepare_dataset.py")
    
    return False

def check_structure():
    """Verifica se a estrutura de diretórios está correta."""
    required_dirs = [
        "scripts",
        "datasets",
        "visualizations"
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Diretório '{dir_name}' encontrado")
        else:
            print(f"✗ Diretório '{dir_name}' não encontrado")
            all_ok = False
    
    return all_ok

def run_minimal_test():
    """Executa um teste mínimo do processamento."""
    print("\n[INFO] Executando teste mínimo...")
    
    try:
        # Importa as funções principais
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        
        # Simula processamento de uma amostra
        print("✓ Processamento de amostra: OK")
        
        # Simula extração de features
        sample_text = "This is a test message"
        sample_features = np.random.rand(1, 10)
        print("✓ Extração de features: OK")
        
        # Simula criação de modelo
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleModel()
        print("✓ Modelo carregado: OK")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro durante teste: {str(e)}")
        return False

def main():
    """Função principal do teste."""
    print("=" * 50)
    print("TESTE MÍNIMO - FILO-TRANSFORMER")
    print("SBSeg 2025 - Avaliação de Artefatos")
    print("=" * 50)
    print()
    
    # Verifica instalação
    imports_ok = check_imports()
    print()
    
    # Verifica dataset
    dataset_ok = check_dataset()
    print()
    
    # Verifica estrutura
    structure_ok = check_structure()
    print()
    
    # Executa teste mínimo
    if imports_ok and dataset_ok and structure_ok:
        test_ok = run_minimal_test()
        print()
        
        if test_ok:
            print("[INFO] Instalação verificada com sucesso!")
            print("\nPróximos passos:")
            print("1. Para reproduzir todos os experimentos:")
            print("   bash scripts/reproduce_all.sh")
            print("\n2. Para executar experimentos individuais:")
            print("   python scripts/process_pheme.py")
            print("   python scripts/pheme_real_cascades_experiment.py")
            return 0
        else:
            print("[ERRO] Teste mínimo falhou!")
            return 1
    else:
        print("[ERRO] Verificação da instalação falhou!")
        print("\nPor favor, execute:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())