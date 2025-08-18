#!/usr/bin/env python3
"""
test_reproducibility.py

Script para testar a reprodutibilidade das configurações de aleatoriedade.
Verifica se a função set_global_seed garante resultados determinísticos.
"""

import numpy as np
import torch
import random
import sys
import os

# Adiciona o diretório dos scripts ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa a função de seed do main_experiment
from main_experiment import set_global_seed


def test_reproducibility(seed=42):
    """
    Testa se a função set_global_seed garante reprodutibilidade.
    
    Args:
        seed (int): Semente para teste
    
    Returns:
        bool: True se os resultados são reprodutíveis
    """
    print(f"Testando reprodutibilidade com seed={seed}")
    print("-" * 50)
    
    # Primeira execução
    set_global_seed(seed)
    
    # Gera números aleatórios
    python_random_1 = [random.random() for _ in range(5)]
    numpy_random_1 = np.random.randn(5)
    torch_random_1 = torch.randn(5)
    
    # Se GPU disponível, testa também
    if torch.cuda.is_available():
        torch_cuda_random_1 = torch.randn(5).cuda()
    else:
        torch_cuda_random_1 = None
    
    # Segunda execução com mesma seed
    set_global_seed(seed)
    
    # Gera números aleatórios novamente
    python_random_2 = [random.random() for _ in range(5)]
    numpy_random_2 = np.random.randn(5)
    torch_random_2 = torch.randn(5)
    
    if torch.cuda.is_available():
        torch_cuda_random_2 = torch.randn(5).cuda()
    else:
        torch_cuda_random_2 = None
    
    # Verifica se são idênticos
    python_match = python_random_1 == python_random_2
    numpy_match = np.allclose(numpy_random_1, numpy_random_2)
    torch_match = torch.allclose(torch_random_1, torch_random_2)
    
    if torch_cuda_random_1 is not None:
        cuda_match = torch.allclose(torch_cuda_random_1, torch_cuda_random_2)
    else:
        cuda_match = True  # Sem GPU, considera como match
    
    # Imprime resultados
    print("📊 Resultados do Teste de Reprodutibilidade:")
    print(f"  Python random: {'✅ Reprodutível' if python_match else '❌ NÃO reprodutível'}")
    print(f"  NumPy random:  {'✅ Reprodutível' if numpy_match else '❌ NÃO reprodutível'}")
    print(f"  PyTorch CPU:   {'✅ Reprodutível' if torch_match else '❌ NÃO reprodutível'}")
    
    if torch.cuda.is_available():
        print(f"  PyTorch CUDA:  {'✅ Reprodutível' if cuda_match else '❌ NÃO reprodutível'}")
    else:
        print("  PyTorch CUDA:  ⚠️ GPU não disponível")
    
    all_match = python_match and numpy_match and torch_match and cuda_match
    
    print("-" * 50)
    if all_match:
        print("✅ SUCESSO: Todas as bibliotecas são reprodutíveis!")
    else:
        print("❌ FALHA: Algumas bibliotecas não são reprodutíveis!")
    
    # Mostra exemplos de valores
    print("\n📈 Exemplos de valores gerados:")
    print(f"  Python: {python_random_1[0]:.6f}")
    print(f"  NumPy:  {numpy_random_1[0]:.6f}")
    print(f"  Torch:  {torch_random_1[0].item():.6f}")
    
    return all_match


def test_deterministic_operations():
    """
    Testa se operações determinísticas estão configuradas corretamente.
    """
    print("\n🔍 Testando configurações determinísticas do PyTorch:")
    print("-" * 50)
    
    # Verifica configurações
    if hasattr(torch.backends.cudnn, 'deterministic'):
        det = torch.backends.cudnn.deterministic
        print(f"  cudnn.deterministic: {det} {'✅' if det else '❌'}")
    
    if hasattr(torch.backends.cudnn, 'benchmark'):
        bench = torch.backends.cudnn.benchmark
        print(f"  cudnn.benchmark: {bench} {'✅' if not bench else '❌'}")
    
    # Verifica variável de ambiente
    pythonhashseed = os.environ.get('PYTHONHASHSEED')
    print(f"  PYTHONHASHSEED: {pythonhashseed} {'✅' if pythonhashseed == '42' else '⚠️'}")
    
    print("-" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("TESTE DE REPRODUTIBILIDADE - FILO-TRANSFORMER")
    print("=" * 50)
    
    # Testa reprodutibilidade
    is_reproducible = test_reproducibility(seed=42)
    
    # Testa configurações determinísticas
    test_deterministic_operations()
    
    print("\n" + "=" * 50)
    if is_reproducible:
        print("✅ Sistema configurado corretamente para reprodutibilidade!")
        sys.exit(0)
    else:
        print("❌ Problemas detectados na configuração de reprodutibilidade!")
        sys.exit(1)