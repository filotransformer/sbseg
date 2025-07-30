#!/usr/bin/env python3
"""
Script para preparar o dataset PHEME para o projeto Filo-Transformer.

Este script orquestra:
1. Descompactação do arquivo phemernrdataset.tar.bz2
2. Chamada do processo de processamento dos dados (process_pheme.py)
3. Verificação de integridade dos arquivos processados
"""

import os
import sys
import subprocess
import tarfile
from pathlib import Path
import json
from tqdm import tqdm
import shutil

def check_processed_exists() -> bool:
    """
    Verifica se o dataset já foi processado.
    
    Returns:
        bool: True se o dataset processado existe, False caso contrário
    """
    processed_path = Path("datasets/processed")
    required_files = [
        "pheme_processed_cascades.csv",
        "pheme_simplified.csv",
        "pheme_metadata.json"
    ]
    
    if not processed_path.exists():
        return False
        
    for file in required_files:
        if not (processed_path / file).exists():
            return False
            
    return True

def extract_tar_bz2(tar_path: str, extract_to: str) -> bool:
    """
    Extrai um arquivo tar.bz2.
    
    Args:
        tar_path: Caminho para o arquivo tar.bz2
        extract_to: Diretório onde extrair
        
    Returns:
        bool: True se a extração foi bem-sucedida, False caso contrário
    """
    try:
        print("📦 Extraindo dataset PHEME...")
        print(f"   Arquivo: {tar_path}")
        print(f"   Destino: {extract_to}")
        print("   Aguarde, este processo pode levar alguns minutos...")
        
        # Conta o número total de arquivos para a barra de progresso
        with tarfile.open(tar_path, "r:bz2") as tar:
            members = tar.getmembers()
            total_files = len(members)
            
        # Extrai com barra de progresso
        with tarfile.open(tar_path, "r:bz2") as tar:
            with tqdm.tqdm(total=total_files, desc="Extraindo arquivos", unit="arquivo") as pbar:
                for member in members:
                    tar.extract(member, path=extract_to)
                    pbar.update(1)
                    
        print("✅ Extração concluída!")
        return True
        
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo não encontrado: {tar_path}")
        return False
    except tarfile.TarError as e:
        print(f"❌ Erro ao extrair arquivo tar.bz2: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro durante a extração: {e}")
        return False

def verify_extraction(datasets_dir: Path) -> bool:
    """
    Verifica se a extração foi bem-sucedida.
    
    Args:
        datasets_dir: Diretório datasets
        
    Returns:
        bool: True se a extração está correta
    """
    pheme_dir = datasets_dir / "pheme-rnr-dataset"
    
    if not pheme_dir.exists():
        print("❌ Diretório pheme-rnr-dataset não foi criado")
        return False
    
    # Verifica se tem os eventos esperados
    expected_events = ['charliehebdo', 'ferguson', 'germanwings-crash', 
                      'ottawashooting', 'sydneysiege']
    
    for event in expected_events:
        event_path = pheme_dir / event
        if not event_path.exists():
            print(f"❌ Evento não encontrado: {event}")
            return False
    
    print("✅ Estrutura do dataset verificada com sucesso!")
    return True

def run_processing_script() -> bool:
    """
    Executa o script de processamento process_pheme.py.
    
    Returns:
        bool: True se o processamento foi bem-sucedido
    """
    try:
        print("\n🔄 Iniciando processamento dos dados PHEME...")
        print("   Este processo pode levar alguns minutos...")
        
        # Executa o script de processamento
        result = subprocess.run(
            [sys.executable, "scripts/process_pheme.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Processamento concluído com sucesso!")
            return True
        else:
            print("❌ Erro durante o processamento:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Erro ao executar script de processamento: {e}")
        return False

def check_dataset_integrity() -> bool:
    """
    Verifica a integridade do dataset processado.
    
    Returns:
        bool: True se o dataset está íntegro
    """
    processed_path = Path("datasets/processed")
    
    # Verifica arquivos principais
    required_files = {
        "pheme_processed_cascades.csv": "Dataset principal com cascatas processadas",
        "pheme_simplified.csv": "Versão simplificada do dataset",
        "pheme_metadata.json": "Metadados do processamento"
    }
    
    print("\n🔍 Verificando integridade dos arquivos processados...")
    
    all_ok = True
    for filename, description in required_files.items():
        filepath = processed_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} ({size_mb:.2f} MB) - {description}")
        else:
            print(f"❌ {filename} - NÃO ENCONTRADO")
            all_ok = False
    
    # Verifica metadados
    if all_ok:
        try:
            with open(processed_path / "pheme_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            print(f"\n📊 Estatísticas do dataset:")
            print(f"   Total de cascatas: {metadata.get('total_cascades', 'N/A')}")
            print(f"   Eventos: {', '.join(metadata.get('events', []))}")
            print(f"   Cascatas com reactions: {metadata.get('stats', {}).get('cascades_with_reactions', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️  Não foi possível ler metadados: {e}")
    
    return all_ok

def cleanup_temp_files(datasets_dir: Path) -> None:
    """
    Remove arquivos temporários para economizar espaço.
    
    Args:
        datasets_dir: Diretório datasets
    """
    pheme_extracted_dir = datasets_dir / "pheme-rnr-dataset"
    
    if pheme_extracted_dir.exists():
        try:
            print("\n🗑️  Removendo arquivos temporários...")
            shutil.rmtree(pheme_extracted_dir)
            print("✅ Arquivos temporários removidos")
        except Exception as e:
            print(f"⚠️  Não foi possível remover arquivos temporários: {e}")

def main():
    """Função principal do script."""
    print("=" * 60)
    print("🔧 PREPARAÇÃO DO DATASET - FILO-TRANSFORMER")
    print("SBSeg 2025 - Artefato de Pesquisa")
    print("=" * 60)
    print()
    
    # Verifica se o dataset já foi processado
    if check_processed_exists():
        print("✅ Dataset já foi processado!")
        print("📁 Localização: datasets/processed/")
        print("\nPara reprocessar, remova a pasta datasets/processed e execute novamente.")
        return 0
    
    # Verifica diretórios
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("❌ Diretório 'datasets' não encontrado!")
        return 1
    
    # Verifica arquivo tar.bz2
    tar_path = datasets_dir / "phemernrdataset.tar.bz2"
    if not tar_path.exists():
        print("❌ Arquivo phemernrdataset.tar.bz2 não encontrado!")
        print(f"   Esperado em: {tar_path}")
        print("\n📋 INSTRUÇÕES:")
        print("1. Baixe o arquivo phemernrdataset.tar.bz2")
        print("2. Coloque-o no diretório datasets/")
        print("3. Execute este script novamente")
        return 1
    
    # Mostra tamanho do arquivo
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"📦 Arquivo encontrado: phemernrdataset.tar.bz2 ({size_mb:.2f} MB)")
    
    # Extrai o arquivo
    print("\n" + "="*50)
    print("ETAPA 1: Extração do arquivo tar.bz2")
    print("="*50)
    
    if not extract_tar_bz2(str(tar_path), str(datasets_dir)):
        print("❌ Falha na extração do arquivo!")
        return 1
    
    # Verifica extração
    if not verify_extraction(datasets_dir):
        print("❌ Falha na verificação da extração!")
        return 1
    
    # Processa os dados
    print("\n" + "="*50)
    print("ETAPA 2: Processamento dos dados")
    print("="*50)
    
    if not run_processing_script():
        print("❌ Falha no processamento dos dados!")
        return 1
    
    # Verifica integridade
    print("\n" + "="*50)
    print("ETAPA 3: Verificação de integridade")
    print("="*50)
    
    if not check_dataset_integrity():
        print("❌ Falha na verificação de integridade!")
        return 1
    
    # Limpa arquivos temporários
    cleanup_temp_files(datasets_dir)
    
    # Sucesso!
    print("\n" + "="*60)
    print("✅ DATASET PREPARADO COM SUCESSO!")
    print("="*60)
    print("\n📁 Arquivos processados em: datasets/processed/")
    print("\n🚀 Próximos passos:")
    print("   1. Execute um teste rápido:")
    print("      python scripts/quick_test.py")
    print("\n   2. Execute todos os experimentos:")
    print("      bash scripts/reproduce_all.sh")
    print("\n   3. Visualize os resultados:")
    print("      Verifique a pasta results/ após a execução")
    
    return 0

if __name__ == "__main__":
    # Verifica dependências
    try:
        import tqdm
    except ImportError:
        print("❌ Dependência não encontrada: tqdm")
        print("Execute: pip install tqdm")
        sys.exit(1)
    
    sys.exit(main())