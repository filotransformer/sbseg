#!/usr/bin/env python3
"""
Script para baixar e descompactar os dados processados do PHEME do Google Drive.
Este script é executado automaticamente durante a instalação se os dados não estiverem presentes.
"""

import os
import sys
import subprocess
import tarfile
from pathlib import Path
import requests
from tqdm import tqdm

def download_file_from_google_drive(file_id: str, destination: str) -> bool:
    """
    Baixa um arquivo do Google Drive usando o file_id.
    Lida corretamente com arquivos grandes que requerem confirmação.
    
    Args:
        file_id: ID do arquivo no Google Drive
        destination: Caminho de destino para salvar o arquivo
        
    Returns:
        bool: True se o download foi bem-sucedido, False caso contrário
    """
    
    def get_confirm_token(response_text):
        """Extrai o token de confirmação da resposta HTML do Google Drive."""
        import re
        # Procura por padrões de confirmação na resposta HTML
        patterns = [
            r'name="confirm" value="([^"]+)"',
            r'"downloadUrl":"([^"]+)"',
            r'confirm=([^&]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1)
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, "wb") as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Baixando") as pbar:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # Para arquivos sem content-length, mostra progresso por chunks
                downloaded = 0
                with tqdm(unit='B', unit_scale=True, desc="Baixando") as pbar:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))

    try:
        session = requests.Session()
        
        # Primeira tentativa: download direto
        URL = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(URL, stream=True)
        
        # Verifica se é um arquivo pequeno (download direto)
        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type:
            print("📥 Download direto...")
            save_response_content(response, destination)
            return True
        
        # Para arquivos grandes, precisa lidar com a página de confirmação
        print("📥 Arquivo grande detectado, processando confirmação...")
        response_text = response.text
        
        # Tenta diferentes métodos para arquivos grandes
        confirm_token = get_confirm_token(response_text)
        
        if confirm_token:
            # Método 1: Usar token de confirmação
            confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
            response = session.get(confirm_url, stream=True)
        else:
            # Método 2: Usar parâmetros alternativos
            alt_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            response = session.get(alt_url, stream=True)
        
        # Verifica se agora temos o arquivo
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            print("❌ Não foi possível contornar a proteção do Google Drive")
            print("💡 Tente baixar manualmente e colocar em datasets/pheme_processed_data.tar.gz")
            return False
        
        save_response_content(response, destination)
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o download: {e}")
        return False

def validate_tar_file(tar_path: str) -> bool:
    """
    Valida se o arquivo baixado é realmente um tar.gz válido.
    
    Args:
        tar_path: Caminho para o arquivo tar.gz
        
    Returns:
        bool: True se é um arquivo tar.gz válido, False caso contrário
    """
    try:
        # Verifica se é um arquivo tar.gz válido
        with tarfile.open(tar_path, "r:gz") as tar:
            # Tenta listar o conteúdo sem extrair
            tar.getnames()
        return True
    except Exception:
        return False

def extract_tar_gz(tar_path: str, extract_to: str) -> bool:
    """
    Extrai um arquivo tar.gz.
    
    Args:
        tar_path: Caminho para o arquivo tar.gz
        extract_to: Diretório onde extrair
        
    Returns:
        bool: True se a extração foi bem-sucedida, False caso contrário
    """
    try:
        # Primeiro valida se é um arquivo tar.gz válido
        if not validate_tar_file(tar_path):
            print("❌ Arquivo baixado não é um tar.gz válido!")
            print("💡 Provavelmente foi baixada uma página HTML do Google Drive")
            
            # Mostra o conteúdo do arquivo para debug
            with open(tar_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)  # Primeiros 500 caracteres
                if '<html' in content.lower() or '<!doctype' in content.lower():
                    print("🔍 Confirmado: arquivo contém HTML em vez de dados binários")
                    print("📄 Início do arquivo:")
                    print(content[:200] + "...")
            
            return False
        
        print("📦 Extraindo dados...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        return True
    except Exception as e:
        print(f"❌ Erro durante a extração: {e}")
        return False

def check_dataset_exists() -> bool:
    """
    Verifica se o dataset processado já existe.
    
    Returns:
        bool: True se o dataset existe, False caso contrário
    """
    processed_path = Path("datasets/processed")
    required_files = [
        "pheme_processed_cascades.csv",
        "pheme_all_tweets.csv",
        "pheme_complete.jsonl",
        "dataset_stats.json"
    ]
    
    if not processed_path.exists():
        return False
        
    for file in required_files:
        if not (processed_path / file).exists():
            return False
            
    return True

def main():
    """Função principal do script."""
    print("=" * 60)
    print("🔄 DOWNLOAD DE DADOS PROCESSADOS - FILO-TRANSFORMER")
    print("SBSeg 2025 - Artefato de Pesquisa")
    print("=" * 60)
    print()
    
    # Verifica se os dados já existem
    if check_dataset_exists():
        print("✅ Dados processados já estão presentes!")
        print("📁 Localização: datasets/processed/")
        return 0
    
    print("📥 Dados processados não encontrados. Iniciando download...")
    print()
    
    # Cria diretório datasets se não existir
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # ID do arquivo no Google Drive (extraído da URL que você forneceu)
    # NOTA: Você precisa substituir este ID pelo ID real do seu arquivo
    file_id = "1efPvPpN8wHkaTs6Y8p5j9XkWwLfbgV-3"  # Substituir pelo ID real
    
    # Caminho para salvar o arquivo compactado
    tar_path = datasets_dir / "pheme_processed_data.tar.gz"
    
    print(f"🌐 Baixando de: Google Drive")
    print(f"📁 Salvando em: {tar_path}")
    print()
    
    # Faz o download
    if not download_file_from_google_drive(file_id, str(tar_path)):
        print("❌ Falha no download automático!")
        print()
        print("📋 INSTRUÇÕES PARA DOWNLOAD MANUAL:")
        print("1. Acesse: https://drive.google.com/file/d/1efPvPpN8wHkaTs6Y8p5j9XkWwLfbgV-3/view?usp=sharing")
        print("2. Clique em 'Baixar' ou 'Download'")
        print("3. Salve o arquivo como: datasets/pheme_processed_data.tar.gz")
        print("4. Execute este script novamente")
        print()
        print("💡 O arquivo deve ter aproximadamente 205MB")
        return 1
    
    print("✅ Download concluído!")
    print()
    
    # Extrai o arquivo
    if not extract_tar_gz(str(tar_path), str(datasets_dir)):
        print("❌ Falha na extração!")
        print()
        print("🔧 POSSÍVEIS SOLUÇÕES:")
        print("1. O arquivo pode estar corrompido - tente baixar novamente")
        print("2. Baixe manualmente do link:")
        print("   https://drive.google.com/file/d/1efPvPpN8wHkaTs6Y8p5j9XkWwLfbgV-3/view?usp=sharing")
        print("3. Certifique-se de que o arquivo tem ~205MB (não 2KB)")
        print("4. Se o problema persistir, contate os autores")
        return 1
    
    print("✅ Extração concluída!")
    print()
    
    # Remove o arquivo compactado para economizar espaço
    try:
        tar_path.unlink()
        print("🗑️  Arquivo compactado removido para economizar espaço")
    except Exception as e:
        print(f"⚠️  Não foi possível remover o arquivo compactado: {e}")
    
    # Verifica se tudo foi extraído corretamente
    if check_dataset_exists():
        print("✅ Dados processados instalados com sucesso!")
        print("📁 Localização: datasets/processed/")
        print()
        print("🚀 Agora você pode executar os experimentos:")
        print("   python scripts/quick_test.py")
        print("   bash scripts/reproduce_all.sh")
        return 0
    else:
        print("❌ Erro: Dados não foram extraídos corretamente!")
        return 1

if __name__ == "__main__":
    # Adiciona requirements necessários
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("❌ Dependências não encontradas!")
        print("Execute: pip install requests tqdm")
        sys.exit(1)
    
    sys.exit(main()) 