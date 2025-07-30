# Guia de Configuração para Windows (WSL2)

Este guia detalha como configurar o ambiente no Windows para executar o Filo-Transformer.

## Pré-requisitos

- Windows 10 versão 2004 ou superior (Build 19041+)
- Windows 11 (qualquer versão)

## Passo 1: Instalar WSL2

Abra o PowerShell como Administrador e execute:

```powershell
# Instalar WSL2 com Ubuntu 24.04
wsl --install -d Ubuntu-24.04
```

Reinicie o computador quando solicitado.

## Passo 2: Configurar Ubuntu no WSL

1. Abra o Ubuntu pelo menu Iniciar
2. Crie um usuário e senha quando solicitado
3. Atualize o sistema:

```bash
sudo apt update && sudo apt upgrade -y
```

## Passo 3: Instalar Dependências do Sistema

```bash
# Pacotes essenciais
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    python3-dev

# Verificar instalação
python3 --version  # Deve mostrar Python 3.8+
git --version
```

## Passo 4: Clonar e Configurar o Projeto

```bash
# Navegar para home
cd ~

# Clonar repositório
git clone https://github.com/filotransformer/sbseg.git
cd sbseg

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências Python
pip install --upgrade pip
pip install -r requirements.txt
```

## Passo 5: Preparar Dataset

```bash
# Preparar dados (apenas primeira vez)
python scripts/prepare_dataset.py
```

## Passo 6: Testar Instalação

```bash
# Teste mínimo
python scripts/quick_test.py
```

## Dicas Importantes

### Acessar arquivos do WSL no Windows
- Os arquivos ficam em: `\\wsl$\Ubuntu-24.04\home\seu_usuario\sbseg`
- Ou digite no Explorer: `\\wsl$`

### Usar GPU (se disponível)
O WSL2 suporta GPU NVIDIA. Para usar:
1. Instale os drivers NVIDIA para WSL
2. O PyTorch detectará automaticamente

### Problemas Comuns

**Erro de memória**
- Aumente a memória do WSL criando `.wslconfig` em `C:\Users\SeuUsuario\`:
```
[wsl2]
memory=8GB
processors=4
```

**Erro de permissão**
- Use `sudo` quando necessário
- Garanta que está no ambiente virtual: `source venv/bin/activate`

**Velocidade lenta**
- Trabalhe dentro do sistema de arquivos do WSL (`/home/usuario/`)
- Evite acessar arquivos do Windows (`/mnt/c/`)

## Verificação Final

Execute para verificar se tudo está funcionando:

```bash
# Dentro do WSL, na pasta sbseg com venv ativado
bash scripts/reproduce_all.sh
```

Se todos os passos foram seguidos corretamente, o sistema estará pronto para uso!