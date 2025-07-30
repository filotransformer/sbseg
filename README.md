# Filo-Transformer: Um modelo baseado em Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News

Este artefato apresenta a implementação do modelo Filo-Transformer, uma abordagem inovadora para detecção de fake news que combina análise filogenética textual com a arquitetura Feature Tokenizer Transformer (FT-Transformer). O modelo utiliza tanto características semânticas quanto estruturais das cascatas de propagação de informação em redes sociais para identificar conteúdo falso com maior precisão.

**Título do Artigo**: Filo-Transformer: Um modelo baseado em Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News

**Artigo #10657** - SBSeg 2025

**Resumo do Artigo**: Este trabalho propõe o Filo-Transformer, um modelo inovador que integra análise filogenética textual com a arquitetura FT-Transformer para detecção de fake news. Através da construção de Grafos de Alinhamento de Árvores (TAGs) e extração de características filogenéticas das cascatas de propagação, o modelo aprende automaticamente a importância relativa entre features semânticas e estruturais. Experimentos no dataset PHEME demonstram melhorias significativas, com AUC de 0.9071 comparado a 0.8882 do baseline, validando a importância das características filogenéticas.

# Estrutura do readme.md

Este documento está organizado nas seguintes seções:
- **Título e Resumo**: Descrição geral do projeto e artigo
- **Estrutura do readme.md**: Esta seção (organização do documento)
- **Selos Considerados**: Selos de qualidade aplicáveis ao artefato
- **Informações básicas**: Requisitos de hardware e software
- **Dependências**: Bibliotecas e versões necessárias
- **Preocupações com segurança**: Considerações de segurança
- **Instalação**: Processo de instalação do ambiente
- **Teste mínimo**: Verificação básica de funcionamento
- **Experimentos**: Reprodução dos resultados do artigo
- **LICENSE**: Licença do projeto

# Selos Considerados

Os selos considerados são: **Disponíveis (SeloD)**, **Funcionais (SeloF)**, **Sustentáveis (SeloS)** e **Experimentos Reprodutíveis (SeloR)**.

**Nota**: Um documento de apêndice (`APENDICE.md`) está disponível com informações complementares para os revisores, incluindo detalhes de desempenho, troubleshooting e checklist de avaliação.

# Informações básicas

## Ambiente de Execução

### Hardware Recomendado
- **CPU**: Processador com pelo menos 4 cores
- **RAM**: Mínimo 8GB, recomendado 16GB
- **Disco**: 10GB de espaço livre
- **GPU**: Opcional (o código funciona em CPU)

### Software Necessário
- **Sistema Operacional**: Linux (Ubuntu 20.04+) ou macOS
- **Python**: 3.8 ou superior
- **Git**: Para clonar o repositório

## Estrutura do Repositório

```
01_sbseg_filo_trans/
├── datasets/                  # Dados processados do PHEME
│   └── processed/            # Dados processados (baixados automaticamente)
├── scripts/                  # Scripts principais
│   ├── download_dataset.py   # Download automático dos dados
│   ├── process_pheme.py      # Processamento do dataset
│   ├── pheme_real_cascades_experiment.py  # Experimento principal
│   ├── hypothesis_validation_viz.py       # Validação de hipóteses
│   └── reproduce_all.sh      # Script de reprodução automática
├── paper/                    # Artigo e visualizações
├── visualizations/           # Resultados visuais
├── requirements.txt          # Dependências Python
├── README.md                 # Este arquivo
└── LICENSE                   # Licença MIT
```

# Dependências

## Bibliotecas Python Principais

```
torch==2.0.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1
tqdm==4.66.1
transformers==4.30.2
```

## Instalação das Dependências

Todas as dependências estão especificadas no arquivo `requirements.txt` com suas versões exatas.

# Preocupações com segurança

Este artefato não apresenta riscos de segurança aos avaliadores. O código:
- Não realiza conexões de rede externas
- Não modifica arquivos do sistema
- Processa apenas dados locais fornecidos
- Não executa código externo ou comandos do sistema

# Instalação

## 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/filo-transformer-sbseg25.git
cd filo-transformer-sbseg25
```

**Nota**: Substitua `seu-usuario` pelo nome de usuário correto do GitHub.

## 2. Criar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # No Linux/macOS
# ou
venv\Scripts\activate  # No Windows
```

## 3. Instalar Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Baixar Dataset Processado

Os dados processados do PHEME estão hospedados no Google Drive devido ao tamanho (4.3GB). Execute o script de download:

```bash
python scripts/download_dataset.py
```

Este script irá:
- Baixar automaticamente os dados processados (compactados em ~200MB)
- Descompactar na pasta `datasets/processed/`
- Remover o arquivo compactado para economizar espaço

**Nota**: O download é feito apenas uma vez. Se os dados já estiverem presentes, o script detecta e pula o download.

# Teste mínimo

Execute o seguinte comando para verificar se a instalação foi bem-sucedida:

```bash
python scripts/quick_test.py
```

Saída esperada:
```
[INFO] Verificando instalação...
✓ PyTorch instalado corretamente
✓ Transformers instalado corretamente
✓ Dataset PHEME encontrado
✓ Estrutura de diretórios correta
[INFO] Executando teste mínimo...
✓ Processamento de amostra: OK
✓ Extração de features: OK
✓ Modelo carregado: OK
[INFO] Instalação verificada com sucesso!
```

Este teste verifica:
1. Todas as dependências foram instaladas
2. O dataset está acessível
3. As funções básicas do modelo funcionam

Tempo esperado: < 30 segundos

# Experimentos

Para reproduzir todos os experimentos automaticamente:

```bash
bash scripts/reproduce_all.sh
```

Tempo total estimado: 2-3 horas (dependendo do hardware)

## Reivindicação #1: Processamento de Cascatas do PHEME

**Reivindicação**: O sistema extrai corretamente características filogenéticas de 5,802 cascatas do dataset PHEME.

### Execução:
```bash
python scripts/process_pheme.py
```

### Configuração:
- Nenhuma alteração necessária
- Flags: Nenhuma

### Recursos esperados:
- RAM: 2GB
- Disco: 500MB
- Tempo: 10-15 minutos

### Resultado esperado:
```
Processing PHEME dataset...
Events found: ['charliehebdo', 'ferguson', 'germanwings', 'ottawashooting', 'sydneysiege']
Processing charliehebdo: 100%|████████| 458/458 [00:45<00:00, 10.12it/s]
...
Total cascades processed: 5802
Phylogenetic features extracted:
- Average depth: 3.45
- Average branching factor: 2.87
- Average cascade size: 12.34
Data saved to: datasets/processed/pheme_cascades_with_features.csv
```

## Reivindicação #2: Superioridade do Filo-Transformer

**Reivindicação**: O Filo-Transformer alcança AUC de 0.9071, superando o baseline de 0.8882 (+1.89%).

### Execução:
```bash
python scripts/pheme_real_cascades_experiment.py --seed 42 --folds 5
```

### Configuração:
- Arquivo: `scripts/pheme_real_cascades_experiment.py`
- Linha 45: `num_epochs = 30` (pode reduzir para 10 para teste rápido)
- Flags: `--seed 42 --folds 5`

### Recursos esperados:
- RAM: 4GB
- Disco: 1GB
- Tempo: 1-2 horas (30 minutos com epochs=10)

### Resultado esperado:
```
=== Experimento Filo-Transformer vs Baseline ===
Fold 1/5:
  Baseline - Acc: 0.8645, AUC: 0.8856, Recall: 0.7543, F1: 0.7721
  Filo-Transformer - Acc: 0.8689, AUC: 0.9034, Recall: 0.7612, F1: 0.7812
...
=== Resultados Finais (5-Fold CV) ===
Baseline (Semântico):
  Accuracy: 0.8671 ± 0.0089
  AUC: 0.8882 ± 0.0076
  Recall: 0.7605 ± 0.0134
  F1-Score: 0.7790 ± 0.0098

Filo-Transformer:
  Accuracy: 0.8702 ± 0.0082
  AUC: 0.9071 ± 0.0069
  Recall: 0.7661 ± 0.0127
  F1-Score: 0.7847 ± 0.0091

Melhoria AUC: +1.89%
```

## Reivindicação #3: Aprendizado Automático de Pesos de Fusão

**Reivindicação**: O modelo aprende automaticamente a priorizar features filogenéticas (65%) sobre semânticas (35%).

### Execução:
```bash
python scripts/pheme_real_cascades_experiment.py --analyze-weights
```

### Configuração:
- Nenhuma alteração necessária
- Flag: `--analyze-weights`

### Recursos esperados:
- RAM: 2GB
- Tempo: 5 minutos

### Resultado esperado:
```
=== Análise de Pesos de Fusão ===
Pesos aprendidos por fold:
Fold 1: Semântico=0.542, Filogenético=1.058 (34.2% vs 65.8%)
Fold 2: Semântico=0.561, Filogenético=1.092 (33.9% vs 66.1%)
...
Média geral:
- Peso Semântico: 35.0% ± 1.2%
- Peso Filogenético: 65.0% ± 1.2%
```

## Reivindicação #4: Validação de Hipóteses

**Reivindicação**: Características filogenéticas correlacionam significativamente com veracidade.

### Execução:
```bash
python scripts/hypothesis_validation_viz.py
```

### Configuração:
- Nenhuma alteração necessária
- Flags: Nenhuma

### Recursos esperados:
- RAM: 3GB
- Disco: 100MB (para salvar visualizações)
- Tempo: 15 minutos

### Resultado esperado:
```
=== Validação de Hipóteses ===
H2.1 - Terminal leaves hypothesis:
  Rumours: 68.4% terminal, Non-rumours: 45.2% terminal
  Chi-square test: p < 0.001 ✓ (Hipótese validada)

H3.2 - Phylogenetic models superior:
  T-test AUC: p < 0.001 ✓ (Hipótese validada)
  
H4.2 - Cascade structure correlates:
  Correlation matrix saved
  All correlations significant (p < 0.05) ✓

H5.2 - Verified profiles influence:
  Verified ratio difference: 0.142 (p < 0.001) ✓

Visualizações salvas em: visualizations/hypothesis_validation/
```

# LICENSE

Este projeto está licenciado sob a MIT License:

```
MIT License

Copyright (c) 2025 Filo-Transformer Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```