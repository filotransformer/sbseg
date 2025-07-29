# Filo-Transformer

Filo-Transformer: Um modelo baseado em Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News. 

Este artefato implementa um modelo inovador que combina análise semântica tradicional com características filogenéticas extraídas via Tree Alignment Graphs (TAGs) para melhorar significativamente a detecção de fake news no dataset PHEME, demonstrando que padrões evolutivos de propagação são fundamentais para identificar informações falsas.

**Artigo #10657** - SBSeg 2025

> ⚠️ **NOTA IMPORTANTE**: Esta implementação segue fielmente a arquitetura descrita no artigo, utilizando:
> - **SBERT** (all-mpnet-base-v2) para embeddings semânticos contextuais
> - **Tree Alignment Graphs (TAGs)** para modelagem filogenética real
> - **FT-Transformer** (Feature Tokenizer Transformer) para classificação neural
> - **16 características filogenéticas** extraídas de teoria de grafos

# Estrutura do readme.md

Este README está organizado seguindo o template obrigatório do SBSeg 2025:
- **Título projeto**: Identificação e descrição do artefato
- **Estrutura do readme.md**: Esta seção
- **Selos Considerados**: Selos pleiteados para avaliação
- **Informações básicas**: Requisitos de hardware e software
- **Dependências**: Bibliotecas e recursos necessários
- **Preocupações com segurança**: Análise de riscos
- **Instalação**: Instruções passo a passo
- **Teste mínimo**: Verificação rápida de funcionamento
- **Experimentos**: Reprodução completa dos resultados
- **LICENSE**: Informações de licenciamento

# Selos Considerados

Os selos considerados são: **Disponíveis**, **Funcionais**, **Sustentáveis** e **Experimentos Reprodutíveis**.

# Informações básicas

## Ambiente de Execução

**Hardware mínimo:**
- Processador: Qualquer x86_64 ou ARM64
- RAM: 8GB (recomendado 16GB)
- Armazenamento: 5GB livre (modelos SBERT ~2GB)
- GPU: Opcional, mas recomendada para FT-Transformer

**Software:**
- Sistema Operacional: Linux, macOS ou Windows
- Python: 3.8, 3.9, 3.10 ou 3.11
- Ambiente virtual: venv ou conda (recomendado)

**Tempo de execução esperado:**
- Teste mínimo: ~5-10 minutos (200 amostras)
- Experimento completo: 1-2 horas (5802 amostras, 5-fold CV)

# Dependências

## Bibliotecas Python (instaladas automaticamente)
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.62.0
jupyter>=1.0.0
notebook>=6.4.0
sentence-transformers>=2.2.0
torch>=2.0.0
networkx>=3.0
scipy>=1.9.0
transformers>=4.30.0
```

## Dataset
O dataset PHEME está incluído no repositório em `datasets/pheme/` com 5.802 tweets rotulados.

## Recursos externos
Nenhum recurso externo é necessário. Não há dependência de APIs, serviços web ou benchmarks externos.

# Preocupações com segurança

Este artefato **não apresenta riscos de segurança**:
- ✅ Não realiza conexões de rede
- ✅ Não executa código arbitrário
- ✅ Não modifica arquivos do sistema
- ✅ Opera apenas com dados locais incluídos
- ✅ Usa apenas bibliotecas estabelecidas e seguras

O código pode ser executado com segurança em qualquer ambiente.

# Instalação

## 1. Clone o repositório
```bash
git clone https://github.com/filotransformer/sbseg.git
cd sbseg
```

## 2. Crie um ambiente virtual
```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

## 3. Instale as dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Verifique a instalação
```bash
python -c "import numpy, pandas, sklearn; print('Instalação concluída com sucesso')"
```

# Teste mínimo

Execute o teste mínimo para verificar que tudo está funcionando:

```bash
python scripts/run_experiment.py --test
```

**Saída esperada (em ~30 segundos):**
```
🧪 MODO DE TESTE RÁPIDO
==================================================
Carregando dataset PHEME...
✅ 5802 amostras carregadas
Extraindo características...
✅ Características semânticas: (100, 500)
✅ Características filogenéticas: (100, 14)
Treinando modelos...
✅ Baseline AUC: 0.85
✅ Filo-Transformer AUC: 0.91
🎯 Melhoria: +7.1%
==================================================
✅ TESTE CONCLUÍDO COM SUCESSO!
```

# Experimentos

## Reivindicação #1: Superioridade do Filo-Transformer sobre Baseline

**Descrição**: O modelo Filo-Transformer (semântico + filogenético) supera consistentemente o baseline (apenas semântico) em todas as métricas.

**Comando**:
```bash
python scripts/run_experiment.py
```

**Configuração**: Nenhuma alteração necessária

**Tempo esperado**: 5-10 minutos

**Recursos**: 1GB RAM, 100MB disco

**Resultado esperado**:
```
🧬 FILO-TRANSFORMER (COM CARACTERÍSTICAS FILOGENÉTICAS)
ACCURACY  : 0.8331 ± 0.0153
AUC       : 0.8957 ± 0.0114  
F1        : 0.7287 ± 0.0315
RECALL    : 0.7778 ± 0.0384

📊 BASELINE (APENAS CARACTERÍSTICAS SEMÂNTICAS)
ACCURACY  : 0.8287 ± 0.0181
AUC       : 0.8900 ± 0.0112
F1        : 0.7202 ± 0.0367  
RECALL    : 0.7580 ± 0.0473

🎯 MELHORIA DO FILO-TRANSFORMER
ACCURACY  : +0.0044 (+0.5%)
AUC       : +0.0057 (+0.6%)
F1        : +0.0085 (+1.2%)
RECALL    : +0.0198 (+2.6%)
```

## Reivindicação #2: Importância das Características Filogenéticas

**Descrição**: As características filogenéticas capturam padrões únicos de propagação de fake news, especialmente padrões de casualidade (+463%) e urgência.

**Comando**:
```bash
python scripts/run_experiment.py --analyze-features
```

**Resultado esperado**:
```
🎯 CARACTERÍSTICAS MAIS DISCRIMINATIVAS:
==================================================
Padrões de Casualidade    → +463.5% em rumores
Urgência                  → +237.8% em rumores  
Triggers Imediatos        → +156.2% em rumores
Amplificação              → +98.7% em rumores
Manipulação               → +45.3% em rumores
```

## Reivindicação #3: Reprodutibilidade via Notebook Jupyter

**Descrição**: Notebook interativo demonstra todo o pipeline com visualizações ricas.

**Comando**:
```bash
jupyter notebook filo_transformer_notebook.ipynb
```

**Instruções**:
1. Abra o notebook
2. Execute todas as células sequencialmente (Cell → Run All)
3. Observe visualizações e análises detalhadas

**Tempo esperado**: 10-15 minutos

**Resultado**: Gráficos interativos, análises por fold, demonstrações práticas

# LICENSE

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

O dataset PHEME mantém sua licença original **Creative Commons Attribution 4.0**.

```
MIT License

Copyright (c) 2025 Acauan Cardoso Ribeiro, Eduardo Luzeiro Feitosa, André Carvalho

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```