# Documentação do Código - Filo-Transformer

## Visão Geral

Este documento descreve a organização e funcionamento dos principais componentes do sistema Filo-Transformer.

## Estrutura de Arquivos

### Scripts Principais

#### 1. `scripts/prepare_dataset.py`
**Função**: Prepara o dataset PHEME para uso no projeto
- Descompacta o arquivo `phemernrdataset.tar.bz2`
- Chama `process_pheme.py` para processar os dados
- Verifica integridade dos arquivos gerados
- Remove arquivos temporários

**Funções principais**:
- `check_processed_exists()`: Verifica se dados já foram processados
- `extract_tar_bz2()`: Extrai arquivo compactado com progresso
- `run_processing_script()`: Executa processamento dos dados
- `check_dataset_integrity()`: Valida arquivos gerados

#### 2. `scripts/process_pheme.py`
**Função**: Processa dataset PHEME extraindo características filogenéticas
- Lê arquivos JSON do PHEME
- Constrói árvores de cascata de tweets
- Extrai 12 características filogenéticas
- Gera arquivos CSV processados

**Classes principais**:
- `PHEMEProcessor`: Classe principal para processamento
  - `extract_tweet_data()`: Extrai dados de tweets individuais
  - `build_cascade_tree()`: Constrói estrutura de árvore
  - `extract_cascade_features()`: Extrai características filogenéticas

#### 3. `scripts/ft_transformer.py`
**Função**: Implementa o modelo FT-Transformer
- Arquitetura baseada em Transformer
- Processa características semânticas e filogenéticas
- Implementa fusão automática de features

**Classes principais**:
- `FTTransformer`: Modelo principal
  - Tokenização de features
  - Multi-head attention
  - Fusão aprendível de modalidades

#### 4. `scripts/pheme_real_cascades_experiment.py`
**Função**: Experimento principal comparando baseline vs Filo-Transformer
- Carrega dados processados
- Gera embeddings semânticos com Sentence-BERT
- Treina e avalia modelos
- Implementa 5-fold cross-validation

**Funções principais**:
- `load_pheme_data()`: Carrega dataset processado
- `train_model()`: Treina modelo com early stopping
- `evaluate_model()`: Avalia desempenho
- `run_experiment()`: Executa experimento completo

#### 5. `scripts/hypothesis_validation_viz.py`
**Função**: Valida as 4 hipóteses do artigo
- H2.1: Terminal leaves hypothesis
- H3.2: Modelos filogenéticos superiores
- H4.2: Estrutura correlaciona com veracidade
- H5.2: Influência de perfis verificados

**Funções principais**:
- `validate_h2_1()`: Análise de folhas terminais
- `validate_h3_2()`: Comparação de modelos
- `validate_h4_2()`: Matriz de correlação
- `validate_h5_2()`: Análise de usuários verificados

#### 6. `scripts/quick_test.py`
**Função**: Teste mínimo de instalação
- Verifica dependências instaladas
- Valida estrutura de diretórios
- Testa funcionalidades básicas

### Scripts de Automação

#### `scripts/reproduce_all.sh`
Script bash que automatiza toda a reprodução:
1. Verifica ambiente e dependências
2. Prepara dados se necessário
3. Executa todos experimentos em sequência
4. Gera resumo dos resultados

## Fluxo de Dados

```
phemernrdataset.tar.bz2
    |
    v
prepare_dataset.py
    |
    v
process_pheme.py
    |
    v
datasets/processed/
    - pheme_processed_cascades.csv
    - pheme_simplified.csv
    - pheme_metadata.json
    |
    v
pheme_real_cascades_experiment.py
    |
    v
results/
    - Métricas de desempenho
    - Pesos de fusão aprendidos
    |
    v
hypothesis_validation_viz.py
    |
    v
visualizations/hypothesis/
    - Gráficos e validações
```

## Características Filogenéticas Extraídas

1. **cascade_size**: Número total de tweets na cascata
2. **cascade_depth**: Profundidade máxima da árvore
3. **cascade_breadth**: Número de respostas diretas ao tweet fonte
4. **cascade_lifetime**: Tempo entre primeiro e último tweet
5. **level_X_count**: Número de nós em cada nível (0-4)
6. **unique_users**: Número de usuários únicos
7. **user_diversity**: Razão unique_users/cascade_size
8. **verified_ratio**: Proporção de usuários verificados

## Parâmetros do Modelo

### FT-Transformer
- **d_model**: 192 (dimensão interna)
- **n_heads**: 8 (cabeças de atenção)
- **n_layers**: 3 (camadas Transformer)
- **d_ff**: 512 (dimensão feed-forward)
- **dropout**: 0.1

### Treinamento
- **learning_rate**: 0.001
- **batch_size**: 32
- **epochs**: 30 (padrão), 5 (teste rápido)
- **early_stopping_patience**: 10
- **optimizer**: Adam

## Modularização

O código está organizado em módulos independentes:
- Processamento de dados separado do modelo
- Experimentos isolados e reproduzíveis
- Visualizações geradas independentemente
- Scripts de automação para facilitar reprodução

## Extensibilidade

Para adicionar novos experimentos:
1. Crie novo script em `scripts/`
2. Use `load_pheme_data()` para carregar dados
3. Implemente sua lógica
4. Adicione ao `reproduce_all.sh` se necessário