# Apêndice - Filo-Transformer SBSeg'25

## Informações Complementares para Revisores

Este documento complementa o README.md com informações adicionais para facilitar o processo de avaliação do artefato "Filo-Transformer: Um modelo baseado em Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News" (Artigo #10657).

## 1. Recursos Específicos

### 1.1 Hardware
O projeto foi desenvolvido e testado nos seguintes ambientes:
- **Desenvolvimento**: Intel Core i7-9750H, 16GB RAM, Ubuntu 20.04
- **Testes**: Google Colab (GPU Tesla T4 opcional)
- **Requisito mínimo**: 8GB RAM, CPU com 4 cores

### 1.2 Dataset
Os dados processados do PHEME são baixados automaticamente via Google Drive (~200MB compactado, 4.3GB descompactado). O script `download_dataset.py` gerencia todo o processo automaticamente. Não são necessárias chaves de acesso ou credenciais especiais.

### 1.3 Recursos Externos
**Nenhum recurso externo é necessário**. O artefato:
- Não requer conexão com internet após instalação das dependências
- Não utiliza APIs externas
- Não requer chaves de acesso ou credenciais
- Todo processamento é feito localmente

## 2. Informações de Desempenho

### 2.1 Tempos de Execução Detalhados

| Experimento | Amostras | CPU (4 cores) | GPU (opcional) |
|-------------|----------|---------------|----------------|
| Teste mínimo | 200 | ~30 segundos | ~20 segundos |
| Reivindicação #1 | 5,802 | 10-15 minutos | 8-10 minutos |
| Reivindicação #2 | 5,802 | 1-2 horas | 45-60 minutos |
| Reivindicação #3 | - | 5 minutos | 5 minutos |
| Reivindicação #4 | 5,802 | 15 minutos | 12 minutos |
| **Total (todos)** | - | 2-3 horas | 1.5-2 horas |

### 2.2 Uso de Memória

- Processamento do dataset: ~2GB RAM
- Treinamento do modelo: ~4GB RAM
- Pico máximo observado: ~6GB RAM

## 3. Estrutura Detalhada do Código

### 3.1 Módulos Principais

```
scripts/
├── ft_transformer.py         # Implementação do FT-Transformer
│   └── Classes: FTTransformer, FusionFTTransformer
├── tag_construction.py        # Construção de TAGs (Tree Alignment Graphs)
│   └── Funções: build_tag(), extract_phylogenetic_features()
├── pheme_dataset_processor.py # Processamento do dataset PHEME
│   └── Classes: PHEMEProcessor, CascadeExtractor
└── run_experiment.py          # Orquestrador principal dos experimentos
    └── Funções: main(), train_model(), evaluate()
```

### 3.2 Fluxo de Execução

1. **Processamento**: `process_pheme.py` → Extrai cascatas e features
2. **Treinamento**: `pheme_real_cascades_experiment.py` → Treina e avalia modelos
3. **Validação**: `hypothesis_validation_viz.py` → Testa hipóteses estatísticas
4. **Visualização**: `visualize_*.py` → Gera gráficos e análises

## 4. Detalhes Técnicos Adicionais

### 4.1 Parâmetros do Modelo

```python
# Configurações padrão do FT-Transformer
config = {
    "num_features": 24,           # Features filogenéticas
    "text_embedding_dim": 768,    # Dimensão BERT
    "d_token": 64,               # Dimensão dos tokens
    "n_blocks": 3,               # Blocos transformer
    "attention_n_heads": 8,      # Cabeças de atenção
    "attention_dropout": 0.2,    # Dropout
    "ffn_d_hidden": 256,        # Dimensão hidden FFN
}
```

### 4.2 Seeds e Reprodutibilidade

- Seed padrão: 42 (pode ser alterada via `--seed`)
- Seeds fixas em: PyTorch, NumPy, Python random
- Resultados determinísticos garantidos

## 5. Troubleshooting

### 5.1 Problemas Comuns

1. **Erro de memória**: Reduza batch_size ou use modo de teste
2. **Dependências**: Use exatamente as versões do requirements.txt
3. **Dataset não encontrado**: Execute `python scripts/download_dataset.py`
4. **Falha no download**: Verifique conexão com internet ou tente novamente

### 5.2 Modos de Execução Alternativos

```bash
# Modo de teste rápido (200 amostras)
python scripts/pheme_real_cascades_experiment.py --test

# Apenas 1 fold para teste
python scripts/pheme_real_cascades_experiment.py --folds 1

# Sem GPU
python scripts/pheme_real_cascades_experiment.py --no-cuda
```

## 6. Contato para Suporte

Em caso de dúvidas durante a avaliação, os autores estão disponíveis através da plataforma HotCRP.

## 7. Checklist para Revisores

- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas via requirements.txt
- [ ] Teste mínimo executado com sucesso
- [ ] Dataset processado baixado em `datasets/processed/`
- [ ] Pelo menos uma reivindicação reproduzida
- [ ] Resultados salvos em `results/`

## 8. Notas Finais

- O código está otimizado para clareza, não apenas desempenho
- Comentários em inglês seguem padrão científico
- Visualizações são salvas automaticamente em `visualizations/`
- Logs detalhados ajudam no debugging se necessário 