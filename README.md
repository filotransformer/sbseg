# Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News

**Resumo:** Este artefato implementa o modelo Filo-Transformer apresentado no artigo "Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News" submetido ao XXV Simpósio Brasileiro de Cibersegurança (SBSeg 2025). O modelo combina reconstrução filogenética usando Tree Alignment Graphs (TAGs) com arquitetura FT-Transformer para detecção de fake news, alcançando AUC 0.9489 e F1 0.8393 no benchmark PHEME.

**Título do Artigo:** Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News

**Resumo do Artigo:** Propomos um pipeline que combina embeddings semânticos, reconstrução filogenética com Tree Alignment Graphs (TAGs) e um FT-Transformer supervisionado para detecção de rumores e fake news. A abordagem modela a propagação de informações como uma árvore filogenética, extraindo características evolutivas que capturam padrões de mutação e recombinação típicos de desinformação.

# Estrutura do readme.md

Este README está organizado seguindo o template obrigatório do SBSeg 2025:

- **Selos Considerados**: Quais selos de qualidade são solicitados
- **Informações básicas**: Ambiente de execução e requisitos
- **Dependências**: Software e bibliotecas necessárias
- **Preocupações com segurança**: Considerações de segurança
- **Instalação**: Processo de instalação passo a passo
- **Teste mínimo**: Verificação básica da instalação
- **Experimentos**: Reprodução completa dos resultados do artigo
- **LICENSE**: Informações de licenciamento

# Selos Considerados

Os selos considerados para este artefato são: **Disponíveis**, **Funcionais**, **Sustentáveis** e **Experimentos Reprodutíveis**.

# Informações básicas

## Ambiente de Execução

**Requisitos de Hardware:**
- CPU: Qualquer arquitetura x86_64 moderna
- RAM: Mínimo 8GB, recomendado 16GB
- Armazenamento: 2GB de espaço livre
- GPU: Opcional (CUDA compatível para aceleração)

**Requisitos de Software:**
- Sistema Operacional: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- Python: 3.10 ou superior
- pip: Gerenciador de pacotes Python
- Git: Para clonagem do repositório

**Ambientes Testados:**
- Ubuntu 22.04 LTS com Python 3.10
- Google Colab (notebook alternativo)
- WSL2 no Windows

## Tempo de Execução Estimado

- **Teste mínimo**: 2-5 minutos
- **Experimento completo**: 30-60 minutos (dependendo do hardware)
- **Apenas baseline**: 15-30 minutos

# Dependências

## Bibliotecas Python (requirements.txt)

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
networkx>=2.8.0
node2vec>=0.4.6
tensorflow>=2.16.0
scikit-learn>=1.1.0
transformers>=4.21.0
openai>=1.0.0
google-generativeai>=0.3.0
```

## Dependências Opcionais

- **OpenAI API Key**: Para embeddings GPT (fallback para SBERT se indisponível)
- **CUDA/cuDNN**: Para aceleração GPU do TensorFlow
- **Graphviz**: Para visualizações avançadas de grafos

## Dataset

O dataset PHEME v.7 está incluído no repositório em formato CSV pré-processado:
- **Origem**: PHEME dataset (Figshare, Creative Commons Attribution)
- **Formato**: CSV com colunas 'text' e 'label' (1=rumor, 0=não-rumor)
- **Eventos**: 5 eventos (charliehebdo, ferguson, germanwings, ottawashooting, sydneysiege)
- **Total**: ~6000 amostras combinadas

# Preocupações com segurança

## Considerações de Segurança

1. **Chave API OpenAI**: Se fornecida, será usada para gerar embeddings via API. A chave deve ser mantida segura e não compartilhada.

2. **Execução de Código**: O artefato executa apenas código Python científico padrão sem operações de sistema perigosas.

3. **Dados de Entrada**: O dataset PHEME é público e não contém informações sensíveis.

4. **Conexões de Rede**: Conexões são feitas apenas para:
   - Download de modelos HuggingFace (transformers)
   - API OpenAI (se chave fornecida)
   - Instalação de dependências via pip

## Recomendações

- Execute em ambiente virtual Python isolado
- Revise o arquivo `.env` antes de adicionar chaves API
- Use firewall para restringir conexões de rede se necessário

# Instalação

## 1. Clonar o Repositório

```bash
git clone https://github.com/filotransformer/sbseg.git
cd sbseg
```

## 2. Criar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

## 3. Instalar Dependências

```bash
# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt
```

## 4. Configurar Chave OpenAI (Opcional)

```bash
# Criar arquivo .env (opcional)
echo "OPENAI_API_KEY=sua_chave_aqui" > .env
```

**Nota**: Se a chave OpenAI não for fornecida, o sistema automaticamente usará embeddings SBERT.

## 5. Verificar Instalação

```bash
python scripts/minimal_test.py
```

Ao final deste processo, você deve ver "🎉 ALL TESTS PASSED!" indicando que a instalação foi bem-sucedida.

# Teste mínimo

Execute o teste mínimo para verificar se a instalação está funcionando corretamente:

```bash
python scripts/minimal_test.py
```

## O que o Teste Verifica

1. **Importação de Bibliotecas**: Verifica se todas as dependências estão instaladas
2. **Módulos Filo-Transformer**: Testa importação dos módulos customizados
3. **Funcionalidade Básica**: 
   - Gera embeddings para 4 textos de exemplo
   - Constrói grafo filogenético
   - Extrai características TAG
4. **Acesso ao Dataset**: Verifica se o dataset PHEME pode ser carregado
5. **Dependências Opcionais**: Informa sobre Node2Vec e OpenAI API

## Resultado Esperado

```
=============================================================
FILO-TRANSFORMER MINIMAL TEST
=============================================================
Testing imports...
✓ numpy
✓ pandas
✓ tensorflow 2.16.1
✓ networkx
✓ scikit-learn
✓ transformers
✓ openai

Testing Filo-Transformer modules...
✓ config
✓ embeddings
✓ graph_builder
✓ features
✓ model

Testing basic functionality...
✓ Configuration created
✓ SBERT embeddings: (4, 768)
✓ Graph built: 4 nodes, X edges
✓ Features extracted: (4, 79)

Testing dataset access...
✓ Dataset loaded: XXXX samples
✓ Required columns present
✓ Label distribution: {0: XXXX, 1: XXXX}

=============================================================
🎉 ALL TESTS PASSED!
The Filo-Transformer installation is working correctly.
You can now run the full experiments.
=============================================================
```

**Tempo esperado**: 2-5 minutos

# Experimentos

Esta seção descreve como reproduzir os resultados apresentados no artigo.

## Estrutura dos Experimentos

O artigo apresenta as seguintes reivindicações principais:

1. **Reivindicação #1**: O Filo-Transformer supera modelos baseline em métricas de detecção de fake news
2. **Reivindicação #2**: As características filogenéticas (TAG) melhoram significativamente a performance
3. **Reivindicação #3**: O modelo alcança AUC 0.9489 e F1 0.8393 no dataset PHEME

## Experimento Completo

### Comando Principal

```bash
python scripts/run_experiment.py
```

### Parâmetros Disponíveis

```bash
python scripts/run_experiment.py --help

# Exemplos:
# Executar apenas Filo-Transformer
python scripts/run_experiment.py --skip-baseline

# Executar apenas baseline
python scripts/run_experiment.py --skip-filo

# Usar dataset específico
python scripts/run_experiment.py --dataset datasets/pheme/charliehebdo.csv

# Salvar resultados em diretório específico
python scripts/run_experiment.py --output meus_resultados
```

### Processo de Execução

1. **Carregamento dos Dados**: Carrega dataset PHEME (~6000 amostras)
2. **Geração de Embeddings**: 
   - GPT embeddings (se API key disponível)
   - SBERT embeddings (fallback)
3. **Cross-Validation 5-fold**: Para garantir robustez estatística
4. **Para cada fold**:
   - Constrói grafo filogenético
   - Extrai características TAG (79 dimensões)
   - Treina modelo Filo-Transformer
   - Avalia performance
5. **Comparação com Baseline**: Executa modelo sem características filogenéticas
6. **Salva Resultados**: Métricas, curvas ROC, e resumo estatístico

### Tempo de Execução Esperado

- **Com GPU**: 30-45 minutos
- **Apenas CPU**: 45-60 minutos
- **Google Colab**: 25-35 minutos

### Recursos Utilizados

- **RAM**: 4-8GB durante execução
- **CPU**: 80-100% durante treinamento
- **Disco**: ~500MB para resultados intermediários

## Reivindicação #1: Superioridade do Filo-Transformer

### Execução

```bash
python scripts/run_experiment.py --output resultados_completos
```

### Resultado Esperado

```
=============================================================
FILO-TRANSFORMER FINAL RESULTS
=============================================================
Accuracy:  0.8888 ± 0.0156
AUC:       0.9489 ± 0.0089
Recall:    0.8530 ± 0.0234
F1-Score:  0.8393 ± 0.0178

=============================================================
BASELINE TRANSFORMER FINAL RESULTS
=============================================================
Accuracy:  0.8542 ± 0.0198
AUC:       0.9234 ± 0.0134
Recall:    0.8123 ± 0.0289
F1-Score:  0.8056 ± 0.0234
```

### Arquivos Gerados

- `resultados_completos/filo_transformer_results.json`: Resultados detalhados
- `resultados_completos/baseline_results.json`: Resultados baseline
- `resultados_completos/summary.json`: Resumo comparativo

## Reivindicação #2: Impacto das Características TAG

### Execução de Ablation Study

```bash
# Filo-Transformer completo
python scripts/run_experiment.py --skip-baseline --output com_tag

# Baseline sem TAG
python scripts/run_experiment.py --skip-filo --output sem_tag
```

### Análise

Compare os arquivos `com_tag/summary.json` e `sem_tag/summary.json` para verificar o impacto das características filogenéticas.

## Reivindicação #3: Performance no PHEME

### Dataset Individual por Evento

```bash
# Teste em cada evento separadamente
for evento in charliehebdo ferguson germanwings ottawashooting sydneysiege; do
    python scripts/run_experiment.py \
        --dataset datasets/pheme/${evento}.csv \
        --output resultados_${evento}
done
```

### Dataset Completo

```bash
python scripts/run_experiment.py \
    --dataset datasets/pheme/pheme_all.csv \
    --output resultados_pheme_completo
```

## Reprodução via Notebook (Alternativo)

Para usuários que preferem Jupyter Notebook:

```bash
# Instalar Jupyter
pip install jupyter

# Executar notebook
jupyter notebook filo_transformer_notebook.ipynb
```

Ou usar o [Google Colab](https://colab.research.google.com/drive/1KAVv9DYrWz-FnOf6X6toRrwN5CW8bIiv?usp=sharing).

## Validação dos Resultados

### Métricas Esperadas (Dataset Completo)

| Modelo | Accuracy | AUC | Recall | F1-Score |
|--------|----------|-----|---------|----------|
| **Filo-Transformer** | **0.8888** | **0.9489** | **0.8530** | **0.8393** |
| Baseline | 0.8542 | 0.9234 | 0.8123 | 0.8056 |

### Tolerâncias Aceitáveis

Devido à natureza estocástica do treinamento, variações de ±0.02 nas métricas são normais.

### Solução de Problemas

1. **Resultados muito diferentes**: Verifique se a mesma versão do dataset está sendo usada
2. **Erro de memória**: Reduza batch_size no arquivo `src/filo_transformer/config.py`
3. **Execução muito lenta**: Verifique se TensorFlow está usando GPU corretamente

# LICENSE

Este projeto é distribuído sob a **Licença MIT**. Veja o arquivo [LICENSE](LICENSE) para detalhes completos.

O dataset PHEME mantém sua licença original **Creative Commons Attribution**.

## Citação

Se usar este artefato em sua pesquisa, por favor cite:

```bibtex
@inproceedings{filo-transformer-2025,
    title={Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News},
    author={[Autores]},
    booktitle={XXV Simpósio Brasileiro de Cibersegurança (SBSeg 2025)},
    year={2025}
}
```