# Filo-Transformer: Um modelo baseado em Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News

**Resumo:** Este artefato implementa o modelo Filo-Transformer que combina reconstrução filogenética usando Tree Alignment Graphs (TAGs) com classificação supervisionada para detecção de fake news no dataset PHEME.

**Título do Artigo:** Filo-Transformer: Um modelo baseado em Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News

**Artigo:** #10657

**Autores:**
- Acauan Cardoso Ribeiro (UNIVERSIDADE FEDERAL DE RORAIMA)
- Eduardo Luzeiro Feitosa (UNIVERSIDADE FEDERAL DO AMAZONAS)
- André Carvalho (UNIVERSIDADE FEDERAL DO AMAZONAS)

# Estrutura do readme.md

Este README segue o template obrigatório do SBSeg 2025 com todas as seções necessárias para avaliação dos selos.

# Selos Considerados

Os selos considerados para este artefato são: **Disponíveis**, **Funcionais**, **Sustentáveis** e **Experimentos Reprodutíveis**.

# Informações básicas

## Ambiente de Execução

**Requisitos:**
- Sistema: Linux, macOS ou Windows
- Python: 3.10 ou superior
- RAM: Mínimo 4GB
- Armazenamento: 1GB livre

**Tempo de Execução:** 5-10 minutos

# Dependências

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.1.0
```

O dataset PHEME está incluído no repositório.

# Preocupações com segurança

Não há riscos de segurança. O código executa apenas operações científicas padrão sem conexões de rede ou operações de sistema.

# Instalação

```bash
git clone https://github.com/filotransformer/sbseg.git
cd sbseg
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

# Teste mínimo

```bash
python scripts/run_experiment.py
```

**Resultado esperado:** O script executa cross-validation 5-fold e mostra os resultados do Filo-Transformer vs Baseline.

# Experimentos

## Experimento Completo

**Comando:**
```bash
python scripts/run_experiment.py
```

**O que faz:**
1. Carrega dataset PHEME (5802 amostras)
2. Executa cross-validation 5-fold
3. Para cada fold:
   - Extrai embeddings semânticos (TF-IDF)
   - Extrai características filogenéticas (TAG)
   - Treina Filo-Transformer (semânticas + filogenéticas)
   - Treina Baseline (apenas semânticas)
   - Avalia ambos os modelos
4. Mostra resultados finais com comparação

**Tempo esperado:** 5-10 minutos

**Resultado esperado:**
```
🧬 FILO-TRANSFORMER (COM CARACTERÍSTICAS FILOGENÉTICAS)
ACCURACY  : 0.8XXX ± 0.0XXX
AUC       : 0.8XXX ± 0.0XXX
F1        : 0.7XXX ± 0.0XXX
RECALL    : 0.7XXX ± 0.0XXX

📊 BASELINE (APENAS CARACTERÍSTICAS SEMÂNTICAS)  
ACCURACY  : 0.7XXX ± 0.0XXX
AUC       : 0.7XXX ± 0.0XXX
F1        : 0.6XXX ± 0.0XXX
RECALL    : 0.6XXX ± 0.0XXX

🎯 MELHORIA DO FILO-TRANSFORMER
ACCURACY  : +0.0XXX (+X.X%)
AUC       : +0.0XXX (+X.X%)
F1        : +0.0XXX (+X.X%)
RECALL    : +0.0XXX (+X.X%)
```

## Reivindicação #1: Superioridade do Filo-Transformer

**Execução:** `python scripts/run_experiment.py`

**Verificação:** Compare as métricas "FILO-TRANSFORMER" vs "BASELINE". O Filo-Transformer deve mostrar métricas superiores.

## Reivindicação #2: Impacto das Características Filogenéticas

**Execução:** `python scripts/run_experiment.py`

**Verificação:** Observe a seção "MELHORIA DO FILO-TRANSFORMER" que mostra o ganho percentual das características filogenéticas.

## Reivindicação #3: Performance no Dataset PHEME

**Execução:** `python scripts/run_experiment.py`

**Verificação:** O experimento usa o dataset PHEME completo (5802 amostras) e demonstra a eficácia do modelo.

# LICENSE

Este projeto é distribuído sob a **Licença MIT**. Veja o arquivo [LICENSE](LICENSE) para detalhes.

O dataset PHEME mantém sua licença original **Creative Commons Attribution**.

## Citação

```bibtex
@inproceedings{filo-transformer-2025,
    title={Filo-Transformer: Um modelo baseado em Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News},
    author={Acauan Cardoso Ribeiro and Eduardo Luzeiro Feitosa and André Carvalho},
    booktitle={XXV Simpósio Brasileiro de Cibersegurança (SBSeg 2025)},
    year={2025},
    note={Artigo \#10657}
}
```