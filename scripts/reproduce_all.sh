#!/bin/bash
# reproduce_all.sh - Script para reproduzir todos os experimentos do Filo-Transformer
# SBSeg 2025 - Avaliação de Artefatos

set -e  # Para em caso de erro

echo "=============================================="
echo "Filo-Transformer - Reprodução Completa"
echo "SBSeg 2025 - Avaliação de Artefatos"
echo "=============================================="
echo

# Verifica se está no diretório correto
if [ ! -f "requirements.txt" ]; then
    echo "❌ Erro: Execute este script no diretório raiz do projeto"
    exit 1
fi

# Verifica se está no ambiente virtual
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  AVISO: Ambiente virtual não detectado!"
    echo "Por favor, ative o ambiente virtual primeiro:"
    echo "  source venv/bin/activate"
    exit 1
fi

# Prepara dados se necessário
echo "📥 Verificando dados processados..."
if [ ! -d "datasets/processed" ] || [ ! -f "datasets/processed/pheme_processed_cascades.csv" ]; then
    echo "Dados não encontrados. Preparando automaticamente..."
    python scripts/prepare_dataset.py
    if [ $? -ne 0 ]; then
        echo "❌ Erro: Falha na preparação dos dados!"
        exit 1
    fi
else
    echo "✅ Dados processados já estão presentes"
fi
echo

# Função para exibir tempo decorrido
show_elapsed_time() {
    local start=$1
    local end=$(date +%s)
    local elapsed=$((end - start))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    echo "Tempo decorrido: ${minutes}m ${seconds}s"
}

# Início da execução
START_TIME=$(date +%s)

echo "📁 Criando diretórios necessários..."
mkdir -p datasets/processed
mkdir -p visualizations/hypothesis_validation
mkdir -p results

echo
echo "=============================================="
echo "ETAPA 1: Processamento do Dataset PHEME"
echo "=============================================="
echo "Extraindo características filogenéticas das cascatas..."
echo

STEP_START=$(date +%s)
python scripts/process_pheme.py
show_elapsed_time $STEP_START

echo
echo "=============================================="
echo "ETAPA 2: Experimento Principal"
echo "=============================================="
echo "Comparando Baseline vs Filo-Transformer..."
echo

STEP_START=$(date +%s)
python scripts/pheme_real_cascades_experiment.py --seed 42 --folds 5 | tee results/main_experiment.log
show_elapsed_time $STEP_START

echo
echo "=============================================="
echo "ETAPA 3: Análise de Pesos de Fusão"
echo "=============================================="
echo "Analisando pesos aprendidos automaticamente..."
echo

STEP_START=$(date +%s)
python scripts/pheme_real_cascades_experiment.py --analyze-weights | tee results/fusion_weights.log
show_elapsed_time $STEP_START

echo
echo "=============================================="
echo "ETAPA 4: Validação de Hipóteses"
echo "=============================================="
echo "Validando hipóteses do artigo..."
echo

STEP_START=$(date +%s)
python scripts/hypothesis_validation_viz.py | tee results/hypothesis_validation.log
show_elapsed_time $STEP_START

echo
echo "=============================================="
echo "RESUMO DOS RESULTADOS"
echo "=============================================="

# Extrai resultados principais dos logs
echo "📊 Resultado Principal:"
grep -A 10 "Resultados Finais" results/main_experiment.log || echo "Ver results/main_experiment.log"

echo
echo "⚖️ Pesos de Fusão:"
grep -A 5 "Média geral" results/fusion_weights.log || echo "Ver results/fusion_weights.log"

echo
echo "✅ Hipóteses Validadas:"
grep "Hipótese validada" results/hypothesis_validation.log || echo "Ver results/hypothesis_validation.log"

echo
echo "=============================================="
echo "REPRODUÇÃO COMPLETA!"
echo "=============================================="
show_elapsed_time $START_TIME

echo
echo "📂 Arquivos gerados:"
echo "  - datasets/processed/pheme_cascades_with_features.csv"
echo "  - results/main_experiment.log"
echo "  - results/fusion_weights.log"
echo "  - results/hypothesis_validation.log"
echo "  - visualizations/hypothesis_validation/*.html"

echo
echo "🎉 Todos os experimentos foram reproduzidos com sucesso!"