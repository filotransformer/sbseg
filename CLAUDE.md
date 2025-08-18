# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic research implementation of **Filo-Transformer**, a novel deep learning architecture for fake news detection that combines semantic and phylogenetic features from social media cascades. The project analyzes tweet propagation patterns using the PHEME dataset.

## Essential Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Prepare dataset (required before first run)
python scripts/prepare_dataset.py
```

### Testing and Validation
```bash
# Quick functionality test
python scripts/quick_test.py

# Verify dataset integrity
python scripts/verify_dataset_integrity.py
```

### Running Experiments
```bash
# Main optimized experiment
python scripts/main_experiment.py

# Run all experiments (full reproduction)
bash scripts/reproduce_all.sh

# Individual experiments
python scripts/pheme_real_cascades_experiment.py          # Baseline (semantic only)
python scripts/pheme_real_cascades_experiment_tags.py     # Filo-Transformer (semantic + phylogenetic)
python scripts/hypothesis_validation_viz.py               # Statistical validation
```

## Architecture Overview

### Core Model: Filo-Transformer
- **Location**: `scripts/ft_transformer.py`
- **Design**: Feature Tokenizer Transformer that fuses semantic embeddings (384-dim) with phylogenetic features (12 or 70-dim TAGs)
- **Key Innovation**: Automatic learning of modality importance through attention mechanisms

### Data Processing Pipeline
1. **PHEME Dataset Processing**: `scripts/process_pheme.py` and `scripts/process_pheme_with_tags.py`
   - Extracts tweet cascades and builds propagation trees
   - Generates semantic embeddings using Sentence-BERT
   
2. **Phylogenetic Feature Extraction**: `scripts/tag_construction.py`
   - Creates Tree Alignment Graphs (TAGs) from cascade structures
   - Computes 12 basic or 70 extended phylogenetic features

### Experimental Framework
- **Cross-validation**: 5-fold stratified setup in all experiments
- **Hyperparameters**: Optimal configurations stored in `OPTIMAL_CONFIG` dictionary
- **Metrics**: AUC, F1, Precision, Recall with statistical significance testing

### Key Files Structure
```
scripts/
├── ft_transformer.py           # Filo-Transformer model implementation
├── process_pheme*.py           # Dataset processing modules
├── tag_construction.py         # Phylogenetic feature extraction
├── main_experiment.py          # Optimized main experiment
├── pheme_real_cascades_*.py   # Individual experiment scripts
└── hypothesis_validation_viz.py # Statistical analysis
```

## Important Considerations

- **Python Version**: Requires Python 3.8+
- **GPU Support**: Automatically detects and uses CUDA if available
- **Memory Requirements**: Dataset processing may require significant RAM (8GB+ recommended)
- **Reproducibility**: All experiments use fixed random seeds for deterministic results
- **Windows Users**: Must use WSL2 (see SETUP_WSL.md for detailed instructions)