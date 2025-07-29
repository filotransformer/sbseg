# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for the **Filo-Transformer** model, submitted to the **Brazilian Cybersecurity Symposium (SBSeg 2025)**. The project combines phylogenetic Tree Alignment Graphs (TAGs) with Transformers for rumor and fake news identification.

**Key Components:**
- Phylogenetic reconstruction using semantic embeddings
- Tree Alignment Graph (TAG) generation with graph-based features  
- FT-Transformer architecture with dual input (text embeddings + TAG features)
- Evaluation on PHEME dataset (5 events: charliehebdo, ferguson, germanwings, ottawashooting, sydneysiege)

## Development Setup

### Environment Setup
```bash
# Create Python 3.10+ virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Key Dependencies
- `tensorflow>=2.16` - Deep learning framework
- `networkx` - Graph operations and phylogenetic tree construction
- `node2vec` - Graph embeddings for TAG features
- `scikit-learn` - ML utilities and metrics
- `transformers` - HuggingFace transformers for SBERT
- `openai` - GPT embeddings (requires API key)

### API Configuration
Set `OPENAI_API_KEY` environment variable or in `.env` file for GPT embeddings. The system automatically falls back to SBERT if OpenAI API key is unavailable.

## Running Experiments

### Primary Notebook
Execute `filo_transformer_notebook.ipynb` sequentially:
1. **Setup cells** - Install dependencies and configure environment
2. **Filo-Transformer pipeline** - Main model with phylogenetic features
3. **Baseline pipeline** - FT-Transformer without phylogenetic features  
4. **Comparative analysis** - ROC curves, boxplots, radar charts
5. **Graph visualization** - TAG visualization and ego graph analysis

### Dataset Structure
```
datasets/pheme/
├── charliehebdo.csv
├── ferguson.csv  
├── germanwings.csv
├── ottawashooting.csv
├── sydneysiege.csv
└── pheme_all.csv  # Combined dataset
```

Each CSV contains:
- `text` - Tweet content
- `label` - Binary classification (1=rumor/fake, 0=non-rumor/real)

## Architecture Overview

### Filo-Transformer Pipeline
1. **Text Embedding**: GPT-3.5/SBERT embeddings of tweet text
2. **Phylogenetic Reconstruction**: K-NN graph construction based on semantic similarity
3. **TAG Feature Extraction**: 79-dimensional features including:
   - Graph centrality measures (PageRank, betweenness, closeness)
   - Phylogenetic metrics (depth, recombination degree, mutation rate)
   - Community detection features
   - Node2Vec graph embeddings (64-dim)
4. **FT-Transformer**: Dual-input architecture processing text embeddings + TAG features
5. **Classification**: Binary output for rumor detection

### Key Hyperparameters
- `SIMILARITY_VALUE = 0.75` - Threshold for edge creation in phylogenetic graph
- `GPT_MODEL_EMBEDDINGS = "text-embedding-3-large"` - OpenAI embedding model
- 5-fold stratified cross-validation with `random_state=4321`

### Model Architecture Details
- **FT-Transformer blocks**: Multi-head attention + feed-forward networks
- **Input dimensions**: Text embeddings (3072-dim for GPT) + TAG features (79-dim)
- **Training**: Early stopping on validation AUC, learning rate reduction on plateau
- **Optimization**: Adam optimizer with 5e-5 learning rate

## Code Organization

### Core Functions
- `encode_gpt()` / `encode_sbert()` - Text embedding generation
- `attrs_tag_extended()` - Comprehensive TAG feature extraction
- `build_dual_ft()` - Dual-input FT-Transformer construction
- `main_filo()` - Complete pipeline execution with cross-validation

### Graph Analysis
- TAG construction using K-NN graphs with cosine similarity
- Phylogenetic metrics inspired by evolutionary biology
- Graph visualization with NetworkX and matplotlib
- Ego graph analysis for hypothesis testing

### Baseline Comparison  
- `main_BASE()` - FT-Transformer without phylogenetic features
- Direct comparison of architectures on same train/test splits

## Expected Results

**PHEME Dataset Performance:**
- **Filo-Transformer**: AUC 0.9489, F1 0.8393, Accuracy 0.8888
- **Baseline (no phylogeny)**: Lower performance across all metrics
- Cross-validation provides statistical significance testing

## File Structure

```
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies  
├── filo_transformer_notebook.ipynb     # Main experiment notebook
├── datasets/pheme/                     # PHEME dataset files
├── images/                            # Figures and visualizations
│   ├── arquitetura-filo-transformer_new.png
│   ├── radar_new.png
│   └── roc_ok.png
└── docs/                              # Academic paper
    └── paper_sbseg_filo-trans.tex
```

## Important Notes

- The notebook is designed for both local execution and Google Colab
- Graph construction can be memory-intensive for large datasets  
- Cross-validation ensures reproducible results with fixed random seeds
- Visualization functions handle both individual ego graphs and full network views
- The system gracefully handles missing OpenAI API keys by falling back to SBERT

## Research Context

This work addresses rumor detection through phylogenetic analysis of information propagation patterns. The key hypothesis is that fake news exhibits distinct evolutionary patterns (mutations, recombinations) compared to legitimate news when modeled as phylogenetic trees.