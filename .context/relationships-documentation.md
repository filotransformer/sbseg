# Filo-Transformer Entity Relationships Documentation

## Overview
This document details all relationships between entities in the Filo-Transformer codebase for fake news detection using semantic and phylogenetic features.

## Core Relationships

### 1. Model Inheritance and Composition

#### FTTransformer ← FiloTransformerTAGs
- **Type**: Inheritance/Extension
- **Description**: FiloTransformerTAGs extends FTTransformer with advanced fusion mechanisms
- **Key Additions**:
  - Multi-head attention fusion
  - Sigmoid gating mechanism
  - Grouped phylogenetic tokenizers (10 features per group)
  - Fusion weight tracking and analysis

#### FTTransformer → FTTransformerClassifier
- **Type**: Composition/Wrapper
- **Description**: FTTransformerClassifier wraps FTTransformer with sklearn-compatible interface
- **Purpose**: Provides training loop, early stopping, and standard ML interface

### 2. Data Processing Pipeline

#### PHEMEDataset → PHEMEProcessor → Cascade
- **Type**: One-to-Many Processing
- **Flow**: Raw dataset → Processor → Individual cascades
- **Operations**:
  - Extract JSON tweets
  - Build propagation trees
  - Generate cascade objects

#### PHEMEProcessor ← PHEMEAdvancedProcessor
- **Type**: Inheritance/Extension
- **Enhanced Features**:
  - TAGConstructor integration
  - SentenceTransformer embeddings
  - 70 phylogenetic features (vs 12 basic)

#### PHEMEAdvancedProcessor → TAGConstructor
- **Type**: Dependency/Usage
- **Purpose**: Extract advanced phylogenetic features
- **Data Flow**: Cascade tree → TAGs → 70 features

### 3. Feature Generation

#### Cascade → SemanticFeatures
- **Type**: One-to-One Generation
- **Process**: Concatenate source + reaction texts → SBERT encoding → 384-dim vectors
- **Model**: all-MiniLM-L6-v2

#### Cascade → PhylogeneticFeatures
- **Type**: One-to-One Extraction
- **Basic**: 12 structural features (size, depth, breadth, etc.)
- **Advanced**: 70 TAGs features (centrality, community, evolution)

#### TAGConstructor → PhylogeneticFeatures
- **Type**: Many-to-One Aggregation
- **Process**: Graph metrics → Statistical aggregation (mean, max, min, std)

### 4. Model Input Relationships

#### BaselineTransformer ← SemanticFeatures
- **Type**: Single Input Processing
- **Dimension**: 384 → 256 (through tokenizer)
- **Purpose**: Semantic-only baseline

#### FiloTransformerTAGs ← SemanticFeatures + PhylogeneticFeatures
- **Type**: Multi-Modal Input Processing
- **Semantic**: 384 → 256 (single tokenizer)
- **Phylogenetic**: 70 × 1 → 70 × 256 (individual tokenizers)
- **Fusion**: Attention-based combination

### 5. Training and Evaluation

#### ExperimentRunner → Models
- **Type**: Orchestration
- **Models Evaluated**:
  - BaselineTransformer (semantic only)
  - FiloTransformerTAGs (full model)
- **Process**: 5-fold cross-validation with early stopping

#### ExperimentRunner → OptimalConfig
- **Type**: Configuration Dependency
- **Parameters**:
  - Batch size: 16
  - Learning rate: 3e-5
  - Model dimensions: 256
  - Attention heads: 8
  - Transformer layers: 3

#### HypothesisValidator → ExperimentRunner
- **Type**: Validation Dependency
- **Purpose**: Statistical validation of scientific hypotheses
- **Outputs**: Interactive visualizations and significance tests

## Data Flow Relationships

### Processing Pipeline
1. **Raw Data**: PHEME JSON files
2. **Extraction**: PHEMEProcessor.extract_tweet_data()
3. **Tree Building**: PHEMEProcessor.build_cascade_tree()
4. **Feature Extraction**:
   - Basic: extract_cascade_features() → 12 features
   - Advanced: TAGConstructor.extract_phylogenetic_features() → 70 features
5. **Semantic Encoding**: SentenceTransformer.encode() → 384-dim vectors

### Training Pipeline
1. **Data Loading**: Features → DataLoader (batch_size=16)
2. **Forward Pass**:
   - Tokenization: Features → Token embeddings
   - Attention: Multi-head self-attention
   - Fusion: Modality combination (Filo-Transformer only)
   - Classification: Final linear layer → Binary output
3. **Optimization**: Adam optimizer (lr=3e-5)
4. **Early Stopping**: Patience=15 epochs

### Inference Pipeline
1. **New Cascade**: Raw tweet data
2. **Feature Extraction**:
   - Semantic: Text → SBERT → 384-dim
   - Phylogenetic: Tree → TAGs → 70-dim
3. **Model Prediction**:
   - Tokenization and fusion
   - Transformer processing
   - Binary classification (fake/real)

## Key Dependency Relationships

### External Dependencies
- **PyTorch**: Core deep learning framework
- **scikit-learn**: ML utilities, metrics, cross-validation
- **NetworkX**: Graph operations for TAGs
- **SentenceTransformers**: Semantic embeddings
- **Pandas/NumPy**: Data manipulation

### Internal Module Dependencies
```
ft_transformer.py
├── Used by: FTTransformerClassifier, FiloTransformerTAGs
└── Dependencies: PyTorch

tag_construction.py
├── Used by: PHEMEAdvancedProcessor
└── Dependencies: NetworkX, NumPy

process_pheme.py
├── Used by: prepare_dataset.py
└── Dependencies: Pandas, JSON, NetworkX

main_experiment.py
├── Uses: All model classes, processors
└── Dependencies: scikit-learn, PyTorch

hypothesis_validation_viz.py
├── Uses: Experiment results
└── Dependencies: Plotly, SciPy
```

## Performance Impact Relationships

### Feature Importance
- **Semantic Features**: ~34% average importance
- **Phylogenetic Features**: ~66% average importance
- **Learned automatically** through attention mechanism

### Model Performance
- **Baseline (Semantic only)**: AUC ~0.90
- **Filo-Transformer (Both)**: AUC ~0.92
- **Improvement**: 2-5% across all metrics

### Computational Relationships
- **CPU Processing**: Viable but slower
- **GPU Processing**: 2-3x speedup
- **Memory Requirements**: 8GB minimum
- **Processing Time**: ~15-20 minutes (main experiment)

## Configuration Relationships

### Hyperparameter Dependencies
- **d_model (256)** → All tokenizer output dimensions
- **n_heads (8)** → Must divide d_model evenly
- **n_layers (3)** → Transformer depth
- **dropout (0.2)** → Applied throughout model
- **batch_size (16)** → Memory usage and convergence

### Dataset Relationships
- **5 Events**: charliehebdo, ferguson, germanwings-crash, ottawashooting, sydneysiege
- **2 Classes**: rumours (1), non-rumours (0)
- **5802 Total Cascades**: Varying sizes and structures
- **Cross-validation**: Stratified by class label

## Validation Relationships

### Hypothesis Testing
1. **H2.1**: Terminal leaves → Higher fake news probability
2. **H3.2**: Filo-Transformer → Better than baseline
3. **H4.2**: Cascade structure → Correlates with veracity
4. **H5.2**: Verified users → Lower fake news spread

### Statistical Tests
- **Paired t-test**: Model comparison
- **Correlation analysis**: Feature-label relationships
- **Mann-Whitney U**: Distribution comparisons
- **Effect size**: Cohen's d calculations