# Filo-Transformer Architecture Documentation

This directory contains comprehensive documentation of the Filo-Transformer codebase architecture, including entity relationships, data flows, and system diagrams.

## 📁 Contents

### 1. Entity Relationship Diagram (`entity-relationship-diagram.mmd`)
Complete ER diagram showing all entities in the system and their relationships, including:
- Core model entities (FTTransformer, BaselineTransformer, etc.)
- Data processing components
- Dataset structures
- Experiment management entities

### 2. Class Diagram (`class-diagram.mmd`)
Object-oriented class structure showing:
- Class attributes and methods
- Inheritance relationships
- Composition and dependencies
- External library integrations

### 3. Sequence Diagram (`sequence-diagram.mmd`)
End-to-end pipeline execution flow:
- Dataset preparation sequence
- Model training workflow
- Experiment execution
- Hypothesis validation process

### 4. Data Flow Diagram (`data-flow-diagram.mmd`)
Visual representation of data transformation:
- Input data sources
- Processing stages
- Feature extraction
- Model architectures
- Output generation

### 5. Component Diagram (`component-diagram.mmd`)
System component architecture:
- Data layer components
- Processing modules
- Model components
- Training infrastructure
- Evaluation systems

### 6. State Diagram (`state-diagram.mmd`)
Pipeline state transitions:
- Initialization states
- Dataset preparation workflow
- Feature extraction process
- Model training states
- Evaluation and validation

### 7. Relationships Documentation (`relationships-documentation.md`)
Detailed textual documentation of all relationships:
- Model inheritance hierarchies
- Data processing pipelines
- Feature generation flows
- Training and evaluation dependencies
- Configuration relationships
- Performance impact analysis

## 🔧 Viewing the Diagrams

These diagrams are in Mermaid format and can be viewed using:

### Online Tools
- [Mermaid Live Editor](https://mermaid.live)
- [GitHub](https://github.com) (renders .mmd files automatically)

### VS Code Extensions
- Mermaid Preview
- Markdown Preview Mermaid Support

### Command Line
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Generate PNG/SVG
mmdc -i entity-relationship-diagram.mmd -o entity-relationship.png
```

## 🏗️ Architecture Overview

### Core Components

1. **Models**
   - `FTTransformer`: Base transformer architecture
   - `FiloTransformerTAGs`: Extended version with fusion mechanisms
   - `BaselineTransformer`: Semantic-only baseline

2. **Data Processing**
   - `PHEMEProcessor`: Basic feature extraction (12 features)
   - `PHEMEAdvancedProcessor`: TAGs feature extraction (70 features)
   - `TAGConstructor`: Phylogenetic analysis engine

3. **Features**
   - Semantic: 384-dimensional SBERT embeddings
   - Phylogenetic Basic: 12 cascade structure features
   - Phylogenetic Advanced: 70 TAGs-based features

4. **Training Infrastructure**
   - 5-fold stratified cross-validation
   - Early stopping mechanism
   - Optimal hyperparameter configuration

## 📊 Key Relationships

### Data Flow
```
PHEME Dataset → Processing → Feature Extraction → Model Training → Evaluation
```

### Model Hierarchy
```
FTTransformer
├── FTTransformerClassifier (sklearn wrapper)
└── FiloTransformerTAGs (advanced fusion)
```

### Feature Pipeline
```
Cascades → Semantic Embeddings (384-dim)
        → Phylogenetic Features (12 or 70-dim)
        → Multi-modal Fusion → Classification
```

## 🎯 Performance Metrics

- **Baseline (Semantic only)**: AUC ~0.90
- **Filo-Transformer**: AUC ~0.92
- **Improvement**: 2-5% across all metrics
- **Fusion Weights**: ~66% phylogenetic, ~34% semantic

## 📝 Notes

- All diagrams represent the current implementation as of the latest commit
- Hyperparameters shown are the optimal configurations from experiments
- External dependencies (PyTorch, scikit-learn, etc.) are marked appropriately
- The architecture supports both CPU and GPU execution