"""
Configuration settings for Filo-Transformer experiments.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class FiloTransformerConfig:
    """Configuration parameters for Filo-Transformer model and experiments."""
    
    # Dataset settings
    dataset_path: str = "datasets/pheme/pheme_all.csv"
    
    # Embedding settings  
    gpt_model: str = "text-embedding-3-large"
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_batch_size: int = 100
    
    # Graph construction
    similarity_threshold: float = 0.75
    knn_k: int = 5
    
    # Model architecture
    n_transformer_blocks: int = 2
    n_attention_heads: int = 8
    dropout_rate: float = 0.2
    learning_rate: float = 5e-5
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    cv_folds: int = 5
    random_state: int = 4321
    early_stopping_patience: int = 10
    lr_reduction_patience: int = 3
    lr_reduction_factor: float = 0.2
    min_lr: float = 1e-7
    
    # TAG features
    node2vec_dimensions: int = 64
    node2vec_walk_length: int = 10
    node2vec_num_walks: int = 50
    node2vec_window: int = 5
    node2vec_min_count: int = 1
    
    # OpenAI API
    openai_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load OpenAI API key from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
    
    @property
    def use_gpt_embeddings(self) -> bool:
        """Whether to use GPT embeddings (if API key available)."""
        return self.openai_api_key is not None
        
    @property
    def embedding_model_name(self) -> str:
        """Get the embedding model name being used."""
        return self.gpt_model if self.use_gpt_embeddings else self.sbert_model