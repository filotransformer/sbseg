"""
Text embedding modules for Filo-Transformer.

This module provides implementations for both GPT and SBERT text embeddings
used in the phylogenetic analysis pipeline.
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import openai
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

from .config import FiloTransformerConfig

warnings.filterwarnings('ignore', '.*set_learning_phase.*', category=UserWarning)


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        pass


class GPTEmbedder(BaseEmbedder):
    """
    GPT-based text embedder using OpenAI's embedding API.
    
    This class provides text embeddings using OpenAI's text-embedding models,
    which are used as the semantic foundation for phylogenetic reconstruction.
    """
    
    def __init__(self, config: FiloTransformerConfig):
        """
        Initialize GPT embedder.
        
        Args:
            config: Configuration object containing API key and model settings
            
        Raises:
            RuntimeError: If OpenAI API key is not available
        """
        self.config = config
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not available for GPT embeddings")
        
        openai.api_key = config.openai_api_key
        self.model = config.gpt_model
        self.batch_size = config.embedding_batch_size
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate GPT embeddings for input texts.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Normalized embeddings array with shape (n_texts, embedding_dim)
        """
        print(f"Generating GPT embeddings using {self.model}...")
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = openai.embeddings.create(
                model=self.model,
                input=batch
            )
            
            batch_embeddings = [
                datum.embedding for datum in response.data
            ]
            embeddings.append(np.array(batch_embeddings, dtype=np.float32))
            
            # Respect rate limits
            time.sleep(1)
        
        result = np.vstack(embeddings)
        print(f"Generated {result.shape[0]} embeddings with dimension {result.shape[1]}")
        return result


class SBERTEmbedder(BaseEmbedder):
    """
    SBERT-based text embedder using HuggingFace transformers.
    
    This class provides text embeddings using Sentence-BERT models as a fallback
    when OpenAI API is not available.
    """
    
    def __init__(self, config: FiloTransformerConfig):
        """
        Initialize SBERT embedder.
        
        Args:
            config: Configuration object containing model settings
        """
        self.config = config
        self.batch_size = config.embedding_batch_size
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.sbert_model)
        self.model = TFAutoModel.from_pretrained(
            config.sbert_model, 
            from_pt=True
        )
        self.model.trainable = False
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate SBERT embeddings for input texts.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Normalized embeddings array with shape (n_texts, embedding_dim)
        """
        print(f"Generating SBERT embeddings using {self.config.sbert_model}...")
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='tf',
                max_length=512
            )
            
            # Get model outputs
            outputs = self.model(encoded)
            
            # Mean pooling with attention mask
            attention_mask = tf.cast(
                encoded['attention_mask'][..., None], 
                tf.float32
            )
            summed = tf.reduce_sum(
                outputs.last_hidden_state * attention_mask, 
                axis=1
            )
            counts = tf.reduce_sum(attention_mask, axis=1)
            
            batch_embeddings = (summed / counts).numpy()
            embeddings.append(batch_embeddings)
        
        result = np.vstack(embeddings)
        print(f"Generated {result.shape[0]} embeddings with dimension {result.shape[1]}")
        return result


def get_embedder(config: FiloTransformerConfig) -> BaseEmbedder:
    """
    Factory function to get appropriate embedder based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        GPTEmbedder if API key available, otherwise SBERTEmbedder
    """
    if config.use_gpt_embeddings:
        return GPTEmbedder(config)
    else:
        print("OpenAI API key not available. Using SBERT embeddings.")
        return SBERTEmbedder(config)