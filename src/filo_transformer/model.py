"""
Filo-Transformer and Baseline model implementations.

This module contains the neural network architectures for both the main
Filo-Transformer model and the baseline transformer without phylogenetic features.
"""

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    GlobalAveragePooling1D, Concatenate, Dense, Dropout, 
    LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .config import FiloTransformerConfig


def ft_transformer_block(
    x: tf.Tensor, 
    dim: int, 
    heads: int, 
    ff_dim: int, 
    dropout_rate: float
) -> tf.Tensor:
    """
    Single FT-Transformer block with multi-head attention and feed-forward layers.
    
    Args:
        x: Input tensor
        dim: Model dimension
        heads: Number of attention heads
        ff_dim: Feed-forward layer dimension
        dropout_rate: Dropout rate
        
    Returns:
        Output tensor after transformer block
    """
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=heads, 
        key_dim=dim // heads, 
        dropout=dropout_rate
    )(x, x)
    
    # Add & normalize
    x_att = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed-forward network
    ff_output = Dense(ff_dim, activation='gelu')(x_att)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(dim)(ff_output)
    
    # Add & normalize
    output = LayerNormalization(epsilon=1e-6)(x_att + ff_output)
    
    return output


class FiloTransformer:
    """
    Filo-Transformer model with dual input (text embeddings + TAG features).
    
    This model combines text embeddings with phylogenetic Tree Alignment Graph
    features for enhanced rumor detection performance.
    """
    
    def __init__(self, config: FiloTransformerConfig):
        """
        Initialize Filo-Transformer.
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config
        
    def build_model(self, d_text: int, d_tag: int) -> Model:
        """
        Build the dual-input Filo-Transformer model.
        
        Args:
            d_text: Dimension of text embeddings
            d_tag: Dimension of TAG features
            
        Returns:
            Compiled Keras model
        """
        # Text input
        text_input = Input(shape=(1, d_text), name='text_input')
        inputs = [text_input]
        
        # Start with text representation
        x = text_input
        
        # Add TAG features if available
        if d_tag > 0:
            tag_input = Input(shape=(1, d_tag), name='tag_input')
            inputs.append(tag_input)
            
            # Project TAG features to text embedding dimension
            projected_tag = Dense(
                d_text, 
                activation='relu', 
                name='tag_projection'
            )(tag_input)
            
            # Concatenate text and projected TAG features
            x = Concatenate(axis=1)([text_input, projected_tag])
        
        # Apply transformer blocks
        for i in range(self.config.n_transformer_blocks):
            x = ft_transformer_block(
                x=x,
                dim=d_text,
                heads=self.config.n_attention_heads,
                ff_dim=2 * d_text,
                dropout_rate=self.config.dropout_rate
            )
        
        # Global pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def get_callbacks(self) -> list:
        """
        Get training callbacks for the model.
        
        Returns:
            List of Keras callbacks
        """
        early_stopping = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            patience=self.config.lr_reduction_patience,
            factor=self.config.lr_reduction_factor,
            min_lr=self.config.min_lr,
            verbose=1
        )
        
        return [early_stopping, reduce_lr]


class BaselineTransformer:
    """
    Baseline FT-Transformer model without phylogenetic features.
    
    This model uses only text embeddings for comparison with the full
    Filo-Transformer architecture.
    """
    
    def __init__(self, config: FiloTransformerConfig):
        """
        Initialize Baseline Transformer.
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config
        
    def build_model(self, d_text: int) -> Model:
        """
        Build the single-input baseline transformer model.
        
        Args:
            d_text: Dimension of text embeddings
            
        Returns:
            Compiled Keras model
        """
        # Single text input
        text_input = Input(shape=(1, d_text), name='text_input')
        x = text_input
        
        # Apply transformer blocks
        for i in range(self.config.n_transformer_blocks):
            x = ft_transformer_block(
                x=x,
                dim=d_text,
                heads=self.config.n_attention_heads,
                ff_dim=2 * d_text,
                dropout_rate=self.config.dropout_rate
            )
        
        # Global pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        # Create and compile model
        model = Model(inputs=text_input, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def get_callbacks(self) -> list:
        """
        Get training callbacks for the model.
        
        Returns:
            List of Keras callbacks
        """
        early_stopping = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            patience=self.config.lr_reduction_patience,
            factor=self.config.lr_reduction_factor,
            min_lr=self.config.min_lr,
            verbose=1
        )
        
        return [early_stopping, reduce_lr]