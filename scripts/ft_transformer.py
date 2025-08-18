#!/usr/bin/env python3
"""
FT-Transformer (Feature Tokenizer Transformer) para classificação de fake news.

Implementa o modelo descrito no artigo que processa características semânticas
e filogenéticas através de auto-atenção para classificação binária.

Autor: Acauan Cardoso Ribeiro, Eduardo Luzeiro Feitosa, André Carvalho
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer para dados tabulares.
    
    Arquitetura baseada em Transformer que trata cada característica como um token,
    aplicando auto-atenção entre características semânticas e filogenéticas.
    """
    
    def __init__(self, 
                 n_semantic_features: int,
                 n_phylogenetic_features: int,
                 d_model: int = 192,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 512,
                 dropout: float = 0.1):
        """
        Inicializa o FT-Transformer.
        
        Args:
            n_semantic_features: Número de dimensões do embedding semântico
            n_phylogenetic_features: Número de características filogenéticas
            d_model: Dimensão interna do modelo
            n_heads: Número de cabeças de atenção
            n_layers: Número de camadas do Transformer
            d_ff: Dimensão da rede feed-forward
            dropout: Taxa de dropout
        """
        super().__init__()
        
        self.n_semantic_features = n_semantic_features
        self.n_phylogenetic_features = n_phylogenetic_features
        self.d_model = d_model
        
        # Tokenização de características
        # Projeção linear para características semânticas
        self.semantic_tokenizer = nn.Linear(n_semantic_features, d_model)
        
        # Projeção linear individual para cada característica filogenética
        self.phylogenetic_tokenizers = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_phylogenetic_features)
        ])
        
        # Token CLS para agregação global
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding (learnable)
        max_tokens = 1 + 1 + n_phylogenetic_features  # CLS + semantic + phylogenetic
        self.positional_encoding = nn.Parameter(torch.randn(1, max_tokens, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Camada de classificação
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(), 
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, 1)
        )
        
        # Inicialização
        self._init_weights()
    
    def _init_weights(self):
        """
        Inicializa pesos do modelo usando Xavier/Glorot para camadas lineares.
        
        Returns:
            None
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias) 
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Inicialização especial para CLS token e positional encoding
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_encoding, std=0.02)
    
    def forward(self, semantic_features: torch.Tensor, 
                phylogenetic_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do FT-Transformer.
        
        Args:
            semantic_features: Tensor de embeddings semânticos (batch_size, n_semantic_features)
            phylogenetic_features: Tensor de características filogenéticas (batch_size, n_phylogenetic_features)
            
        Returns:
            Logits para classificação binária (batch_size, 1)
        """
        batch_size = semantic_features.size(0)
        
        # 1. Tokenização
        tokens = []
        
        # Token CLS
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls_tokens)
        
        # Token semântico (todo o embedding como um único token)
        semantic_token = self.semantic_tokenizer(semantic_features).unsqueeze(1)
        tokens.append(semantic_token)
        
        # Tokens filogenéticos (um token por característica)
        if self.n_phylogenetic_features > 0:
            for i in range(self.n_phylogenetic_features):
                feature = phylogenetic_features[:, i:i+1] 
                token = self.phylogenetic_tokenizers[i](feature).unsqueeze(1)
                tokens.append(token)
        
        # Concatenar todos os tokens
        x = torch.cat(tokens, dim=1)  # (batch_size, n_tokens, d_model)
        
        # 2. Adicionar positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # 3. Passar pelo Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 4. Usar token CLS para classificação
        cls_output = x[:, 0, :]  # (batch_size, d_model)
        
        # 5. Classificação
        logits = self.classifier(cls_output)  # (batch_size, 1)
        
        return logits
    
    def predict_proba(self, semantic_features: torch.Tensor,
                     phylogenetic_features: torch.Tensor) -> torch.Tensor:
        """
        Prediz probabilidades usando sigmoid.
        
        Args:
            semantic_features: Tensor de embeddings semânticos (batch_size, n_semantic_features)
            phylogenetic_features: Tensor de características filogenéticas (batch_size, n_phylogenetic_features)
        
        Returns:
            torch.Tensor: Probabilidades no intervalo [0, 1] onde 1 indica fake news
        """
        logits = self.forward(semantic_features, phylogenetic_features)
        return torch.sigmoid(logits)


class FTTransformerClassifier:
    """
    Wrapper sklearn-like para o FT-Transformer.
    
    Fornece interface compatível com scikit-learn para facilitar integração
    com pipelines de machine learning existentes.
    """
    
    def __init__(self,
                 n_semantic_features: int,
                 n_phylogenetic_features: int,
                 d_model: int = 192,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 n_epochs: int = 50,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 early_stopping_patience: int = 10,
                 verbose: bool = True):
        """
        Inicializa o classificador FT-Transformer.
        
        Args:
            n_semantic_features: Número de dimensões do embedding semântico
            n_phylogenetic_features: Número de características filogenéticas
            d_model: Dimensão interna do modelo
            n_heads: Número de cabeças de atenção
            n_layers: Número de camadas do Transformer
            d_ff: Dimensão da rede feed-forward
            dropout: Taxa de dropout
            learning_rate: Taxa de aprendizado para o otimizador
            batch_size: Tamanho do batch para treinamento
            n_epochs: Número de épocas de treinamento
            device: Dispositivo para execução ('cuda' ou 'cpu')
            early_stopping_patience: Paciência para early stopping
            verbose: Se True, imprime informações durante treinamento
        """
        self.n_semantic_features = n_semantic_features
        self.n_phylogenetic_features = n_phylogenetic_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        
        self.model = None
        self.optimizer = None
        self.best_model_state = None
        
    def _prepare_data(self, X_semantic: np.ndarray, X_phylo: np.ndarray, 
                     y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepara dados numpy para tensores PyTorch.
        
        Args:
            X_semantic: Array numpy com embeddings semânticos
            X_phylo: Array numpy com características filogenéticas
            y: Array numpy com labels (opcional)
        
        Returns:
            tuple: Tensores PyTorch (X_semantic, X_phylo, y ou None)
        """
        X_semantic_tensor = torch.FloatTensor(X_semantic).to(self.device)
        X_phylo_tensor = torch.FloatTensor(X_phylo).to(self.device)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
            return X_semantic_tensor, X_phylo_tensor, y_tensor
        
        return X_semantic_tensor, X_phylo_tensor, None
    
    def fit(self, X_semantic: np.ndarray, X_phylo: np.ndarray, y: np.ndarray,
            X_val_semantic: Optional[np.ndarray] = None,
            X_val_phylo: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):
        """
        Treina o modelo FT-Transformer.
        
        Args:
            X_semantic: Embeddings semânticos de treino
            X_phylo: Características filogenéticas de treino
            y: Labels de treino (0 ou 1)
            X_val_semantic: Embeddings semânticos de validação (opcional)
            X_val_phylo: Características filogenéticas de validação (opcional)
            y_val: Labels de validação (opcional)
        """
        # Inicializar modelo
        self.model = FTTransformer(
            n_semantic_features=self.n_semantic_features,
            n_phylogenetic_features=self.n_phylogenetic_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Scheduler para reduzir learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Calcular class weights baseado na distribuição real
        pos_count = np.sum(y)
        neg_count = len(y) - pos_count
        pos_weight = torch.tensor([neg_count / pos_count]).to(self.device) if pos_count > 0 else torch.tensor([1.0]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Preparar dados
        X_sem_tensor, X_phylo_tensor, y_tensor = self._prepare_data(X_semantic, X_phylo, y)
        
        # Validação
        use_validation = X_val_semantic is not None and X_val_phylo is not None and y_val is not None
        if use_validation:
            X_val_sem_tensor, X_val_phylo_tensor, y_val_tensor = self._prepare_data(
                X_val_semantic, X_val_phylo, y_val
            )
        
        # Treinamento
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            # Mini-batches
            indices = torch.randperm(len(X_sem_tensor))
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                
                X_sem_batch = X_sem_tensor[batch_indices]
                X_phylo_batch = X_phylo_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                
                self.optimizer.zero_grad()
                outputs = self.model(X_sem_batch, X_phylo_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches
            
            # Validation
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_sem_tensor, X_val_phylo_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # Scheduler step
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if self.verbose and epoch % 10 == 0:
                if use_validation:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Restaurar melhor modelo se validação foi usada
        if use_validation and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self
    
    def predict_proba(self, X_semantic: np.ndarray, X_phylo: np.ndarray) -> np.ndarray:
        """
        Prediz probabilidades.
        
        Returns:
            Array com probabilidades para classe negativa e positiva
        """
        self.model.eval()
        X_sem_tensor, X_phylo_tensor, _ = self._prepare_data(X_semantic, X_phylo)
        
        with torch.no_grad():
            proba_positive = self.model.predict_proba(X_sem_tensor, X_phylo_tensor)
            proba_positive = proba_positive.cpu().numpy()
        
        # Retornar probabilidades para ambas as classes
        proba_negative = 1 - proba_positive
        return np.hstack([proba_negative, proba_positive])
    
    def predict(self, X_semantic: np.ndarray, X_phylo: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Prediz classes.
        
        Args:
            X_semantic: Embeddings semânticos
            X_phylo: Características filogenéticas  
            threshold: Limiar de decisão (default 0.5)
        
        Returns:
            Array com predições binárias (0 ou 1)
        """
        self.model.eval()
        X_sem_tensor, X_phylo_tensor, _ = self._prepare_data(X_semantic, X_phylo)
        
        with torch.no_grad():
            logits = self.model(X_sem_tensor, X_phylo_tensor)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
        
        return (proba >= threshold).astype(int)