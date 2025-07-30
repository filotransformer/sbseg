"""
pheme_tags_experiment.py

Experimento do Filo-Transformer usando features filogenéticas avançadas (TAGs).
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurações otimizadas
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparâmetros
BATCH_SIZE = 16  # Menor batch size para features mais complexas
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 3
DROPOUT = 0.2
PATIENCE = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Dispositivo: {DEVICE}")

class FiloTransformerTAG(nn.Module):
    """
    Filo-Transformer otimizado para features TAGs.
    """
    def __init__(self, num_semantic_features, num_phylo_features, num_classes=2,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        
        # Normalização de entrada
        self.semantic_norm = nn.LayerNorm(num_semantic_features)
        self.phylo_norm = nn.LayerNorm(num_phylo_features)
        
        # Projeções com regularização
        self.semantic_proj = nn.Sequential(
            nn.Linear(num_semantic_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
        self.phylo_proj = nn.Sequential(
            nn.Linear(num_phylo_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
        # Attention-based fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature enhancement layers
        self.enhancement = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classificador com skip connection
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, semantic_features, phylo_features):
        batch_size = semantic_features.size(0)
        
        # Normalização
        semantic_features = self.semantic_norm(semantic_features)
        phylo_features = self.phylo_norm(phylo_features)
        
        # Projeções
        semantic_embed = self.semantic_proj(semantic_features)
        phylo_embed = self.phylo_proj(phylo_features)
        
        # Cross-attention entre modalidades
        attended_semantic, _ = self.cross_attention(
            semantic_embed.unsqueeze(1),
            phylo_embed.unsqueeze(1),
            phylo_embed.unsqueeze(1)
        )
        attended_semantic = attended_semantic.squeeze(1)
        
        # Fusão com enhancement
        fused = torch.cat([attended_semantic, phylo_embed], dim=-1)
        enhanced = self.enhancement(fused)
        
        # Residual connection
        enhanced = enhanced + self.residual_weight * semantic_embed
        
        # Transformer processing
        tokens = enhanced.unsqueeze(1)
        transformed = self.transformer(tokens)
        
        # Global pooling
        output = transformed.mean(dim=1)
        
        return self.classifier(output)

def load_pheme_tags_data():
    """Carrega dados processados com TAGs"""
    base_path = Path("datasets/processed")
    
    # Carrega features filogenéticas
    df = pd.read_csv(base_path / "pheme_processed_cascades_tags.csv")
    
    # Carrega embeddings semânticos
    embeddings_df = pd.read_pickle(base_path / "pheme_semantic_embeddings.pkl")
    
    # Merge
    df = df.merge(embeddings_df, on='source_tweet_id')
    
    # Extrai features
    phylo_cols = [col for col in df.columns 
                  if any(x in col for x in ['_mean', '_max', '_min', '_std', 
                                           'cascade_', 'num_', 'density_', 'is_', 'components'])]
    
    # Converte embeddings de lista para array
    semantic_features = np.array(df['semantic_embedding'].tolist())
    phylo_features = df[phylo_cols].values
    labels = df['label'].values
    
    print(f"\nFeatures carregadas:")
    print(f"  Semânticas: {semantic_features.shape}")
    print(f"  Filogenéticas TAGs: {phylo_features.shape}")
    print(f"  Total de features filogenéticas: {len(phylo_cols)}")
    
    return semantic_features, phylo_features, labels, phylo_cols

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """Treina o modelo com early stopping e scheduler"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for semantic, phylo, labels in train_loader:
            semantic = semantic.to(DEVICE)
            phylo = phylo.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(semantic, phylo)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for semantic, phylo, labels in val_loader:
                semantic = semantic.to(DEVICE)
                phylo = phylo.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(semantic, phylo)
                probs = torch.softmax(outputs, dim=1)
                
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Métricas
        val_auc = roc_auc_score(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, val_preds)
        
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                  f"Val AUC = {val_auc:.4f}, Val Acc = {val_acc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Restaura melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, test_loader):
    """Avalia o modelo"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for semantic, phylo, labels in test_loader:
            semantic = semantic.to(DEVICE)
            phylo = phylo.to(DEVICE)
            
            outputs = model(semantic, phylo)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'auc': roc_auc_score(all_labels, all_probs)
    }

def main():
    # Carrega dados com TAGs
    print("Carregando dados PHEME com features TAGs...")
    try:
        X_semantic, X_phylo, y, phylo_cols = load_pheme_tags_data()
    except FileNotFoundError:
        print("\n❌ ERRO: Dataset com TAGs não encontrado!")
        print("Execute primeiro: python scripts/process_pheme_with_tags.py")
        return
    
    # Normalização
    scaler_semantic = StandardScaler()
    scaler_phylo = StandardScaler()
    
    X_semantic = scaler_semantic.fit_transform(X_semantic)
    X_phylo = scaler_phylo.fit_transform(X_phylo)
    
    # 5-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    
    print("\n" + "="*60)
    print("EXPERIMENTO FILO-TRANSFORMER COM TAGs")
    print("="*60)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_semantic, y), 1):
        print(f"\nFold {fold}/5")
        print("-" * 40)
        
        # Split data
        X_semantic_train = X_semantic[train_idx]
        X_phylo_train = X_phylo[train_idx]
        y_train = y[train_idx]
        
        X_semantic_test = X_semantic[test_idx]
        X_phylo_test = X_phylo[test_idx]
        y_test = y[test_idx]
        
        # Validation split
        val_size = int(0.2 * len(X_semantic_train))
        val_indices = np.random.choice(len(X_semantic_train), val_size, replace=False)
        train_indices = np.setdiff1d(np.arange(len(X_semantic_train)), val_indices)
        
        # Datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_semantic_train[train_indices]),
            torch.FloatTensor(X_phylo_train[train_indices]),
            torch.LongTensor(y_train[train_indices])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_semantic_train[val_indices]),
            torch.FloatTensor(X_phylo_train[val_indices]),
            torch.LongTensor(y_train[val_indices])
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_semantic_test),
            torch.FloatTensor(X_phylo_test),
            torch.LongTensor(y_test)
        )
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Modelo
        model = FiloTransformerTAG(
            num_semantic_features=X_semantic.shape[1],
            num_phylo_features=X_phylo.shape[1],
            num_classes=2
        ).to(DEVICE)
        
        # Treina
        model = train_model(model, train_loader, val_loader)
        
        # Avalia
        metrics = evaluate_model(model, test_loader)
        results.append(metrics)
        
        print(f"Resultados Fold {fold}:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
    
    # Resultados finais
    print("\n" + "="*60)
    print("RESULTADOS FINAIS (5-fold CV)")
    print("="*60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.upper()}: {mean:.4f} (±{std:.4f})")
    
    # Salva resultados
    results_data = {
        'model': 'Filo-Transformer with TAGs',
        'features': {
            'semantic_dim': int(X_semantic.shape[1]),
            'phylo_dim': int(X_phylo.shape[1]),
            'phylo_features': phylo_cols[:20]  # Primeiras 20 features
        },
        'results': results,
        'average_metrics': {
            metric: {
                'mean': float(np.mean([r[metric] for r in results])),
                'std': float(np.std([r[metric] for r in results]))
            }
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/pheme_tags_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResultados salvos em: results/pheme_tags_results.json")

if __name__ == "__main__":
    main()