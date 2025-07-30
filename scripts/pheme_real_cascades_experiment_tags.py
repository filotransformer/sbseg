"""
pheme_real_cascades_experiment_tags.py

Versão atualizada do experimento principal usando features TAGs avançadas.
Compara Baseline (apenas semântico) vs Filo-Transformer com 70 features filogenéticas.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurações
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Hyperparâmetros otimizados para TAGs
BATCH_SIZE = 16
LEARNING_RATE = 5e-5  # Menor LR para features complexas
NUM_EPOCHS = 100
D_MODEL = 256  # Maior capacidade para 70 features
N_HEADS = 8
N_LAYERS = 3
DROPOUT = 0.25
PATIENCE = 20
WEIGHT_DECAY = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Dispositivo: {DEVICE}")

class BaselineTransformer(nn.Module):
    """Baseline usando apenas features semânticas"""
    def __init__(self, num_features, num_classes=2, d_model=D_MODEL, 
                 n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, features):
        x = self.input_proj(features).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class FiloTransformerTAGs(nn.Module):
    """
    Filo-Transformer otimizado para 70 features TAGs.
    Usa mecanismo de fusão adaptativo com gating.
    """
    def __init__(self, num_semantic_features, num_phylo_features, num_classes=2,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        
        # Feature-specific tokenizers
        self.semantic_tokenizer = nn.Sequential(
            nn.Linear(num_semantic_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
        # Phylogenetic feature groups (para 70 features)
        features_per_group = 10
        num_groups = (num_phylo_features + features_per_group - 1) // features_per_group
        
        self.phylo_tokenizers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(min(features_per_group, num_phylo_features - i * features_per_group), d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.Dropout(dropout)
            )
            for i in range(num_groups)
        ])
        
        # Learnable fusion weights with attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads // 2,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism for adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        max_tokens = 2 + num_groups  # CLS + semantic + phylo groups
        self.pos_encoding = nn.Parameter(torch.randn(1, max_tokens, d_model))
        
        # Transformer with pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head with residual
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Analysis weights
        self.semantic_weight = nn.Parameter(torch.tensor(0.5))
        self.phylo_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, semantic_features, phylo_features):
        batch_size = semantic_features.size(0)
        
        # Tokenize semantic features
        semantic_tokens = self.semantic_tokenizer(semantic_features).unsqueeze(1)
        
        # Tokenize phylogenetic features in groups
        phylo_tokens = []
        features_per_group = 10
        for i, tokenizer in enumerate(self.phylo_tokenizers):
            start_idx = i * features_per_group
            end_idx = min(start_idx + features_per_group, phylo_features.size(1))
            group_features = phylo_features[:, start_idx:end_idx]
            tokens = tokenizer(group_features).unsqueeze(1)
            phylo_tokens.append(tokens)
        
        phylo_tokens = torch.cat(phylo_tokens, dim=1)
        
        # Attention-based fusion
        fused_phylo, _ = self.fusion_attention(
            semantic_tokens,
            phylo_tokens,
            phylo_tokens
        )
        
        # Adaptive gating
        gate_input = torch.cat([
            semantic_tokens.mean(dim=1),
            fused_phylo.mean(dim=1)
        ], dim=-1)
        gate = self.fusion_gate(gate_input).unsqueeze(1)
        
        # Apply gating
        fused_features = semantic_tokens * gate + fused_phylo * (1 - gate)
        
        # Update analysis weights
        with torch.no_grad():
            self.semantic_weight.data = gate.mean().item()
            self.phylo_weight.data = 1 - gate.mean().item()
        
        # Prepare tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, fused_features], dim=1)
        
        # Add positional encoding
        tokens = tokens + self.pos_encoding[:, :tokens.size(1), :]
        
        # Transformer processing
        encoded = self.transformer(tokens)
        
        # Use CLS token with residual
        cls_output = encoded[:, 0]
        pre_output = self.pre_classifier(cls_output)
        
        # Residual connection
        combined = torch.cat([cls_output, pre_output], dim=-1)
        
        return self.classifier(combined)

def load_pheme_tags_data():
    """Carrega dados processados com TAGs"""
    base_path = Path("datasets/processed")
    
    # Verifica se existem os arquivos TAGs
    tags_csv = base_path / "pheme_processed_cascades_tags.csv"
    tags_pkl = base_path / "pheme_semantic_embeddings.pkl"
    
    if not tags_csv.exists() or not tags_pkl.exists():
        print("❌ Arquivos TAGs não encontrados!")
        print("Execute primeiro: python scripts/process_pheme_with_tags.py")
        return None, None, None, None
    
    # Carrega features filogenéticas
    df = pd.read_csv(tags_csv)
    
    # Carrega embeddings semânticos
    embeddings_df = pd.read_pickle(tags_pkl)
    
    # Converte source_tweet_id para o mesmo tipo em ambos os dataframes
    df['source_tweet_id'] = df['source_tweet_id'].astype(str)
    embeddings_df['source_tweet_id'] = embeddings_df['source_tweet_id'].astype(str)
    
    # Merge
    df = df.merge(embeddings_df, on='source_tweet_id')
    
    # Identifica colunas de features
    phylo_cols = [col for col in df.columns 
                  if any(x in col for x in ['_mean', '_max', '_min', '_std', 
                                           'cascade_', 'num_', 'density_', 'is_', 'components'])
                  and col != 'source_tweet_id']
    
    # Extrai features
    semantic_features = np.array(df['semantic_embedding'].tolist())
    phylo_features = df[phylo_cols].values
    labels = df['label'].values
    
    print(f"\nDataset TAGs carregado:")
    print(f"  Total de cascatas: {len(df)}")
    print(f"  Features semânticas: {semantic_features.shape}")
    print(f"  Features filogenéticas TAGs: {phylo_features.shape}")
    print(f"  Distribuição de classes: {np.bincount(labels)}")
    
    return semantic_features, phylo_features, labels, phylo_cols

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    """Treina o modelo com early stopping e regularização"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            if len(batch) == 2:  # Baseline
                features, labels = batch
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
            else:  # Filo-Transformer
                semantic, phylo, labels = batch
                semantic = semantic.to(DEVICE)
                phylo = phylo.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(semantic, phylo)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:  # Baseline
                    features, labels = batch
                    features, labels = features.to(DEVICE), labels.to(DEVICE)
                    outputs = model(features)
                else:  # Filo-Transformer
                    semantic, phylo, labels = batch
                    semantic = semantic.to(DEVICE)
                    phylo = phylo.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(semantic, phylo)
                
                probs = torch.softmax(outputs, dim=1)
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                  f"Val Acc = {val_acc:.4f}, Val AUC = {val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, test_loader):
    """Avalia o modelo no conjunto de teste"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:  # Baseline
                features, labels = batch
                features = features.to(DEVICE)
                outputs = model(features)
            else:  # Filo-Transformer
                semantic, phylo, labels = batch
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
    # Carrega dados
    print("Carregando dados PHEME com TAGs...")
    X_semantic, X_phylo, y, phylo_cols = load_pheme_tags_data()
    
    if X_semantic is None:
        return
    
    # Normalização robusta para features complexas
    scaler_semantic = StandardScaler()
    scaler_phylo = RobustScaler()  # Mais robusto para outliers
    
    X_semantic = scaler_semantic.fit_transform(X_semantic)
    X_phylo = scaler_phylo.fit_transform(X_phylo)
    
    # Configuração do experimento
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    baseline_results = []
    filo_results = []
    fusion_weights = []
    
    print("\n" + "="*60)
    print("EXPERIMENTO: Baseline vs Filo-Transformer com TAGs")
    print("="*60)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_semantic, y), 1):
        print(f"\nFold {fold}/5")
        print("-" * 50)
        
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
        
        # Datasets - Baseline
        train_dataset_baseline = TensorDataset(
            torch.FloatTensor(X_semantic_train[train_indices]),
            torch.LongTensor(y_train[train_indices])
        )
        val_dataset_baseline = TensorDataset(
            torch.FloatTensor(X_semantic_train[val_indices]),
            torch.LongTensor(y_train[val_indices])
        )
        test_dataset_baseline = TensorDataset(
            torch.FloatTensor(X_semantic_test),
            torch.LongTensor(y_test)
        )
        
        # Datasets - Filo-Transformer
        train_dataset_filo = TensorDataset(
            torch.FloatTensor(X_semantic_train[train_indices]),
            torch.FloatTensor(X_phylo_train[train_indices]),
            torch.LongTensor(y_train[train_indices])
        )
        val_dataset_filo = TensorDataset(
            torch.FloatTensor(X_semantic_train[val_indices]),
            torch.FloatTensor(X_phylo_train[val_indices]),
            torch.LongTensor(y_train[val_indices])
        )
        test_dataset_filo = TensorDataset(
            torch.FloatTensor(X_semantic_test),
            torch.FloatTensor(X_phylo_test),
            torch.LongTensor(y_test)
        )
        
        # DataLoaders
        train_loader_baseline = DataLoader(train_dataset_baseline, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_baseline = DataLoader(val_dataset_baseline, batch_size=BATCH_SIZE)
        test_loader_baseline = DataLoader(test_dataset_baseline, batch_size=BATCH_SIZE)
        
        train_loader_filo = DataLoader(train_dataset_filo, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_filo = DataLoader(val_dataset_filo, batch_size=BATCH_SIZE)
        test_loader_filo = DataLoader(test_dataset_filo, batch_size=BATCH_SIZE)
        
        # Treina Baseline
        print("\nTreinando Baseline (apenas semântico)...")
        baseline_model = BaselineTransformer(
            num_features=X_semantic_train.shape[1],
            num_classes=2
        ).to(DEVICE)
        
        baseline_model = train_model(baseline_model, train_loader_baseline, val_loader_baseline)
        baseline_metrics = evaluate_model(baseline_model, test_loader_baseline)
        baseline_results.append(baseline_metrics)
        
        print(f"Baseline - AUC: {baseline_metrics['auc']:.4f}, "
              f"F1: {baseline_metrics['f1']:.4f}, "
              f"Acc: {baseline_metrics['accuracy']:.4f}")
        
        # Treina Filo-Transformer
        print("\nTreinando Filo-Transformer com TAGs...")
        filo_model = FiloTransformerTAGs(
            num_semantic_features=X_semantic_train.shape[1],
            num_phylo_features=X_phylo_train.shape[1],
            num_classes=2
        ).to(DEVICE)
        
        filo_model = train_model(filo_model, train_loader_filo, val_loader_filo)
        filo_metrics = evaluate_model(filo_model, test_loader_filo)
        filo_results.append(filo_metrics)
        
        # Captura pesos de fusão
        semantic_w = filo_model.semantic_weight.item()
        phylo_w = filo_model.phylo_weight.item()
        fusion_weights.append({'semantic': semantic_w, 'phylo': phylo_w})
        
        print(f"Filo-Transformer - AUC: {filo_metrics['auc']:.4f}, "
              f"F1: {filo_metrics['f1']:.4f}, "
              f"Acc: {filo_metrics['accuracy']:.4f}")
        print(f"Pesos de fusão - Semântico: {semantic_w:.3f}, Filogenético: {phylo_w:.3f}")
    
    # Resultados finais
    print("\n" + "="*60)
    print("RESULTADOS FINAIS (5-fold CV)")
    print("="*60)
    
    print("\nBaseline (apenas features semânticas):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in baseline_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric.upper()}: {mean:.4f} (±{std:.4f})")
    
    print("\nFilo-Transformer (semânticas + filogenéticas TAGs):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in filo_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric.upper()}: {mean:.4f} (±{std:.4f})")
    
    print("\nMelhoria do Filo-Transformer sobre Baseline:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        baseline_mean = np.mean([r[metric] for r in baseline_results])
        filo_mean = np.mean([r[metric] for r in filo_results])
        improvement = ((filo_mean - baseline_mean) / baseline_mean) * 100
        print(f"  {metric.upper()}: {improvement:+.2f}%")
    
    print("\nPesos de fusão médios:")
    avg_semantic = np.mean([w['semantic'] for w in fusion_weights])
    avg_phylo = np.mean([w['phylo'] for w in fusion_weights])
    print(f"  Semântico: {avg_semantic:.1%}")
    print(f"  Filogenético: {avg_phylo:.1%}")
    
    # Salva resultados
    results = {
        'baseline': baseline_results,
        'filo_transformer': filo_results,
        'fusion_weights': fusion_weights,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'num_phylo_features': len(phylo_cols),
            'phylo_features_sample': phylo_cols[:20]
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/pheme_real_cascades_tags_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados salvos em: results/pheme_real_cascades_tags_results.json")

if __name__ == "__main__":
    import sys
    
    # Verifica flag --analyze-weights
    if '--analyze-weights' in sys.argv:
        print("\n" + "="*60)
        print("ANÁLISE DE PESOS DE FUSÃO")
        print("="*60)
        print("\nExecutando experimento completo para análise de pesos...")
        print("(Use os resultados salvos em results/pheme_real_cascades_tags_results.json)")
    
    main()