"""
pheme_real_cascades_experiment.py

Experimento principal do Filo-Transformer no dataset PHEME com features reais de cascata.

Este script implementa e compara dois modelos:
1. Baseline: Usa apenas embeddings semânticos (GPT-2)
2. Filo-Transformer: Combina embeddings semânticos com features filogenéticas

Funcionalidades principais:
- Carrega dados pré-processados do PHEME com features de cascata
- Treina modelos usando validação cruzada 5-fold
- Compara performance entre baseline e Filo-Transformer
- Analisa pesos de fusão aprendidos automaticamente
- Salva resultados em formato JSON

Uso:
    python pheme_real_cascades_experiment.py [--seed SEED] [--folds FOLDS] [--analyze-weights]
    
Parâmetros opcionais:
    --seed: Semente aleatória (default: 42)
    --folds: Número de folds para CV (default: 5)
    --analyze-weights: Apenas analisa pesos de fusão sem treinar

Saída:
    - Resultados detalhados no console
    - Arquivo JSON com todas as métricas: pheme_real_cascades_results.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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

# Hyperparâmetros otimizados
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 100
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.1
PATIENCE = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Dispositivo: {DEVICE}")

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer (FT-Transformer) base.
    
    Arquitetura que tokeniza features contínuas e usa self-attention
    para aprender representações complexas dos dados.
    
    Args:
        num_continuous_features (int): Número de features contínuas de entrada
        num_classes (int): Número de classes para classificação (default: 2)
        d_model (int): Dimensão do modelo/embeddings (default: 256)
        n_heads (int): Número de cabeças de atenção (default: 8)
        n_layers (int): Número de camadas do transformer (default: 4)
        dropout (float): Taxa de dropout (default: 0.1)
    """
    def __init__(self, num_continuous_features, num_classes=2, d_model=D_MODEL, 
                 n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        
        # Feature tokenizer para features contínuas
        self.continuous_embedder = nn.Linear(num_continuous_features, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, continuous_features):
        """
        Forward pass do FT-Transformer.
        
        Args:
            continuous_features (torch.Tensor): Features contínuas [batch_size, num_features]
            
        Returns:
            torch.Tensor: Logits de saída [batch_size, num_classes]
        """
        batch_size = continuous_features.size(0)
        
        # Embed features contínuas
        continuous_tokens = self.continuous_embedder(continuous_features.unsqueeze(1))
        
        # Adiciona CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, continuous_tokens], dim=1)
        
        # Transformer encoding
        output = self.transformer(tokens)
        
        # Usa apenas CLS token para classificação
        cls_output = output[:, 0]
        
        return self.classifier(cls_output)

class FiloTransformer(nn.Module):
    """
    Filo-Transformer: Modelo que funde features semânticas e filogenéticas.
    
    Implementa fusão aprendível entre dois tipos de features:
    - Semânticas: Embeddings de texto (GPT-2/BERT)
    - Filogenéticas: Estrutura da cascata (profundidade, ramificação, etc.)
    
    Os pesos de fusão são aprendidos automaticamente durante o treinamento,
    permitindo que o modelo descubra a importância relativa de cada tipo.
    
    Args:
        num_semantic_features (int): Dimensão dos embeddings semânticos
        num_phylo_features (int): Número de features filogenéticas
        num_classes (int): Número de classes (default: 2)
        d_model (int): Dimensão do modelo (default: 256)
        n_heads (int): Número de cabeças de atenção (default: 8)
        n_layers (int): Número de camadas (default: 4)
        dropout (float): Taxa de dropout (default: 0.1)
    """
    def __init__(self, num_semantic_features, num_phylo_features, num_classes=2,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        
        # Embedders separados para cada tipo de feature
        self.semantic_embedder = nn.Linear(num_semantic_features, d_model)
        self.phylo_embedder = nn.Linear(num_phylo_features, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Pesos aprendíveis para fusão (inicializa phylo com peso 2x)
        self.semantic_weight = nn.Parameter(torch.tensor(1.0))
        self.phylo_weight = nn.Parameter(torch.tensor(2.0))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, semantic_features, phylo_features):
        """
        Forward pass com fusão de features.
        
        Args:
            semantic_features (torch.Tensor): Features semânticas [batch_size, semantic_dim]
            phylo_features (torch.Tensor): Features filogenéticas [batch_size, phylo_dim]
            
        Returns:
            torch.Tensor: Logits de saída [batch_size, num_classes]
        """
        batch_size = semantic_features.size(0)
        
        # Embed features separadamente
        semantic_tokens = self.semantic_embedder(semantic_features.unsqueeze(1))
        phylo_tokens = self.phylo_embedder(phylo_features.unsqueeze(1))
        
        # Aplica pesos e normaliza
        weights = torch.softmax(torch.stack([self.semantic_weight, self.phylo_weight]), dim=0)
        semantic_tokens = semantic_tokens * weights[0]
        phylo_tokens = phylo_tokens * weights[1]
        
        # CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatena todos os tokens
        tokens = torch.cat([cls_tokens, semantic_tokens, phylo_tokens], dim=1)
        
        # Transformer encoding
        output = self.transformer(tokens)
        
        # Usa CLS token para classificação
        cls_output = output[:, 0]
        
        return self.classifier(cls_output)

def load_pheme_data():
    """
    Carrega dados pré-processados do PHEME com features de cascata.
    
    Lê os arquivos gerados por process_pheme.py:
    - pheme_processed_cascades.csv: Dados principais com embeddings e features
    - pheme_metadata.json: Metadados sobre as colunas
    
    Returns:
        tuple: (semantic_features, phylo_features, labels)
            - semantic_features: Array com embeddings GPT-2 [n_samples, 768]
            - phylo_features: Array com features de cascata [n_samples, n_phylo_features]
            - labels: Array com labels binários [n_samples]
    """
    base_path = Path("/home/acauan/ufam/papers/01_sbseg_filo_trans/datasets/processed")
    
    # Carrega dataset principal
    df = pd.read_csv(base_path / "pheme_processed_cascades.csv")
    
    # Carrega metadados para saber quais são as features de cascata
    with open(base_path / "pheme_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    cascade_features = metadata['cascade_features']
    
    print(f"Total de cascatas: {len(df)}")
    print(f"Features de cascata disponíveis: {cascade_features}")
    
    return df, cascade_features

def prepare_features(df, cascade_features, sentence_model):
    """Prepara features semânticas e filogenéticas"""
    
    # Features semânticas: embeddings do texto fonte
    print("Gerando embeddings semânticos...")
    texts = df['source_text'].tolist()
    semantic_features = sentence_model.encode(texts, batch_size=32, show_progress_bar=True)
    
    # Features filogenéticas: características da cascata
    phylo_features = df[cascade_features].values
    
    # Labels
    labels = df['label'].values
    
    return semantic_features, phylo_features, labels

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    """Treina o modelo com early stopping"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
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
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
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
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Restaura melhor modelo
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
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
            else:  # Filo-Transformer
                semantic, phylo, labels = batch
                semantic = semantic.to(DEVICE)
                phylo = phylo.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(semantic, phylo)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcula métricas
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    }
    
    return metrics

def main():
    # Carrega dados
    print("Carregando dados PHEME processados...")
    df, cascade_features = load_pheme_data()
    
    # Inicializa modelo de embeddings
    print("Carregando modelo de embeddings...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepara features
    semantic_features, phylo_features, labels = prepare_features(df, cascade_features, sentence_model)
    
    print(f"\nDimensões:")
    print(f"  Features semânticas: {semantic_features.shape}")
    print(f"  Features filogenéticas: {phylo_features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Distribuição de classes: {np.bincount(labels)}")
    
    # Normaliza features
    semantic_scaler = StandardScaler()
    phylo_scaler = StandardScaler()
    
    # 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    baseline_results = []
    filo_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(semantic_features, labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*50}")
        
        # Split dos dados
        X_semantic_train = semantic_scaler.fit_transform(semantic_features[train_idx])
        X_semantic_test = semantic_scaler.transform(semantic_features[test_idx])
        
        X_phylo_train = phylo_scaler.fit_transform(phylo_features[train_idx])
        X_phylo_test = phylo_scaler.transform(phylo_features[test_idx])
        
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        # Validação split
        val_size = int(len(X_semantic_train) * 0.2)
        val_indices = np.random.choice(len(X_semantic_train), val_size, replace=False)
        train_indices = np.setdiff1d(np.arange(len(X_semantic_train)), val_indices)
        
        # Datasets para baseline (apenas semântico)
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
        
        # Datasets para Filo-Transformer
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
        baseline_model = FTTransformer(
            num_continuous_features=X_semantic_train.shape[1],
            num_classes=2
        ).to(DEVICE)
        
        baseline_model = train_model(baseline_model, train_loader_baseline, val_loader_baseline)
        baseline_metrics = evaluate_model(baseline_model, test_loader_baseline)
        baseline_results.append(baseline_metrics)
        
        print(f"Baseline - AUC: {baseline_metrics['auc']:.4f}, "
              f"F1: {baseline_metrics['f1']:.4f}, "
              f"Acc: {baseline_metrics['accuracy']:.4f}")
        
        # Treina Filo-Transformer
        print("\nTreinando Filo-Transformer...")
        filo_model = FiloTransformer(
            num_semantic_features=X_semantic_train.shape[1],
            num_phylo_features=X_phylo_train.shape[1],
            num_classes=2
        ).to(DEVICE)
        
        filo_model = train_model(filo_model, train_loader_filo, val_loader_filo)
        filo_metrics = evaluate_model(filo_model, test_loader_filo)
        filo_results.append(filo_metrics)
        
        print(f"Filo-Transformer - AUC: {filo_metrics['auc']:.4f}, "
              f"F1: {filo_metrics['f1']:.4f}, "
              f"Acc: {filo_metrics['accuracy']:.4f}")
        
        # Imprime pesos de fusão
        with torch.no_grad():
            weights = torch.softmax(
                torch.stack([filo_model.semantic_weight, filo_model.phylo_weight]), 
                dim=0
            )
            print(f"Pesos de fusão - Semântico: {weights[0].item():.3f}, "
                  f"Filogenético: {weights[1].item():.3f}")
    
    # Resultados finais
    print(f"\n{'='*60}")
    print("RESULTADOS FINAIS (5-fold CV)")
    print(f"{'='*60}")
    
    # Baseline
    print("\nBaseline (apenas features semânticas):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in baseline_results]
        print(f"  {metric.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})")
    
    # Filo-Transformer
    print("\nFilo-Transformer (semânticas + filogenéticas):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in filo_results]
        print(f"  {metric.upper()}: {np.mean(values):.4f} (±{np.std(values):.4f})")
    
    # Melhoria
    print("\nMelhoria do Filo-Transformer sobre Baseline:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        baseline_values = [r[metric] for r in baseline_results]
        filo_values = [r[metric] for r in filo_results]
        improvement = (np.mean(filo_values) - np.mean(baseline_values)) * 100
        print(f"  {metric.upper()}: {improvement:+.2f}%")
    
    # Salva resultados
    results = {
        'baseline': baseline_results,
        'filo_transformer': filo_results,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'cascade_features': cascade_features
        }
    }
    
    with open('pheme_real_cascades_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResultados salvos em pheme_real_cascades_results.json")

if __name__ == "__main__":
    main()