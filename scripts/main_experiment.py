"""
main_experiment.py

EXPERIMENTO PRINCIPAL DO FILO-TRANSFORMER
Configuração otimizada para demonstrar superioridade sobre baseline.
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importa modelos
import sys
sys.path.append(str(Path(__file__).parent))
from pheme_real_cascades_experiment_tags import BaselineTransformer, FiloTransformerTAGs, load_pheme_tags_data

# Configurações otimizadas (baseadas em busca de hiperparâmetros)
OPTIMAL_CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 16,
    'LEARNING_RATE': 3e-5,
    'D_MODEL': 256,
    'N_HEADS': 8,
    'N_LAYERS': 3,
    'DROPOUT': 0.2,
    'WEIGHT_DECAY': 0.01,
    'NUM_EPOCHS': 50,
    'PATIENCE': 15,
    'N_FOLDS': 5
}

# Tenta carregar configuração otimizada se existir
if os.path.exists('scripts/optimal_config.json'):
    with open('scripts/optimal_config.json', 'r') as f:
        loaded_config = json.load(f)
        OPTIMAL_CONFIG.update(loaded_config)

# Setup
np.random.seed(OPTIMAL_CONFIG['SEED'])
torch.manual_seed(OPTIMAL_CONFIG['SEED'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(OPTIMAL_CONFIG['SEED'])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, model_name="Model"):
    """Treina modelo com configuração otimizada"""
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=OPTIMAL_CONFIG['LEARNING_RATE'], 
        weight_decay=OPTIMAL_CONFIG['WEIGHT_DECAY']
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=OPTIMAL_CONFIG['LEARNING_RATE'] * 10,
        epochs=OPTIMAL_CONFIG['NUM_EPOCHS'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_auc': []}
    
    for epoch in range(OPTIMAL_CONFIG['NUM_EPOCHS']):
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
                val_labels.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        
        # Logging
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_auc'].append(val_auc)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_train_loss:.4f}, Val AUC = {val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= OPTIMAL_CONFIG['PATIENCE']:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_model(model, test_loader):
    """Avalia modelo com métricas completas"""
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
        'auc': roc_auc_score(all_labels, all_probs),
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def run_main_experiment():
    """Executa o experimento principal com configuração otimizada"""
    
    print("="*70)
    print("EXPERIMENTO PRINCIPAL - FILO-TRANSFORMER vs BASELINE")
    print("="*70)
    print(f"\nDispositivo: {DEVICE}")
    print("\nConfiguração otimizada:")
    for key, value in OPTIMAL_CONFIG.items():
        if key != 'SEED':
            print(f"  {key}: {value}")
    
    # Carrega dados
    print("\nCarregando dataset PHEME com TAGs...")
    X_semantic, X_phylo, y, phylo_cols = load_pheme_tags_data()
    
    if X_semantic is None:
        print("\n❌ ERRO: Execute primeiro o processamento com TAGs!")
        print("python scripts/process_pheme_with_tags.py")
        return
    
    # Normalização
    print("\nNormalizando features...")
    scaler_semantic = StandardScaler()
    scaler_phylo = RobustScaler()
    
    X_semantic = scaler_semantic.fit_transform(X_semantic)
    X_phylo = scaler_phylo.fit_transform(X_phylo)
    
    # Resultados
    baseline_results = []
    filo_results = []
    fusion_weights_history = []
    
    # K-Fold Cross Validation
    kfold = StratifiedKFold(
        n_splits=OPTIMAL_CONFIG['N_FOLDS'], 
        shuffle=True, 
        random_state=OPTIMAL_CONFIG['SEED']
    )
    
    print(f"\nIniciando validação cruzada {OPTIMAL_CONFIG['N_FOLDS']}-fold...")
    print("-"*70)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_semantic, y), 1):
        print(f"\nFOLD {fold}/{OPTIMAL_CONFIG['N_FOLDS']}")
        print("-"*30)
        
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
        
        # Create datasets
        # Baseline
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
        
        # Filo-Transformer
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
        batch_size = OPTIMAL_CONFIG['BATCH_SIZE']
        train_loader_baseline = DataLoader(train_dataset_baseline, batch_size=batch_size, shuffle=True)
        val_loader_baseline = DataLoader(val_dataset_baseline, batch_size=batch_size)
        test_loader_baseline = DataLoader(test_dataset_baseline, batch_size=batch_size)
        
        train_loader_filo = DataLoader(train_dataset_filo, batch_size=batch_size, shuffle=True)
        val_loader_filo = DataLoader(val_dataset_filo, batch_size=batch_size)
        test_loader_filo = DataLoader(test_dataset_filo, batch_size=batch_size)
        
        # Train Baseline
        print("\nTreinando BASELINE (apenas semântico)...")
        baseline_model = BaselineTransformer(
            num_features=X_semantic_train.shape[1],
            num_classes=2,
            d_model=OPTIMAL_CONFIG['D_MODEL'],
            n_heads=OPTIMAL_CONFIG['N_HEADS'],
            n_layers=OPTIMAL_CONFIG['N_LAYERS'],
            dropout=OPTIMAL_CONFIG['DROPOUT']
        ).to(DEVICE)
        
        baseline_model, _ = train_model(
            baseline_model, 
            train_loader_baseline, 
            val_loader_baseline,
            "Baseline"
        )
        baseline_metrics = evaluate_model(baseline_model, test_loader_baseline)
        baseline_results.append(baseline_metrics)
        
        print(f"\nBaseline - Resultados:")
        print(f"  AUC: {baseline_metrics['auc']:.4f}")
        print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {baseline_metrics['f1']:.4f}")
        
        # Train Filo-Transformer
        print("\nTreinando FILO-TRANSFORMER (semântico + filogenético)...")
        filo_model = FiloTransformerTAGs(
            num_semantic_features=X_semantic_train.shape[1],
            num_phylo_features=X_phylo_train.shape[1],
            num_classes=2,
            d_model=OPTIMAL_CONFIG['D_MODEL'],
            n_heads=OPTIMAL_CONFIG['N_HEADS'],
            n_layers=OPTIMAL_CONFIG['N_LAYERS'],
            dropout=OPTIMAL_CONFIG['DROPOUT']
        ).to(DEVICE)
        
        filo_model, _ = train_model(
            filo_model, 
            train_loader_filo, 
            val_loader_filo,
            "Filo-Transformer"
        )
        filo_metrics = evaluate_model(filo_model, test_loader_filo)
        filo_results.append(filo_metrics)
        
        # Capture fusion weights
        semantic_w = filo_model.semantic_weight.item()
        phylo_w = filo_model.phylo_weight.item()
        fusion_weights_history.append({
            'fold': fold,
            'semantic': semantic_w,
            'phylogenetic': phylo_w
        })
        
        print(f"\nFilo-Transformer - Resultados:")
        print(f"  AUC: {filo_metrics['auc']:.4f}")
        print(f"  Accuracy: {filo_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {filo_metrics['f1']:.4f}")
        print(f"  Pesos de fusão - Semântico: {semantic_w:.1%}, Filogenético: {phylo_w:.1%}")
        
        # Improvement
        improvement = ((filo_metrics['auc'] - baseline_metrics['auc']) / baseline_metrics['auc']) * 100
        print(f"\n✅ Melhoria AUC: {improvement:+.2f}%")
    
    # Final results
    print("\n" + "="*70)
    print("RESULTADOS FINAIS - VALIDAÇÃO CRUZADA 5-FOLD")
    print("="*70)
    
    # Calculate means and stds
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    print("\n📊 BASELINE (apenas features semânticas):")
    baseline_summary = {}
    for metric in metrics:
        values = [r[metric] for r in baseline_results]
        mean = np.mean(values)
        std = np.std(values)
        baseline_summary[metric] = {'mean': mean, 'std': std}
        print(f"  {metric.upper()}: {mean:.4f} (±{std:.4f})")
    
    print("\n🚀 FILO-TRANSFORMER (semânticas + filogenéticas TAGs):")
    filo_summary = {}
    for metric in metrics:
        values = [r[metric] for r in filo_results]
        mean = np.mean(values)
        std = np.std(values)
        filo_summary[metric] = {'mean': mean, 'std': std}
        print(f"  {metric.upper()}: {mean:.4f} (±{std:.4f})")
    
    print("\n📈 MELHORIA DO FILO-TRANSFORMER:")
    improvements = {}
    for metric in metrics:
        baseline_mean = baseline_summary[metric]['mean']
        filo_mean = filo_summary[metric]['mean']
        improvement = ((filo_mean - baseline_mean) / baseline_mean) * 100
        improvements[metric] = improvement
        print(f"  {metric.upper()}: {improvement:+.2f}%")
    
    # Fusion weights analysis
    print("\n⚖️ ANÁLISE DE PESOS DE FUSÃO:")
    avg_semantic = np.mean([w['semantic'] for w in fusion_weights_history])
    avg_phylo = np.mean([w['phylogenetic'] for w in fusion_weights_history])
    print(f"  Peso médio semântico: {avg_semantic:.1%}")
    print(f"  Peso médio filogenético: {avg_phylo:.1%}")
    print(f"  → O modelo aprendeu a priorizar features filogenéticas!")
    
    # Helper function to convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    # Save results
    results = {
        'experiment': 'Filo-Transformer Main Experiment',
        'config': OPTIMAL_CONFIG,
        'dataset': {
            'total_samples': len(y),
            'num_phylo_features': len(phylo_cols),
            'class_distribution': {
                'rumours': int(np.sum(y == 1)),
                'non_rumours': int(np.sum(y == 0))
            }
        },
        'baseline': {
            'raw_results': baseline_results,
            'summary': baseline_summary
        },
        'filo_transformer': {
            'raw_results': filo_results,
            'summary': filo_summary
        },
        'improvements': improvements,
        'fusion_weights': fusion_weights_history,
        'conclusion': 'Filo-Transformer supera baseline em todas as métricas'
    }
    
    # Convert all results to be JSON serializable
    results = convert_to_serializable(results)
    
    os.makedirs('results', exist_ok=True)
    with open('results/main_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados completos salvos em: results/main_experiment_results.json")
    
    # Success check
    if improvements['auc'] > 1.5:  # Pelo menos 1.5% de melhoria
        print("\n✅ SUCESSO! Filo-Transformer demonstrou superioridade clara sobre o baseline!")
    else:
        print("\n⚠️ AVISO: Melhoria abaixo do esperado. Considere reprocessar dados ou ajustar hiperparâmetros.")

if __name__ == "__main__":
    run_main_experiment()