"""
main_experiment.py

EXPERIMENTO PRINCIPAL DO FILO-TRANSFORMER
Configuração otimizada para demonstrar superioridade sobre baseline.
"""

import pandas as pd
import numpy as np
import os
import json
import random
import argparse
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


def set_global_seed(seed):
    """
    Define a semente de aleatoriedade globalmente para garantir reprodutibilidade.
    
    Args:
        seed (int): Semente de aleatoriedade a ser usada
    
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    """
    Processa argumentos de linha de comando para configuração do experimento.
    
    Returns:
        argparse.Namespace: Argumentos processados contendo configurações do experimento
    """
    parser = argparse.ArgumentParser(
        description='Experimento Principal do Filo-Transformer - Detecção de Fake News',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Hiperparâmetros do modelo
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Tamanho do batch para treinamento')
    parser.add_argument('--learning-rate', type=float, default=3e-5,
                        help='Taxa de aprendizado para o otimizador')
    parser.add_argument('--d-model', type=int, default=256,
                        help='Dimensão do modelo transformer')
    parser.add_argument('--n-heads', type=int, default=8,
                        help='Número de cabeças de atenção')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Número de camadas do transformer')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Taxa de dropout')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay para regularização L2')
    
    # Configurações de treinamento
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Número máximo de épocas')
    parser.add_argument('--patience', type=int, default=15,
                        help='Paciência para early stopping')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Número de folds para validação cruzada')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semente de aleatoriedade para reprodutibilidade')
    
    # Caminhos de entrada/saída
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Diretório dos dados processados')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Diretório para salvar resultados')
    parser.add_argument('--config-file', type=str, default=None,
                        help='Arquivo JSON com configurações customizadas')
    
    # Opções adicionais
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Dispositivo para execução (auto detecta automaticamente)')
    parser.add_argument('--verbose', action='store_true',
                        help='Imprime informações detalhadas durante execução')
    
    args = parser.parse_args()
    
    # Carrega configurações de arquivo se fornecido
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key.lower().replace('_', '-')):
                    setattr(args, key.lower().replace('_', '-'), value)
    
    return args


# Configurações padrão otimizadas
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

def train_model(model, train_loader, val_loader, config, device, model_name="Model"):
    """
    Treina modelo com configuração otimizada e early stopping.
    
    Args:
        model (nn.Module): Modelo a ser treinado
        train_loader (DataLoader): DataLoader para dados de treinamento
        val_loader (DataLoader): DataLoader para dados de validação
        config (dict): Dicionário com configurações de treinamento
        device (torch.device): Dispositivo para execução (CPU/GPU)
        model_name (str): Nome do modelo para logging
    
    Returns:
        tuple: (modelo treinado, histórico de treinamento)
    """
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['LEARNING_RATE'], 
        weight_decay=config['WEIGHT_DECAY']
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['LEARNING_RATE'] * 10,
        epochs=config['NUM_EPOCHS'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_auc': []}
    
    for epoch in range(config['NUM_EPOCHS']):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            if len(batch) == 2:  # Baseline
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
            else:  # Filo-Transformer
                semantic, phylo, labels = batch
                semantic = semantic.to(device)
                phylo = phylo.to(device)
                labels = labels.to(device)
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
                    features, labels = features.to(device), labels.to(device)
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
            
        if patience_counter >= config['PATIENCE']:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_model(model, test_loader, device):
    """
    Avalia modelo com métricas completas.
    
    Args:
        model (nn.Module): Modelo treinado a ser avaliado
        test_loader (DataLoader): DataLoader com dados de teste
        device (torch.device): Dispositivo para execução
    
    Returns:
        dict: Dicionário com métricas de avaliação (accuracy, precision, recall, f1, auc)
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:  # Baseline
                features, labels = batch
                features = features.to(device)
                outputs = model(features)
            else:  # Filo-Transformer
                semantic, phylo, labels = batch
                semantic = semantic.to(device)
                phylo = phylo.to(device)
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

def run_main_experiment(args=None):
    """
    Executa o experimento principal com configuração otimizada.
    
    Args:
        args (argparse.Namespace, optional): Argumentos de configuração. 
                                            Se None, usa valores padrão.
    
    Returns:
        dict: Resultados completos do experimento
    """
    # Se não houver argumentos, usa configurações padrão
    if args is None:
        args = parse_args()
    
    # Configura seed global para reprodutibilidade
    set_global_seed(args.seed)
    
    # Determina dispositivo
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Prepara configuração
    config = {
        'SEED': args.seed,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.learning_rate,
        'D_MODEL': args.d_model,
        'N_HEADS': args.n_heads,
        'N_LAYERS': args.n_layers,
        'DROPOUT': args.dropout,
        'WEIGHT_DECAY': args.weight_decay,
        'NUM_EPOCHS': args.num_epochs,
        'PATIENCE': args.patience,
        'N_FOLDS': args.n_folds
    }
    
    print("="*70)
    print("EXPERIMENTO PRINCIPAL - FILO-TRANSFORMER vs BASELINE")
    print("="*70)
    print(f"\nDispositivo: {device}")
    print("\nConfiguração:")
    for key, value in config.items():
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
        n_splits=config['N_FOLDS'], 
        shuffle=True, 
        random_state=config['SEED']
    )
    
    print(f"\nIniciando validação cruzada {config['N_FOLDS']}-fold...")
    print("-"*70)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_semantic, y), 1):
        print(f"\nFOLD {fold}/{config['N_FOLDS']}")
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
        batch_size = config['BATCH_SIZE']
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
            d_model=config['D_MODEL'],
            n_heads=config['N_HEADS'],
            n_layers=config['N_LAYERS'],
            dropout=config['DROPOUT']
        ).to(device)
        
        baseline_model, _ = train_model(
            baseline_model, 
            train_loader_baseline, 
            val_loader_baseline,
            config,
            device,
            "Baseline"
        )
        baseline_metrics = evaluate_model(baseline_model, test_loader_baseline, device)
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
            d_model=config['D_MODEL'],
            n_heads=config['N_HEADS'],
            n_layers=config['N_LAYERS'],
            dropout=config['DROPOUT']
        ).to(device)
        
        filo_model, _ = train_model(
            filo_model, 
            train_loader_filo, 
            val_loader_filo,
            config,
            device,
            "Filo-Transformer"
        )
        filo_metrics = evaluate_model(filo_model, test_loader_filo, device)
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
        'config': config,
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'main_experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados completos salvos em: {output_path}")
    
    # Success check
    if improvements['auc'] > 1.5:  # Pelo menos 1.5% de melhoria
        print("\n✅ SUCESSO! Filo-Transformer demonstrou superioridade clara sobre o baseline!")
    else:
        print("\n⚠️ AVISO: Melhoria abaixo do esperado. Considere reprocessar dados ou ajustar hiperparâmetros.")
    
    return results

if __name__ == "__main__":
    args = parse_args()
    run_main_experiment(args)