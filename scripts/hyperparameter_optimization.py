"""
hyperparameter_optimization.py

Busca automática dos melhores hiperparâmetros para o Filo-Transformer.
Usa Optuna para otimização bayesiana eficiente.
"""

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Importa o modelo
import sys
sys.path.append(str(Path(__file__).parent))
from pheme_real_cascades_experiment_tags import FiloTransformerTAGs, load_pheme_tags_data

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial):
    """Função objetivo para Optuna otimizar"""
    
    # Hiperparâmetros para buscar
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    d_model = trial.suggest_categorical('d_model', [128, 192, 256])
    n_heads = trial.suggest_categorical('n_heads', [4, 6, 8])
    n_layers = trial.suggest_int('n_layers', 2, 4)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.4)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)
    
    # Carrega dados
    X_semantic, X_phylo, y, _ = load_pheme_tags_data()
    
    # Normalização
    scaler_semantic = StandardScaler()
    scaler_phylo = RobustScaler()
    X_semantic = scaler_semantic.fit_transform(X_semantic)
    X_phylo = scaler_phylo.fit_transform(X_phylo)
    
    # Validação cruzada simplificada (3 folds para velocidade)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    auc_scores = []
    
    for train_idx, val_idx in kfold.split(X_semantic, y):
        # Dados de treino e validação
        X_sem_train = torch.FloatTensor(X_semantic[train_idx])
        X_phy_train = torch.FloatTensor(X_phylo[train_idx])
        y_train = torch.LongTensor(y[train_idx])
        
        X_sem_val = torch.FloatTensor(X_semantic[val_idx])
        X_phy_val = torch.FloatTensor(X_phylo[val_idx])
        y_val = torch.LongTensor(y[val_idx])
        
        # Modelo
        model = FiloTransformerTAGs(
            num_semantic_features=X_semantic.shape[1],
            num_phylo_features=X_phylo.shape[1],
            num_classes=2,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        ).to(DEVICE)
        
        # Treinamento simplificado
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Treina por menos épocas para velocidade
        num_epochs = 20
        for epoch in range(num_epochs):
            model.train()
            for i in range(0, len(X_sem_train), batch_size):
                batch_sem = X_sem_train[i:i+batch_size].to(DEVICE)
                batch_phy = X_phy_train[i:i+batch_size].to(DEVICE)
                batch_y = y_train[i:i+batch_size].to(DEVICE)
                
                outputs = model(batch_sem, batch_phy)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Avaliação
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_sem_val.to(DEVICE), X_phy_val.to(DEVICE))
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            auc = roc_auc_score(y_val.numpy(), val_probs)
            auc_scores.append(auc)
    
    # Retorna AUC médio (negativo para minimização)
    return np.mean(auc_scores)

def find_best_hyperparameters(n_trials=50):
    """Executa busca de hiperparâmetros"""
    
    print("Iniciando busca de hiperparâmetros com Optuna...")
    print(f"Dispositivo: {DEVICE}")
    print(f"Número de trials: {n_trials}")
    
    # Cria estudo
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    
    # Otimiza
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Resultados
    print("\n" + "="*60)
    print("MELHORES HIPERPARÂMETROS ENCONTRADOS")
    print("="*60)
    
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print(f"\nMelhor AUC médio: {study.best_value:.4f}")
    
    # Salva resultados
    results = {
        'best_params': best_params,
        'best_auc': study.best_value,
        'n_trials': n_trials,
        'all_trials': [
            {
                'params': trial.params,
                'value': trial.value,
                'number': trial.number
            }
            for trial in study.trials
        ]
    }
    
    with open('results/hyperparameter_search.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados salvos em: results/hyperparameter_search.json")
    
    # Cria arquivo de configuração otimizada
    config = {
        'BATCH_SIZE': best_params['batch_size'],
        'LEARNING_RATE': best_params['learning_rate'],
        'D_MODEL': best_params['d_model'],
        'N_HEADS': best_params['n_heads'],
        'N_LAYERS': best_params['n_layers'],
        'DROPOUT': best_params['dropout'],
        'WEIGHT_DECAY': best_params['weight_decay'],
        'NUM_EPOCHS': 50,  # Fixo para experimento final
        'PATIENCE': 15,    # Fixo
        'N_FOLDS': 5       # Fixo
    }
    
    with open('scripts/optimal_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nConfiguração otimizada salva em: scripts/optimal_config.json")
    
    return best_params

if __name__ == "__main__":
    # Instala Optuna se necessário
    try:
        import optuna
    except ImportError:
        print("Instalando Optuna...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        import optuna
    
    # Executa busca
    find_best_hyperparameters(n_trials=30)  # Reduzido para demonstração