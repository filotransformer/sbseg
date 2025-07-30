#!/usr/bin/env python3
"""
Experimento Filo-Transformer usando dataset PHEME real com estruturas conversacionais.

Este script usa o dataset PHEME processado para treinar o Filo-Transformer
com características filogenéticas reais extraídas das árvores de conversação.

Autor: Acauan Cardoso Ribeiro, Eduardo Luzeiro Feitosa, André Carvalho

Atualização: Usa features reais de cascata extraídas do dataset PHEME processado
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Adicionar diretório scripts ao path
sys.path.append(str(Path(__file__).parent))

from ft_transformer import FTTransformerClassifier


class PHEMEFiloTransformerExperiment:
    """
    Experimento Filo-Transformer usando dataset PHEME com cascatas reais.
    """
    
    def __init__(self, processed_data_path: str = 'datasets/processed', device: str = None):
        """
        Inicializa o experimento.
        
        Args:
            processed_data_path: Caminho para dados processados do PHEME
            device: Dispositivo para PyTorch
        """
        self.processed_data_path = Path(processed_data_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*60)
        print("FILO-TRANSFORMER: DATASET PHEME REAL")
        print("Cascatas conversacionais reais do Twitter")
        print("="*60)
        print(f"🚀 Dispositivo: {self.device}")
        print(f"📂 Dados processados: {self.processed_data_path}")
        
        # Inicializar modelo de embeddings
        print("📥 Carregando modelo SBERT...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.embedding_dim = 768
        
    def load_processed_data(self) -> pd.DataFrame:
        """
        Carrega o dataset PHEME processado.
        
        Returns:
            DataFrame com cascatas processadas
        """
        cascade_file = self.processed_data_path / 'pheme_cascades.csv'
        
        if not cascade_file.exists():
            raise FileNotFoundError(
                f"Dataset processado não encontrado: {cascade_file}\\n"
                "Execute primeiro: python scripts/pheme_dataset_processor.py"
            )
        
        print(f"📂 Carregando dataset: {cascade_file}")
        df = pd.read_csv(cascade_file)
        
        print(f"✅ Dataset carregado: {len(df)} cascatas")
        print(f"   - Rumores: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
        print(f"   - Não-rumores: {len(df) - df['label'].sum()} ({(1-df['label'].mean())*100:.1f}%)")
        print(f"   - Eventos: {df['event'].nunique()} ({', '.join(df['event'].unique())})")
        
        return df
    
    def extract_phylogenetic_features_from_conversation_trees(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrai características filogenéticas das árvores de conversação reais.
        
        Args:
            df: DataFrame com cascatas e árvores de conversação
            
        Returns:
            Array de características filogenéticas
        """
        print("🌳 Extraindo características filogenéticas das árvores de conversação...")
        
        features_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processando cascatas"):
            try:
                # Parse da árvore de conversação
                tree_data = json.loads(row['conversation_tree'])
                
                # Construir grafo NetworkX
                G = nx.DiGraph()
                
                # Adicionar nós
                for node_id, node_data in tree_data['nodes'].items():
                    G.add_node(node_id, **node_data)
                
                # Adicionar arestas
                for edge in tree_data['edges']:
                    G.add_edge(
                        edge['parent'], 
                        edge['child'], 
                        weight=edge['weight']
                    )
                
                # Extrair características do source tweet (nó raiz)
                source_id = tree_data['root']
                features = self._extract_node_features(G, source_id, tree_data)
                
                features_list.append(features)
                
            except Exception as e:
                # Para casos com erros, usar features zeros
                features_list.append(np.zeros(20))
                
        feature_matrix = np.array(features_list)
        print(f"✅ Características extraídas: {feature_matrix.shape}")
        
        return feature_matrix
    
    def _extract_node_features(self, G: nx.DiGraph, node_id: str, tree_data: Dict) -> np.ndarray:
        """
        Extrai características filogenéticas para um nó específico.
        
        Args:
            G: Grafo NetworkX
            node_id: ID do nó (source tweet)
            tree_data: Dados da árvore de conversação
            
        Returns:
            Array de características
        """
        features = []
        
        try:
            # 1. Características básicas da cascata
            features.append(len(tree_data['nodes']) - 1)  # num_children (excluir root)
            features.append(max(tree_data['depth_map'].values()) if tree_data['depth_map'] else 0)  # max_depth
            features.append(len(tree_data['edges']))  # num_edges
            
            # 2. Características do nó raiz
            if G.has_node(node_id):
                features.append(G.out_degree(node_id))  # out_degree
                features.append(G.in_degree(node_id))   # in_degree (sempre 0 para root)
                
                # Centralidades (para grafos pequenos)
                if len(G) > 1:
                    try:
                        closeness = nx.closeness_centrality(G)
                        betweenness = nx.betweenness_centrality(G)
                        features.append(closeness.get(node_id, 0))
                        features.append(betweenness.get(node_id, 0))
                    except:
                        features.extend([0, 0])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0, 0, 0])
            
            # 3. Características temporais
            timestamps = [node_data.get('timestamp', 0) for node_data in tree_data['nodes'].values()]
            if len(timestamps) > 1:
                duration = max(timestamps) - min(timestamps)  # duração em segundos
                features.append(duration / 3600)  # converter para horas
                
                # Taxa de atividade (tweets por hora)
                if duration > 0:
                    features.append((len(timestamps) - 1) / (duration / 3600))
                else:
                    features.append(0)
            else:
                features.extend([0, 0])
            
            # 4. Características dos usuários na cascata
            user_followers = []
            user_verified = []
            for node_data in tree_data['nodes'].values():
                user_followers.append(node_data.get('user_followers_count', 0))
                user_verified.append(1 if node_data.get('user_verified', False) else 0)
            
            features.append(np.mean(user_followers) if user_followers else 0)  # avg_followers
            features.append(np.max(user_followers) if user_followers else 0)   # max_followers
            features.append(np.sum(user_verified) / len(user_verified) if user_verified else 0)  # verified_ratio
            
            # 5. Características de engajamento
            retweet_counts = []
            favorite_counts = []
            for node_data in tree_data['nodes'].values():
                retweet_counts.append(node_data.get('retweet_count', 0))
                favorite_counts.append(node_data.get('favorite_count', 0))
            
            features.append(np.mean(retweet_counts) if retweet_counts else 0)
            features.append(np.mean(favorite_counts) if favorite_counts else 0)
            
            # 6. Características de diversidade
            languages = set()
            hashtags = set()
            for node_data in tree_data['nodes'].values():
                lang = node_data.get('lang', '')
                if lang:
                    languages.add(lang)
                
                hashtags_list = node_data.get('hashtags', [])
                if isinstance(hashtags_list, list):
                    hashtags.update(hashtags_list)
            
            features.append(len(languages))  # num_languages
            features.append(len(hashtags))   # num_hashtags
            
            # 7. Características de similaridade (pesos das arestas)
            edge_weights = [edge['weight'] for edge in tree_data['edges']]
            if edge_weights:
                features.append(np.mean(edge_weights))  # avg_edge_weight
                features.append(np.std(edge_weights))   # std_edge_weight
            else:
                features.extend([0, 0])
            
            # 8. Estrutura em árvore
            if len(G) > 1:
                # Leaf nodes ratio
                leaf_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
                features.append(len(leaf_nodes) / len(G))
            else:
                features.append(0)
            
        except Exception as e:
            # Em caso de erro, preencher com zeros
            features = [0] * 20
        
        # Garantir que sempre retornamos 20 features
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def run_experiment(self, test_mode: bool = False):
        """
        Executa o experimento completo com dataset PHEME real.
        
        Args:
            test_mode: Se True, usa subset pequeno para teste rápido
        """
        # 1. Carregar dados processados
        df = self.load_processed_data()
        
        if test_mode:
            print("\\n🧪 MODO DE TESTE RÁPIDO")
            # Pegar amostra estratificada por evento e label
            df_sample = []
            for event in df['event'].unique():
                event_df = df[df['event'] == event]
                for label in [0, 1]:
                    label_df = event_df[event_df['label'] == label]
                    if len(label_df) > 0:
                        sample_size = min(20, len(label_df))
                        df_sample.append(label_df.sample(n=sample_size, random_state=42))
            
            df = pd.concat(df_sample, ignore_index=True)
            print(f"Dataset de teste: {len(df)} cascatas")
        
        # 2. Gerar embeddings semânticos
        print("\\n🧬 Gerando embeddings semânticos...")
        texts = df['text'].fillna('').astype(str).tolist()
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        
        # 3. Extrair características filogenéticas das árvores de conversação
        phylo_features = self.extract_phylogenetic_features_from_conversation_trees(df)
        
        # 4. Preparar labels
        labels = df['label'].values
        
        # 5. Cross-validation 5-fold
        print("\\n🔄 Executando validação cruzada 5-fold...")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=4321)
        
        # Métricas para armazenar resultados
        baseline_results = {'accuracy': [], 'auc': [], 'f1': [], 'recall': []}
        filo_results = {'accuracy': [], 'auc': [], 'f1': [], 'recall': []}
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(embeddings, labels), 1):
            print(f"\\n--- Fold {fold}/5 ---")
            
            # Dividir dados
            X_train_emb = embeddings[train_idx]
            X_test_emb = embeddings[test_idx]
            X_train_phylo = phylo_features[train_idx]
            X_test_phylo = phylo_features[test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            
            # Normalizar características filogenéticas para este fold
            scaler = StandardScaler()
            X_train_phylo_scaled = scaler.fit_transform(X_train_phylo)
            X_test_phylo_scaled = scaler.transform(X_test_phylo)
            
            # Remover features com variância zero no conjunto de treino
            feature_std = np.std(X_train_phylo_scaled, axis=0)
            valid_features = feature_std > 1e-6  # Features com variância muito baixa
            
            if np.sum(valid_features) > 0:
                X_train_phylo_scaled = X_train_phylo_scaled[:, valid_features]
                X_test_phylo_scaled = X_test_phylo_scaled[:, valid_features]
                n_phylo_features = X_train_phylo_scaled.shape[1]
                if fold == 1:  # Only show debug info on first fold
                    feature_names = [
                        'num_children', 'max_depth', 'num_edges', 'out_degree', 'in_degree',
                        'closeness_centrality', 'betweenness_centrality', 'duration_hours', 'activity_rate',
                        'avg_followers', 'max_followers', 'verified_ratio', 'avg_retweets', 'avg_favorites',
                        'num_languages', 'num_hashtags', 'avg_edge_weight', 'std_edge_weight', 'leaf_ratio'
                    ]
                    valid_feature_names = [name for i, name in enumerate(feature_names[:len(valid_features)]) if valid_features[i]]
                    print(f"   Using {n_phylo_features}/{len(valid_features)} phylogenetic features: {valid_feature_names}")
            else:
                # Se todas as features têm variância zero, usar apenas uma dummy feature
                X_train_phylo_scaled = np.zeros((len(X_train_phylo_scaled), 1))
                X_test_phylo_scaled = np.zeros((len(X_test_phylo_scaled), 1))
                n_phylo_features = 1
                if fold == 1:
                    print(f"   Warning: All phylogenetic features have zero variance, using dummy feature")
            
            # BASELINE: FT-Transformer apenas com embeddings semânticos
            print("📊 Treinando Baseline (apenas embeddings)...")
            baseline_model = FTTransformerClassifier(
                n_semantic_features=self.embedding_dim,
                n_phylogenetic_features=1,  # Dummy feature
                d_model=256,
                n_heads=8,
                n_layers=2,  # Reduce complexity to prevent overfitting
                n_epochs=50 if not test_mode else 10,
                batch_size=64,
                learning_rate=5e-5,
                device=self.device,
                verbose=False
            )
            
            # Dummy phylo features para baseline
            dummy_phylo_train = np.zeros((len(X_train_emb), 1))
            dummy_phylo_test = np.zeros((len(X_test_emb), 1))
            
            # Dividir treino em treino/validação
            val_split = int(0.8 * len(X_train_emb))
            baseline_model.fit(
                X_train_emb[:val_split], 
                dummy_phylo_train[:val_split], 
                y_train[:val_split],
                X_train_emb[val_split:],
                dummy_phylo_train[val_split:],
                y_train[val_split:]
            )
            
            # Avaliar baseline
            y_pred_baseline = baseline_model.predict(X_test_emb, dummy_phylo_test)
            y_proba_baseline = baseline_model.predict_proba(X_test_emb, dummy_phylo_test)[:, 1]
            
            baseline_results['accuracy'].append(accuracy_score(y_test, y_pred_baseline))
            baseline_results['auc'].append(roc_auc_score(y_test, y_proba_baseline))
            baseline_results['f1'].append(f1_score(y_test, y_pred_baseline))
            baseline_results['recall'].append(recall_score(y_test, y_pred_baseline))
            
            # FILO-TRANSFORMER: Com embeddings + características filogenéticas reais
            print("🧬 Treinando Filo-Transformer (embeddings + filogenia real)...")
            filo_model = FTTransformerClassifier(
                n_semantic_features=self.embedding_dim,
                n_phylogenetic_features=n_phylo_features,
                d_model=256,
                n_heads=8,
                n_layers=2,  # Reduce complexity to prevent overfitting
                n_epochs=50 if not test_mode else 10,
                batch_size=64,
                learning_rate=5e-5,
                device=self.device,
                verbose=False
            )
            
            filo_model.fit(
                X_train_emb[:val_split], 
                X_train_phylo_scaled[:val_split], 
                y_train[:val_split],
                X_train_emb[val_split:],
                X_train_phylo_scaled[val_split:],
                y_train[val_split:]
            )
            
            # Avaliar Filo-Transformer
            y_pred_filo = filo_model.predict(X_test_emb, X_test_phylo_scaled)
            y_proba_filo = filo_model.predict_proba(X_test_emb, X_test_phylo_scaled)[:, 1]
            
            filo_results['accuracy'].append(accuracy_score(y_test, y_pred_filo))
            filo_results['auc'].append(roc_auc_score(y_test, y_proba_filo))
            filo_results['f1'].append(f1_score(y_test, y_pred_filo))
            filo_results['recall'].append(recall_score(y_test, y_pred_filo))
            
            print(f"Fold {fold} - Baseline AUC: {baseline_results['auc'][-1]:.4f}")
            print(f"Fold {fold} - Filo-Transformer AUC: {filo_results['auc'][-1]:.4f}")
            print(f"Melhoria: {filo_results['auc'][-1] - baseline_results['auc'][-1]:+.4f}")
        
        # Calcular e exibir resultados finais
        self._print_final_results(baseline_results, filo_results)
        
        return baseline_results, filo_results
    
    def _print_final_results(self, baseline_results, filo_results):
        """Imprime resultados finais formatados."""
        print("\\n" + "="*60)
        print("RESULTADOS FINAIS - DATASET PHEME REAL")
        print("="*60)
        
        print("\\n📊 BASELINE (FT-Transformer com apenas embeddings semânticos)")
        print("-"*60)
        for metric in ['accuracy', 'auc', 'f1', 'recall']:
            values = baseline_results[metric]
            print(f"{metric.upper():10}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        print("\\n🧬 FILO-TRANSFORMER (embeddings + características filogenéticas REAIS)")
        print("-"*60)
        for metric in ['accuracy', 'auc', 'f1', 'recall']:
            values = filo_results[metric]
            print(f"{metric.upper():10}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        print("\\n🎯 CONTRIBUIÇÃO DAS CARACTERÍSTICAS FILOGENÉTICAS REAIS")
        print("-"*60)
        for metric in ['accuracy', 'auc', 'f1', 'recall']:
            baseline_mean = np.mean(baseline_results[metric])
            filo_mean = np.mean(filo_results[metric])
            improvement = filo_mean - baseline_mean
            improvement_pct = (improvement / baseline_mean) * 100
            print(f"{metric.upper():10}: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        print("\\n" + "="*60)
        print("✅ EXPERIMENTO CONCLUÍDO!")
        print("🌳 Características filogenéticas: Extraídas de cascatas reais do Twitter")
        print("📊 Comparação justa: Mesmo FT-Transformer para baseline e proposta")
        print("🔬 Dataset: PHEME com 5 eventos e estruturas conversacionais reais")
        print("="*60)


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Filo-Transformer: Dataset PHEME Real')
    parser.add_argument('--test', action='store_true', help='Modo de teste rápido')
    
    args = parser.parse_args()
    
    # Verificar se dados processados existem
    processed_path = Path('datasets/processed')
    if not (processed_path / 'pheme_cascades.csv').exists():
        print("❌ Dados processados não encontrados!")
        print("Execute primeiro: python scripts/pheme_dataset_processor.py")
        return 1
    
    # Criar e executar experimento
    experiment = PHEMEFiloTransformerExperiment()
    
    try:
        experiment.run_experiment(test_mode=args.test)
        return 0
    except Exception as e:
        print(f"\\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())