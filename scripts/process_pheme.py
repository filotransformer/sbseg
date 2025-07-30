"""
PHEME Dataset Processor for Filo-Transformer

Processa a base PHEME extraindo:
1. Tweets fonte e suas reactions
2. Estrutura de cascata/thread
3. Features temporais e estruturais
4. Metadados para análise filogenética
"""

import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

class PHEMEProcessor:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.events = ['charliehebdo', 'ferguson', 'germanwings-crash', 
                      'ottawashooting', 'sydneysiege']
        
    def extract_tweet_data(self, tweet_path: Path) -> Dict:
        """Extrai dados de um tweet JSON"""
        with open(tweet_path, 'r', encoding='utf-8') as f:
            tweet = json.load(f)
        
        return {
            'tweet_id': tweet['id_str'],
            'text': tweet['text'],
            'user_id': tweet['user']['id_str'],
            'user_followers': tweet['user']['followers_count'],
            'user_friends': tweet['user']['friends_count'],
            'user_verified': tweet['user']['verified'],
            'created_at': tweet['created_at'],
            'timestamp': datetime.strptime(tweet['created_at'], 
                                         '%a %b %d %H:%M:%S %z %Y').timestamp(),
            'retweet_count': tweet.get('retweet_count', 0),
            'favorite_count': tweet.get('favorite_count', 0),
            'in_reply_to': tweet.get('in_reply_to_status_id_str'),
            'hashtags': [h['text'] for h in tweet.get('entities', {}).get('hashtags', [])],
            'urls': [u['expanded_url'] for u in tweet.get('entities', {}).get('urls', [])],
            'user_mentions': [m['screen_name'] for m in tweet.get('entities', {}).get('user_mentions', [])]
        }
    
    def build_cascade_tree(self, source_id: str, reactions: List[Dict]) -> Dict:
        """Constrói estrutura de árvore da cascata de tweets"""
        # Estrutura: {node_id: {'level': int, 'children': [node_ids], 'parent': node_id}}
        tree = {
            source_id: {'level': 0, 'children': [], 'parent': None, 'is_source': True}
        }
        
        # Adiciona reactions
        for reaction in reactions:
            reaction_id = reaction['tweet_id']
            parent_id = reaction.get('in_reply_to', source_id)
            
            # Se o parent não existe na árvore, assume que é o source
            if parent_id not in tree:
                parent_id = source_id
            
            parent_level = tree[parent_id]['level']
            tree[reaction_id] = {
                'level': parent_level + 1,
                'children': [],
                'parent': parent_id,
                'is_source': False
            }
            tree[parent_id]['children'].append(reaction_id)
        
        return tree
    
    def extract_cascade_features(self, tree: Dict, tweets_data: Dict[str, Dict]) -> Dict:
        """Extrai features filogenéticas da cascata"""
        
        # Features estruturais
        node_count = len(tree)
        depths = [node['level'] for node in tree.values()]
        level_1_nodes = [nid for nid, node in tree.items() if node['level'] == 1]
        branching_factors = [len(node['children']) for node in tree.values()]
        
        features = {
            'cascade_size': node_count,
            'cascade_depth': max(depths) if depths else 0,
            'cascade_breadth': len(level_1_nodes),
            'avg_branching_factor': np.mean(branching_factors),
            'max_branching_factor': max(branching_factors) if branching_factors else 0,
        }
        
        # Features temporais
        timestamps = [tweets_data[n]['timestamp'] for n in tree.keys() if n in tweets_data]
        if len(timestamps) > 1:
            features['cascade_lifetime'] = max(timestamps) - min(timestamps)
            features['avg_time_between_tweets'] = np.mean(np.diff(sorted(timestamps))) if len(timestamps) > 1 else 0
        else:
            features['cascade_lifetime'] = 0
            features['avg_time_between_tweets'] = 0
        
        # Features de propagação por nível
        level_counts = defaultdict(int)
        for node in tree.values():
            level_counts[node['level']] += 1
        
        for i in range(5):  # Primeiros 5 níveis
            features[f'level_{i}_count'] = level_counts.get(i, 0)
        
        # Features de usuários
        user_counts = defaultdict(int)
        verified_count = 0
        total_followers = []
        
        for node_id in tree.keys():
            if node_id in tweets_data:
                user_id = tweets_data[node_id]['user_id']
                user_counts[user_id] += 1
                if tweets_data[node_id]['user_verified']:
                    verified_count += 1
                total_followers.append(tweets_data[node_id]['user_followers'])
        
        features['unique_users'] = len(user_counts)
        features['user_diversity'] = features['unique_users'] / features['cascade_size'] if features['cascade_size'] > 0 else 0
        features['verified_ratio'] = verified_count / features['cascade_size'] if features['cascade_size'] > 0 else 0
        features['avg_user_followers'] = np.mean(total_followers) if total_followers else 0
        features['max_user_followers'] = max(total_followers) if total_followers else 0
        
        return features
    
    def process_event(self, event: str) -> pd.DataFrame:
        """Processa todos os tweets de um evento"""
        event_path = self.base_path / event
        all_cascades = []
        
        for label in ['rumours', 'non-rumours']:
            label_path = event_path / label
            if not label_path.exists():
                continue
                
            for tweet_dir in label_path.iterdir():
                if not tweet_dir.is_dir():
                    continue
                
                # Extrai source tweet
                source_path = tweet_dir / 'source-tweet' / f'{tweet_dir.name}.json'
                if not source_path.exists():
                    continue
                
                try:
                    source_data = self.extract_tweet_data(source_path)
                    
                    # Extrai reactions
                    reactions_dir = tweet_dir / 'reactions'
                    reactions_data = []
                    
                    if reactions_dir.exists():
                        for reaction_file in reactions_dir.glob('*.json'):
                            try:
                                reaction_data = self.extract_tweet_data(reaction_file)
                                reactions_data.append(reaction_data)
                            except:
                                continue
                    
                    # Constrói grafo da cascata
                    tweets_dict = {source_data['tweet_id']: source_data}
                    for r in reactions_data:
                        tweets_dict[r['tweet_id']] = r
                    
                    tree = self.build_cascade_tree(source_data['tweet_id'], reactions_data)
                    
                    # Extrai features
                    cascade_features = self.extract_cascade_features(tree, tweets_dict)
                    
                    # Combina tudo
                    cascade_data = {
                        'event': event,
                        'source_tweet_id': source_data['tweet_id'],
                        'source_text': source_data['text'],
                        'label': 1 if label == 'rumours' else 0,
                        'source_timestamp': source_data['timestamp'],
                        **cascade_features,
                        'reaction_texts': ' '.join([r['text'] for r in reactions_data]),
                        'all_texts': source_data['text'] + ' ' + ' '.join([r['text'] for r in reactions_data]),
                        'tree_structure': tree,  # Para reconstruir a estrutura depois
                    }
                    
                    all_cascades.append(cascade_data)
                    
                except Exception as e:
                    print(f"Erro processando {tweet_dir}: {e}")
                    continue
        
        return pd.DataFrame(all_cascades)
    
    def process_all_events(self) -> pd.DataFrame:
        """Processa todos os eventos e combina em um único dataset"""
        all_dfs = []
        
        for event in self.events:
            print(f"Processando evento: {event}")
            event_df = self.process_event(event)
            all_dfs.append(event_df)
            print(f"  - {len(event_df)} cascatas processadas")
        
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Adiciona features agregadas
        final_df['has_reactions'] = final_df['cascade_size'] > 1
        final_df['is_viral'] = final_df['cascade_size'] > final_df['cascade_size'].quantile(0.75)
        
        return final_df

def main():
    # Caminho base da dataset PHEME
    base_path = "/home/acauan/ufam/papers/01_sbseg_filo_trans/datasets/pheme-rnr-dataset"
    
    processor = PHEMEProcessor(base_path)
    
    print("Iniciando processamento da base PHEME...")
    df = processor.process_all_events()
    
    # Salva dataset processado
    output_path = "/home/acauan/ufam/papers/01_sbseg_filo_trans/datasets/processed"
    os.makedirs(output_path, exist_ok=True)
    
    # CSV principal
    df.to_csv(f"{output_path}/pheme_processed_cascades.csv", index=False)
    
    # Salva também versão com menos colunas para análise rápida
    df_simplified = df[['event', 'source_tweet_id', 'source_text', 'label', 
                       'cascade_size', 'cascade_depth', 'cascade_lifetime',
                       'unique_users', 'verified_ratio', 'has_reactions']]
    df_simplified.to_csv(f"{output_path}/pheme_simplified.csv", index=False)
    
    # Estatísticas
    print("\nEstatísticas do dataset processado:")
    print(f"Total de cascatas: {len(df)}")
    print(f"Rumours: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Non-rumours: {len(df) - df['label'].sum()} ({(1-df['label'].mean())*100:.1f}%)")
    print(f"\nDistribuição por evento:")
    print(df.groupby('event')['label'].agg(['count', 'sum', 'mean']))
    
    # Salva metadados em JSON
    metadata = {
        'total_cascades': int(len(df)),
        'events': list(df['event'].unique()),
        'features': list(df.columns),
        'cascade_features': [col for col in df.columns if col.startswith(('cascade_', 'level_', 'user_', 'unique_', 'verified_'))],
        'stats': {
            'avg_cascade_size': float(df['cascade_size'].mean()),
            'avg_cascade_depth': float(df['cascade_depth'].mean()),
            'cascades_with_reactions': int(df['has_reactions'].sum()),
            'viral_cascades': int(df['is_viral'].sum())
        }
    }
    
    with open(f"{output_path}/pheme_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset salvo em: {output_path}")
    print("Arquivos gerados:")
    print("  - pheme_processed_cascades.csv (dataset completo)")
    print("  - pheme_simplified.csv (versão simplificada)")
    print("  - pheme_metadata.json (metadados)")

if __name__ == "__main__":
    main()