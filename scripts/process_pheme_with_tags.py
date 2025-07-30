"""
process_pheme_with_tags.py

Processamento avançado do dataset PHEME usando Tree Alignment Graphs (TAGs).
Este script extrai features filogenéticas complexas necessárias para o Filo-Transformer.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import sys

# Adiciona o diretório scripts ao path para importar tag_construction
sys.path.append(str(Path(__file__).parent))
from tag_construction import TAGConstructor

class PHEMEAdvancedProcessor:
    """Processador avançado do dataset PHEME com TAGs"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.events = ['charliehebdo', 'ferguson', 'germanwings-crash', 
                      'ottawashooting', 'sydneysiege']
        self.tag_constructor = TAGConstructor(similarity_threshold=0.7)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_tweet_data(self, json_path: Path) -> dict:
        """Extrai dados de um tweet JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            'tweet_id': str(data.get('id', '')),
            'text': data.get('text', ''),
            'user_id': str(data.get('user', {}).get('id', '')),
            'user_followers': data.get('user', {}).get('followers_count', 0),
            'user_verified': data.get('user', {}).get('verified', False),
            'created_at': data.get('created_at', ''),
            'in_reply_to': str(data.get('in_reply_to_status_id', ''))
        }
    
    def build_cascade_with_embeddings(self, source_id: str, tweets_data: dict) -> tuple:
        """
        Constrói cascata completa com embeddings e timestamps.
        
        Returns:
            tuple: (embeddings, timestamps, post_ids, texts)
        """
        # Ordena tweets por timestamp
        sorted_tweets = sorted(tweets_data.items(), 
                             key=lambda x: x[1]['created_at'])
        
        embeddings = []
        timestamps = []
        post_ids = []
        texts = []
        
        # Processa cada tweet na ordem temporal
        for i, (tweet_id, tweet_data) in enumerate(sorted_tweets):
            # Gera embedding do texto
            text = tweet_data['text']
            embedding = self.sbert_model.encode(text, normalize_embeddings=True)
            
            embeddings.append(embedding)
            timestamps.append(i * 60)  # Simula timestamps em segundos
            post_ids.append(tweet_id)
            texts.append(text)
        
        return (np.array(embeddings), 
                np.array(timestamps), 
                post_ids, 
                texts)
    
    def process_cascade_with_tags(self, source_data: dict, reactions_data: list, 
                                 label: int) -> dict:
        """
        Processa uma cascata completa usando TAGs.
        
        Returns:
            dict: Features completas incluindo TAGs
        """
        # Combina todos os tweets
        tweets_dict = {source_data['tweet_id']: source_data}
        for r in reactions_data:
            tweets_dict[r['tweet_id']] = r
        
        # Constrói dados para TAG
        embeddings, timestamps, post_ids, texts = self.build_cascade_with_embeddings(
            source_data['tweet_id'], tweets_dict
        )
        
        # Constrói TAG
        tag = self.tag_constructor.build_tag(
            embeddings=embeddings,
            timestamps=timestamps,
            post_ids=post_ids
        )
        
        # Extrai features filogenéticas avançadas
        phylo_features_df = self.tag_constructor.extract_phylogenetic_features(tag)
        
        # Agrega features por cascata (média, max, min, std)
        cascade_features = {}
        
        # Features numéricas para agregar
        numeric_cols = [col for col in phylo_features_df.columns 
                       if col not in ['node_id', 'community_id']]
        
        for col in numeric_cols:
            values = phylo_features_df[col].values
            cascade_features[f'{col}_mean'] = np.mean(values)
            cascade_features[f'{col}_max'] = np.max(values)
            cascade_features[f'{col}_min'] = np.min(values)
            cascade_features[f'{col}_std'] = np.std(values)
        
        # Features básicas
        cascade_features.update({
            'source_tweet_id': source_data['tweet_id'],
            'source_text': source_data['text'],
            'label': label,
            'cascade_size': len(tweets_dict),
            'num_nodes_tag': tag.number_of_nodes(),
            'num_edges_tag': tag.number_of_edges(),
            'density_tag': nx.density(tag),
            'is_connected': nx.is_weakly_connected(tag),
            'num_components': nx.number_weakly_connected_components(tag),
        })
        
        # Embeddings semânticos (média dos embeddings da cascata)
        cascade_features['semantic_embedding'] = embeddings.mean(axis=0).tolist()
        
        return cascade_features
    
    def process_event(self, event: str) -> pd.DataFrame:
        """Processa todos os tweets de um evento com TAGs"""
        event_path = self.base_path / event
        all_cascades = []
        
        print(f"\nProcessando evento: {event}")
        
        for label in ['rumours', 'non-rumours']:
            label_path = event_path / label
            if not label_path.exists():
                continue
            
            label_value = 1 if label == 'rumours' else 0
            tweet_dirs = list(label_path.iterdir())
            
            for tweet_dir in tqdm(tweet_dirs, desc=f"  {label}"):
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
                    
                    # Processa com TAGs
                    cascade_data = self.process_cascade_with_tags(
                        source_data, reactions_data, label_value
                    )
                    cascade_data['event'] = event
                    
                    all_cascades.append(cascade_data)
                    
                except Exception as e:
                    print(f"    Erro processando {tweet_dir.name}: {str(e)}")
                    continue
        
        return pd.DataFrame(all_cascades)
    
    def process_all_events(self) -> pd.DataFrame:
        """Processa todos os eventos"""
        all_dfs = []
        
        for event in self.events:
            event_df = self.process_event(event)
            all_dfs.append(event_df)
            print(f"  - {len(event_df)} cascatas processadas")
        
        return pd.concat(all_dfs, ignore_index=True)

def main():
    # Caminho base da dataset PHEME
    base_path = "datasets/pheme-rnr-dataset"
    
    processor = PHEMEAdvancedProcessor(base_path)
    
    print("Iniciando processamento avançado da base PHEME com TAGs...")
    print("Este processo pode demorar 20-30 minutos...")
    
    df = processor.process_all_events()
    
    # Salva dataset processado
    output_path = "datasets/processed"
    os.makedirs(output_path, exist_ok=True)
    
    # Separa embeddings semânticos das outras features
    semantic_cols = [col for col in df.columns if col == 'semantic_embedding']
    phylo_cols = [col for col in df.columns 
                  if any(x in col for x in ['_mean', '_max', '_min', '_std', 
                                           'cascade_', 'num_', 'density_', 'is_', 'components'])]
    meta_cols = ['event', 'source_tweet_id', 'source_text', 'label']
    
    # CSV principal (sem embeddings para reduzir tamanho)
    df_main = df[meta_cols + phylo_cols]
    df_main.to_csv(f"{output_path}/pheme_processed_cascades_tags.csv", index=False)
    
    # Salva embeddings separadamente
    embeddings_df = df[['source_tweet_id'] + semantic_cols]
    embeddings_df.to_pickle(f"{output_path}/pheme_semantic_embeddings.pkl")
    
    # Estatísticas
    print("\nEstatísticas do dataset processado com TAGs:")
    print(f"Total de cascatas: {len(df)}")
    print(f"Rumours: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Non-rumours: {len(df) - df['label'].sum()} ({(1-df['label'].mean())*100:.1f}%)")
    print(f"Features filogenéticas extraídas: {len(phylo_cols)}")
    
    # Salva metadados
    metadata = {
        'total_cascades': int(len(df)),
        'events': list(df['event'].unique()),
        'phylogenetic_features': phylo_cols,
        'num_phylo_features': len(phylo_cols),
        'tag_features': [col for col in phylo_cols if 'tag' in col],
        'centrality_features': [col for col in phylo_cols if 'centrality' in col],
        'community_features': [col for col in phylo_cols if 'community' in col],
        'mutation_features': [col for col in phylo_cols if 'mutation' in col],
        'stats': {
            'avg_cascade_size': float(df['cascade_size'].mean()),
            'avg_nodes_tag': float(df['num_nodes_tag'].mean()),
            'avg_edges_tag': float(df['num_edges_tag'].mean()),
            'connected_cascades': int(df['is_connected'].sum())
        }
    }
    
    with open(f"{output_path}/pheme_metadata_tags.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset com TAGs salvo em: {output_path}")
    print("Arquivos gerados:")
    print("  - pheme_processed_cascades_tags.csv (features filogenéticas)")
    print("  - pheme_semantic_embeddings.pkl (embeddings)")
    print("  - pheme_metadata_tags.json (metadados)")
    
    # Mostra exemplo de features
    print(f"\nExemplo de features filogenéticas extraídas:")
    for i, feat in enumerate(phylo_cols[:10]):
        print(f"  {i+1}. {feat}")
    print(f"  ... e mais {len(phylo_cols)-10} features")

if __name__ == "__main__":
    main()