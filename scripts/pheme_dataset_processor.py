#!/usr/bin/env python3
"""
Processador completo do dataset PHEME com estruturas conversacionais reais.

Extrai tweets, suas reações e metadados para construir TAGs verdadeiros
baseados nas estruturas de conversação do Twitter.

Autor: Acauan Cardoso Ribeiro, Eduardo Luzeiro Feitosa, André Carvalho
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PHEMEDatasetProcessor:
    """
    Processador do dataset PHEME para extrair estruturas conversacionais completas.
    
    Converte a estrutura hierárquica original em dados estruturados para o Filo-Transformer.
    """
    
    def __init__(self, dataset_path: str = 'datasets/pheme-rnr-dataset'):
        """
        Inicializa o processador.
        
        Args:
            dataset_path: Caminho para o dataset PHEME
        """
        self.dataset_path = Path(dataset_path)
        self.events = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
        
        # Estatísticas do dataset
        self.stats = {
            'total_source_tweets': 0,
            'total_reactions': 0,
            'total_cascades': 0,
            'events': {}
        }
    
    def parse_tweet_json(self, json_path: str) -> Dict:
        """
        Parse um arquivo JSON de tweet e extrai informações relevantes.
        
        Args:
            json_path: Caminho para o arquivo JSON
            
        Returns:
            Dicionário com informações do tweet
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                tweet_data = json.load(f)
            
            # Extrair informações essenciais
            tweet_info = {
                'id': tweet_data.get('id_str', ''),
                'text': tweet_data.get('text', ''),
                'created_at': tweet_data.get('created_at', ''),
                'user_id': tweet_data.get('user', {}).get('id_str', ''),
                'user_screen_name': tweet_data.get('user', {}).get('screen_name', ''),
                'user_followers_count': tweet_data.get('user', {}).get('followers_count', 0),
                'user_friends_count': tweet_data.get('user', {}).get('friends_count', 0),
                'user_verified': tweet_data.get('user', {}).get('verified', False),
                'retweet_count': tweet_data.get('retweet_count', 0),
                'favorite_count': tweet_data.get('favorite_count', 0),
                'in_reply_to_status_id': tweet_data.get('in_reply_to_status_id_str', None),
                'in_reply_to_user_id': tweet_data.get('in_reply_to_user_id_str', None),
                'lang': tweet_data.get('lang', ''),
                'hashtags': [ht['text'] for ht in tweet_data.get('entities', {}).get('hashtags', [])],
                'user_mentions': [um['screen_name'] for um in tweet_data.get('entities', {}).get('user_mentions', [])],
                'urls': [url['expanded_url'] for url in tweet_data.get('entities', {}).get('urls', [])],
                'geo_enabled': tweet_data.get('user', {}).get('geo_enabled', False),
                'coordinates': tweet_data.get('coordinates', None),
                'place': tweet_data.get('place', None)
            }
            
            # Converter timestamp
            if tweet_info['created_at']:
                try:
                    dt = datetime.strptime(tweet_info['created_at'], '%a %b %d %H:%M:%S %z %Y')
                    tweet_info['timestamp'] = dt.timestamp()
                    tweet_info['created_at_parsed'] = dt.isoformat()
                except:
                    tweet_info['timestamp'] = 0
                    tweet_info['created_at_parsed'] = ''
            else:
                tweet_info['timestamp'] = 0
                tweet_info['created_at_parsed'] = ''
                
            return tweet_info
            
        except Exception as e:
            logger.error(f"Erro ao processar {json_path}: {e}")
            return {}
    
    def process_cascade(self, cascade_path: Path, event: str, label: str) -> Dict:
        """
        Processa uma cascata individual (source tweet + reactions).
        
        Args:
            cascade_path: Caminho para o diretório da cascata
            event: Nome do evento
            label: Label (rumour ou non-rumour)
            
        Returns:
            Dicionário com informações completas da cascata
        """
        cascade_info = {
            'cascade_id': cascade_path.name,
            'event': event,
            'label': 1 if label == 'rumours' else 0,
            'source_tweet': {},
            'reactions': [],
            'conversation_tree': {}
        }
        
        # Processar source tweet
        source_path = cascade_path / 'source-tweet' / f'{cascade_path.name}.json'
        if source_path.exists():
            cascade_info['source_tweet'] = self.parse_tweet_json(str(source_path))
            cascade_info['source_tweet']['is_source'] = True
            cascade_info['source_tweet']['depth'] = 0
            cascade_info['source_tweet']['parent_id'] = None
        
        # Processar reactions
        reactions_path = cascade_path / 'reactions'
        if reactions_path.exists():
            for reaction_file in reactions_path.glob('*.json'):
                reaction_info = self.parse_tweet_json(str(reaction_file))
                if reaction_info:
                    reaction_info['is_source'] = False
                    # Determinar profundidade baseada em reply relationships
                    if reaction_info['in_reply_to_status_id']:
                        reaction_info['parent_id'] = reaction_info['in_reply_to_status_id']
                        # Por simplicidade, assumir depth 1 para todas as reações
                        # Em implementação completa, calcularíamos depth recursivamente
                        reaction_info['depth'] = 1
                    else:
                        reaction_info['parent_id'] = cascade_info['cascade_id']
                        reaction_info['depth'] = 1
                    
                    cascade_info['reactions'].append(reaction_info)
        
        # Construir árvore de conversação
        cascade_info['conversation_tree'] = self._build_conversation_tree(
            cascade_info['source_tweet'], 
            cascade_info['reactions']
        )
        
        return cascade_info
    
    def _build_conversation_tree(self, source_tweet: Dict, reactions: List[Dict]) -> Dict:
        """
        Constrói uma árvore de conversação baseada nas relações de reply.
        
        Args:
            source_tweet: Tweet original
            reactions: Lista de reações
            
        Returns:
            Estrutura em árvore da conversação
        """
        tree = {
            'root': source_tweet['id'],
            'nodes': {},
            'edges': [],
            'depth_map': {},
            'temporal_order': []
        }
        
        # Adicionar nó raiz
        tree['nodes'][source_tweet['id']] = source_tweet
        tree['depth_map'][source_tweet['id']] = 0
        
        # Ordenar reações por timestamp
        reactions_sorted = sorted(reactions, key=lambda x: x.get('timestamp', 0))
        tree['temporal_order'] = [source_tweet['id']] + [r['id'] for r in reactions_sorted]
        
        # Processar reações
        for reaction in reactions_sorted:
            reaction_id = reaction['id']
            tree['nodes'][reaction_id] = reaction
            
            # Determinar parent
            parent_id = reaction.get('parent_id', source_tweet['id'])
            if parent_id not in tree['nodes']:
                parent_id = source_tweet['id']  # Fallback para root
            
            # Adicionar aresta
            tree['edges'].append({
                'parent': parent_id,
                'child': reaction_id,
                'weight': self._calculate_similarity_weight(
                    tree['nodes'][parent_id], 
                    reaction
                )
            })
            
            # Calcular depth
            parent_depth = tree['depth_map'].get(parent_id, 0)
            tree['depth_map'][reaction_id] = parent_depth + 1
        
        return tree
    
    def _calculate_similarity_weight(self, parent_tweet: Dict, child_tweet: Dict) -> float:
        """
        Calcula um peso de similaridade baseado em características dos tweets.
        
        Args:
            parent_tweet: Tweet pai
            child_tweet: Tweet filho
            
        Returns:
            Peso de similaridade entre 0 e 1
        """
        weight = 0.5  # Base weight
        
        # Temporal proximity (tweets mais próximos no tempo são mais similares)
        time_diff = abs(child_tweet.get('timestamp', 0) - parent_tweet.get('timestamp', 0))
        if time_diff > 0:
            # Normalizar para escala de horas (peso maior para tweets próximos)
            time_weight = max(0, 1 - (time_diff / 3600))  # 1 hora = peso 0
            weight += 0.2 * time_weight
        
        # User relationship (mesmo usuário ou menções)
        if child_tweet.get('user_id') == parent_tweet.get('user_id'):
            weight += 0.2
        elif parent_tweet.get('user_screen_name') in child_tweet.get('user_mentions', []):
            weight += 0.15
        
        # Content similarity (hashtags comuns)
        parent_hashtags = set(parent_tweet.get('hashtags', []))
        child_hashtags = set(child_tweet.get('hashtags', []))
        if parent_hashtags and child_hashtags:
            hashtag_similarity = len(parent_hashtags & child_hashtags) / len(parent_hashtags | child_hashtags)
            weight += 0.15 * hashtag_similarity
        
        return min(1.0, weight)
    
    def process_all_events(self) -> List[Dict]:
        """
        Processa todos os eventos do dataset PHEME.
        
        Returns:
            Lista com todas as cascatas processadas
        """
        all_cascades = []
        
        for event in self.events:
            logger.info(f"Processando evento: {event}")
            event_path = self.dataset_path / event
            
            if not event_path.exists():
                logger.warning(f"Evento {event} não encontrado em {event_path}")
                continue
            
            event_stats = {'rumours': 0, 'non-rumours': 0, 'total_reactions': 0}
            
            # Processar rumours e non-rumours
            for label in ['rumours', 'non-rumours']:
                label_path = event_path / label
                
                if not label_path.exists():
                    continue
                
                # Processar cada cascata
                cascade_dirs = [d for d in label_path.iterdir() if d.is_dir()]
                
                for cascade_dir in tqdm(cascade_dirs, desc=f"{event}-{label}"):
                    cascade_info = self.process_cascade(cascade_dir, event, label)
                    
                    if cascade_info['source_tweet']:  # Só adicionar se source tweet existe
                        all_cascades.append(cascade_info)
                        event_stats[label] += 1
                        event_stats['total_reactions'] += len(cascade_info['reactions'])
            
            self.stats['events'][event] = event_stats
            logger.info(f"Evento {event}: {event_stats}")
        
        # Atualizar estatísticas globais
        self.stats['total_cascades'] = len(all_cascades)
        self.stats['total_source_tweets'] = len(all_cascades)
        self.stats['total_reactions'] = sum(len(c['reactions']) for c in all_cascades)
        
        logger.info(f"Processamento concluído: {self.stats}")
        return all_cascades
    
    def create_flat_dataset(self, cascades: List[Dict]) -> pd.DataFrame:
        """
        Cria um dataset plano com todos os tweets (source + reactions).
        
        Args:
            cascades: Lista de cascatas processadas
            
        Returns:
            DataFrame com todos os tweets
        """
        rows = []
        
        for cascade in cascades:
            # Adicionar source tweet
            source = cascade['source_tweet'].copy()
            source.update({
                'cascade_id': cascade['cascade_id'],
                'event': cascade['event'],
                'label': cascade['label'],
                'conversation_tree': json.dumps(cascade['conversation_tree'])
            })
            rows.append(source)
            
            # Adicionar reactions
            for reaction in cascade['reactions']:
                reaction_row = reaction.copy()
                reaction_row.update({
                    'cascade_id': cascade['cascade_id'],
                    'event': cascade['event'],
                    'label': cascade['label'],
                    'conversation_tree': json.dumps(cascade['conversation_tree'])
                })
                rows.append(reaction_row)
        
        return pd.DataFrame(rows)
    
    def create_cascade_dataset(self, cascades: List[Dict]) -> pd.DataFrame:
        """
        Cria um dataset com uma linha por cascata (apenas source tweets).
        
        Args:
            cascades: Lista de cascatas processadas
            
        Returns:
            DataFrame com source tweets e metadados da cascata
        """
        rows = []
        
        for cascade in cascades:
            if not cascade['source_tweet']:
                continue
                
            source = cascade['source_tweet'].copy()
            
            # Adicionar estatísticas da cascata
            source.update({
                'cascade_id': cascade['cascade_id'],
                'event': cascade['event'],
                'label': cascade['label'],
                'num_reactions': len(cascade['reactions']),
                'max_depth': max([r.get('depth', 1) for r in cascade['reactions']] + [0]),
                'cascade_duration': self._calculate_cascade_duration(cascade),
                'unique_users': len(set([cascade['source_tweet'].get('user_id', '')] + 
                                       [r.get('user_id', '') for r in cascade['reactions']])),
                'conversation_tree': json.dumps(cascade['conversation_tree'])
            })
            
            rows.append(source)
        
        return pd.DataFrame(rows)
    
    def _calculate_cascade_duration(self, cascade: Dict) -> float:
        """
        Calcula a duração da cascata em horas.
        
        Args:
            cascade: Informações da cascata
            
        Returns:
            Duração em horas
        """
        if not cascade['reactions']:
            return 0.0
        
        source_time = cascade['source_tweet'].get('timestamp', 0)
        reaction_times = [r.get('timestamp', 0) for r in cascade['reactions']]
        
        if source_time == 0 or not reaction_times:
            return 0.0
        
        return (max(reaction_times) - source_time) / 3600  # Converter para horas
    
    def save_processed_data(self, cascades: List[Dict], output_dir: str = 'datasets/processed'):
        """
        Salva os dados processados em diferentes formatos.
        
        Args:
            cascades: Lista de cascatas processadas
            output_dir: Diretório de saída
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Dataset plano (todos os tweets)
        flat_df = self.create_flat_dataset(cascades)
        flat_df.to_csv(output_path / 'pheme_all_tweets.csv', index=False)
        logger.info(f"Dataset plano salvo: {len(flat_df)} tweets")
        
        # 2. Dataset de cascatas (apenas source tweets)
        cascade_df = self.create_cascade_dataset(cascades)
        cascade_df.to_csv(output_path / 'pheme_cascades.csv', index=False)
        logger.info(f"Dataset de cascatas salvo: {len(cascade_df)} cascatas")
        
        # 3. Metadados completos (JSON Lines)
        with open(output_path / 'pheme_complete.jsonl', 'w', encoding='utf-8') as f:
            for cascade in cascades:
                f.write(json.dumps(cascade, ensure_ascii=False) + '\n')
        logger.info(f"Metadados completos salvos: {len(cascades)} cascatas")
        
        # 4. Estatísticas
        with open(output_path / 'dataset_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info("Estatísticas salvas")
        
        return {
            'flat_dataset': output_path / 'pheme_all_tweets.csv',
            'cascade_dataset': output_path / 'pheme_cascades.csv',
            'complete_metadata': output_path / 'pheme_complete.jsonl',
            'statistics': output_path / 'dataset_stats.json'
        }


def main():
    """Função principal para processar o dataset PHEME."""
    
    print("="*60)
    print("PROCESSADOR DO DATASET PHEME")
    print("Extração de estruturas conversacionais para Filo-Transformer")
    print("="*60)
    
    # Inicializar processador
    processor = PHEMEDatasetProcessor()
    
    # Processar todos os eventos
    print("\n🔄 Processando eventos...")
    cascades = processor.process_all_events()
    
    # Salvar dados processados
    print("\n💾 Salvando dados processados...")
    saved_files = processor.save_processed_data(cascades)
    
    print("\n✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"📊 Total de cascatas: {processor.stats['total_cascades']}")
    print(f"📊 Total de tweets: {processor.stats['total_source_tweets'] + processor.stats['total_reactions']}")
    print(f"📊 Total de reações: {processor.stats['total_reactions']}")
    
    print("\n📁 Arquivos gerados:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")
    
    print("\n🎯 Próximos passos:")
    print("1. Use 'pheme_cascades.csv' para treinar o Filo-Transformer")
    print("2. Use 'pheme_complete.jsonl' para análises detalhadas de cascatas")
    print("3. Use as árvores de conversação para extrair características filogenéticas")


if __name__ == "__main__":
    main()