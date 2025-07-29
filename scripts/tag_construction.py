#!/usr/bin/env python3
"""
Construção de Tree Alignment Graphs (TAGs) para análise filogenética de textos.

Este módulo implementa a construção de TAGs conforme descrito no artigo,
capturando relações evolutivas complexas entre textos em cascatas de informação.

Autor: Acauan Cardoso Ribeiro, Eduardo Luzeiro Feitosa, André Carvalho
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import pandas as pd
from typing import List, Dict, Tuple, Optional


class TAGConstructor:
    """
    Constrói Tree Alignment Graphs (TAGs) a partir de embeddings semânticos.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Inicializa o construtor de TAGs.
        
        Args:
            similarity_threshold: Limiar de similaridade para considerar relação entre posts
        """
        self.similarity_threshold = similarity_threshold
        
    def build_tag(self, embeddings: np.ndarray, timestamps: List[float], 
                  post_ids: List[str], reply_structure: Optional[Dict] = None) -> nx.DiGraph:
        """
        Constrói o TAG a partir dos embeddings e metadados.
        
        Args:
            embeddings: Array de embeddings semânticos (n_posts, embedding_dim)
            timestamps: Lista de timestamps dos posts
            post_ids: Lista de IDs dos posts
            reply_structure: Dicionário opcional com estrutura de respostas
            
        Returns:
            TAG como grafo direcionado NetworkX
        """
        n_posts = len(embeddings)
        
        # 1. Calcular matriz de similaridade
        similarity_matrix = cosine_similarity(embeddings)
        
        # 2. Criar grafo base
        G = nx.DiGraph()
        
        # Adicionar nós com atributos
        for i, post_id in enumerate(post_ids):
            G.add_node(post_id, 
                      embedding=embeddings[i],
                      timestamp=timestamps[i],
                      index=i)
        
        # 3. Identificar posts semente (mais antigos ou sem pais)
        sorted_indices = np.argsort(timestamps)
        
        # 4. Construir árvores de propagação
        for i in sorted_indices[1:]:  # Skip primeiro (mais antigo)
            current_id = post_ids[i]
            current_time = timestamps[i]
            
            # Encontrar potenciais ancestrais (posts anteriores)
            potential_ancestors = []
            
            for j in sorted_indices:
                if timestamps[j] >= current_time:
                    break
                    
                ancestor_id = post_ids[j]
                similarity = similarity_matrix[i, j]
                
                # Considerar estrutura de resposta se disponível
                if reply_structure and current_id in reply_structure:
                    if ancestor_id == reply_structure[current_id]:
                        # Resposta direta tem prioridade máxima
                        potential_ancestors.append((ancestor_id, 1.0, j))
                        break
                elif similarity >= self.similarity_threshold:
                    potential_ancestors.append((ancestor_id, similarity, j))
            
            # Conectar ao ancestral mais provável
            if potential_ancestors:
                # Ordenar por similaridade (descendente)
                potential_ancestors.sort(key=lambda x: x[1], reverse=True)
                best_ancestor = potential_ancestors[0]
                
                # Adicionar aresta com peso = similaridade
                G.add_edge(best_ancestor[0], current_id, 
                          weight=best_ancestor[1],
                          mutation_distance=1.0 - best_ancestor[1])
        
        # 5. Identificar e adicionar recombinações
        self._add_recombinations(G, similarity_matrix, post_ids, timestamps)
        
        return G
    
    def _add_recombinations(self, G: nx.DiGraph, similarity_matrix: np.ndarray,
                           post_ids: List[str], timestamps: List[float]):
        """
        Identifica e adiciona arestas de recombinação ao grafo.
        
        Uma recombinação ocorre quando um post tem alta similaridade com
        múltiplos ancestrais de diferentes ramos.
        """
        recombination_threshold = 0.8
        
        for i, post_id in enumerate(post_ids):
            if G.in_degree(post_id) > 0:  # Já tem um pai principal
                continue
                
            # Encontrar posts de diferentes linhagens
            high_similarity_ancestors = []
            
            for j, other_id in enumerate(post_ids):
                if i == j or timestamps[j] >= timestamps[i]:
                    continue
                    
                if similarity_matrix[i, j] >= recombination_threshold:
                    # Verificar se pertence a linhagem diferente
                    ancestors_i = set(nx.ancestors(G, post_id))
                    ancestors_j = set(nx.ancestors(G, other_id))
                    
                    if len(ancestors_i & ancestors_j) == 0:  # Linhagens diferentes
                        high_similarity_ancestors.append((other_id, similarity_matrix[i, j]))
            
            # Adicionar arestas de recombinação (tipo especial)
            for ancestor_id, sim in high_similarity_ancestors[:2]:  # Max 2 pais recombinantes
                G.add_edge(ancestor_id, post_id,
                          weight=sim,
                          edge_type='recombination',
                          mutation_distance=1.0 - sim)
    
    def extract_phylogenetic_features(self, G: nx.DiGraph) -> pd.DataFrame:
        """
        Extrai atributos filogenéticos do TAG conforme Tabela do artigo.
        
        Returns:
            DataFrame com atributos para cada nó
        """
        features = []
        
        # Calcular métricas globais uma vez
        if len(G) > 0:
            pagerank = nx.pagerank(G)
            
            # Componentes para betweenness e closeness
            if nx.is_weakly_connected(G):
                betweenness = nx.betweenness_centrality(G)
                closeness = nx.closeness_centrality(G)
            else:
                # Calcular por componente
                betweenness = {}
                closeness = {}
                for component in nx.weakly_connected_components(G):
                    subgraph = G.subgraph(component)
                    betweenness.update(nx.betweenness_centrality(subgraph))
                    closeness.update(nx.closeness_centrality(subgraph))
            
            # Detectar comunidades no grafo não-direcionado
            G_undirected = G.to_undirected()
            communities = list(nx.community.louvain_communities(G_undirected))
            node_to_community = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = i
        
        for node in G.nodes():
            node_features = {
                'node_id': node,
                
                # Centralidade & Importância
                'pagerank': pagerank.get(node, 0),
                'closeness': closeness.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                
                # Grau & Estrutura Local
                'degree_normal': G.degree(node),
                'degree_in': G.in_degree(node),
                'degree_out': G.out_degree(node),
                'is_leaf': int(G.out_degree(node) == 0),
                
                # Relações Ancestrais/Descendentes
                'n_ancestors': len(nx.ancestors(G, node)),
                'n_descendants': len(nx.descendants(G, node)),
                'subtree_size': len(nx.descendants(G, node)) + 1,
                
                # Evolução, Profundidade e Recombinação
                'depth_normal': self._calculate_depth(G, node),
                'recomb_degree': self._calculate_recombination_degree(G, node),
                'entropy_ancestors': self._calculate_ancestor_entropy(G, node),
                'mutation_rate': self._calculate_mutation_rate(G, node),
            }
            
            # Diversidade & Comunidade
            node_features['num_communities'] = len(communities)
            node_features['community_id'] = node_to_community.get(node, -1)
            node_features['gini_similarity'] = self._calculate_gini_similarity(G, node)
            
            features.append(node_features)
        
        return pd.DataFrame(features)
    
    def _calculate_depth(self, G: nx.DiGraph, node: str) -> int:
        """Calcula a profundidade do nó na árvore."""
        try:
            # Encontrar todos os caminhos desde as raízes
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            if not roots:
                return 0
            
            max_depth = 0
            for root in roots:
                if nx.has_path(G, root, node):
                    paths = nx.all_simple_paths(G, root, node)
                    for path in paths:
                        max_depth = max(max_depth, len(path) - 1)
            
            return max_depth
        except:
            return 0
    
    def _calculate_recombination_degree(self, G: nx.DiGraph, node: str) -> int:
        """Conta quantas arestas de recombinação o nó possui."""
        recomb_edges = 0
        for pred in G.predecessors(node):
            edge_data = G.get_edge_data(pred, node)
            if edge_data.get('edge_type') == 'recombination':
                recomb_edges += 1
        return recomb_edges
    
    def _calculate_ancestor_entropy(self, G: nx.DiGraph, node: str) -> float:
        """Calcula a entropia da distribuição de ancestrais."""
        ancestors = list(nx.ancestors(G, node))
        if not ancestors:
            return 0.0
        
        # Contar ancestrais por nível
        level_counts = defaultdict(int)
        for ancestor in ancestors:
            depth = self._calculate_depth(G, ancestor)
            level_counts[depth] += 1
        
        # Calcular entropia
        total = len(ancestors)
        entropy = 0.0
        for count in level_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_mutation_rate(self, G: nx.DiGraph, node: str) -> float:
        """Calcula a taxa média de mutação nas arestas incidentes."""
        mutation_distances = []
        
        for pred in G.predecessors(node):
            edge_data = G.get_edge_data(pred, node)
            if 'mutation_distance' in edge_data:
                mutation_distances.append(edge_data['mutation_distance'])
        
        return np.mean(mutation_distances) if mutation_distances else 0.0
    
    def _calculate_gini_similarity(self, G: nx.DiGraph, node: str) -> float:
        """Calcula o coeficiente de Gini da similaridade com vizinhos."""
        similarities = []
        
        # Coletar similaridades com todos os vizinhos
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            if 'weight' in edge_data:
                similarities.append(edge_data['weight'])
        
        for neighbor in G.predecessors(node):
            edge_data = G.get_edge_data(neighbor, node)
            if 'weight' in edge_data:
                similarities.append(edge_data['weight'])
        
        if not similarities:
            return 0.0
        
        # Calcular coeficiente de Gini
        similarities = sorted(similarities)
        n = len(similarities)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * similarities)) / (n * np.sum(similarities)) - (n + 1) / n