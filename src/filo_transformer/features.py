"""
Tree Alignment Graph (TAG) feature extraction module.

This module implements comprehensive phylogenetic and graph-based feature extraction
for the Filo-Transformer model, including centrality measures, evolutionary metrics,
and graph embeddings.
"""

import warnings
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community

from .config import FiloTransformerConfig

# Suppress Node2Vec warnings
warnings.filterwarnings('ignore', category=UserWarning, module='node2vec')


def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient for measuring inequality in distributions.
    
    Args:
        values: Array of numerical values
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    arr = np.sort(values.astype(float))
    n = arr.size
    
    if n == 0 or np.allclose(arr, 0):
        return 0.0
    
    cum = np.cumsum(arr)
    if cum[-1] == 0:
        return 0.0
    
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr)) / (n * cum[-1]) - (n + 1) / n


class TAGFeatureExtractor:
    """
    Extracts Tree Alignment Graph (TAG) features from phylogenetic graphs.
    
    This class implements a comprehensive set of phylogenetic and graph-based
    features that capture evolutionary patterns and structural properties of
    the information propagation network.
    """
    
    def __init__(self, config: FiloTransformerConfig):
        """
        Initialize the TAG feature extractor.
        
        Args:
            config: Configuration object containing Node2Vec and other parameters
        """
        self.config = config
        
    def extract_features(
        self,
        graph: nx.DiGraph,
        node_ids: List[int],
        similarity_matrix: np.ndarray,
        nodes_in_similarity: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Extract comprehensive TAG features for specified nodes.
        
        This method computes 79-dimensional feature vectors including:
        - Graph centrality measures (PageRank, betweenness, closeness)
        - Phylogenetic metrics (depth, recombination, mutation rate)
        - Community detection features
        - Node2Vec graph embeddings (64-dimensional)
        
        Args:
            graph: The phylogenetic graph
            node_ids: List of node IDs to extract features for
            similarity_matrix: Pairwise similarity matrix between nodes
            nodes_in_similarity: Mapping of node IDs to similarity matrix indices
            
        Returns:
            DataFrame with TAG features for each node (79 columns)
        """
        if not node_ids:
            return self._get_empty_features_dataframe(node_ids)
        
        if nodes_in_similarity is None:
            nodes_in_similarity = node_ids
        
        # Create mapping from node ID to similarity matrix index
        similarity_mapping = {
            node_id: idx 
            for idx, node_id in enumerate(nodes_in_similarity) 
            if node_id in node_ids
        }
        
        print(f"Extracting TAG features for {len(node_ids)} nodes...")
        
        # Initialize feature dataframe
        df = pd.DataFrame(index=node_ids)
        
        # Extract all feature groups
        self._extract_centrality_features(df, graph)
        self._extract_degree_features(df, graph)
        self._extract_phylogenetic_features(df, graph)
        self._extract_similarity_features(df, graph, similarity_matrix, similarity_mapping)
        self._extract_community_features(df, graph)
        self._extract_depth_features(df, graph)
        self._extract_evolutionary_features(df, graph, similarity_matrix, similarity_mapping)
        self._extract_graph_embeddings(df, graph, node_ids)
        
        # Ensure all features are present and fill missing values
        df = self._ensure_complete_features(df)
        
        print(f"Extracted {df.shape[1]} features for {df.shape[0]} nodes")
        return df
    
    def _extract_centrality_features(self, df: pd.DataFrame, graph: nx.DiGraph) -> None:
        """Extract graph centrality features."""
        # PageRank
        pagerank_scores = nx.pagerank(graph, weight='weight') if graph.number_of_nodes() > 0 else {}
        df['pagerank'] = df.index.map(lambda n: pagerank_scores.get(n, 0.0))
        
        # Betweenness and closeness centrality
        if graph.number_of_nodes() > 0:
            df['betweenness'] = pd.Series(nx.betweenness_centrality(graph))
            df['closeness'] = pd.Series(nx.closeness_centrality(graph))
        else:
            df['betweenness'] = 0.0
            df['closeness'] = 0.0
    
    def _extract_degree_features(self, df: pd.DataFrame, graph: nx.DiGraph) -> None:
        """Extract degree-based features."""
        max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 1
        
        df['deg_norm'] = df.index.map(
            lambda n: graph.degree(n) / max_degree if max_degree > 0 and graph.has_node(n) else 0.0
        )
        df['deg_in'] = df.index.map(lambda n: graph.in_degree(n) if graph.has_node(n) else 0)
        df['deg_out'] = df.index.map(lambda n: graph.out_degree(n) if graph.has_node(n) else 0)
    
    def _extract_phylogenetic_features(self, df: pd.DataFrame, graph: nx.DiGraph) -> None:
        """Extract phylogenetic tree features."""
        # Ancestor and descendant counts
        df['n_anc'] = df.index.map(lambda n: len(list(nx.ancestors(graph, n))) if graph.has_node(n) else 0)
        df['n_desc'] = df.index.map(lambda n: len(list(nx.descendants(graph, n))) if graph.has_node(n) else 0)
        
        # Leaf nodes and recombination degree
        df['is_leaf'] = df.index.map(lambda n: graph.out_degree(n) == 0 if graph.has_node(n) else True)
        df['recomb_degree'] = df.index.map(lambda n: max(0, graph.in_degree(n) - 1) if graph.has_node(n) else 0)
        
        # Subtree size
        df['subtree_size'] = df.index.map(lambda n: len(nx.descendants(graph, n)) + 1 if graph.has_node(n) else 1)
    
    def _extract_similarity_features(
        self, 
        df: pd.DataFrame, 
        graph: nx.DiGraph, 
        similarity_matrix: np.ndarray,
        similarity_mapping: Dict[int, int]
    ) -> None:
        """Extract similarity-based features."""
        for node in df.index:
            if node not in similarity_mapping or not graph.has_node(node):
                df.at[node, 'gini_sim'] = 0.0
                continue
                
            # Get neighbors and their similarities
            neighbors = [v for v in graph.neighbors(node) if v in similarity_mapping]
            if neighbors:
                neighbor_similarities = [
                    similarity_matrix[similarity_mapping[node], similarity_mapping[neighbor]]
                    for neighbor in neighbors
                ]
                df.at[node, 'gini_sim'] = gini_coefficient(np.array(neighbor_similarities))
            else:
                df.at[node, 'gini_sim'] = 0.0
    
    def _extract_community_features(self, df: pd.DataFrame, graph: nx.DiGraph) -> None:
        """Extract community detection features."""
        if graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
            undirected_graph = graph.to_undirected()
            communities = community.greedy_modularity_communities(undirected_graph)
            
            # Map nodes to community IDs
            community_mapping = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_mapping[node] = i
            
            df['num_comms'] = df.index.map(lambda n: 1 if n in community_mapping else 0)
        else:
            df['num_comms'] = 0.0
    
    def _extract_depth_features(self, df: pd.DataFrame, graph: nx.DiGraph) -> None:
        """Extract depth-based features."""
        if graph.number_of_nodes() == 0:
            df['depth_norm'] = 0.0
            return
            
        # Find root nodes (no predecessors)
        roots = [n for n, d in graph.in_degree() if d == 0]
        
        # Calculate depth from roots
        depth_mapping = {}
        for root in roots:
            distances = nx.single_source_shortest_path_length(graph, root)
            for node, distance in distances.items():
                depth_mapping[node] = min(depth_mapping.get(node, np.inf), distance)
        
        # Normalize depths
        if depth_mapping:
            max_depth = max(depth_mapping.values())
            if max_depth > 0:
                df['depth_norm'] = df.index.map(lambda n: depth_mapping.get(n, 0) / max_depth)
            else:
                df['depth_norm'] = 0.0
        else:
            df['depth_norm'] = 0.0
    
    def _extract_evolutionary_features(
        self,
        df: pd.DataFrame,
        graph: nx.DiGraph,
        similarity_matrix: np.ndarray,
        similarity_mapping: Dict[int, int]
    ) -> None:
        """Extract evolutionary features (entropy, mutation rate)."""
        entropy_scores = {}
        mutation_rates = {}
        
        for node in df.index:
            if node not in similarity_mapping or not graph.has_node(node):
                entropy_scores[node] = 0.0
                mutation_rates[node] = 0.0
                continue
            
            predecessors = [p for p in graph.predecessors(node) if p in similarity_mapping]
            
            if predecessors:
                # Calculate entropy of ancestor similarities
                similarities = np.array([
                    similarity_matrix[similarity_mapping[node], similarity_mapping[pred]]
                    for pred in predecessors
                ])
                
                if similarities.sum() > 0:
                    probabilities = similarities / similarities.sum()
                    entropy_scores[node] = -np.sum(probabilities * np.log(probabilities + 1e-12))
                else:
                    entropy_scores[node] = 0.0
                
                # Calculate mutation rate (1 - mean similarity to ancestors)
                mutation_rates[node] = 1 - np.mean(similarities)
            else:
                entropy_scores[node] = 0.0
                mutation_rates[node] = 0.0
        
        df['entropy_anc'] = pd.Series(entropy_scores)
        df['mut_rate'] = pd.Series(mutation_rates)
    
    def _extract_graph_embeddings(self, df: pd.DataFrame, graph: nx.DiGraph, node_ids: List[int]) -> None:
        """Extract Node2Vec graph embeddings."""
        embedding_dim = self.config.node2vec_dimensions
        embeddings = np.zeros((len(node_ids), embedding_dim))
        
        try:
            from node2vec import Node2Vec
            
            if graph.number_of_nodes() > 1 and graph.number_of_edges() > 0:
                # Convert node IDs to strings for Node2Vec
                valid_nodes = [str(n) for n in node_ids if graph.has_node(n)]
                
                if len(valid_nodes) > 1:
                    subgraph = graph.subgraph(valid_nodes).copy()
                    
                    if subgraph.number_of_nodes() > 1 and subgraph.number_of_edges() > 0:
                        # Train Node2Vec model
                        node2vec = Node2Vec(
                            subgraph,
                            dimensions=embedding_dim,
                            walk_length=self.config.node2vec_walk_length,
                            num_walks=self.config.node2vec_num_walks,
                            workers=1,
                            quiet=True
                        )
                        
                        model = node2vec.fit(
                            window=self.config.node2vec_window,
                            min_count=self.config.node2vec_min_count
                        )
                        
                        # Extract embeddings
                        node_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
                        for node_str in model.wv.index_to_key:
                            original_node_id = int(node_str)
                            if original_node_id in node_to_index:
                                embeddings[node_to_index[original_node_id], :] = model.wv[node_str]
        
        except ImportError:
            print("Node2Vec not available. Using zero embeddings.")
        except Exception as e:
            print(f"Error computing Node2Vec embeddings: {e}")
        
        # Add embedding features to dataframe
        for i in range(embedding_dim):
            df[f'graph_emb_{i}'] = embeddings[:, i] if len(node_ids) > 0 else 0.0
    
    def _ensure_complete_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all expected features are present and handle missing values."""
        # Define expected feature columns
        base_features = [
            'pagerank', 'deg_norm', 'deg_in', 'deg_out', 'n_anc', 'n_desc',
            'gini_sim', 'num_comms', 'depth_norm', 'is_leaf', 'recomb_degree',
            'entropy_anc', 'mut_rate', 'subtree_size', 'closeness', 'betweenness'
        ]
        
        embedding_features = [f'graph_emb_{i}' for i in range(self.config.node2vec_dimensions)]
        all_features = base_features + embedding_features
        
        # Reindex to ensure all columns are present
        df = df.reindex(columns=all_features, fill_value=0.0)
        
        # Fill any remaining NaN values
        df = df.fillna(0.0)
        
        return df.astype(float)
    
    def _get_empty_features_dataframe(self, node_ids: List[int]) -> pd.DataFrame:
        """Return empty dataframe with correct feature structure."""
        base_features = [
            'pagerank', 'deg_norm', 'deg_in', 'deg_out', 'n_anc', 'n_desc',
            'gini_sim', 'num_comms', 'depth_norm', 'is_leaf', 'recomb_degree',
            'entropy_anc', 'mut_rate', 'subtree_size', 'closeness', 'betweenness'
        ]
        
        embedding_features = [f'graph_emb_{i}' for i in range(self.config.node2vec_dimensions)]
        all_features = base_features + embedding_features
        
        return pd.DataFrame(columns=all_features, index=node_ids).astype(float)