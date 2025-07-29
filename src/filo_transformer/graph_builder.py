"""
Phylogenetic graph construction module.

This module implements the construction of phylogenetic graphs from text embeddings
using k-NN similarity and cosine distance metrics.
"""

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from typing import List, Tuple

from .config import FiloTransformerConfig


class PhylogeneticGraphBuilder:
    """
    Builds phylogenetic graphs from text embeddings for TAG feature extraction.
    
    This class constructs directed graphs representing evolutionary relationships
    between texts based on their semantic similarity, which forms the foundation
    for phylogenetic feature extraction.
    """
    
    def __init__(self, config: FiloTransformerConfig):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration object containing similarity thresholds and k-NN parameters
        """
        self.config = config
        self.similarity_threshold = config.similarity_threshold
        self.knn_k = config.knn_k
        
    def build_graph(
        self, 
        embeddings: np.ndarray, 
        node_ids: List[int]
    ) -> Tuple[nx.DiGraph, np.ndarray]:
        """
        Build a phylogenetic graph from text embeddings.
        
        This method creates a directed graph where edges represent evolutionary
        relationships based on semantic similarity between texts.
        
        Args:
            embeddings: Normalized text embeddings with shape (n_texts, embedding_dim)
            node_ids: List of node IDs corresponding to original dataset indices
            
        Returns:
            Tuple of (directed graph, similarity matrix)
        """
        print(f"Building phylogenetic graph for {len(embeddings)} nodes...")
        
        # Create empty directed graph
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        
        # Compute similarity matrix
        similarity_matrix = embeddings.dot(embeddings.T)
        np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity
        
        if len(embeddings) <= 1:
            print("Warning: Insufficient nodes for edge creation")
            return G, similarity_matrix
        
        # Use k-NN to determine potential edges
        k = min(self.knn_k, len(embeddings) - 1)
        knn_graph = kneighbors_graph(
            embeddings,
            n_neighbors=k,
            mode='connectivity',
            metric='cosine',
            include_self=False
        ).tocoo()
        
        # Add edges based on similarity threshold
        edges_added = 0
        for i, j in zip(knn_graph.row, knn_graph.col):
            similarity = similarity_matrix[i, j]
            
            if similarity >= self.similarity_threshold:
                source_node = node_ids[i]
                target_node = node_ids[j]
                
                G.add_edge(
                    source_node, 
                    target_node, 
                    weight=similarity
                )
                edges_added += 1
        
        print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        print(f"Edge density: {edges_added / (len(embeddings) * (len(embeddings) - 1)):.4f}")
        
        return G, similarity_matrix
    
    def build_fold_graph(
        self,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray, 
        train_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> Tuple[nx.DiGraph, np.ndarray, List[int]]:
        """
        Build graph for a specific cross-validation fold.
        
        This method combines train and test embeddings to build a single graph
        that includes all samples from the current fold, enabling transductive
        learning for TAG feature extraction.
        
        Args:
            train_embeddings: Training set embeddings
            test_embeddings: Test set embeddings  
            train_indices: Original dataset indices for training samples
            test_indices: Original dataset indices for test samples
            
        Returns:
            Tuple of (fold graph, similarity matrix, combined node indices)
        """
        # Combine embeddings and indices
        combined_embeddings = np.vstack([train_embeddings, test_embeddings])
        combined_indices = np.concatenate([train_indices, test_indices])
        
        # Build graph for the entire fold
        fold_graph, similarity_matrix = self.build_graph(
            combined_embeddings,
            combined_indices.tolist()
        )
        
        return fold_graph, similarity_matrix, combined_indices.tolist()
    
    def get_graph_statistics(self, graph: nx.DiGraph) -> dict:
        """
        Compute basic statistics about the constructed graph.
        
        Args:
            graph: The phylogenetic graph
            
        Returns:
            Dictionary containing graph statistics
        """
        if graph.number_of_nodes() == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'density': 0.0,
                'avg_in_degree': 0.0,
                'avg_out_degree': 0.0,
                'num_connected_components': 0
            }
        
        # Basic metrics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        
        # Degree statistics
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        
        avg_in_degree = np.mean(in_degrees) if in_degrees else 0.0
        avg_out_degree = np.mean(out_degrees) if out_degrees else 0.0
        
        # Connected components (using undirected version)
        undirected = graph.to_undirected()
        num_components = nx.number_connected_components(undirected)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges, 
            'density': density,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'num_connected_components': num_components
        }