#!/usr/bin/env python3
"""
Main script to run Filo-Transformer experiments.

This script runs both the Filo-Transformer model (with phylogenetic features)
and the baseline model (without phylogenetic features) for comparison.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, f1_score

from filo_transformer.config import FiloTransformerConfig
from filo_transformer.embeddings import get_embedder
from filo_transformer.graph_builder import PhylogeneticGraphBuilder
from filo_transformer.features import TAGFeatureExtractor
from filo_transformer.model import FiloTransformer, BaselineTransformer


class Preprocessor:
    """Simple feature normalizer."""
    
    def __init__(self):
        from tensorflow.keras.layers import Normalization
        self.normalizer = Normalization()
    
    def fit(self, X):
        if X.size > 0:
            self.normalizer.adapt(X)
    
    def transform(self, X):
        return self.normalizer(X).numpy() if X.size > 0 else X


def run_filo_transformer_experiment(config: FiloTransformerConfig) -> dict:
    """
    Run the complete Filo-Transformer experiment with cross-validation.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with experiment results
    """
    print("=" * 60)
    print("FILO-TRANSFORMER EXPERIMENT")
    print("=" * 60)
    
    # Load dataset
    print(f"Loading dataset: {config.dataset_path}")
    df = pd.read_csv(config.dataset_path)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].to_numpy()
    
    print(f"Dataset loaded: {len(texts)} samples")
    print(f"Label distribution: {np.unique(labels, return_counts=True)}")
    
    # Generate embeddings
    print(f"Generating embeddings using: {config.embedding_model_name}")
    embedder = get_embedder(config)
    embeddings = embedder.encode(texts)
    
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Cross-validation
    skf = StratifiedKFold(
        n_splits=config.cv_folds, 
        shuffle=True, 
        random_state=config.random_state
    )
    
    results = {
        'accuracy': [],
        'auc': [],
        'recall': [],
        'f1': [],
        'roc_data': []
    }
    
    graph_builder = PhylogeneticGraphBuilder(config)
    feature_extractor = TAGFeatureExtractor(config)
    model_builder = FiloTransformer(config)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels), 1):
        print(f"\n--- Fold {fold}/{config.cv_folds} ---")
        
        # Clear TensorFlow session
        import tensorflow as tf
        tf.keras.backend.clear_session()
        
        # Split data
        X_train_emb = embeddings[train_idx]
        X_test_emb = embeddings[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        # Build phylogenetic graph for this fold
        fold_graph, similarity_matrix, combined_indices = graph_builder.build_fold_graph(
            X_train_emb, X_test_emb, train_idx, test_idx
        )
        
        # Extract TAG features
        tag_features_df = feature_extractor.extract_features(
            fold_graph,
            combined_indices,
            similarity_matrix,
            combined_indices
        )
        
        # Split TAG features
        train_tag_features = tag_features_df.loc[train_idx].values
        test_tag_features = tag_features_df.loc[test_idx].values
        
        # Handle NaN values
        train_tag_features = np.nan_to_num(train_tag_features, nan=0.0)
        test_tag_features = np.nan_to_num(test_tag_features, nan=0.0)
        
        # Normalize TAG features
        tag_preprocessor = Preprocessor()
        tag_preprocessor.fit(train_tag_features)
        train_tag_norm = tag_preprocessor.transform(train_tag_features)
        test_tag_norm = tag_preprocessor.transform(test_tag_features)
        
        # Prepare model inputs
        d_text = embeddings.shape[1]
        d_tag = train_tag_norm.shape[1] if train_tag_norm.ndim > 1 else 0
        
        train_inputs = [X_train_emb[:, None, :]]
        test_inputs = [X_test_emb[:, None, :]]
        
        if d_tag > 0:
            train_inputs.append(train_tag_norm.reshape(-1, 1, d_tag).astype(np.float32))
            test_inputs.append(test_tag_norm.reshape(-1, 1, d_tag).astype(np.float32))
        
        # Build and train model
        model = model_builder.build_model(d_text, d_tag)
        if fold == 1:
            model.summary()
        
        # Train model
        history = model.fit(
            x=train_inputs,
            y=y_train,
            validation_data=(test_inputs, y_test),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=model_builder.get_callbacks(),
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc, test_auc, test_recall = model.evaluate(test_inputs, y_test, verbose=0)
        predictions_proba = model.predict(test_inputs)
        predictions_binary = (predictions_proba > 0.5).astype(int).flatten()
        
        # Calculate additional metrics
        recall_sklearn = recall_score(y_test, predictions_binary)
        f1_sklearn = f1_score(y_test, predictions_binary)
        
        # Store results
        results['accuracy'].append(test_acc)
        results['auc'].append(test_auc)
        results['recall'].append(recall_sklearn)
        results['f1'].append(f1_sklearn)
        results['roc_data'].append((y_test, predictions_proba.flatten()))
        
        print(f"Fold {fold} Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  Recall: {recall_sklearn:.4f}")
        print(f"  F1: {f1_sklearn:.4f}")
    
    # Print final results
    print(f"\n{'='*60}")
    print("FILO-TRANSFORMER FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"AUC:       {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print(f"Recall:    {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}")
    print(f"F1-Score:  {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    
    return results


def run_baseline_experiment(config: FiloTransformerConfig) -> dict:
    """
    Run the baseline transformer experiment without phylogenetic features.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with experiment results
    """
    print("=" * 60)
    print("BASELINE TRANSFORMER EXPERIMENT")
    print("=" * 60)
    
    # Load dataset
    print(f"Loading dataset: {config.dataset_path}")
    df = pd.read_csv(config.dataset_path)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].to_numpy()
    
    print(f"Dataset loaded: {len(texts)} samples")
    
    # Generate embeddings
    print(f"Generating embeddings using: {config.embedding_model_name}")
    embedder = get_embedder(config)
    embeddings = embedder.encode(texts)
    
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Cross-validation
    skf = StratifiedKFold(
        n_splits=config.cv_folds, 
        shuffle=True, 
        random_state=config.random_state
    )
    
    results = {
        'accuracy': [],
        'auc': [],
        'recall': [],
        'f1': [],
        'roc_data': []
    }
    
    model_builder = BaselineTransformer(config)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels), 1):
        print(f"\n--- Fold {fold}/{config.cv_folds} ---")
        
        # Clear TensorFlow session
        import tensorflow as tf
        tf.keras.backend.clear_session()
        
        # Split data
        X_train = embeddings[train_idx]
        X_test = embeddings[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
        # Prepare inputs (add sequence dimension)
        train_inputs = X_train[:, None, :]
        test_inputs = X_test[:, None, :]
        
        # Build and train model
        model = model_builder.build_model(embeddings.shape[1])
        if fold == 1:
            model.summary()
        
        # Train model
        history = model.fit(
            x=train_inputs,
            y=y_train,
            validation_data=(test_inputs, y_test),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=model_builder.get_callbacks(),
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc, test_auc, test_recall = model.evaluate(test_inputs, y_test, verbose=0)
        predictions_proba = model.predict(test_inputs)
        predictions_binary = (predictions_proba > 0.5).astype(int).flatten()
        
        # Calculate additional metrics
        recall_sklearn = recall_score(y_test, predictions_binary)
        f1_sklearn = f1_score(y_test, predictions_binary)
        
        # Store results
        results['accuracy'].append(test_acc)
        results['auc'].append(test_auc)
        results['recall'].append(recall_sklearn)
        results['f1'].append(f1_sklearn)
        results['roc_data'].append((y_test, predictions_proba.flatten()))
        
        print(f"Fold {fold} Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  Recall: {recall_sklearn:.4f}")
        print(f"  F1: {f1_sklearn:.4f}")
    
    # Print final results
    print(f"\n{'='*60}")
    print("BASELINE TRANSFORMER FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"AUC:       {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print(f"Recall:    {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}")
    print(f"F1-Score:  {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    
    return results


def save_results(filo_results: dict, baseline_results: dict, output_dir: str):
    """Save experiment results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    with open(f"{output_dir}/filo_transformer_results.json", 'w') as f:
        # Convert ROC data to serializable format
        serializable_filo = filo_results.copy()
        serializable_filo['roc_data'] = [
            (y_true.tolist(), y_pred.tolist()) 
            for y_true, y_pred in filo_results['roc_data']
        ]
        json.dump(serializable_filo, f, indent=2)
    
    with open(f"{output_dir}/baseline_results.json", 'w') as f:
        serializable_baseline = baseline_results.copy()
        serializable_baseline['roc_data'] = [
            (y_true.tolist(), y_pred.tolist()) 
            for y_true, y_pred in baseline_results['roc_data']
        ]
        json.dump(serializable_baseline, f, indent=2)
    
    # Save summary
    summary = {
        'filo_transformer': {
            'accuracy_mean': float(np.mean(filo_results['accuracy'])),
            'accuracy_std': float(np.std(filo_results['accuracy'])),
            'auc_mean': float(np.mean(filo_results['auc'])),
            'auc_std': float(np.std(filo_results['auc'])),
            'recall_mean': float(np.mean(filo_results['recall'])),
            'recall_std': float(np.std(filo_results['recall'])),
            'f1_mean': float(np.mean(filo_results['f1'])),
            'f1_std': float(np.std(filo_results['f1']))
        },
        'baseline': {
            'accuracy_mean': float(np.mean(baseline_results['accuracy'])),
            'accuracy_std': float(np.std(baseline_results['accuracy'])),
            'auc_mean': float(np.mean(baseline_results['auc'])),
            'auc_std': float(np.std(baseline_results['auc'])),
            'recall_mean': float(np.mean(baseline_results['recall'])),
            'recall_std': float(np.std(baseline_results['recall'])),
            'f1_mean': float(np.mean(baseline_results['f1'])),
            'f1_std': float(np.std(baseline_results['f1']))
        }
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}")


def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Run Filo-Transformer experiments')
    parser.add_argument('--dataset', type=str, default='datasets/pheme/pheme_all.csv',
                      help='Path to dataset CSV file')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    parser.add_argument('--skip-baseline', action='store_true',
                      help='Skip baseline experiment')
    parser.add_argument('--skip-filo', action='store_true',
                      help='Skip Filo-Transformer experiment')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = FiloTransformerConfig()
    config.dataset_path = args.dataset
    
    print(f"Using embedding model: {config.embedding_model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output directory: {args.output}")
    
    # Run experiments
    filo_results = None
    baseline_results = None
    
    if not args.skip_filo:
        start_time = time.time()
        filo_results = run_filo_transformer_experiment(config)
        filo_time = time.time() - start_time
        print(f"Filo-Transformer experiment completed in {filo_time:.2f} seconds")
    
    if not args.skip_baseline:
        start_time = time.time()
        baseline_results = run_baseline_experiment(config)
        baseline_time = time.time() - start_time
        print(f"Baseline experiment completed in {baseline_time:.2f} seconds")
    
    # Save results
    if filo_results and baseline_results:
        save_results(filo_results, baseline_results, args.output)
    
    print("Experiments completed successfully!")


if __name__ == '__main__':
    main()