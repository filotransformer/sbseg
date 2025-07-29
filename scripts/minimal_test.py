#!/usr/bin/env python3
"""
Minimal test script for Filo-Transformer installation verification.

This script performs a quick test to verify that all dependencies are installed
correctly and the basic functionality works.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ tensorflow {tf.__version__}")
    except ImportError as e:
        print(f"✗ tensorflow: {e}")
        return False
    
    try:
        import networkx as nx
        print("✓ networkx")
    except ImportError as e:
        print(f"✗ networkx: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✓ transformers")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        import openai
        print("✓ openai")
    except ImportError as e:
        print(f"✗ openai: {e}")
        return False
    
    return True


def test_filo_transformer_modules():
    """Test that Filo-Transformer modules can be imported."""
    print("\nTesting Filo-Transformer modules...")
    
    try:
        from filo_transformer.config import FiloTransformerConfig
        print("✓ config")
    except ImportError as e:
        print(f"✗ config: {e}")
        return False
    
    try:
        from filo_transformer.embeddings import SBERTEmbedder
        print("✓ embeddings")
    except ImportError as e:
        print(f"✗ embeddings: {e}")
        return False
    
    try:
        from filo_transformer.graph_builder import PhylogeneticGraphBuilder
        print("✓ graph_builder")
    except ImportError as e:
        print(f"✗ graph_builder: {e}")
        return False
    
    try:
        from filo_transformer.features import TAGFeatureExtractor
        print("✓ features")
    except ImportError as e:
        print(f"✗ features: {e}")
        return False
    
    try:
        from filo_transformer.model import FiloTransformer
        print("✓ model")
    except ImportError as e:
        print(f"✗ model: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality with minimal data."""
    print("\nTesting basic functionality...")
    
    try:
        import os
        # Force CPU usage to avoid GPU issues
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        import numpy as np
        import pandas as pd
        from filo_transformer.config import FiloTransformerConfig
        from filo_transformer.graph_builder import PhylogeneticGraphBuilder
        from filo_transformer.features import TAGFeatureExtractor
        
        # Create minimal test data
        test_texts = [
            "This is a fake news example.",
            "This is real news content.", 
            "Another fake news sample.",
            "Real news article text."
        ]
        test_labels = [1, 0, 1, 0]
        
        # Test configuration
        config = FiloTransformerConfig()
        config.openai_api_key = None  # Force SBERT usage
        print("✓ Configuration created")
        
        # Create dummy embeddings instead of using SBERT to avoid GPU issues
        np.random.seed(42)
        embeddings = np.random.rand(4, 768).astype(np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        print(f"✓ Mock embeddings created: {embeddings.shape}")
        
        # Test graph building
        graph_builder = PhylogeneticGraphBuilder(config)
        graph, similarity_matrix = graph_builder.build_graph(
            embeddings, 
            list(range(len(test_texts)))
        )
        print(f"✓ Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Test feature extraction  
        feature_extractor = TAGFeatureExtractor(config)
        features_df = feature_extractor.extract_features(
            graph,
            list(range(len(test_texts))),
            similarity_matrix
        )
        print(f"✓ Features extracted: {features_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_access():
    """Test that the dataset can be accessed."""
    print("\nTesting dataset access...")
    
    try:
        import pandas as pd
        
        # Check if dataset file exists
        dataset_path = Path(__file__).parent.parent / 'datasets' / 'pheme' / 'pheme_all.csv'
        
        if not dataset_path.exists():
            print(f"✗ Dataset not found at: {dataset_path}")
            return False
        
        # Try to load dataset
        df = pd.read_csv(dataset_path)
        print(f"✓ Dataset loaded: {len(df)} samples")
        
        # Check required columns
        if 'text' not in df.columns:
            print("✗ Missing 'text' column in dataset")
            return False
        
        if 'label' not in df.columns:
            print("✗ Missing 'label' column in dataset")
            return False
        
        print(f"✓ Required columns present")
        print(f"✓ Label distribution: {df['label'].value_counts().to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset access failed: {e}")
        return False


def check_optional_dependencies():
    """Check optional dependencies."""
    print("\nChecking optional dependencies...")
    
    # Node2Vec
    try:
        import node2vec
        print("✓ node2vec (graph embeddings available)")
    except ImportError:
        print("! node2vec not available (graph embeddings will be zero)")
    
    # OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✓ OPENAI_API_KEY found (GPT embeddings available)")
    else:
        print("! OPENAI_API_KEY not found (will use SBERT embeddings)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("FILO-TRANSFORMER MINIMAL TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Import tests failed. Please install missing dependencies.")
        return False
    
    # Test Filo-Transformer modules
    if not test_filo_transformer_modules():
        all_passed = False
        print("\n❌ Module tests failed. Check Python path and module structure.")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
        print("\n❌ Functionality tests failed.")
        return False
    
    # Test dataset access
    if not test_dataset_access():
        all_passed = False
        print("\n❌ Dataset access failed.")
        return False
    
    # Check optional dependencies
    check_optional_dependencies()
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("The Filo-Transformer installation is working correctly.")
        print("You can now run the full experiments.")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues before running experiments.")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)