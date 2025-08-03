"""
Unit tests for Enhanced GPTCache components.
"""

import pytest
import numpy as np
import tempfile
import os
import yaml
from unittest.mock import Mock, patch

from enhanced_cache import (
    EnhancedCache, ConfigManager, PCAEmbeddingWrapper, 
    ContextAwareSimilarityEvaluation, FederatedTauTuner, CacheEntry
)


class TestConfigManager:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("")
        
        try:
            config = ConfigManager(f.name)
            assert config.get('tau') == 0.8
            assert config.get('pca_enabled') is True
            assert config.get('pca_dim') == 128
        finally:
            os.unlink(f.name)
    
    def test_custom_config(self):
        """Test custom configuration loading."""
        config_data = {
            'tau': 0.9,
            'pca_enabled': False,
            'pca_dim': 64
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump(config_data, f)
        
        try:
            config = ConfigManager(f.name)
            assert config.get('tau') == 0.9
            assert config.get('pca_enabled') is False
            assert config.get('pca_dim') == 64
        finally:
            os.unlink(f.name)
    
    def test_config_update(self):
        """Test configuration update."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump({'tau': 0.8}, f)
        
        try:
            config = ConfigManager(f.name)
            config.update('tau', 0.9)
            assert config.get('tau') == 0.9
            
            # Verify file was updated
            with open(f.name, 'r') as f_read:
                updated_config = yaml.safe_load(f_read)
            assert updated_config['tau'] == 0.9
        finally:
            os.unlink(f.name)


class TestPCAEmbeddingWrapper:
    """Test PCA embedding wrapper."""
    
    def setup_method(self):
        """Setup test method."""
        self.config = Mock()
        self.config.get.side_effect = lambda key, default=None: {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': 768,
            'pca_dim': 128,
            'pca_enabled': True
        }.get(key, default)
        
        self.wrapper = PCAEmbeddingWrapper(self.config)
    
    @patch('enhanced_cache.SentenceTransformer')
    def test_encode_without_pca(self, mock_transformer):
        """Test encoding without PCA."""
        self.config.get.side_effect = lambda key, default=None: {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': 768,
            'pca_dim': 128,
            'pca_enabled': False
        }.get(key, default)
        
        wrapper = PCAEmbeddingWrapper(self.config)
        mock_emb = np.random.rand(768)
        wrapper.embedding_model.encode.return_value = mock_emb
        
        result = wrapper.encode("test query")
        
        assert np.array_equal(result, mock_emb)
        wrapper.embedding_model.encode.assert_called_once_with("test query")
    
    @patch('enhanced_cache.SentenceTransformer')
    def test_encode_with_pca(self, mock_transformer):
        """Test encoding with PCA."""
        mock_emb = np.random.rand(768)
        compressed_emb = np.random.rand(128)
        
        self.wrapper.embedding_model.encode.return_value = mock_emb
        self.wrapper.pca = Mock()
        self.wrapper.pca.transform.return_value = compressed_emb.reshape(1, -1)
        
        result = self.wrapper.encode("test query")
        
        assert np.array_equal(result, compressed_emb)
        self.wrapper.pca.transform.assert_called_once()
    
    @patch('enhanced_cache.SentenceTransformer')
    def test_train_pca(self, mock_transformer):
        """Test PCA training."""
        training_texts = ["text1", "text2", "text3"]
        mock_embs = [np.random.rand(768) for _ in training_texts]
        
        self.wrapper.embedding_model.encode.side_effect = mock_embs
        
        with patch('builtins.open', create=True) as mock_open:
            with patch('pickle.dump') as mock_dump:
                self.wrapper.train_pca(training_texts)
        
        assert self.wrapper.is_trained is True
        assert self.wrapper.pca is not None
        mock_dump.assert_called_once()


class TestContextAwareSimilarityEvaluation:
    """Test context-aware similarity evaluation."""
    
    def setup_method(self):
        """Setup test method."""
        self.config = Mock()
        self.config.get.side_effect = lambda key, default=None: {
            'tau': 0.8,
            'tau_context': 0.7
        }.get(key, default)
        
        self.embedding_wrapper = Mock()
        self.evaluator = ContextAwareSimilarityEvaluation(self.config, self.embedding_wrapper)
    
    def test_evaluation_with_context(self):
        """Test similarity evaluation with context."""
        query_emb = np.array([1.0, 0.0, 0.0])
        context_emb = np.array([0.0, 1.0, 0.0])
        cache_query_emb = np.array([0.9, 0.1, 0.0])
        cache_context_emb = np.array([0.1, 0.9, 0.0])
        
        src_dict = {'query_emb': query_emb, 'context_emb': context_emb}
        cache_dict = {'query_emb': cache_query_emb, 'context_emb': cache_context_emb}
        
        with patch('enhanced_cache.cosine_similarity') as mock_cosine:
            mock_cosine.side_effect = [[[0.9]], [[0.8]]]  # query_sim, context_sim
            
            result = self.evaluator.evaluation(src_dict, cache_dict)
            
            assert result == 0.8  # min(0.9, 0.8)
    
    def test_evaluation_without_context(self):
        """Test similarity evaluation without context."""
        query_emb = np.array([1.0, 0.0, 0.0])
        cache_query_emb = np.array([0.9, 0.1, 0.0])
        
        src_dict = {'query_emb': query_emb, 'context_emb': None}
        cache_dict = {'query_emb': cache_query_emb, 'context_emb': None}
        
        with patch('enhanced_cache.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = [[0.9]]
            
            result = self.evaluator.evaluation(src_dict, cache_dict)
            
            assert result == 0.9
    
    def test_evaluation_missing_embeddings(self):
        """Test evaluation with missing embeddings."""
        src_dict = {'query_emb': None, 'context_emb': None}
        cache_dict = {'query_emb': None, 'context_emb': None}
        
        result = self.evaluator.evaluation(src_dict, cache_dict)
        
        assert result == 0.0
    
    def test_range(self):
        """Test similarity range."""
        min_val, max_val = self.evaluator.range()
        assert min_val == 0.0
        assert max_val == 1.0


class TestFederatedTauTuner:
    """Test federated tau tuning."""
    
    def setup_method(self):
        """Setup test method."""
        self.config = Mock()
        self.config.get.side_effect = lambda key, default=None: {
            'tau': 0.8,
            'fl_round': 10,
            'fl_clients': 3,
            'fl_grid_search_range': [0.6, 0.7, 0.8, 0.9]
        }.get(key, default)
        
        self.tuner = FederatedTauTuner(self.config)
    
    def test_record_query(self):
        """Test query recording."""
        self.tuner.record_query("test query", "test context", True, "test response")
        
        assert self.tuner.query_count == 1
        assert len(self.tuner.client_metrics[0]) == 1
        
        entry = self.tuner.client_metrics[0][0]
        assert entry['query'] == "test query"
        assert entry['context'] == "test context"
        assert entry['hit'] is True
        assert entry['response'] == "test response"
    
    def test_client_assignment(self):
        """Test client assignment (round-robin)."""
        for i in range(6):
            self.tuner.record_query(f"query{i}", f"context{i}", True, f"response{i}")
        
        # Should be assigned to clients 0, 1, 2, 0, 1, 2
        assert len(self.tuner.client_metrics[0]) == 2
        assert len(self.tuner.client_metrics[1]) == 2
        assert len(self.tuner.client_metrics[2]) == 2
    
    def test_federated_update_trigger(self):
        """Test federated update triggering."""
        # Record queries up to fl_round
        for i in range(10):
            self.tuner.record_query(f"query{i}", f"context{i}", True, f"response{i}")
        
        # Should trigger federated update
        assert self.tuner.query_count == 10
    
    def test_get_current_tau(self):
        """Test getting current tau value."""
        assert self.tuner.get_current_tau() == 0.8
        
        # Update tau
        self.tuner.global_tau = 0.9
        assert self.tuner.get_current_tau() == 0.9


class TestEnhancedCache:
    """Test enhanced cache functionality."""
    
    def setup_method(self):
        """Setup test method."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            yaml.dump({
                'tau': 0.8,
                'tau_context': 0.7,
                'pca_enabled': False,  # Disable PCA for faster tests
                'cache_size': 100,
                'cache_ttl': 3600
            }, f)
            self.config_path = f.name
    
    def teardown_method(self):
        """Teardown test method."""
        if os.path.exists(self.config_path):
            os.unlink(self.config_path)
    
    @patch('enhanced_cache.SentenceTransformer')
    def test_cache_put_and_get(self, mock_transformer):
        """Test basic cache put and get operations."""
        cache = EnhancedCache(self.config_path)
        
        # Mock embeddings
        mock_emb = np.random.rand(768)
        cache.embedding_wrapper.embedding_model.encode.return_value = mock_emb
        
        # Put entry
        cache.put("test query", "test response", "test context")
        
        # Get entry
        result = cache.get("test query", "test context")
        
        assert result == "test response"
        assert len(cache.cache_entries) == 1
    
    @patch('enhanced_cache.SentenceTransformer')
    def test_cache_miss(self, mock_transformer):
        """Test cache miss behavior."""
        cache = EnhancedCache(self.config_path)
        
        # Mock embeddings
        mock_emb = np.random.rand(768)
        cache.embedding_wrapper.embedding_model.encode.return_value = mock_emb
        
        # Try to get non-existent entry
        result = cache.get("test query", "test context")
        
        assert result is None
    
    @patch('enhanced_cache.SentenceTransformer')
    def test_cache_clear(self, mock_transformer):
        """Test cache clearing."""
        cache = EnhancedCache(self.config_path)
        
        # Mock embeddings
        mock_emb = np.random.rand(768)
        cache.embedding_wrapper.embedding_model.encode.return_value = mock_emb
        
        # Put some entries
        cache.put("query1", "response1", "context1")
        cache.put("query2", "response2", "context2")
        
        assert len(cache.cache_entries) == 2
        
        # Clear cache
        cache.clear()
        
        assert len(cache.cache_entries) == 0
        assert len(cache.cache_order) == 0
    
    @patch('enhanced_cache.SentenceTransformer')
    def test_cache_stats(self, mock_transformer):
        """Test cache statistics."""
        cache = EnhancedCache(self.config_path)
        
        # Mock embeddings
        mock_emb = np.random.rand(768)
        cache.embedding_wrapper.embedding_model.encode.return_value = mock_emb
        
        # Put an entry
        cache.put("test query", "test response", "test context")
        
        stats = cache.get_stats()
        
        assert stats['size'] == 1
        assert stats['max_size'] == 100
        assert 'current_tau' in stats
        assert 'query_count' in stats
        assert 'pca_enabled' in stats
        assert 'pca_trained' in stats
    
    @patch('enhanced_cache.SentenceTransformer')
    def test_check_hit(self, mock_transformer):
        """Test check_hit method."""
        cache = EnhancedCache(self.config_path)
        
        # Mock embeddings
        mock_emb = np.random.rand(768)
        cache.embedding_wrapper.embedding_model.encode.return_value = mock_emb
        
        # Check non-existent entry
        assert cache.check_hit("test query", "test context") is False
        
        # Put entry
        cache.put("test query", "test response", "test context")
        
        # Check existing entry
        assert cache.check_hit("test query", "test context") is True


class TestCacheEntry:
    """Test cache entry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        query_emb = np.array([1.0, 2.0, 3.0])
        context_emb = np.array([4.0, 5.0, 6.0])
        metadata = {'source': 'test'}
        
        entry = CacheEntry(
            query_emb=query_emb,
            context_emb=context_emb,
            response="test response",
            timestamp=1234567890.0,
            metadata=metadata
        )
        
        assert np.array_equal(entry.query_emb, query_emb)
        assert np.array_equal(entry.context_emb, context_emb)
        assert entry.response == "test response"
        assert entry.timestamp == 1234567890.0
        assert entry.metadata == metadata


if __name__ == "__main__":
    pytest.main([__file__]) 