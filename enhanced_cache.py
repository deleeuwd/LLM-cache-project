"""
Enhanced GPTCache - Federated, Context-Aware Semantic Caching
Extends GPTCache v0.1.24 with context-aware filtering, PCA compression, and federated tau tuning.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pickle
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import yaml
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import faiss
from sentence_transformers import SentenceTransformer
import gptcache
from gptcache import Cache
from gptcache.similarity_evaluation import SimilarityEvaluation


@dataclass
class CacheEntry:
    """Enhanced cache entry with context embeddings."""
    query_emb: np.ndarray
    context_emb: np.ndarray
    response: str
    timestamp: float
    metadata: Dict[str, Any]


class ConfigManager:
    """Manages configuration for the enhanced cache."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'tau': 0.8,
                'tau_context': 0.7,
                'pca_dim': 128,
                'pca_enabled': True,
                'pca_training_samples': 5000,
                'fl_round': 100,
                'fl_clients': 5,
                'fl_grid_search_range': [0.6, 0.7, 0.8, 0.9],
                'cache_size': 10000,
                'cache_ttl': 3600,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'embedding_dim': 768,
                'log_level': 'INFO',
                'log_file': 'enhanced_cache.log'
            }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['log_file']),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def update(self, key: str, value: Any):
        """Update configuration value."""
        # Convert numpy values to native Python types to avoid YAML serialization issues
        if hasattr(value, 'item'):  # Check if it's a numpy scalar
            value = value.item()
        
        self.config[key] = value
        # Save to file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


class PCAEmbeddingWrapper:
    """Wrapper for embedding function with PCA compression."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.pca = None
        self.embedding_model = SentenceTransformer(config.get('embedding_model'))
        self.original_dim = config.get('embedding_dim')
        self.pca_dim = config.get('pca_dim')
        self.is_trained = False
        
    def train_pca(self, training_texts: List[str]):
        """Train PCA model on a set of texts."""
        if not self.config.get('pca_enabled'):
            return
            
        logging.info(f"Training PCA on {len(training_texts)} samples...")
        
        # Get embeddings for training texts
        embeddings = []
        for text in training_texts:
            emb = self.embedding_model.encode(text)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Train PCA
        self.pca = PCA(n_components=self.pca_dim)
        self.pca.fit(embeddings)
        self.is_trained = True
        
        # Save PCA model
        with open('pca.pkl', 'wb') as f:
            pickle.dump(self.pca, f)
        
        logging.info(f"PCA trained and saved. Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text with optional PCA compression."""
        emb = self.embedding_model.encode(text)
        
        if self.config.get('pca_enabled') and self.pca is not None:
            emb = self.pca.transform(emb.reshape(1, -1)).flatten()
        
        return emb
    
    def load_pca(self):
        """Load pre-trained PCA model."""
        if os.path.exists('pca.pkl'):
            with open('pca.pkl', 'rb') as f:
                self.pca = pickle.load(f)
            self.is_trained = True
            logging.info("PCA model loaded from file")


class ContextAwareSimilarityEvaluation(SimilarityEvaluation):
    """Context-aware similarity evaluation for GPTCache."""
    
    def __init__(self, config: ConfigManager, embedding_wrapper: PCAEmbeddingWrapper):
        self.config = config
        self.embedding_wrapper = embedding_wrapper
        self.tau = config.get('tau')
        self.tau_context = config.get('tau_context')
    
    def evaluation(self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any]) -> float:
        """Evaluate similarity between source and cache entry."""
        # Get embeddings
        query_emb = src_dict.get('query_emb')
        context_emb = src_dict.get('context_emb')
        cache_query_emb = cache_dict.get('query_emb')
        cache_context_emb = cache_dict.get('context_emb')
        
        if query_emb is None or cache_query_emb is None:
            return 0.0
        
        # Check for dimension mismatch and handle it
        if query_emb.shape[0] != cache_query_emb.shape[0]:
            # If dimensions don't match, we can't compare them directly
            # This happens when PCA is trained after some entries are already cached
            # For now, return a low similarity score
            return 0.1
        
        # Calculate query similarity
        query_sim = cosine_similarity(
            query_emb.reshape(1, -1), 
            cache_query_emb.reshape(1, -1)
        )[0][0]
        
        # Calculate context similarity if available
        context_sim = 1.0  # Default to perfect match if no context
        if context_emb is not None and cache_context_emb is not None:
            if context_emb.shape[0] != cache_context_emb.shape[0]:
                # Dimension mismatch for context embeddings
                context_sim = 0.1
            else:
                context_sim = cosine_similarity(
                    context_emb.reshape(1, -1), 
                    cache_context_emb.reshape(1, -1)
                )[0][0]
        
        # Return minimum of both similarities (both must exceed thresholds)
        return min(query_sim, context_sim)
    
    def range(self) -> Tuple[float, float]:
        """Return similarity range."""
        return 0.0, 1.0


class FederatedTauTuner:
    """Simulates federated learning for tau threshold tuning."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.query_count = 0
        self.client_metrics = defaultdict(list)
        self.global_tau = config.get('tau')
        self.fl_round = config.get('fl_round')
        self.fl_clients = config.get('fl_clients')
        self.grid_search_range = config.get('fl_grid_search_range')
    
    def record_query(self, query: str, context: str, hit: bool, response: str):
        """Record a query for federated learning."""
        self.query_count += 1
        
        # Simulate client assignment (round-robin)
        client_id = self.query_count % self.fl_clients
        
        # Store metrics for this client
        self.client_metrics[client_id].append({
            'query': query,
            'context': context,
            'hit': hit,
            'response': response,
            'timestamp': time.time()
        })
        
        # Check if it's time for federated update
        if self.query_count % self.fl_round == 0:
            self._simulate_federated_tau_update()
    
    def _simulate_federated_tau_update(self):
        """Simulate federated tau update using FedAvg."""
        logging.info(f"Starting federated tau update after {self.query_count} queries")
        
        client_taus = []
        
        # Each client performs grid search
        for client_id in range(self.fl_clients):
            if client_id in self.client_metrics:
                best_tau = self._grid_search_tau(client_id)
                client_taus.append(best_tau)
        
        if client_taus:
            # FedAvg: average the tau deltas
            tau_deltas = [tau - self.global_tau for tau in client_taus]
            avg_delta = np.mean(tau_deltas)
            
            # Update global tau
            old_tau = self.global_tau
            self.global_tau = max(0.1, min(0.95, self.global_tau + avg_delta))
            
            logging.info(f"Federated tau update: {old_tau:.3f} -> {self.global_tau:.3f}")
            
            # Update config
            self.config.update('tau', self.global_tau)
    
    def _grid_search_tau(self, client_id: int) -> float:
        """Perform grid search for optimal tau for a client."""
        client_data = self.client_metrics[client_id]
        
        best_f1 = 0.0
        best_tau = self.global_tau
        
        for tau in self.grid_search_range:
            # Simulate cache behavior with this tau
            hits = 0
            correct_hits = 0
            total_queries = len(client_data)
            
            for entry in client_data:
                # Simplified simulation - in practice, you'd need actual similarity computation
                if np.random.random() < tau:  # Simulate hit
                    hits += 1
                    if entry['hit']:  # Assume ground truth is available
                        correct_hits += 1
            
            if hits > 0:
                precision = correct_hits / hits
                recall = correct_hits / total_queries
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = tau
        
        return best_tau
    
    def get_current_tau(self) -> float:
        """Get current tau value."""
        return self.global_tau


class EnhancedCache:
    """Enhanced GPTCache with context-aware filtering, PCA compression, and federated tuning."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.embedding_wrapper = PCAEmbeddingWrapper(self.config)
        self.federated_tuner = FederatedTauTuner(self.config)
        
        # Initialize cache storage
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.cache_order = deque()  # For LRU eviction
        self.max_size = self.config.get('cache_size')
        self.ttl = self.config.get('cache_ttl')
        
        # Load PCA if available
        self.embedding_wrapper.load_pca()
        
        # Initialize similarity evaluator
        self.similarity_evaluator = ContextAwareSimilarityEvaluation(
            self.config, self.embedding_wrapper
        )
        
        logging.info("Enhanced GPTCache initialized")
    
    def get(self, query: str, context: str = "") -> Optional[str]:
        """Get response from cache if available."""
        # Generate embeddings
        query_emb = self.embedding_wrapper.encode(query)
        context_emb = self.embedding_wrapper.encode(context) if context else None
        
        # Check for cache hit
        best_match = None
        best_score = 0.0
        
        for cache_id, entry in self.cache_entries.items():
            # Check TTL
            if time.time() - entry.timestamp > self.ttl:
                continue
            
            # Evaluate similarity
            src_dict = {'query_emb': query_emb, 'context_emb': context_emb}
            cache_dict = {'query_emb': entry.query_emb, 'context_emb': entry.context_emb}
            
            score = self.similarity_evaluator.evaluation(src_dict, cache_dict)
            
            if score > best_score:
                best_score = score
                best_match = entry
        
        # Check if score exceeds thresholds
        tau = self.federated_tuner.get_current_tau()
        tau_context = self.config.get('tau_context')
        
        if best_match and best_score >= tau:
            # Record hit for federated learning
            self.federated_tuner.record_query(query, context, True, best_match.response)
            logging.info(f"Cache hit with score {best_score:.3f}")
            return best_match.response
        
        # Record miss for federated learning
        self.federated_tuner.record_query(query, context, False, "")
        logging.info(f"Cache miss with best score {best_score:.3f}")
        return None
    
    def put(self, query: str, response: str, context: str = "", metadata: Dict[str, Any] = None):
        """Store query-response pair in cache."""
        # Generate embeddings
        query_emb = self.embedding_wrapper.encode(query)
        context_emb = self.embedding_wrapper.encode(context) if context else None
        
        # Create cache entry
        entry = CacheEntry(
            query_emb=query_emb,
            context_emb=context_emb,
            response=response,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Generate cache ID (simple hash)
        cache_id = str(hash(query + context))
        
        # Add to cache
        self.cache_entries[cache_id] = entry
        self.cache_order.append(cache_id)
        
        # LRU eviction if needed
        if len(self.cache_entries) > self.max_size:
            oldest_id = self.cache_order.popleft()
            del self.cache_entries[oldest_id]
        
        logging.info(f"Stored in cache: {query[:50]}...")
    
    def check_hit(self, query: str, context: str = "") -> bool:
        """Check if query exists in cache."""
        return self.get(query, context) is not None
    
    def clear(self):
        """Clear all cache entries."""
        self.cache_entries.clear()
        self.cache_order.clear()
        logging.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache_entries),
            'max_size': self.max_size,
            'current_tau': self.federated_tuner.get_current_tau(),
            'query_count': self.federated_tuner.query_count,
            'pca_enabled': self.config.get('pca_enabled'),
            'pca_trained': self.embedding_wrapper.is_trained
        }
    
    def train_pca(self, training_texts: List[str]):
        """Train PCA model on provided texts."""
        # Clear cache before training PCA to avoid dimension mismatches
        logging.info("Clearing cache before PCA training to avoid dimension mismatches")
        self.clear()
        
        # Train PCA
        self.embedding_wrapper.train_pca(training_texts)


# Backward compatibility - expose original GPTCache API
def create_cache(config_path: str = "config.yaml") -> EnhancedCache:
    """Create an enhanced cache instance."""
    return EnhancedCache(config_path)


# Example usage
if __name__ == "__main__":
    # Create enhanced cache
    cache = EnhancedCache()
    
    # Example queries
    queries = [
        ("What is machine learning?", "AI and data science context"),
        ("How does neural networks work?", "Deep learning context"),
        ("What is Python?", "Programming language context")
    ]
    
    responses = [
        "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        "Neural networks are computational models inspired by biological neural networks.",
        "Python is a high-level, interpreted programming language known for its simplicity."
    ]
    
    # Store in cache
    for (query, context), response in zip(queries, responses):
        cache.put(query, response, context)
    
    # Test retrieval
    test_query = "Tell me about machine learning"
    test_context = "AI and data science context"
    
    result = cache.get(test_query, test_context)
    print(f"Query: {test_query}")
    print(f"Result: {result}")
    print(f"Cache stats: {cache.get_stats()}") 