# Enhanced GPTCache - Federated, Context-Aware Semantic Caching

An enhanced version of GPTCache (v0.1.24) that adds support for context-aware filtering, PCA-compressed embeddings, and federated τ-threshold tuning while preserving compatibility with the original API.

## 🚀 Features

### ✨ Core Enhancements

- **Context-Aware Filtering**: Requires both query and context embeddings to match for cache hits
- **PCA Embedding Compression**: Reduces embedding dimensionality from 768→128 dimensions to save memory
- **Federated τ-Tuning**: Simulates federated learning to personalize similarity thresholds per user
- **API Compatibility**: Maintains full compatibility with original GPTCache API

### 🧠 Advanced Capabilities

- **Dual-Threshold Matching**: Separate thresholds for query (`tau`) and context (`tau_context`) similarity
- **Memory Optimization**: Configurable PCA compression with preserved semantic meaning
- **Adaptive Thresholds**: Federated learning updates similarity thresholds based on user behavior
- **Comprehensive Benchmarking**: Three workload types (repetitive, contextual, novel) with detailed metrics

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (for embedding models)
- 2GB+ disk space

## 🛠️ Installation

### Option 1: Direct Installation

```bash
# Clone the repository
git clone <repository-url>
cd enhanced-gptcache

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t enhanced-gptcache .

# Run container
docker run -it enhanced-gptcache
```

## ⚙️ Configuration

The system uses a central `config.yaml` file for all configuration:

```yaml
# Similarity thresholds
tau: 0.8                    # Main similarity threshold
tau_context: 0.7           # Context embedding threshold

# PCA compression settings
pca_dim: 128               # Reduced embedding size
pca_enabled: true          # Enable/disable PCA
pca_training_samples: 5000 # Training samples for PCA

# Federated learning settings
fl_round: 100              # Queries before federated update
fl_clients: 5              # Number of simulated clients
fl_grid_search_range: [0.6, 0.7, 0.8, 0.9]

# Cache settings
cache_size: 10000          # Maximum cache entries
cache_ttl: 3600           # Time to live (seconds)
```

## 🚀 Quick Start

### Basic Usage

```python
from enhanced_cache import EnhancedCache

# Create cache instance
cache = EnhancedCache()

# Store query-response pairs with context
cache.put(
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    context="AI and data science context"
)

# Retrieve from cache
result = cache.get(
    query="Tell me about machine learning",
    context="AI and data science context"
)

print(result)  # Returns cached response if similarity threshold met
```

### Training PCA Model

```python
# Prepare training texts
training_texts = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain deep learning",
    # ... more training examples
]

# Train PCA model
cache.train_pca(training_texts)
```

### Checking Cache Statistics

```python
stats = cache.get_stats()
print(f"Cache size: {stats['size']}")
print(f"Current tau: {stats['current_tau']}")
print(f"PCA enabled: {stats['pca_enabled']}")
```

## 🧪 Benchmarking

### Run Comprehensive Benchmarks

```bash
# Run all workloads with all variants
python benchmark_runner.py --queries 1000 --variant all --workload all

# Run specific workload
python benchmark_runner.py --workload repetitive --queries 500

# Export results
python benchmark_runner.py --export both --plots
```

### Benchmark Workloads

1. **Repetitive**: High cache hit potential with repeated queries
2. **Contextual**: Varying contexts with similar queries
3. **Novel**: Mostly unique queries to test cache miss behavior

### Metrics Collected

- **Hit Rate**: Percentage of successful cache retrievals
- **F1 Score**: Harmonic mean of precision and recall
- **Latency**: Mean, P95, and P99 response times
- **Memory Usage**: RAM consumption in MB
- **Cache Size**: Number of stored entries

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_enhanced_cache.py

# Run with coverage
pytest --cov=enhanced_cache tests/
```

### Test Coverage

- Configuration management
- PCA embedding compression
- Context-aware similarity evaluation
- Federated tau tuning
- Cache operations (put, get, clear)
- API compatibility

## 📊 Performance Comparison

### Memory Usage
- **Original GPTCache**: ~768 dimensions per embedding
- **Enhanced GPTCache**: ~128 dimensions per embedding (83% reduction)

### Context-Aware Filtering
- **Improved Precision**: Reduces false positives by requiring context match
- **Configurable Thresholds**: Separate control for query and context similarity

### Federated Learning
- **Adaptive Thresholds**: Automatically tunes similarity thresholds
- **Personalization**: Simulates per-user optimization

## 🔧 Advanced Usage

### Custom Configuration

```python
# Load custom config
cache = EnhancedCache("custom_config.yaml")

# Update configuration programmatically
cache.config.update('tau', 0.9)
cache.config.update('pca_enabled', False)
```

### Federated Learning Monitoring

```python
# Check federated learning status
stats = cache.get_stats()
print(f"Total queries: {stats['query_count']}")
print(f"Current tau: {stats['current_tau']}")

# Federated updates occur every fl_round queries
# (default: 100 queries)
```

### PCA Model Management

```python
# Check PCA status
stats = cache.get_stats()
print(f"PCA trained: {stats['pca_trained']}")

# Retrain PCA with new data
new_training_texts = ["new query 1", "new query 2", ...]
cache.train_pca(new_training_texts)
```

## 🐳 Docker Usage

### Build and Run

```bash
# Build image
docker build -t enhanced-gptcache .

# Run with custom config
docker run -v $(pwd)/config.yaml:/app/config.yaml enhanced-gptcache

# Run benchmarks
docker run enhanced-gptcache python benchmark_runner.py --queries 500
```

### Development Container

```bash
# Run with volume mounting for development
docker run -it -v $(pwd):/app enhanced-gptcache bash
```

## 📈 Results and Analysis

### Expected Performance

- **Hit Rate**: 60-80% for repetitive workloads, 30-50% for contextual
- **Latency**: <10ms for cache hits, <100ms for cache misses
- **Memory**: 60-80% reduction compared to original GPTCache
- **F1 Score**: 0.7-0.9 depending on workload and configuration

### Benchmark Results

Results are exported to:
- `benchmark_results_YYYYMMDD_HHMMSS.csv`
- `benchmark_results_YYYYMMDD_HHMMSS.json`
- `benchmark_results_YYYYMMDD_HHMMSS.png` (visualization)

## 🔍 Architecture

### Core Components

1. **ConfigManager**: Centralized configuration management
2. **PCAEmbeddingWrapper**: PCA compression for embeddings
3. **ContextAwareSimilarityEvaluation**: Dual-threshold similarity matching
4. **FederatedTauTuner**: Adaptive threshold optimization
5. **EnhancedCache**: Main cache interface with API compatibility

### Data Flow

```
Query + Context → Embedding → PCA Compression → Similarity Check → Cache Hit/Miss
                                                      ↓
                                              Federated Learning
                                                      ↓
                                              Threshold Update
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project extends GPTCache v0.1.24 and maintains compatibility with its license.

## 🙏 Acknowledgments

- Original GPTCache team for the base implementation
- Sentence Transformers for embedding models
- FAISS for efficient similarity search
- Scikit-learn for PCA implementation

## 📞 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Enhanced GPTCache** - Making semantic caching smarter, faster, and more context-aware! 🚀 