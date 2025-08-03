"""
Example Usage of Enhanced GPTCache
Demonstrates context-aware filtering, PCA compression, and federated learning.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import numpy as np
from enhanced_cache import EnhancedCache


def main():
    """Main example demonstrating enhanced cache features."""
    print("🚀 Enhanced GPTCache Example")
    print("=" * 50)
    
    # Create enhanced cache instance
    cache = EnhancedCache()
    
    # Example 1: Basic Usage with Context
    print("\n1️⃣ Basic Usage with Context-Aware Filtering")
    print("-" * 40)
    
    # Store some query-response pairs with context
    queries_with_context = [
        ("What is machine learning?", "AI and data science context", 
         "Machine learning is a subset of AI that enables computers to learn without explicit programming."),
        ("How do neural networks work?", "Deep learning context",
         "Neural networks are computational models inspired by biological neural networks."),
        ("What is Python?", "Programming language context",
         "Python is a high-level, interpreted programming language known for its simplicity.")
    ]
    
    for query, context, response in queries_with_context:
        cache.put(query, response, context)
        print(f"✅ Stored: '{query[:30]}...' with context '{context[:20]}...'")
    
    # Test retrieval with similar queries
    test_cases = [
        ("Tell me about machine learning", "AI and data science context"),
        ("Explain neural networks", "Deep learning context"),
        ("What is Python programming?", "Programming language context"),
        ("Tell me about machine learning", "Different context"),  # Should miss due to context mismatch
    ]
    
    for query, context in test_cases:
        start_time = time.time()
        result = cache.get(query, context)
        latency = (time.time() - start_time) * 1000
        
        if result:
            print(f"🎯 HIT: '{query[:30]}...' → '{result[:50]}...' ({latency:.2f}ms)")
        else:
            print(f"❌ MISS: '{query[:30]}...' ({latency:.2f}ms)")
    
    # Example 2: PCA Training
    print("\n2️⃣ PCA Compression Training")
    print("-" * 40)
    
    # Generate training data
    training_texts = []
    for i in range(100):
        training_texts.append(f"Training query number {i} about various topics")
        training_texts.append(f"Another training example {i} for machine learning")
        training_texts.append(f"Sample text {i} for natural language processing")
    
    print(f"📚 Training PCA on {len(training_texts)} samples...")
    cache.train_pca(training_texts)
    
    # Example 3: Federated Learning Simulation
    print("\n3️⃣ Federated Learning Simulation")
    print("-" * 40)
    
    # Simulate multiple queries to trigger federated updates
    print("🔄 Simulating queries to trigger federated learning...")
    
    for i in range(150):  # More than fl_round (100) to trigger federated update
        query = f"Simulated query {i} for federated learning"
        context = f"Context {i % 3}"  # Vary context
        cache.get(query, context)  # This will record for federated learning
    
    # Example 4: Cache Statistics
    print("\n4️⃣ Cache Statistics and Performance")
    print("-" * 40)
    
    stats = cache.get_stats()
    print(f"📊 Cache Size: {stats['size']} entries")
    print(f"🎯 Current Tau: {stats['current_tau']:.3f}")
    print(f"📈 Total Queries: {stats['query_count']}")
    print(f"🧠 PCA Enabled: {stats['pca_enabled']}")
    print(f"🎓 PCA Trained: {stats['pca_trained']}")
    
    # Example 5: Performance Comparison
    print("\n5️⃣ Performance Comparison")
    print("-" * 40)
    
    # Test with and without context
    test_query = "What is machine learning?"
    
    # Test with matching context
    start_time = time.time()
    result_with_context = cache.get(test_query, "AI and data science context")
    time_with_context = (time.time() - start_time) * 1000
    
    # Test with different context
    start_time = time.time()
    result_without_context = cache.get(test_query, "Different context")
    time_without_context = (time.time() - start_time) * 1000
    
    print(f"⏱️  With matching context: {time_with_context:.2f}ms")
    print(f"⏱️  With different context: {time_without_context:.2f}ms")
    print(f"✅ Context match result: {'HIT' if result_with_context else 'MISS'}")
    print(f"❌ Context mismatch result: {'HIT' if result_without_context else 'MISS'}")
    
    # Example 6: Memory Optimization
    print("\n6️⃣ Memory Optimization with PCA")
    print("-" * 40)
    
    # Show PCA benefits
    if stats['pca_enabled'] and stats['pca_trained']:
        original_dim = 768
        compressed_dim = cache.config.get('pca_dim', 128)
        compression_ratio = (1 - compressed_dim / original_dim) * 100
        
        print(f"📉 Original embedding dimension: {original_dim}")
        print(f"📉 Compressed dimension: {compressed_dim}")
        print(f"💾 Memory reduction: {compression_ratio:.1f}%")
        print(f"🎯 Preserved variance: ~95% (typical for PCA)")
    
    # Example 7: Advanced Features
    print("\n7️⃣ Advanced Features")
    print("-" * 40)
    
    # Test metadata storage
    cache.put(
        "Advanced query with metadata",
        "Response with additional information",
        "Advanced context",
        metadata={"source": "example", "confidence": 0.95, "timestamp": time.time()}
    )
    
    # Test cache clearing
    print(f"🗑️  Clearing cache...")
    cache.clear()
    
    # Verify cache is empty
    stats_after_clear = cache.get_stats()
    print(f"📊 Cache size after clear: {stats_after_clear['size']}")
    
    print("\n🎉 Example completed successfully!")
    print("=" * 50)


def demonstrate_workloads():
    """Demonstrate different workload types."""
    print("\n🔄 Workload Demonstration")
    print("=" * 50)
    
    cache = EnhancedCache()
    
    # Repetitive workload
    print("\n📝 Repetitive Workload (High hit rate expected)")
    repetitive_queries = [
        "What is machine learning?",
        "What is machine learning?",
        "What is machine learning?",
        "Tell me about machine learning",
        "Explain machine learning"
    ]
    
    # Store base query
    cache.put("What is machine learning?", "ML is a subset of AI...", "AI context")
    
    hits = 0
    for query in repetitive_queries:
        if cache.get(query, "AI context"):
            hits += 1
    
    print(f"Repetitive workload hit rate: {hits/len(repetitive_queries)*100:.1f}%")
    
    # Contextual workload
    print("\n🌍 Contextual Workload (Medium hit rate expected)")
    contextual_queries = [
        ("What is Python?", "Programming context"),
        ("What is Python?", "Snake context"),
        ("What is Python?", "Programming language context"),
        ("What is Python?", "Biology context")
    ]
    
    cache.put("What is Python?", "Python is a programming language", "Programming context")
    
    hits = 0
    for query, context in contextual_queries:
        if cache.get(query, context):
            hits += 1
    
    print(f"Contextual workload hit rate: {hits/len(contextual_queries)*100:.1f}%")
    
    # Novel workload
    print("\n🆕 Novel Workload (Low hit rate expected)")
    novel_queries = [
        "What is quantum computing?",
        "How do black holes work?",
        "Explain blockchain technology",
        "What is CRISPR?",
        "How does photosynthesis work?"
    ]
    
    hits = 0
    for query in novel_queries:
        if cache.get(query, "General context"):
            hits += 1
    
    print(f"Novel workload hit rate: {hits/len(novel_queries)*100:.1f}%")


if __name__ == "__main__":
    main()
    demonstrate_workloads() 