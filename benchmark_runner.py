"""
Benchmarking Suite for Enhanced GPTCache
Tests performance across different workloads and configurations.
"""

import os
import time
import json
import csv
import argparse
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from enhanced_cache import EnhancedCache
import gptcache
from gptcache import Cache


class VanillaCache:
    """Vanilla GPTCache wrapper that mimics original behavior without enhancements."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Create a simple cache that ignores context and uses basic similarity
        self.cache_entries = {}
        self.cache_order = []
        self.max_size = 10000
        self.ttl = 3600
        
        # Simple embedding model for vanilla behavior
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Basic similarity threshold (no federated learning)
        self.tau = 0.8
        
        # Mock config for compatibility
        self.config = type('Config', (), {
            'get': lambda key, default=None: {
                'pca_enabled': False,
                'pca_dim': 768,
                'tau': 0.8,
                'fl_round': 1000000
            }.get(key, default),
            'update': lambda key, value: None  # No-op for vanilla
        })()
        
    def encode(self, text: str):
        """Simple encoding without PCA."""
        return self.embedding_model.encode(text)
    
    def similarity(self, emb1, emb2):
        """Basic cosine similarity."""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    
    def get(self, query: str, context: str = "") -> str:
        """Get response from cache (ignores context)."""
        import time
        
        query_emb = self.encode(query)
        
        # Find best match (ignoring context)
        best_match = None
        best_score = 0.0
        
        for cache_id, entry in self.cache_entries.items():
            # Check TTL
            if time.time() - entry['timestamp'] > self.ttl:
                continue
            
            # Simple similarity check (no context filtering)
            score = self.similarity(query_emb, entry['query_emb'])
            
            if score > best_score:
                best_score = score
                best_match = entry
        
        # Return if similarity exceeds threshold
        if best_match and best_score >= self.tau:
            return best_match['response']
        
        return None
    
    def put(self, query: str, response: str, context: str = "", metadata: dict = None):
        """Store query-response pair (ignores context)."""
        import time
        import hashlib
        
        query_emb = self.encode(query)
        
        # Create cache entry (no context embedding)
        entry = {
            'query_emb': query_emb,
            'response': response,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        # Generate cache ID
        cache_id = hashlib.md5(query.encode()).hexdigest()
        
        # Add to cache
        self.cache_entries[cache_id] = entry
        self.cache_order.append(cache_id)
        
        # LRU eviction
        if len(self.cache_entries) > self.max_size:
            oldest_id = self.cache_order.pop(0)
            del self.cache_entries[oldest_id]
    
    def clear(self):
        """Clear cache."""
        self.cache_entries.clear()
        self.cache_order.clear()
    
    def get_stats(self):
        """Get cache statistics."""
        return {
            'size': len(self.cache_entries),
            'max_size': self.max_size,
            'current_tau': self.tau,
            'query_count': 0,  # No tracking in vanilla
            'pca_enabled': False,
            'pca_trained': False
        }


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    workload: str
    variant: str
    hit_rate: float
    f1_score: float
    latency_mean: float
    latency_p95: float
    latency_p99: float
    memory_usage_mb: float
    cache_size: int
    tau: float
    pca_enabled: bool
    pca_dim: int
    federated_enabled: bool
    total_queries: int
    timestamp: str


class WorkloadGenerator:
    """Generates different types of workloads for benchmarking."""
    
    def __init__(self):
        # Sample queries for different domains
        self.queries = {
            'tech': [
                "What is machine learning?",
                "How do neural networks work?",
                "Explain deep learning",
                "What is artificial intelligence?",
                "How does Python work?",
                "What is data science?",
                "Explain computer vision",
                "What is natural language processing?",
                "How do databases work?",
                "What is cloud computing?"
            ],
            'science': [
                "What is quantum physics?",
                "How does photosynthesis work?",
                "Explain the theory of relativity",
                "What is DNA?",
                "How do atoms work?",
                "What is evolution?",
                "Explain gravity",
                "What is chemistry?",
                "How do cells work?",
                "What is astronomy?"
            ],
            'general': [
                "What is the weather like?",
                "How do I cook pasta?",
                "What is the capital of France?",
                "How do I learn to drive?",
                "What is the meaning of life?",
                "How do I make friends?",
                "What is music?",
                "How do I write a book?",
                "What is art?",
                "How do I stay healthy?"
            ]
        }
        
        self.contexts = {
            'tech': "Technology and computer science context",
            'science': "Scientific research and discovery context", 
            'general': "General knowledge and everyday life context"
        }
        
        self.responses = {
            'tech': [
                "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
                "Neural networks are computational models inspired by biological neural networks.",
                "Deep learning uses multiple layers of neural networks to learn complex patterns.",
                "Artificial intelligence is the simulation of human intelligence in machines.",
                "Python is a high-level, interpreted programming language known for its simplicity.",
                "Data science combines statistics, programming, and domain expertise to extract insights.",
                "Computer vision enables machines to interpret and understand visual information.",
                "Natural language processing helps computers understand and generate human language.",
                "Databases are organized collections of structured information.",
                "Cloud computing provides on-demand computing resources over the internet."
            ],
            'science': [
                "Quantum physics describes the behavior of matter and energy at atomic scales.",
                "Photosynthesis is the process by which plants convert light into chemical energy.",
                "The theory of relativity describes the relationship between space, time, and gravity.",
                "DNA is the molecule that carries genetic information in living organisms.",
                "Atoms are the basic building blocks of matter, consisting of protons, neutrons, and electrons.",
                "Evolution is the process by which species change over time through natural selection.",
                "Gravity is a fundamental force that attracts objects toward each other.",
                "Chemistry is the study of matter and the changes it undergoes.",
                "Cells are the basic structural and functional units of all living organisms.",
                "Astronomy is the study of celestial objects and phenomena in the universe."
            ],
            'general': [
                "Weather conditions vary by location and time, check local forecasts for accurate information.",
                "To cook pasta, boil water, add pasta, and cook until al dente, then drain and serve.",
                "Paris is the capital of France, known for its culture, art, and architecture.",
                "Learning to drive involves understanding traffic rules, vehicle operation, and safety practices.",
                "The meaning of life is a philosophical question with many different interpretations.",
                "Making friends involves being open, kind, and finding common interests with others.",
                "Music is an art form using sound and silence organized in time.",
                "Writing a book involves planning, drafting, revising, and editing your ideas.",
                "Art is creative expression that can take many forms and serve various purposes.",
                "Staying healthy involves balanced nutrition, regular exercise, and good sleep habits."
            ]
        }
    
    def generate_repetitive_workload(self, num_queries: int) -> List[Tuple[str, str, str]]:
        """Generate repetitive workload with high cache hit potential."""
        workload = []
        
        # Use a small set of queries that repeat frequently
        core_queries = self.queries['tech'][:5]  # Use first 5 tech queries
        core_contexts = [self.contexts['tech']] * 5
        core_responses = self.responses['tech'][:5]
        
        for i in range(num_queries):
            # Repeat the same queries with some variation
            query_idx = i % len(core_queries)
            query = core_queries[query_idx]
            context = core_contexts[query_idx]
            response = core_responses[query_idx]
            
            # Add some minor variations to test similarity matching
            if i % 10 == 0:  # Every 10th query, add slight variation
                query = query.replace("?", "?").replace("What", "Can you tell me about")
            
            workload.append((query, context, response))
        
        return workload
    
    def generate_contextual_workload(self, num_queries: int) -> List[Tuple[str, str, str]]:
        """Generate contextual workload with varying contexts."""
        workload = []
        
        # Use queries from different domains with different contexts
        all_queries = []
        all_contexts = []
        all_responses = []
        
        for domain in ['tech', 'science', 'general']:
            all_queries.extend(self.queries[domain])
            all_contexts.extend([self.contexts[domain]] * len(self.queries[domain]))
            all_responses.extend(self.responses[domain])
        
        for i in range(num_queries):
            # Cycle through different domains
            idx = i % len(all_queries)
            query = all_queries[idx]
            context = all_contexts[idx]
            response = all_responses[idx]
            
            # Sometimes use similar queries with different contexts
            if i % 20 == 0:
                # Use a tech query with science context
                query = self.queries['tech'][0]
                context = self.contexts['science']
                response = "Machine learning applies scientific principles to enable computers to learn."
            
            workload.append((query, context, response))
        
        return workload
    
    def generate_novel_workload(self, num_queries: int) -> List[Tuple[str, str, str]]:
        """Generate novel workload with mostly unique queries."""
        workload = []
        
        # Create variations of existing queries to simulate novel queries
        base_queries = []
        for domain_queries in self.queries.values():
            base_queries.extend(domain_queries)
        
        for i in range(num_queries):
            if i < len(base_queries):
                # Use original queries for first batch
                query = base_queries[i]
                context = self.contexts['tech'] if i < len(self.queries['tech']) else self.contexts['science']
                response = f"Response for: {query}"
            else:
                # Generate novel queries by combining words
                words = ["advanced", "modern", "sophisticated", "complex", "innovative", "cutting-edge"]
                topics = ["algorithms", "systems", "technologies", "methods", "approaches", "solutions"]
                query = f"What are {np.random.choice(words)} {np.random.choice(topics)}?"
                context = "Novel technology context"
                response = f"Novel response for: {query}"
            
            workload.append((query, context, response))
        
        return workload


class BenchmarkRunner:
    """Runs comprehensive benchmarks on the enhanced cache."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.workload_generator = WorkloadGenerator()
        self.results = []
        
    def run_benchmark(self, workload_type: str, num_queries: int, 
                     variant: str = "enhanced", pca_enabled: bool = True, federated_enabled: bool = False) -> BenchmarkResult:
        """Run a single benchmark."""
        print(f"\nRunning {workload_type} workload with {variant} variant...")
        
        # Create cache based on variant
        if variant == "vanilla":
            cache = VanillaCache(self.config_path)
        else:
            cache = EnhancedCache(self.config_path)
        
        # Configure PCA (only for enhanced variants)
        if variant != "vanilla":
            if not pca_enabled:
                cache.config.update('pca_enabled', False)
            
            # Configure Federated Learning
            if federated_enabled:
                # Enable federated learning with more frequent updates for testing
                cache.config.update('fl_round', 50)  # Update every 50 queries instead of 100
                cache.config.update('fl_clients', 3)  # Reduce clients for faster simulation
            else:
                # Disable federated learning by setting a very high round number
                cache.config.update('fl_round', 1000000)  # Effectively disables federated updates
        
        # Generate workload
        if workload_type == "repetitive":
            workload = self.workload_generator.generate_repetitive_workload(num_queries)
        elif workload_type == "contextual":
            workload = self.workload_generator.generate_contextual_workload(num_queries)
        elif workload_type == "novel":
            workload = self.workload_generator.generate_novel_workload(num_queries)
        else:
            raise ValueError(f"Unknown workload type: {workload_type}")
        
        # Pre-populate cache with some entries
        pre_populate_count = num_queries // 4
        for i in range(pre_populate_count):
            query, context, response = workload[i]
            cache.put(query, response, context)
        
        # Run queries and collect metrics
        latencies = []
        hits = 0
        total_queries = len(workload)
        
        # Track memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for query, context, response in tqdm(workload, desc=f"Running {workload_type}"):
            start_time = time.time()
            
            # Try to get from cache
            cached_response = cache.get(query, context)
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            if cached_response is not None:
                hits += 1
            else:
                # Store in cache if not found
                cache.put(query, response, context)
        
        # Calculate final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        # Calculate metrics
        hit_rate = hits / total_queries
        latency_mean = np.mean(latencies)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        
        # Calculate F1 score (simplified - assume all hits are correct for this benchmark)
        precision = hit_rate
        recall = hit_rate  # Simplified assumption
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Get cache stats
        stats = cache.get_stats()
        
        result = BenchmarkResult(
            workload=workload_type,
            variant=variant,
            hit_rate=hit_rate,
            f1_score=f1_score,
            latency_mean=latency_mean,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            memory_usage_mb=memory_usage,
            cache_size=stats['size'],
            tau=stats['current_tau'],
            pca_enabled=stats['pca_enabled'],
            pca_dim=cache.config.get('pca_dim'),
            federated_enabled=federated_enabled,
            total_queries=total_queries,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        return result
    
    def run_comprehensive_benchmark(self, num_queries: int = 1000):
        """Run comprehensive benchmark across all workloads and variants."""
        print("Starting comprehensive benchmark...")
        
        workloads = ["repetitive", "contextual", "novel"]
        variants = [
            ("vanilla", False, False),          # Original GPTCache behavior
            ("enhanced_no_pca", False, False),  # Context only (no PCA, no federated)
            ("enhanced_pca", True, False),      # Context + PCA (no federated)
            ("enhanced_full", True, True)       # Context + PCA + Federated Learning
        ]
        
        for workload in workloads:
            for variant_name, pca_enabled, federated_enabled in variants:
                try:
                    result = self.run_benchmark(
                        workload, num_queries, variant_name, pca_enabled, federated_enabled
                    )
                    print(f"Completed {workload} - {variant_name}")
                except Exception as e:
                    print(f"Error in {workload} - {variant_name}: {e}")
    
    def export_results(self, format: str = "csv"):
        """Export benchmark results."""
        if format == "csv":
            filename = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    'workload', 'variant', 'hit_rate', 'f1_score', 'latency_mean', 
                    'latency_p95', 'latency_p99', 'memory_usage_mb', 'cache_size', 
                    'tau', 'pca_enabled', 'pca_dim', 'federated_enabled', 'total_queries', 'timestamp'
                ])
                # Write data
                for result in self.results:
                    writer.writerow([
                        result.workload, result.variant, result.hit_rate, result.f1_score,
                        result.latency_mean, result.latency_p95, result.latency_p99,
                        result.memory_usage_mb, result.cache_size, result.tau,
                        result.pca_enabled, result.pca_dim, result.federated_enabled, result.total_queries, result.timestamp
                    ])
            print(f"Results exported to {filename}")
        
        elif format == "json":
            filename = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump([vars(result) for result in self.results], f, indent=2)
            print(f"Results exported to {filename}")
    
    def generate_plots(self):
        """Generate visualization plots for benchmark results."""
        if not self.results:
            print("No results to plot")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame([vars(result) for result in self.results])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced GPTCache Benchmark Results', fontsize=16)
        
        # Hit Rate by Workload and Variant
        ax1 = axes[0, 0]
        pivot_hit = df.pivot_table(values='hit_rate', index='workload', columns='variant', aggfunc='mean')
        pivot_hit.plot(kind='bar', ax=ax1)
        ax1.set_title('Hit Rate by Workload and Variant')
        ax1.set_ylabel('Hit Rate')
        ax1.legend()
        
        # Latency by Workload
        ax2 = axes[0, 1]
        pivot_latency = df.pivot_table(values='latency_mean', index='workload', columns='variant', aggfunc='mean')
        pivot_latency.plot(kind='bar', ax=ax2)
        ax2.set_title('Mean Latency by Workload and Variant')
        ax2.set_ylabel('Latency (ms)')
        ax2.legend()
        
        # Memory Usage
        ax3 = axes[1, 0]
        pivot_memory = df.pivot_table(values='memory_usage_mb', index='workload', columns='variant', aggfunc='mean')
        pivot_memory.plot(kind='bar', ax=ax3)
        ax3.set_title('Memory Usage by Workload and Variant')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.legend()
        
        # F1 Score
        ax4 = axes[1, 1]
        pivot_f1 = df.pivot_table(values='f1_score', index='workload', columns='variant', aggfunc='mean')
        pivot_f1.plot(kind='bar', ax=ax4)
        ax4.set_title('F1 Score by Workload and Variant')
        ax4.set_ylabel('F1 Score')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'benchmark_results_{time.strftime("%Y%m%d_%H%M%S")}.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved as benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.png")


def main():
    """Main function for running benchmarks."""
    parser = argparse.ArgumentParser(description='Enhanced GPTCache Benchmark Runner')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--queries', type=int, default=1000, help='Number of queries per benchmark')
    parser.add_argument('--workload', choices=['repetitive', 'contextual', 'novel', 'all'], 
                       default='all', help='Workload type to benchmark')
    parser.add_argument('--variant', choices=['vanilla', 'enhanced_pca', 'enhanced_no_pca', 'enhanced_full', 'all'],
                       default='all', help='Cache variant to test')
    parser.add_argument('--export', choices=['csv', 'json', 'both'], default='both',
                       help='Export format for results')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(args.config)
    
    # Run benchmarks
    if args.workload == 'all':
        workloads = ['repetitive', 'contextual', 'novel']
    else:
        workloads = [args.workload]
    
    if args.variant == 'all':
        variants = [
            ('vanilla', False, False),          # Original GPTCache behavior
            ('enhanced_no_pca', False, False),  # Context only (no PCA, no federated)
            ('enhanced_pca', True, False),      # Context + PCA (no federated)
            ('enhanced_full', True, True)       # Context + PCA + Federated Learning
        ]
    else:
        if args.variant == 'vanilla':
            variants = [('vanilla', False, False)]
        elif args.variant == 'enhanced_pca':
            variants = [('enhanced_pca', True, False)]
        elif args.variant == 'enhanced_no_pca':
            variants = [('enhanced_no_pca', False, False)]
        elif args.variant == 'enhanced_full':
            variants = [('enhanced_full', True, True)]
        else:
            raise ValueError(f"Unknown variant: {args.variant}")
    
    for workload in workloads:
        for variant_name, pca_enabled, federated_enabled in variants:
            try:
                result = runner.run_benchmark(workload, args.queries, variant_name, pca_enabled, federated_enabled)
                print(f"\n{workload} - {variant_name} Results:")
                print(f"  Hit Rate: {result.hit_rate:.3f}")
                print(f"  F1 Score: {result.f1_score:.3f}")
                print(f"  Mean Latency: {result.latency_mean:.2f} ms")
                print(f"  P95 Latency: {result.latency_p95:.2f} ms")
                print(f"  Memory Usage: {result.memory_usage_mb:.2f} MB")
            except Exception as e:
                print(f"Error in {workload} - {variant_name}: {e}")
    
    # Export results
    if args.export in ['csv', 'both']:
        runner.export_results('csv')
    if args.export in ['json', 'both']:
        runner.export_results('json')
    
    # Generate plots
    if args.plots:
        runner.generate_plots()


if __name__ == "__main__":
    main() 