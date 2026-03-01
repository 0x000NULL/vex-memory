"""
Cache Performance Benchmark
============================

Measures embedding cache performance impact on query latency.

Metrics:
- Uncached query latency (Ollama generation)
- Cached query latency (memory/DB lookup)
- Cache hit rate
- Total time saved
"""

import time
import json
import requests
from typing import List, Dict, Any
import statistics

API_BASE = "http://localhost:8000"


def benchmark_queries(queries: List[str], iterations: int = 3) -> Dict[str, Any]:
    """
    Benchmark query performance with caching.
    
    Args:
        queries: List of queries to test
        iterations: Number of times to repeat each query
    
    Returns:
        Performance metrics
    """
    results = {
        "queries_tested": len(queries),
        "iterations": iterations,
        "first_run_times": [],
        "cached_run_times": [],
        "speedup_factors": [],
    }
    
    # Clear cache to start fresh
    print("Clearing cache...")
    requests.post(f"{API_BASE}/api/cache/clear")
    
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Testing: {query[:50]}...")
        
        # First run (cache miss)
        start = time.time()
        resp = requests.post(f"{API_BASE}/query", json={
            "query": query,
            "max_results": 5
        })
        first_run_time = (time.time() - start) * 1000  # ms
        results["first_run_times"].append(first_run_time)
        print(f"  First run: {first_run_time:.1f}ms (cache miss)")
        
        if not resp.ok:
            print(f"  ERROR: {resp.status_code}")
            continue
        
        # Subsequent runs (cache hits)
        cached_times = []
        for j in range(iterations - 1):
            start = time.time()
            resp = requests.post(f"{API_BASE}/query", json={
                "query": query,
                "max_results": 5
            })
            cached_time = (time.time() - start) * 1000  # ms
            cached_times.append(cached_time)
        
        avg_cached_time = statistics.mean(cached_times)
        results["cached_run_times"].append(avg_cached_time)
        
        speedup = first_run_time / avg_cached_time if avg_cached_time > 0 else 0
        results["speedup_factors"].append(speedup)
        
        print(f"  Cached avg: {avg_cached_time:.1f}ms (cache hit)")
        print(f"  Speedup: {speedup:.1f}x")
    
    # Get final cache stats
    cache_resp = requests.get(f"{API_BASE}/api/cache/stats")
    results["cache_stats"] = cache_resp.json() if cache_resp.ok else {}
    
    # Calculate summary statistics
    if results["first_run_times"]:
        results["summary"] = {
            "avg_first_run_ms": statistics.mean(results["first_run_times"]),
            "avg_cached_run_ms": statistics.mean(results["cached_run_times"]),
            "median_speedup": statistics.median(results["speedup_factors"]),
            "max_speedup": max(results["speedup_factors"]),
            "total_time_saved_ms": sum(results["first_run_times"]) - sum(results["cached_run_times"]),
        }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Pretty print benchmark results."""
    print("\n" + "="*70)
    print("CACHE PERFORMANCE BENCHMARK RESULTS")
    print("="*70)
    
    summary = results.get("summary", {})
    
    print(f"\nQueries tested: {results['queries_tested']}")
    print(f"Iterations per query: {results['iterations']}")
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Average first run (uncached): {summary.get('avg_first_run_ms', 0):.1f}ms")
    print(f"  Average cached run:           {summary.get('avg_cached_run_ms', 0):.1f}ms")
    print(f"  Median speedup:               {summary.get('median_speedup', 0):.1f}x")
    print(f"  Max speedup:                  {summary.get('max_speedup', 0):.1f}x")
    print(f"  Total time saved:             {summary.get('total_time_saved_ms', 0):.1f}ms")
    
    cache_stats = results.get("cache_stats", {})
    print(f"\n💾 Cache Statistics:")
    print(f"  Hit rate:                     {cache_stats.get('hit_rate', 0)*100:.1f}%")
    print(f"  Total hits:                   {cache_stats.get('hits', 0)}")
    print(f"  Total misses:                 {cache_stats.get('misses', 0)}")
    print(f"  Memory cache size:            {cache_stats.get('memory_cache_size', 0)} entries")
    print(f"  Avg cache latency:            {cache_stats.get('avg_latency_ms', 0):.4f}ms")
    
    db_cache = cache_stats.get('db_cache', {})
    if db_cache.get('total_entries'):
        print(f"  DB cache entries:             {db_cache.get('total_entries', 0)}")
    
    print("\n✅ SUCCESS CRITERIA:")
    
    # Check if we met targets
    avg_cached = summary.get('avg_cached_run_ms', float('inf'))
    hit_rate = cache_stats.get('hit_rate', 0)
    
    criteria = [
        ("Query latency (cached) < 100ms", avg_cached < 100, f"{avg_cached:.1f}ms"),
        ("Cache hit rate > 50%", hit_rate > 0.5, f"{hit_rate*100:.1f}%"),
        ("Speedup factor > 2x", summary.get('median_speedup', 0) > 2, f"{summary.get('median_speedup', 0):.1f}x"),
    ]
    
    for criterion, passed, value in criteria:
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {value}")
    
    print("\n" + "="*70)


def main():
    """Run cache performance benchmark."""
    
    # Test queries covering different topics
    test_queries = [
        "What is vex-memory and how does it work?",
        "Explain the architecture of the memory system",
        "How are embeddings generated and stored?",
        "What are the key features?",
        "How does caching improve performance?",
        "What database is used for storage?",
        "How are memories prioritized?",
        "What is the consolidation process?",
        "How does temporal search work?",
        "What are the API endpoints available?",
    ]
    
    print("🚀 Starting Cache Performance Benchmark")
    print(f"Testing {len(test_queries)} unique queries with caching...")
    
    results = benchmark_queries(test_queries, iterations=3)
    print_results(results)
    
    # Save results to file
    with open("/tmp/cache_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📝 Results saved to /tmp/cache_benchmark_results.json")


if __name__ == "__main__":
    main()
