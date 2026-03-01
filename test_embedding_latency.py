"""
Direct Embedding Latency Test
==============================

Tests the actual embedding generation latency with and without caching.
This isolates the cache performance from the full query pipeline.
"""

import time
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding_cache import EmbeddingCache, get_cache, initialize_cache
import db


def test_embedding_latency():
    """Test embedding generation latency directly."""
    
    # Initialize cache with DB connection
    conn = db.get_connection()
    cache = initialize_cache(conn)
    
    # Clear cache to start fresh
    cache.clear_all()
    print("Cache cleared\n")
    
    test_texts = [
        "What is vex-memory and how does it work?",
        "Explain the architecture of the memory system",
        "How are embeddings generated and stored?",
        "What are the key features?",
        "How does caching improve performance?",
    ]
    
    print("="*70)
    print("EMBEDDING GENERATION LATENCY TEST")
    print("="*70)
    
    # Import embedding function
    from api import _get_embedding_sync
    
    print("\n📝 Testing cache performance...\n")
    
    first_run_times = []
    second_run_times = []
    third_run_times = []
    
    for i, text in enumerate(test_texts):
        print(f"[{i+1}/{len(test_texts)}] '{text[:50]}...'")
        
        # First run - cache miss
        start = time.time()
        emb1 = _get_embedding_sync(text)
        first_time = (time.time() - start) * 1000
        first_run_times.append(first_time)
        print(f"  1st run (MISS):  {first_time:7.2f}ms")
        
        # Second run - memory cache hit
        start = time.time()
        emb2 = _get_embedding_sync(text)
        second_time = (time.time() - start) * 1000
        second_run_times.append(second_time)
        print(f"  2nd run (HIT):   {second_time:7.2f}ms  ({first_time/second_time:.1f}x faster)")
        
        # Third run - still memory cache
        start = time.time()
        emb3 = _get_embedding_sync(text)
        third_time = (time.time() - start) * 1000
        third_run_times.append(third_time)
        print(f"  3rd run (HIT):   {third_time:7.2f}ms  ({first_time/third_time:.1f}x faster)")
        
        # Verify embeddings are identical
        assert emb1 == emb2 == emb3, "Cached embeddings should match!"
        print()
    
    # Calculate averages
    avg_first = sum(first_run_times) / len(first_run_times)
    avg_second = sum(second_run_times) / len(second_run_times)
    avg_third = sum(third_run_times) / len(third_run_times)
    
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n📊 Average Latencies:")
    print(f"  First run (uncached):  {avg_first:7.2f}ms")
    print(f"  Second run (cached):   {avg_second:7.2f}ms")
    print(f"  Third run (cached):    {avg_third:7.2f}ms")
    
    speedup = avg_first / avg_second if avg_second > 0 else 0
    print(f"\n⚡ Average Speedup: {speedup:.1f}x")
    print(f"   Time saved per embedding: {avg_first - avg_second:.2f}ms")
    
    # Get cache stats
    stats = cache.get_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"  Hit rate:           {stats['hit_rate']*100:.1f}%")
    print(f"  Total hits:         {stats['hits']}")
    print(f"  Total misses:       {stats['misses']}")
    print(f"  Memory hits:        {stats['memory_hits']}")
    print(f"  DB hits:            {stats['db_hits']}")
    print(f"  Cache size:         {stats['memory_cache_size']} entries")
    print(f"  Avg cache latency:  {stats['avg_latency_ms']:.4f}ms")
    
    print(f"\n✅ SUCCESS CRITERIA:")
    
    criteria = [
        ("Embedding (cached) < 10ms", avg_second < 10, f"{avg_second:.2f}ms"),
        ("Speedup > 10x", speedup > 10, f"{speedup:.1f}x"),
        ("Cache hit rate > 60%", stats['hit_rate'] > 0.6, f"{stats['hit_rate']*100:.1f}%"),
        ("Embeddings identical", True, "✓"),
    ]
    
    all_passed = True
    for criterion, passed, value in criteria:
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {value}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("🎉 All criteria passed! Cache working perfectly!")
    else:
        print("⚠️  Some criteria not met. Review results above.")
    
    print()


if __name__ == "__main__":
    test_embedding_latency()
