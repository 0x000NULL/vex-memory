#!/usr/bin/env python3
"""
Integration test for Phase 3: Adaptive Learning

Tests the complete workflow:
1. Create test memories
2. Make prioritized context queries (logged automatically)
3. Trigger weight optimization
4. Verify learned weights are saved
5. Test analytics endpoints
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"
ANALYTICS_NAMESPACE = "default"  # Analytics uses string namespace names


def create_test_memories(count=20):
    """Create test memories for optimization."""
    print(f"\n📝 Creating {count} test memories...")
    
    memories = []
    for i in range(count):
        response = requests.post(f"{BASE_URL}/api/memories", json={
            "content": f"Test memory {i} about topic {i % 5}. This contains information about feature {i % 3}.",
            "type": "semantic",
            "importance_score": 0.5 + (i % 5) * 0.1
        })
        
        if response.status_code == 200:
            memory = response.json()
            memories.append(memory["id"])
    
    print(f"✅ Created {len(memories)} memories")
    return memories


def make_test_queries(count=60):
    """Make test queries to generate analytics data."""
    print(f"\n🔍 Making {count} test queries...")
    
    queries = [
        "What are the features?",
        "Tell me about topic 0",
        "Information about feature 1",
        "What do we know about topic 2?",
        "Details on feature 0",
    ]
    
    for i in range(count):
        query = queries[i % len(queries)]
        
        # Vary weights slightly to create diverse data
        weights = {
            "similarity": 0.3 + (i % 3) * 0.1,
            "importance": 0.3 + (i % 4) * 0.1,
            "recency": 0.2 + (i % 2) * 0.1
        }
        
        # Note: Not passing namespace to filter - analytics will use "default"
        # In production, you'd map namespace UUID to name for analytics
        response = requests.post(f"{BASE_URL}/api/memories/prioritized-context", json={
            "query": query,
            "token_budget": 2000,
            "model": "gpt-4",
            "weights": weights,
            "limit": 50
        })
        
        if response.status_code != 200:
            print(f"❌ Query {i} failed: {response.status_code}")
            print(response.text)
        
        # Don't overwhelm the server
        if i % 10 == 0:
            time.sleep(0.1)
    
    print(f"✅ Completed {count} queries")


def test_analytics_summary():
    """Test analytics summary endpoint."""
    print("\n📊 Testing analytics summary...")
    
    response = requests.get(f"{BASE_URL}/api/weights/analytics", params={
        "namespace": ANALYTICS_NAMESPACE
    })
    
    if response.status_code == 200:
        summary = response.json()
        print(f"✅ Analytics summary:")
        print(f"   Total queries: {summary['total_queries']}")
        print(f"   Avg tokens used: {summary['avg_tokens_used']:.1f}")
        print(f"   Avg token efficiency: {summary['avg_token_efficiency']:.2%}")
        print(f"   Avg computation time: {summary['avg_computation_time_ms']:.1f}ms")
        return summary
    else:
        print(f"❌ Failed to get analytics: {response.status_code}")
        print(response.text)
        return None


def trigger_optimization():
    """Trigger weight optimization."""
    print("\n🎯 Triggering weight optimization...")
    
    response = requests.post(f"{BASE_URL}/api/weights/optimize", json={
        "namespace": ANALYTICS_NAMESPACE,
        "min_queries": 50
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Optimization complete:")
        print(f"   Weight ID: {result['weight_id']}")
        print(f"   Best weights: {result['best_weights']}")
        print(f"   Objective score: {result['objective_score']:.4f}")
        print(f"   Training queries: {result['metadata']['training_queries']}")
        print(f"   Validation queries: {result['metadata']['validation_queries']}")
        print(f"   Combinations tested: {result['metadata']['combinations_tested']}")
        print(f"   Computation time: {result['metadata']['computation_time_ms']:.1f}ms")
        return result
    else:
        print(f"❌ Optimization failed: {response.status_code}")
        print(response.text)
        return None


def get_learned_weights():
    """Get learned weights for namespace."""
    print("\n📖 Fetching learned weights...")
    
    response = requests.get(f"{BASE_URL}/api/weights/learned/{ANALYTICS_NAMESPACE}")
    
    if response.status_code == 200:
        weights = response.json()
        print(f"✅ Learned weights retrieved:")
        print(f"   Weights: {weights['weights']}")
        print(f"   Objective score: {weights['objective_score']:.4f}")
        print(f"   Trained on {weights['training_queries']} queries")
        return weights
    elif response.status_code == 404:
        print(f"ℹ️  No learned weights found (expected before optimization)")
        return None
    else:
        print(f"❌ Failed to get learned weights: {response.status_code}")
        print(response.text)
        return None


def test_with_learned_weights():
    """Test query using learned weights automatically."""
    print("\n🧪 Testing query with learned weights...")
    
    # Query without explicit weights - should use learned weights
    response = requests.post(f"{BASE_URL}/api/memories/prioritized-context", json={
        "query": "What are the main features?",
        "token_budget": 2000,
        "model": "gpt-4",
        "limit": 50
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Query successful:")
        print(f"   Selected {len(result['memories'])} memories")
        print(f"   Used {result['metadata']['total_tokens']} tokens")
        return result
    else:
        print(f"❌ Query failed: {response.status_code}")
        print(response.text)
        return None


def cleanup():
    """Cleanup test data."""
    print("\n🧹 Cleaning up test data...")
    
    # Delete analytics data
    response = requests.delete(f"{BASE_URL}/api/analytics/{ANALYTICS_NAMESPACE}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Deleted {result['deleted_logs']} query logs")
    else:
        print(f"⚠️  Analytics cleanup: {response.status_code}")
    
    # Note: Memories and learned weights are left for inspection
    # You can manually delete via:
    # DELETE FROM learned_weights WHERE namespace = 'default';
    # DELETE FROM optimization_history WHERE namespace = 'default';


def main():
    """Run complete integration test."""
    print("=" * 70)
    print("🧪 Phase 3 Integration Test: Adaptive Learning")
    print("=" * 70)
    
    try:
        # Step 1: Create test data
        memories = create_test_memories(count=20)
        
        # Step 2: Make queries to generate analytics
        make_test_queries(count=60)
        
        # Step 3: Check analytics
        summary = test_analytics_summary()
        
        if not summary or summary['total_queries'] < 50:
            print("\n❌ Insufficient queries for optimization")
            return
        
        # Step 4: Check for learned weights (should not exist yet)
        get_learned_weights()
        
        # Step 5: Trigger optimization
        optimization_result = trigger_optimization()
        
        if not optimization_result:
            print("\n❌ Optimization failed")
            return
        
        # Step 6: Verify learned weights were saved
        learned_weights = get_learned_weights()
        
        if not learned_weights:
            print("\n❌ Learned weights not found after optimization")
            return
        
        # Step 7: Test query with learned weights
        test_with_learned_weights()
        
        # Success!
        print("\n" + "=" * 70)
        print("✅ All Phase 3 integration tests passed!")
        print("=" * 70)
        print("\n📊 Summary:")
        print(f"   - Created {len(memories)} test memories")
        print(f"   - Executed {summary['total_queries']} queries")
        print(f"   - Optimized weights: {learned_weights['weights']}")
        print(f"   - Objective score: {learned_weights['objective_score']:.4f}")
        print(f"\nℹ️  Test data left in database for inspection.")
        print(f"   Namespace: {ANALYTICS_NAMESPACE}")
        
        # Optionally cleanup
        # cleanup()
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
