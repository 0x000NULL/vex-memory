#!/bin/bash
# Integration test for embedding cache system
# Tests all cache layers and API endpoints

set -e

API_URL="http://localhost:8000"
PASSED=0
FAILED=0

echo "========================================================================"
echo "EMBEDDING CACHE INTEGRATION TEST"
echo "========================================================================"
echo ""

# Helper functions
pass() {
    echo "✅ PASS: $1"
    ((PASSED++))
}

fail() {
    echo "❌ FAIL: $1"
    ((FAILED++))
}

test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    
    echo -n "Testing $name... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" -o /tmp/response.json "$API_URL$endpoint")
    else
        response=$(curl -s -w "%{http_code}" -X POST -H "Content-Type: application/json" -d "$data" -o /tmp/response.json "$API_URL$endpoint")
    fi
    
    if [ "$response" = "200" ]; then
        pass "$name"
        return 0
    else
        fail "$name (HTTP $response)"
        return 1
    fi
}

echo "1. Health Check"
echo "----------------"
test_endpoint "Health endpoint" "GET" "/health" ""
if jq -e '.cache_stats != null' /tmp/response.json > /dev/null 2>&1; then
    pass "Cache stats in health response"
else
    fail "Cache stats missing from health"
fi
echo ""

echo "2. Cache Statistics"
echo "-------------------"
test_endpoint "Get cache stats" "GET" "/api/cache/stats" ""
if jq -e '.hit_rate' /tmp/response.json > /dev/null 2>&1; then
    pass "Cache stats structure valid"
else
    fail "Cache stats structure invalid"
fi
echo ""

echo "3. Cache Operations"
echo "-------------------"

# Clear cache
test_endpoint "Clear cache" "POST" "/api/cache/clear" "{}"
if jq -e '.memory_cleared >= 0' /tmp/response.json > /dev/null 2>&1; then
    pass "Cache clear returns counts"
else
    fail "Cache clear response invalid"
fi

# Warmup cache
warmup_data='{"common_texts": ["test query 1", "test query 2", "test query 3"]}'
test_endpoint "Cache warmup" "POST" "/api/cache/warmup" "$warmup_data"
if jq -e '.cached_count' /tmp/response.json > /dev/null 2>&1; then
    cached=$(jq -r '.cached_count' /tmp/response.json)
    if [ "$cached" -gt 0 ]; then
        pass "Cache warmup cached $cached embeddings"
    else
        fail "Cache warmup cached 0 embeddings"
    fi
else
    fail "Cache warmup response invalid"
fi

# Verify warmup worked
test_endpoint "Stats after warmup" "GET" "/api/cache/stats" ""
memory_size=$(jq -r '.memory_cache_size' /tmp/response.json)
if [ "$memory_size" -gt 0 ]; then
    pass "Memory cache populated ($memory_size entries)"
else
    fail "Memory cache still empty after warmup"
fi

# Evict old entries
test_endpoint "Cache eviction" "POST" "/api/cache/evict" "{}"
if jq -e '.evicted >= 0' /tmp/response.json > /dev/null 2>&1; then
    pass "Cache eviction returns count"
else
    fail "Cache eviction response invalid"
fi

echo ""

echo "4. Query Performance (Cache Hit)"
echo "---------------------------------"

# First query (cache miss after clear)
query_data='{"query": "What is vex-memory?", "max_results": 5}'
echo -n "First query (should miss)... "
start_time=$(date +%s%N)
curl -s -X POST -H "Content-Type: application/json" -d "$query_data" "$API_URL/query" > /tmp/query1.json
end_time=$(date +%s%N)
first_query_ms=$(( (end_time - start_time) / 1000000 ))

if [ $? -eq 0 ]; then
    pass "First query completed (${first_query_ms}ms)"
else
    fail "First query failed"
fi

# Second query (cache hit)
echo -n "Second query (should hit)... "
start_time=$(date +%s%N)
curl -s -X POST -H "Content-Type: application/json" -d "$query_data" "$API_URL/query" > /tmp/query2.json
end_time=$(date +%s%N)
second_query_ms=$(( (end_time - start_time) / 1000000 ))

if [ $? -eq 0 ]; then
    pass "Second query completed (${second_query_ms}ms)"
else
    fail "Second query failed"
fi

# Check if second query was faster (or at least not slower)
if [ "$second_query_ms" -le "$first_query_ms" ]; then
    pass "Second query same speed or faster (cache working)"
else
    echo "⚠️  WARNING: Second query slower ($second_query_ms vs $first_query_ms ms)"
fi

# Verify cache hit
test_endpoint "Stats after queries" "GET" "/api/cache/stats" ""
hits=$(jq -r '.hits' /tmp/response.json)
if [ "$hits" -gt 0 ]; then
    hit_rate=$(jq -r '.hit_rate' /tmp/response.json)
    pass "Cache has hits (hit_rate: $hit_rate)"
else
    fail "Cache has no hits after queries"
fi

echo ""

echo "5. Database Cache Persistence"
echo "------------------------------"

# Check DB cache exists
docker exec vex-memory-db-1 psql -U vex -d vex_memory -c "SELECT COUNT(*) FROM embedding_cache;" > /tmp/db_cache_count.txt 2>&1
if [ $? -eq 0 ]; then
    pass "Database cache table accessible"
    db_count=$(grep -oP '\d+' /tmp/db_cache_count.txt | tail -1)
    echo "   Database has $db_count cached embeddings"
else
    fail "Database cache table not accessible"
fi

echo ""

echo "6. Cache Stats Validation"
echo "-------------------------"

test_endpoint "Final cache stats" "GET" "/api/cache/stats" ""

# Validate required fields
required_fields=("hits" "misses" "hit_rate" "memory_cache_size" "avg_latency_ms")
for field in "${required_fields[@]}"; do
    if jq -e ".$field" /tmp/response.json > /dev/null 2>&1; then
        value=$(jq -r ".$field" /tmp/response.json)
        pass "Stats has $field: $value"
    else
        fail "Stats missing field: $field"
    fi
done

echo ""
echo "========================================================================"
echo "TEST SUMMARY"
echo "========================================================================"
echo ""
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All tests passed! Cache system working correctly."
    exit 0
else
    echo "⚠️  Some tests failed. Review output above."
    exit 1
fi
