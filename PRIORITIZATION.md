# Smart Context Prioritization

Intelligent memory prioritization system that maximizes context relevance within token budgets.

## Overview

Instead of simple concatenation, vex-memory now intelligently selects and prioritizes memories using multi-factor scoring to fit within LLM token budgets while maximizing relevance.

## Features

### Phase 1: Token-Aware Ranking (v1.1.0) ✅

- **Accurate Token Counting**: Uses `tiktoken` for precise token estimation across different LLM models
- **Multi-Factor Scoring**: Combines similarity, importance, and recency into a weighted score
- **Diversity Filtering**: Avoids redundant memories using Jaccard similarity
- **Budget Enforcement**: Guarantees no token budget overruns
- **Graceful Truncation**: Handles oversized memories intelligently
- **High Performance**: <100ms for 1000 memories

### Phase 2: Advanced Diversity + Priorities (v1.2.0) ✅

- **MMR Algorithm**: Maximal Marginal Relevance for better diversity control
- **Entity Extraction**: Automatic extraction of people, organizations, locations, dates, etc.
- **Entity Coverage**: Track and prioritize entity coverage in selected memories
- **Type Priorities**: Configurable priorities for memory types (episodic=1.0, semantic=0.8, procedural=0.6, meta=0.4)
- **Namespace Priorities**: Configurable priorities for namespaces (main=1.0, shared=0.7, isolated=0.3)
- **Weight Presets**: Pre-configured weight sets for common use cases
- **Weight Tuning**: Grid search and benchmarking tools for optimization

## API Usage

### Endpoint: `POST /api/memories/prioritized-context`

Request:
```json
{
  "query": "What did we discuss about the project launch?",
  "token_budget": 4000,
  "model": "gpt-4",
  "weights": {
    "similarity": 0.4,
    "importance": 0.3,
    "recency": 0.2,
    "diversity": 0.1
  },
  "diversity_threshold": 0.7,
  "min_score": 0.5,
  "namespace": "project-alpha",
  "limit": 100
}
```

Parameters:
- `query` (required): Search query for finding relevant memories
- `token_budget` (default: 4000): Maximum tokens to use
- `model` (default: "gpt-4"): LLM model for token counting
- `weights` (optional): Custom scoring weights
- `diversity_threshold` (default: 0.7): Jaccard similarity threshold (0-1)
- `min_score` (optional): Minimum score threshold (0-1)
- `namespace` (optional): Filter by namespace
- `limit` (default: 100): Maximum candidate memories to retrieve

Response:
```json
{
  "memories": [
    {
      "id": "...",
      "content": "...",
      "importance_score": 0.85,
      "_score": 0.78,
      "_score_factors": {
        "similarity": 0.92,
        "importance": 0.85,
        "recency": 0.65
      }
    }
  ],
  "metadata": {
    "total_tokens": 3847,
    "budget": 4000,
    "utilization": 0.96,
    "memories_selected": 12,
    "memories_available": 100,
    "diversity_filtered": 5,
    "average_score": 0.73,
    "weights": {
      "similarity": 0.4,
      "importance": 0.3,
      "recency": 0.2,
      "diversity": 0.1
    },
    "search_type": "prioritized_semantic",
    "model": "gpt-4",
    "diversity_threshold": 0.7
  }
}
```

## Scoring Algorithm

### Multi-Factor Score

Each memory receives a composite score:

```
score = w_sim * similarity + w_imp * importance + w_rec * recency
```

Default weights:
- Similarity: 0.4 (semantic match to query)
- Importance: 0.3 (user-assigned importance)
- Recency: 0.2 (time-based decay)
- Diversity: 0.1 (reserved for Phase 2 enhancements)

### Recency Decay

Uses exponential decay with configurable half-life (default: 30 days):

```
recency_score = e^(-λ * age_days)
where λ = ln(2) / half_life_days
```

### Diversity Filtering

Jaccard similarity between word sets:
```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Memories with similarity > threshold to already-selected memories are filtered out.

## Selection Algorithm

1. **Retrieve Candidates**: Semantic search returns top N memories (default: 100)
2. **Score All**: Calculate multi-factor score for each memory
3. **Sort by Score**: Descending order
4. **Greedy Selection**:
   - For each memory (highest score first):
     - Check token budget (skip if exceeded)
     - Check diversity threshold (skip if too similar)
     - Add to selection
5. **Return**: Selected memories + metadata

## Phase 2: MMR and Entity Coverage (v1.2.0)

### MMR Endpoint: `POST /api/memories/prioritized-mmr`

Maximal Marginal Relevance provides better diversity control by iteratively selecting memories that balance relevance and novelty.

Request:
```json
{
  "query": "project updates",
  "token_budget": 2000,
  "model": "gpt-4",
  "weights": {
    "lambda": 0.7
  }
}
```

Parameters:
- `lambda` (default: 0.7): Balance between relevance (1.0) and diversity (0.0)
  - 1.0 = Pure relevance (identical to greedy)
  - 0.7 = Good balance (recommended)
  - 0.5 = Equal weight to relevance and diversity
  - 0.0 = Maximum diversity (may sacrifice relevance)

### MMR Algorithm

```
For each iteration:
  If first memory:
    Select highest relevance score
  Else:
    For each candidate:
      mmr_score = λ * relevance - (1-λ) * max_similarity_to_selected
    Select memory with highest mmr_score
```

**Performance Note:** MMR has O(n²) complexity due to pairwise similarity calculations. For optimal performance with large datasets (>200 candidates), the implementation automatically limits to the top 200 scored candidates before applying MMR. For datasets with <200 memories, use standard prioritization which is faster.

### Type and Namespace Priorities

Configure priority multipliers for different memory types and namespaces:

```python
from prioritizer import PriorityMappings

# Custom priorities
priorities = PriorityMappings(
    type_priorities={
        "episodic": 1.0,    # Highest priority
        "semantic": 0.8,
        "procedural": 0.6,
        "meta": 0.4         # Lowest priority
    },
    namespace_priorities={
        "main": 1.0,        # Highest priority
        "shared": 0.7,
        "isolated": 0.3     # Lowest priority
    }
)

prioritizer = MemoryPrioritizer(priority_mappings=priorities)
```

Priority multipliers are applied to the base score:
```
final_score = base_score * ((type_priority + namespace_priority) / 2)
```

### Weight Presets

Get optimized weight configurations for common use cases:

```bash
# Get available presets
curl http://localhost:8000/api/weights/presets

# Get recommended weights for a use case
curl http://localhost:8000/api/weights/recommend?use_case=entity_focused
```

Available presets:
- **balanced**: All factors weighted equally (default)
- **relevance_focused**: Prioritizes similarity and importance
- **recency_focused**: Prioritizes recent memories
- **diversity_focused**: Maximizes variety and entity coverage
- **entity_focused**: Prioritizes entity coverage
- **importance_focused**: Prioritizes memory importance

Example response:
```json
{
  "name": "Entity Focused",
  "description": "Prioritizes entity coverage",
  "weights": {
    "similarity": 0.3,
    "importance": 0.25,
    "recency": 0.1,
    "diversity": 0.1,
    "entity_coverage": 0.25
  }
}
```

### Entity Extraction

Automatically extract entities from memory content:

```python
from entity_extractor import EntityExtractor

extractor = EntityExtractor()

# Extract entities
result = extractor.extract("John Smith works at Microsoft in Seattle.")

# Returns:
{
    "entities": [
        {"text": "John Smith", "type": "PERSON", "priority": 0.8},
        {"text": "Microsoft", "type": "ORG", "priority": 0.9},
        {"text": "Seattle", "type": "GPE", "priority": 0.7}
    ],
    "unique_entities": {"john smith", "microsoft", "seattle"},
    "entity_counts": Counter({"PERSON": 1, "ORG": 1, "GPE": 1})
}
```

Supported entity types:
- **PERSON**: People, including fictional
- **ORG**: Organizations, companies, agencies
- **GPE**: Geopolitical entities (countries, cities, states)
- **DATE**: Absolute or relative dates
- **EMAIL**: Email addresses (regex)
- **URL**: Web URLs (regex)
- **PHONE**: Phone numbers (regex)

### Weight Tuning

Use the weight tuner to find optimal configurations:

```python
from weight_tuner import WeightTuner

tuner = WeightTuner()

# Compare all presets
results = tuner.compare_presets(
    memories=test_memories,
    token_budget=2000,
    ground_truth=["mem-1", "mem-2", "mem-3"]
)

# Grid search for best weights
results = tuner.grid_search(
    memories=test_memories,
    token_budget=2000,
    similarity_range=(0.2, 0.6, 0.1),
    importance_range=(0.2, 0.5, 0.1),
    recency_range=(0.1, 0.4, 0.1),
    top_k=10
)

# Best configuration
best = results[0]
print(f"Score: {best.score:.3f}")
print(f"Weights: {best.config.weights}")
```

## Token Estimation

### Supported Models

- GPT-4 family (gpt-4, gpt-4-turbo, gpt-4o)
- GPT-3.5-turbo
- Claude family (approximated with cl100k_base)
- Custom models via encoding name

### Memory Formatting

Memories are formatted for context as:
```
[2024-01-15] Memory content here (importance: 0.85)
```

Token count includes:
- Timestamp (if available)
- Content
- Importance indicator (if > 0.7)
- Compact metadata (if < 100 chars)

## Performance

- **1000 memories**: <100ms
- **Typical query (100 candidates)**: <20ms
- **Token counting overhead**: ~5ms for 100 memories
- **Memory overhead**: ~1MB per 1000 memories

## Edge Cases

### Single Memory Exceeds Budget
- If first memory and high score (>0.7): Truncate with "..."
- Otherwise: Skip

### All Memories Filtered by Diversity
- At least one memory always selected (highest score)

### Empty Results
- Returns empty array with metadata (utilization: 0.0)

### Invalid Namespace UUID
- Logs warning and continues (no filter applied)

## Testing

Run tests:
```bash
# All prioritization tests
docker exec vex-memory-api-1 pytest tests/test_token_estimator.py tests/test_prioritizer.py tests/test_prioritized_context_api.py -v

# Performance tests
docker exec vex-memory-api-1 pytest tests/test_prioritizer.py::TestPerformance -v

# Integration tests
docker exec vex-memory-api-1 pytest tests/test_prioritized_context_api.py::TestPrioritizedContextIntegration -v
```

## Future Phases

### Phase 2: Advanced Diversity + Entity Coverage (Planned)
- Maximal Marginal Relevance (MMR) for better diversity
- Named entity extraction and coverage tracking
- Type and namespace priority weighting
- Adaptive weight tuning based on historical data

### Phase 3: Adaptive Learning (Planned)
- Usage logging and analytics
- Automatic weight optimization
- Per-user/namespace learned weights
- A/B testing framework

## Migration Guide

### From Simple Query

**Before:**
```python
response = requests.post("/api/query", json={
    "query": "project launch",
    "limit": 10
})
```

**After:**
```python
response = requests.post("/api/memories/prioritized-context", json={
    "query": "project launch",
    "token_budget": 2000  # Specify your LLM's context budget
})
```

### Benefits
- **Token Safety**: Never exceed your LLM's limits
- **Better Relevance**: Multi-factor scoring beats simple similarity
- **Less Redundancy**: Diversity filtering removes duplicates
- **Richer Metadata**: Understand why memories were selected

## Configuration

### Custom Weights

Adjust weights based on your use case:

**Recent events prioritized:**
```json
{
  "weights": {
    "similarity": 0.3,
    "importance": 0.2,
    "recency": 0.5,
    "diversity": 0.0
  }
}
```

**High-importance memories:**
```json
{
  "weights": {
    "similarity": 0.3,
    "importance": 0.6,
    "recency": 0.1,
    "diversity": 0.0
  }
}
```

**Maximum diversity:**
```json
{
  "weights": {
    "similarity": 0.5,
    "importance": 0.3,
    "recency": 0.2,
    "diversity": 0.0
  },
  "diversity_threshold": 0.5  # Lower = more aggressive filtering
}
```

### Recency Half-Life

Modify in `prioritizer.py`:
```python
prioritizer = MemoryPrioritizer(
    recency_half_life_days=14.0  # Faster decay for short-term projects
)
```

## Troubleshooting

### Low Utilization (<50%)
- Increase `limit` to retrieve more candidates
- Lower `min_score` threshold
- Decrease `diversity_threshold` to allow more similar memories

### Too Many Similar Memories
- Increase `diversity_threshold` (e.g., 0.8)
- Increase diversity weight in scoring

### Performance Issues
- Reduce `limit` (fewer candidates to score)
- Check database indexes on `embedding` column
- Monitor token estimation overhead

## Contributing

To add support for new models:
1. Add model mapping in `token_estimator.py` -> `SUPPORTED_MODELS`
2. Test token counting accuracy
3. Update documentation

## License

Same as vex-memory main project.

---

## Phase 3: Adaptive Learning (v2.0.0)

### Overview

Vex Memory v2.0.0 introduces **self-improving memory selection** through usage analytics and automatic weight optimization. The system learns from your query patterns to find the optimal weight configuration for your use case.

### How It Works

1. **Usage Tracking**: Every `POST /api/memories/prioritized-context` call is logged (if enabled)
2. **Pattern Analysis**: System analyzes which weights work best for your queries
3. **Optimization**: Grid search finds optimal weights that maximize diversity + token efficiency
4. **Auto-Tuning**: SDK automatically fetches and uses learned weights

### Architecture

```
Query → API → Prioritizer → Selected Memories
         ↓
    Analytics Logger
         ↓
    query_logs table
         ↓
    Weight Optimizer (periodic)
         ↓
    learned_weights table
         ↓
    SDK Auto-Tuning
```

### Usage Analytics

#### What's Logged

Per query:
- Query text (can be sanitized)
- Weights used
- Memories selected (IDs only)
- Token usage (used / budget)
- Performance (computation time)
- Metadata (search type, thresholds, etc.)

#### Privacy Controls

**Disable logging:**
```bash
USAGE_LOGGING_ENABLED=false
```

**Sanitize queries:**
```bash
SANITIZE_QUERIES=true  # Hashes query text
```

**Configure retention:**
```bash
USAGE_LOG_RETENTION_DAYS=90  # Default: 90 days
```

See [PRIVACY.md](PRIVACY.md) for full details.

#### View Analytics

**API:**
```bash
curl "http://localhost:8000/api/weights/analytics?namespace=my-agent"
```

**SDK:**
```python
client = VexMemoryClient()
summary = client.get_analytics_summary("my-agent")

print(f"Total queries: {summary['total_queries']}")
print(f"Avg token efficiency: {summary['avg_token_efficiency']:.2%}")
print(f"Avg memories retrieved: {summary['avg_memories_retrieved']}")
```

**Output:**
```json
{
  "enabled": true,
  "namespace": "my-agent",
  "total_queries": 234,
  "avg_tokens_used": 3456.7,
  "avg_tokens_budget": 4000.0,
  "avg_token_efficiency": 0.864,
  "avg_memories_retrieved": 12.3,
  "avg_memories_dropped": 8.1,
  "avg_computation_time_ms": 42.5,
  "first_query": "2026-02-15T10:00:00Z",
  "last_query": "2026-03-01T08:00:00Z"
}
```

### Weight Optimization

#### When to Optimize

After **50+ queries** (configurable), the system has enough data to learn optimal weights.

#### How Optimization Works

1. **Data Split**: 80% training, 20% validation
2. **Grid Search**: Tests multiple weight combinations
3. **Evaluation**: Each combination scored on validation set
4. **Objective Function**: `diversity_score + token_efficiency`
   - **Diversity**: Average Jaccard distance between selected memories
   - **Token Efficiency**: `tokens_used / tokens_budget`
5. **Selection**: Best-performing weights are saved

#### Trigger Optimization

**API:**
```bash
curl -X POST http://localhost:8000/api/weights/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "my-agent",
    "min_queries": 50
  }'
```

**SDK:**
```python
client = VexMemoryClient()

result = client.trigger_weight_optimization(
    namespace="my-agent",
    min_queries=50
)

print(f"Best weights: {result['best_weights']}")
print(f"Objective score: {result['objective_score']}")
print(f"Training queries: {result['metadata']['training_queries']}")
```

**Output:**
```json
{
  "weight_id": "123e4567-e89b-12d3-a456-426614174000",
  "history_id": "234e5678-f89c-23e4-b567-537725285111",
  "namespace": "my-agent",
  "best_weights": {
    "similarity": 0.45,
    "importance": 0.35,
    "recency": 0.2
  },
  "objective_score": 1.28,
  "metadata": {
    "training_queries": 120,
    "validation_queries": 30,
    "combinations_tested": 50,
    "computation_time_ms": 1250.5,
    "avg_diversity_score": 0.68,
    "avg_token_efficiency": 0.85
  }
}
```

#### Custom Search Space

Override default search space:

```python
result = client.trigger_weight_optimization(
    namespace="my-agent",
    search_space={
        "similarity": [0.3, 0.4, 0.5, 0.6],
        "importance": [0.2, 0.3, 0.4],
        "recency": [0.1, 0.2, 0.3]
    }
)
```

### Auto-Tuning (SDK)

#### Enable Auto-Tuning

```python
client = VexMemoryClient()

# Enable auto-tuning for a namespace
client.enable_auto_tuning(
    namespace="my-agent",
    refresh_interval=3600  # Refresh weights every hour
)

# Now use build_context() normally
context = client.build_context(
    query="What are the latest deployment strategies?",
    token_budget=4000
)

# Automatically uses learned weights (if available)
# Falls back to defaults if no learned weights exist
```

#### How Auto-Tuning Works

1. **Fetch Learned Weights**: On enable, fetches current learned weights
2. **Background Refresh**: Thread refreshes weights every `refresh_interval` seconds
3. **Automatic Application**: `build_context()` uses learned weights automatically
4. **Manual Override**: Explicitly passing `weights=` always takes precedence

#### Weight Priority

```python
# Priority order:
# 1. Explicit user weights (highest priority)
context = client.build_context("query", weights={"similarity": 0.5, ...})

# 2. Learned weights (if auto-tuning enabled)
client.enable_auto_tuning("my-agent")
context = client.build_context("query")  # Uses learned weights

# 3. Server defaults (lowest priority)
context = client.build_context("query")  # Uses server defaults
```

#### Disable Auto-Tuning

```python
client.disable_auto_tuning()
```

### Complete Workflow Example

```python
from vex_memory import VexMemoryClient

# 1. Initialize client
client = VexMemoryClient()

# 2. Use normally (with default weights)
for i in range(100):
    context = client.build_context(
        query=f"Query {i}",
        token_budget=4000,
        namespace="my-agent"
    )
    # Usage is logged automatically

# 3. After enough queries, trigger optimization
result = client.trigger_weight_optimization("my-agent")
print(f"Learned weights: {result['best_weights']}")

# 4. Enable auto-tuning
client.enable_auto_tuning(namespace="my-agent")

# 5. Continue using - now with optimized weights
context = client.build_context(
    query="Important query",
    token_budget=4000
)
# Automatically uses learned weights

# 6. Check performance improvement
summary = client.get_analytics_summary("my-agent")
print(f"Token efficiency improved to: {summary['avg_token_efficiency']:.2%}")
```

### Performance

- **Logging overhead**: <2ms per query (non-blocking)
- **Optimization time**: <5s for 1000 queries
- **Background refresh**: Negligible CPU/memory
- **No impact on query latency**

### Data Management

#### Export Analytics

```python
# JSON export
data = client.export_analytics("my-agent", format="json")

# CSV export
csv_data = client.export_analytics("my-agent", format="csv")
```

#### Delete Analytics (GDPR)

```python
result = client.delete_analytics("my-agent")
print(f"Deleted {result['deleted_logs']} query logs")
```

### Troubleshooting

**Q: Optimization fails with "Insufficient data"**

A: You need at least 50 queries (by default). Check analytics:
```python
summary = client.get_analytics_summary("my-agent")
print(f"Queries: {summary['total_queries']}")
```

**Q: Auto-tuning not improving results**

A: Check if learned weights exist:
```python
try:
    weights = client.get_learned_weights("my-agent")
    print(f"Learned weights: {weights['weights']}")
except VexMemoryAPIError:
    print("No learned weights - trigger optimization first")
```

**Q: How often should I re-optimize?**

A: Recommendations:
- **Initial**: After 50-100 queries
- **Periodic**: Weekly or after major query pattern changes
- **Automatic**: Future versions will support automatic re-optimization

**Q: Can I use different weights for different query types?**

A: Not yet. Currently, one weight configuration per namespace. Future versions may support query-type-specific weights.

### Best Practices

1. **Let it Learn**: Run 50-100 queries before optimizing
2. **Monitor Performance**: Check analytics summary regularly
3. **Re-optimize**: Re-run optimization after major usage changes
4. **Privacy**: Enable query sanitization if queries contain sensitive data
5. **Retention**: Adjust retention period based on your needs (30-90 days recommended)
6. **Manual Override**: Use explicit weights when you know exactly what you want

### Limitations

- **Minimum Data**: Requires 50+ queries for optimization
- **Namespace-Level**: One weight config per namespace (no per-user yet)
- **Static Weights**: Weights don't adapt in real-time (periodic optimization required)
- **Grid Search Only**: No advanced optimization algorithms yet (planned for v2.1.0)

### Future Enhancements (v2.1.0+)

- **User Feedback**: Thumbs up/down on results to improve optimization
- **Online Learning**: Real-time weight adaptation
- **Query Clustering**: Different weights for different query types
- **Bayesian Optimization**: More efficient weight search
- **A/B Testing**: Compare weight configurations automatically

---

For privacy details, see [PRIVACY.md](PRIVACY.md).
For full changelog, see [CHANGELOG.md](CHANGELOG.md).
