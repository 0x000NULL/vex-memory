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
