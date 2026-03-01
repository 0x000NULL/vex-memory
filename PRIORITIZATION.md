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
