# Changelog

All notable changes to vex-memory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-01

### Added - Smart Context Prioritization (Phase 1)

Major new feature: Intelligent memory prioritization with token-aware selection.

#### Core Modules
- **token_estimator.py**: Accurate token counting using `tiktoken`
  - Support for GPT-4, GPT-3.5-turbo, Claude models
  - Memory formatting and token budget calculation
  - Graceful truncation for oversized memories
  - 23 unit tests

- **prioritizer.py**: Multi-factor memory scoring and selection
  - Weighted scoring: similarity (0.4), importance (0.3), recency (0.2), diversity (0.1)
  - Exponential recency decay (30-day half-life)
  - Diversity filtering via Jaccard similarity
  - Greedy selection algorithm
  - 28 unit tests

#### API Endpoints
- **POST /api/memories/prioritized-context**: New intelligent context retrieval
  - Configurable token budgets (never exceeded)
  - Custom scoring weights
  - Diversity threshold control
  - Minimum score filtering
  - Namespace support
  - 25 integration tests

#### Performance
- <100ms for 1000 memories
- <20ms for typical 100-candidate queries
- Token counting overhead: ~5ms per 100 memories

#### Documentation
- [PRIORITIZATION.md](PRIORITIZATION.md): Complete feature guide
  - API usage examples
  - Scoring algorithm details
  - Configuration options
  - Migration guide
  - Troubleshooting

#### Tests
- 78 total tests added
- All tests passing
- Performance benchmarks included

### Changed
- README.md: Added Smart Context Prioritization feature
- requirements.txt: Added tiktoken>=0.5.0

### Technical Details
- Token budget enforcement prevents LLM context overruns
- Multi-factor scoring improves relevance over simple similarity
- Diversity filtering reduces redundancy
- UUID validation for namespace filters
- Fallback to keyword search if embeddings unavailable

## [0.3.1] - 2026-02-XX

### Previous Release
- (See git history for earlier changes)

## [1.2.0] - 2026-03-01

### Added - Advanced Diversity + Priorities (Phase 2)

Enhanced prioritization with MMR, entity extraction, and configurable priorities.

#### New Modules
- **entity_extractor.py**: Automatic entity extraction from text
  - spaCy NER for people, organizations, locations
  - Regex patterns for emails, URLs, phones, dates
  - Entity type priority mapping
  - Coverage calculation and tracking
  - 24 unit tests

- **weight_tuner.py**: Weight optimization and tuning utilities
  - 6 predefined weight presets (balanced, relevance_focused, etc.)
  - Grid search for optimal configurations
  - Preset comparison and benchmarking
  - Custom evaluation functions
  - 23 unit tests

- **PriorityMappings**: Configurable type and namespace priorities
  - Type priorities: episodic (1.0), semantic (0.8), procedural (0.6), meta (0.4)
  - Namespace priorities: main (1.0), shared (0.7), isolated (0.3)
  - Custom priority configurations
  - 30 unit tests

#### Enhanced prioritizer.py
- **MMR Algorithm**: `prioritize_mmr()` method
  - Lambda parameter for relevance/diversity balance
  - Iterative selection for better diversity
  - Same token guarantees as greedy method
  
- **Entity Coverage**: `prioritize_with_entity_coverage()` method
  - Track entity coverage in selected memories
  - Boost scores for uncovered entities
  - Coverage metrics in response

- **Type/Namespace Priorities**: New scoring factors
  - `_type_priority()` method
  - `_namespace_priority()` method
  - Priority multipliers applied to base scores
  - Configurable mappings

#### API Endpoints
- **POST /api/memories/prioritized-mmr**: MMR-based selection
  - Lambda parameter for diversity control
  - Same parameters as prioritized-context
  - Method indicator in metadata

- **GET /api/weights/presets**: List available weight presets
  - Returns 6 preset configurations
  - Name, key, and description for each

- **GET /api/weights/recommend**: Get recommended weights
  - Query param: use_case (balanced, relevance_focused, etc.)
  - Returns optimized weight configuration
  - Ready to use in prioritization requests

#### Python SDK Updates
- **build_context()**: New parameters
  - `use_mmr=False`: Enable MMR selection
  - `mmr_lambda=0.7`: MMR balance parameter
  - Automatically routes to correct endpoint

- **get_weight_presets()**: Get available presets
- **get_recommended_weights(use_case)**: Get optimized weights
- **Integration tests**: 5 new tests for v1.2.0 features

#### Documentation
- Updated PRIORITIZATION.md with Phase 2 features
- Added MMR algorithm explanation
- Added entity extraction guide
- Added weight tuning examples
- Added type/namespace priority docs

#### Performance
- MMR: <100ms for 1000 memories (same as greedy)
- Entity extraction: ~10ms per memory with spaCy
- Weight tuning: Grid search tests 50+ configs in <5s
- All existing performance targets maintained

#### Tests
- 77 new tests (30 priority weighting + 23 weight tuner + 24 entity extractor)
- All 203 tests passing (126 from Phase 1 + 77 Phase 2)
- Integration tests verify MMR and weight endpoints

### Changed
- prioritizer.py: Added MMR, entity coverage, and priority methods
- api.py: Added MMR endpoint and weight configuration endpoints
- SDK client.py: Added MMR support and weight preset methods
- ScoringWeights: Added `entity_coverage` field (default: 0.05)

### Migration from v1.1.0
- Fully backward compatible
- Existing `POST /api/memories/prioritized-context` unchanged
- New features optional
- Default weights adjusted to include entity_coverage:
  - similarity: 0.4 → 0.4
  - importance: 0.3 → 0.3
  - recency: 0.2 → 0.2
  - diversity: 0.1 → 0.05
  - entity_coverage: 0.0 → 0.05 (new)

## Roadmap

### [2.0.0] - Phase 3: Adaptive Learning (Planned)
- Usage logging and analytics
- Automatic weight optimization
- Per-user/namespace learned weights
- A/B testing framework

---

[1.1.0]: https://github.com/0x000NULL/vex-memory/compare/v0.3.1...v1.1.0
[0.3.1]: https://github.com/0x000NULL/vex-memory/releases/tag/v0.3.1
