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

## Roadmap

### [1.2.0] - Phase 2: Advanced Diversity + Entity Coverage (Planned)
- Maximal Marginal Relevance (MMR) for better diversity
- Named entity extraction and coverage tracking
- Type and namespace priority weighting
- Adaptive weight tuning

### [2.0.0] - Phase 3: Adaptive Learning (Planned)
- Usage logging and analytics
- Automatic weight optimization
- Per-user/namespace learned weights
- A/B testing framework

---

[1.1.0]: https://github.com/0x000NULL/vex-memory/compare/v0.3.1...v1.1.0
[0.3.1]: https://github.com/0x000NULL/vex-memory/releases/tag/v0.3.1
