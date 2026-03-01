# Changelog

All notable changes to Vex Memory are documented here.

## [0.3.0] - 2026-02-28

### Added

#### 🤝 Multi-Agent Memory Namespaces
- **Namespace-Based Access Control** — Memories can be organized into namespaces with owner-based permissions
- **Shared Context for Sub-Agents** — Grant read/write access to spawned sub-agents so they inherit your memory context
- **Access Policy Management** — JSONB-based read/write permission arrays per namespace
- **Database Functions** — `can_read_namespace()`, `can_write_namespace()`, and `get_agent_memories()` for efficient access checks
- **API Endpoints**:
  - `POST /namespaces` — Create namespace
  - `GET /namespaces` — List accessible namespaces
  - `GET /namespaces/{id}` — Get namespace details
  - `POST /namespaces/{id}/grant` — Grant access to agent
  - `POST /namespaces/{id}/revoke` — Revoke access
  - `GET /namespaces/{id}/permissions` — View current permissions
- **Default Namespace** — `vex-main` created automatically, existing memories backfilled
- **Migration** — `migrations/add_namespaces.sql` (idempotent, safe to re-run)

#### 🎲 Confidence Scoring System
- **Confidence Scores (0.0-1.0)** — Distinguish verified facts from uncertain assumptions
- **Auto-Tagging Logic** — Linguistic markers analyzed ("is/confirmed" = high, "probably" = medium, "maybe" = low)
- **Memory Type Baselines** — Episodic (0.9), Semantic (0.8), Procedural (0.8), Emotional (0.7)
- **Content Quality Factors**:
  - Specific dates/numbers → +0.02 boost
  - Proper nouns (3+) → +0.03 boost
  - Long detailed content (200+ chars) → +0.05 boost
  - Questions/conditionals → -0.2 penalty
- **Source Metadata Boosts** — Verified sources (+0.05), high importance (+0.05), auto-extraction penalty (-0.1)
- **Retrieval Ranking Updated** — 40% relevance + 30% importance + 20% confidence + 10% recency
- **API Enhancements**:
  - `confidence_score` field in `MemoryCreate` and `MemoryOut` models
  - `min_confidence` query filter
  - `POST /memories/backfill-confidence` — Backfill existing memories
- **Dashboard Visualization** — Confidence distribution histogram, per-memory confidence badges
- **Migration** — `migrations/add_confidence.sql` (idempotent, backfills with heuristics)

#### 📂 Auto-Sync File Watcher (Committed in Earlier Release)
- **Real-Time Sync** — Daemon monitors `~/.openclaw/workspace/memory/*.md` and auto-POSTs to graph DB
- **SHA-256 Deduplication** — Tracks file state to avoid re-syncing unchanged content
- **Intelligent Parsing** — Markdown headers, bullets, paragraphs extracted as separate memories
- **Auto-Inference** — Memory type and importance auto-assigned based on content
- **500ms Debounce** — Waits for complete writes before syncing
- **systemd Service** — Auto-restart, resource-limited (256MB RAM, 20% CPU)
- **Logging** — journalctl + `/tmp/vex-memory-sync.log`

### Improved
- **Schema** — `memory_nodes` now has `confidence_score` and `namespace_id` columns
- **Retriever** — Confidence-weighted ranking in `_calculate_relevance_score()`
- **Documentation** — README sections for namespaces and confidence scoring with full API examples
- **Tests** — Comprehensive test suites added:
  - `tests/test_namespaces.py` — Access control, grant/revoke, namespace queries
  - `tests/test_confidence.py` — Auto-tagging accuracy, linguistic markers, retrieval impact

### Fixed
- **HTTP 201 Handling** — Sync scripts now treat 201 (Created) as success (previously only 200 was accepted)

### Migration Guide
```bash
# Apply v0.3.0 migration (combines namespaces + confidence)
docker exec vex-memory-db-1 psql -U vex -d vex_memory -f /app/migrations/v0.3.0.sql

# OR apply individually (same result)
docker exec vex-memory-db-1 psql -U vex -d vex_memory -f /app/migrations/add_namespaces.sql
docker exec vex-memory-db-1 psql -U vex -d vex_memory -f /app/migrations/add_confidence.sql

# Rebuild API container to include new modules
docker compose build api
docker compose up -d
```

### Breaking Changes
None — all changes are additive and backwards-compatible. Existing memories work without modifications.

---

## [0.2.0] - 2026-02-14

### Added
- **Auto-Extraction Pipeline** — NLP-based extraction of decisions, events, facts, and learnings from raw conversation text without LLM calls
- **Pre-Compaction Dump** — Script to flush important context to the graph DB before LLM context window compaction, preventing memory loss during long sessions (`scripts/pre-compaction-dump.sh`)
- **Smart Startup Recall** — Session-start endpoint (`/context/session-start`) pulls relevant context from graph + vector search based on the user's first message
- **Sleep Consolidation** — Cron-schedulable consolidation engine clusters similar memories and creates summaries, inspired by how the brain consolidates during sleep
- **Emotion Tagging** — Keyword-based sentiment analysis tags memories with dominant emotions (joy, pride, frustration, excitement, concern, relief, curiosity, satisfaction)
- **Importance Decay** — Memories that aren't accessed gradually lose importance over time via Ebbinghaus-inspired decay curves with 30-day half-life
- **Contradiction Detection** — Automatically identifies conflicting memories and creates CONTRADICTS graph edges
- **Semantic Search (Hybrid)** — Combines pgvector cosine similarity with keyword matching; optional Qdrant integration for dedicated vector search at scale
- **Qdrant Integration** — Optional secondary vector store for large-scale deployments (dual-write to pgvector + Qdrant)
- **Conversation Auto-Extraction** — `/memories/extract-from-conversation` endpoint processes multi-turn conversations and extracts structured memories
- **Feedback & Learning** — Track which memories are used/ignored/corrected; importance scores adjust based on observed usefulness
- **Temporal Reasoning** — Natural language date parsing, timeline queries, and change-since endpoints
- **Dashboard** — Real-time web UI at `/dashboard` showing memory stats, types, emotions, entity graphs, and recent activity

### Improved
- Architecture diagram updated with optional Qdrant layer
- README expanded with comprehensive API reference and integration examples
- Docker Compose healthchecks and service dependencies

## [0.1.0] - 2026-02-01

### Added
- Initial release with core memory CRUD, pgvector embeddings, Apache AGE graph
- FastAPI REST API with full CRUD operations
- Docker Compose one-command deployment
- Basic deduplication via embedding similarity
- Entity extraction and graph auto-linking
