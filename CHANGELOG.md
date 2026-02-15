# Changelog

All notable changes to Vex Memory are documented here.

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
