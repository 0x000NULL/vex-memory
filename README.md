# Vex Memory System v2.0

A hybrid **PostgreSQL + Apache AGE (graph) + pgvector (embeddings)** memory system for AI agents. It extracts, consolidates, retrieves, and serves structured memories through a REST API.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Markdown   │────▶│  Extractor   │────▶│ Consolidator │
│    Files    │     │  (NLP/spaCy) │     │  (decay/     │
└─────────────┘     └──────────────┘     │   merging)   │
                                         └──────┬───────┘
                                                │
                    ┌──────────────┐     ┌───────▼───────┐
                    │   FastAPI    │◀───▶│  PostgreSQL   │
                    │   REST API   │     │  + AGE graph  │
                    └──────────────┘     │  + pgvector   │
                                         └───────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| `extractor.py` | NLP pipeline: extracts entities, facts, relationships, emotions from text |
| `consolidator.py` | "Sleep cycle": deduplication, conflict detection, Ebbinghaus forgetting curves, memory merging |
| `retriever.py` | Multi-strategy retrieval: keyword, temporal, entity, procedural, associative, semantic |
| `migrate_flat_files.py` | One-time migration from markdown files to structured JSON/DB |
| `api.py` | FastAPI service with CRUD, query, extract, and entity endpoints |
| `db.py` | PostgreSQL connection management |
| `schema.sql` | Full DDL: tables, indexes, AGE graph, pgvector, triggers, views |

## Quick Start

### Docker Compose (recommended)

```bash
docker compose up --build
```

This starts:
- **PostgreSQL 16** with pgvector + Apache AGE on port `5432`
- **FastAPI** on port `8000`

### Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run API (DB optional — falls back to in-memory)
uvicorn api:app --reload

# Run tests
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (DB status, memory count) |
| `GET` | `/stats` | System statistics |
| `POST` | `/memories` | Create a memory |
| `GET` | `/memories` | List memories (filter by type, importance) |
| `GET` | `/memories/{id}` | Get a single memory |
| `POST` | `/query` | Natural-language memory query |
| `POST` | `/extract?content=...` | Extract memories from raw text |
| `GET` | `/entities` | List known entities |

### Example: Query memories

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I deploy with Docker?", "max_tokens": 2000}'
```

### Example: Store a memory

```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "Ethan prefers Python for backend", "type": "semantic", "importance_score": 0.8}'
```

## Memory Types

Based on cognitive psychology:

- **Episodic** — Specific events: "Ethan said X at time Y"
- **Semantic** — Facts: "Python supports async/await"
- **Procedural** — How-tos: "To deploy, run docker compose up"
- **Emotional** — Preferences: "Ethan prefers Linux over macOS"

## Consolidation Algorithm

The consolidator implements a biologically-inspired "sleep cycle":

1. **Deduplication** — Jaccard similarity (threshold 0.85)
2. **Importance scoring** — Multi-factor: recency × entity importance × content richness × type weight
3. **Forgetting curves** — Ebbinghaus-inspired: `retention = e^(-t/S)` where S varies by memory type (semantic: 365d, procedural: 180d, episodic: 90d, emotional: 60d)
4. **Decay resistance** — Access count, importance, and relationship density slow decay
5. **Conflict detection** — Factual contradictions, temporal inconsistencies, preference evolution
6. **Memory merging** — Highly similar memories consolidated into single richer nodes

## Schema Highlights

- **768-dim vectors** (nomic-embed-text-v1.5) with IVFFlat indexes
- **Apache AGE graph** for traversal queries
- **Ebbinghaus decay** via `calculate_current_relevance()` SQL function
- **Entity registry** with canonical names, aliases, and type classification
- **Conflict tracking** table with severity and resolution status

## Testing

```bash
pytest tests/ -v --tb=short
```

Tests cover:
- Extractor: section splitting, classification, entity extraction, importance scoring
- Consolidator: deduplication, similarity, decay curves, conflict detection, merging
- Retriever: intent inference, keyword/entity/procedural search, query pipeline
- API: all endpoints, error handling, CRUD operations

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://vex:vex_memory_dev@localhost:5432/vex_memory` | PostgreSQL connection string |
| `MEMORY_JSON` | `extracted_memories.json` | Fallback memory file when DB unavailable |

## License

Private — part of the Vex/OpenClaw ecosystem.
