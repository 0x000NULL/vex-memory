<p align="center">
  <h1 align="center">🧠 Vex Memory</h1>
  <p align="center"><strong>A human-inspired memory system for AI agents</strong></p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL 16">
    <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
    <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Qdrant-optional-FF6B6B?logo=qdrant&logoColor=white" alt="Qdrant">
  </p>
</p>

  <p align="center">
    🌐 <a href="https://vexmemory.dev"><strong>vexmemory.dev</strong></a> · 
    📦 <a href="https://github.com/0x000NULL/vex-memory">GitHub</a>
  </p>
</p>

---

Most AI memory systems are just vector stores with a retrieval step. Vex Memory is different — it models memory the way humans actually remember things: important memories stay vivid, unused ones fade via **Ebbinghaus-inspired decay curves**, related concepts form **graph relationships** you can traverse, emotional context gets tagged automatically, and a **consolidation engine** periodically merges and summarizes related memories like your brain does during sleep. The result is a memory system that gets *smarter* over time, not just bigger.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🔻 Memory Decay** | Exponential forgetting curves with 30-day half-life. Frequently accessed memories resist decay. Importance scores adjust automatically over time. |
| **🤖 Auto-Extraction** | NLP pipeline (spaCy NER + pattern matching) extracts decisions, events, facts, and learnings from raw conversation text — no LLM needed. |
| **🔍 Deduplication** | Embedding-based similarity detection (cosine > 0.85) prevents redundant memories. Near-duplicates are merged, preserving the richer content. |
| **😴 Sleep Consolidation** | "Sleep cycle" engine clusters semantically similar memories, creates summaries, and lowers importance of originals. Topic-based consolidation groups by entity. Runs on a configurable cron schedule. |
| **🕸️ Graph Relationships** | Apache AGE property graph for memory traversal. Auto-links similar memories (cosine > 0.7). Manual relationship types: CAUSED_BY, PART_OF, RELATED_TO, PRECEDED, CONTRADICTS, SUPPORTS. |
| **📊 Dashboard** | Real-time web dashboard showing memory stats, types, emotions, and recent activity at `localhost:8000/dashboard`. |
| **💭 Emotional Tagging** | Keyword-based sentiment analysis tags memories with dominant emotions (joy, pride, frustration, excitement, concern, relief, curiosity, satisfaction). |
| **🎯 Smart Startup Recall** | Session-start endpoint pulls relevant context from graph + vector search based on the user's first message, so agents wake up with context. |
| **📈 Feedback Loops** | Track which memories are actually *used*, *ignored*, or *corrected*. Importance scores adjust based on observed usefulness over time. |
| **⏰ Temporal Reasoning** | Natural language date parsing ("last Tuesday", "2 weeks ago", "since January"). Timeline queries and change-since endpoints. |
| **💾 Pre-Compaction Dump** | Script to flush important context to the graph DB before LLM context window compaction, preventing memory loss during long sessions. |
| **🔎 Semantic Search (Hybrid)** | Combines pgvector cosine similarity with keyword matching for best-of-both-worlds retrieval. Optional Qdrant integration for dedicated vector search at scale. |
| **⚡ Contradiction Detection** | Automatically identifies memories that conflict with each other via CONTRADICTS graph edges, helping agents resolve inconsistencies. |
| **🎯 Importance Decay** | Memories that aren't accessed gradually lose importance score over time, keeping the most relevant context surfaced. |

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Your Agent    │────▶│   FastAPI REST    │────▶│   PostgreSQL 16  │
│  (OpenClaw /    │◀────│    API (:8000)    │◀────│                  │
│   LangChain /   │     │                  │     │  ┌────────────┐  │
│   any REST)     │     │  • Store/Query   │     │  │  pgvector  │  │
└─────────────────┘     │  • Context       │     │  │ (embeddings│  │
                        │  • Extract       │     │  │  384-dim)  │  │
┌─────────────────┐     │  • Consolidate   │     │  ├────────────┤  │
│     Ollama      │◀────│  • Graph         │     │  │ Apache AGE │  │
│  (embeddings)   │     │  • Feedback      │     │  │  (graph)   │  │
│  all-minilm     │     │  • Dashboard     │     │  └────────────┘  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                │ (optional)
                                ▼
                        ┌──────────────────┐
                        │     Qdrant       │
                        │  (vector search) │
                        │  :6333 / :6334   │
                        │  Hybrid search   │
                        └──────────────────┘
```

### Hybrid pgvector + Qdrant (Optional)

By default, Vex Memory uses **pgvector** for all embedding storage and similarity search — no extra services needed. For large-scale deployments (100K+ memories) or if you want dedicated vector search infrastructure, you can enable **Qdrant** as a secondary vector store:

```bash
# Enable Qdrant in .env
QDRANT_ENABLED=true
QDRANT_URL=http://localhost:6333

# Add Qdrant to your docker-compose override
docker compose --profile qdrant up -d
```

When Qdrant is enabled, memories are dual-written to both pgvector and Qdrant. Queries use Qdrant for vector search and PostgreSQL for graph traversal and filtering, combining the best of both worlds.

## 🚀 Quick Start

```bash
git clone https://github.com/0x000NULL/vex-memory.git
cd vex-memory
cp .env.example .env    # review and customize
docker compose up -d
```

That's it. API is at `http://localhost:8000`, dashboard at `http://localhost:8000/dashboard`.

> **Note:** Ollama runs on the host for GPU access. Install it separately: `curl -fsSL https://ollama.com/install.sh | sh && ollama pull all-minilm`. The system degrades gracefully without it (keyword search instead of semantic search).

## 📡 API Reference

### Health & Stats

```bash
# Health check
curl http://localhost:8000/health

# System statistics
curl http://localhost:8000/stats
```

### Memory CRUD

```bash
# Create a memory
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "Python 3.12 supports improved error messages", "type": "semantic", "importance_score": 0.7}'

# List memories (with filters)
curl "http://localhost:8000/memories?limit=10&type=semantic&min_importance=0.5"

# Get a specific memory
curl http://localhost:8000/memories/{id}

# Update a memory
curl -X PUT http://localhost:8000/memories/{id} \
  -H "Content-Type: application/json" \
  -d '{"importance_score": 0.9}'

# Delete a memory
curl -X DELETE http://localhost:8000/memories/{id}

# Bulk create
curl -X POST http://localhost:8000/memories/bulk \
  -H "Content-Type: application/json" \
  -d '{"memories": [{"content": "Fact 1", "type": "semantic"}, {"content": "Fact 2", "type": "semantic"}]}'
```

### Querying

```bash
# Semantic/keyword query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I deploy with Docker?", "max_tokens": 2000}'

# Extract structured memories from raw text
curl -X POST "http://localhost:8000/extract?content=We+decided+to+migrate+to+PostgreSQL+16+for+better+performance"
```

### Context Endpoints (Agent Integration)

```bash
# Get context for a conversation message
curl -X POST http://localhost:8000/context \
  -H "Content-Type: application/json" \
  -d '{"message": "What did we decide about the database?", "max_tokens": 2000}'

# Session startup — broad context pull
curl -X POST http://localhost:8000/context/session-start \
  -H "Content-Type: application/json" \
  -d '{"first_message": "Good morning, what were we working on?", "max_tokens": 4000}'

# Recent important memories (no query needed)
curl http://localhost:8000/context/recent
```

### Conversation Auto-Extraction

```bash
# Extract and store memories from conversation messages
curl -X POST http://localhost:8000/memories/extract-from-conversation \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "We decided to use FastAPI for the backend"},
      {"role": "assistant", "content": "Good choice. I deployed the initial version to staging."}
    ],
    "min_score": 0.3
  }'
```

### Timeline & Temporal

```bash
# Get memory timeline for a date range
curl "http://localhost:8000/timeline?start=2026-02-01&end=2026-02-14"

# What changed since a date
curl http://localhost:8000/memories/since/2026-02-10
```

### Emotions

```bash
# Get memories by emotion
curl http://localhost:8000/memories/by-emotion/excitement

# Bulk-tag all untagged memories with emotions
curl -X POST http://localhost:8000/memories/tag-emotions
```

### Graph Relationships

```bash
# Create a relationship
curl -X POST http://localhost:8000/graph/link \
  -H "Content-Type: application/json" \
  -d '{"from_memory_id": "uuid1", "to_memory_id": "uuid2", "relationship_type": "CAUSED_BY"}'

# Auto-link similar memories
curl -X POST http://localhost:8000/graph/auto-link

# Traverse from a memory (2 hops)
curl "http://localhost:8000/graph/traverse/{memory_id}?depth=2"

# Find shortest path between memories
curl "http://localhost:8000/graph/path?from_id=uuid1&to_id=uuid2"

# Get all memories about an entity
curl http://localhost:8000/graph/subgraph/PostgreSQL
```

### Feedback & Learning

```bash
# Record that a memory was useful
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"memory_id": "uuid", "feedback_type": "used"}'

# Apply feedback to adjust importance scores
curl -X POST http://localhost:8000/feedback/apply

# View learning statistics
curl http://localhost:8000/feedback/stats
```

### Maintenance

```bash
# Trigger memory consolidation
curl -X POST "http://localhost:8000/memories/consolidate?similarity_threshold=0.75"

# Run deduplication
curl -X POST "http://localhost:8000/memories/deduplicate?threshold=0.9"

# Recalculate decay factors
curl -X POST http://localhost:8000/memories/decay-update

# Backfill embeddings for memories missing them
curl -X POST "http://localhost:8000/memories/backfill-embeddings?limit=100"
```

### Entities

```bash
# List known entities
curl "http://localhost:8000/entities?limit=50"
```

## 📊 Dashboard

Visit `http://localhost:8000/dashboard` to see a real-time overview of your memory system including memory counts, type distribution, emotional breakdown, and recent activity.

## ⚙️ Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `vex` | Database user |
| `POSTGRES_PASSWORD` | `vex_memory_dev` | Database password |
| `POSTGRES_DB` | `vex_memory` | Database name |
| `DATABASE_URL` | `postgresql://vex:vex_memory_dev@db:5432/vex_memory` | Full connection string |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `EMBED_MODEL` | `all-minilm` | Embedding model name |
| `AUTO_EXTRACT_ENABLED` | `false` | Enable auto-extraction on ingest |
| `AUTO_EXTRACT_THRESHOLD` | `0.5` | Minimum score for auto-extracted memories |
| `VEX_ENV` | `docker` | Environment identifier |
| `QDRANT_ENABLED` | `false` | Enable Qdrant as secondary vector store |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `vex_memories` | Qdrant collection name |

## 🔌 Built For

- **[OpenClaw](https://github.com/openclaw)** — AI agent framework with persistent memory
- **LangChain / LlamaIndex** — plug in via REST endpoints
- **Any REST-capable agent** — standard HTTP API, no SDK required

### Integration Example

```python
import requests

# Store a memory
requests.post("http://localhost:8000/memories", json={
    "content": "User prefers dark mode interfaces",
    "type": "semantic",
    "importance_score": 0.7,
    "source": "conversation"
})

# Get context for a new message
ctx = requests.post("http://localhost:8000/context", json={
    "message": "What are the user's UI preferences?",
    "max_tokens": 2000
}).json()

print(ctx["context"])  # Relevant memories formatted as text
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

[MIT](LICENSE) © 2026
