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
| **🎲 Confidence Scoring** | Distinguishes verified facts (0.9+) from likely assumptions (0.6-0.8) to uncertain inferences (0.3-0.5). Auto-tagged based on linguistic markers ("is" vs "probably" vs "maybe"). Affects retrieval ranking. |

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

## 🤝 Multi-Agent Memory Sharing

Vex Memory supports **namespace-based memory sharing** for multi-agent systems. This eliminates cold starts when spawning sub-agents by granting them read/write access to specific memory namespaces.

### Use Cases

- **Main agent + sub-agents**: Vex spawns sub-agents for tasks (e.g., FIMIL Phase 4 analysis). Sub-agents can access Vex's main namespace for context without duplicating memories.
- **Team collaboration**: Multiple agents working on the same project can share a namespace while maintaining private namespaces for agent-specific context.
- **Permission control**: Grant read-only access to observers, write access to collaborators.

### Quick Example

```bash
# 1. Create a namespace (Vex's main namespace is auto-created as 'vex-main')
curl -X POST http://localhost:8000/namespaces \
  -H "Content-Type: application/json" \
  -d '{
    "name": "project-alpha",
    "owner_agent": "vex",
    "access_policy": {"read": [], "write": []}
  }'

# 2. Grant sub-agent read access
curl -X POST "http://localhost:8000/namespaces/{namespace_id}/grant?grantor_agent=vex" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "sub-agent-123", "permission": "read"}'

# 3. Sub-agent queries memories with access control
curl "http://localhost:8000/memories?agent_id=sub-agent-123&namespace_id={namespace_id}"

# 4. Create a memory in the shared namespace
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Project alpha uses PostgreSQL 16 with pgvector",
    "type": "semantic",
    "namespace_id": "{namespace_id}"
  }'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/namespaces` | POST | Create a new namespace |
| `/namespaces` | GET | List all namespaces (optional `?agent_id=` filter) |
| `/namespaces/{id}` | GET | Get namespace details |
| `/namespaces/{id}/grant` | POST | Grant read/write access to an agent |
| `/namespaces/{id}/revoke` | POST | Revoke access from an agent |
| `/namespaces/{id}/permissions` | GET | Get full permission details |
| `/memories?namespace_id={id}` | GET | List memories in a namespace |
| `/memories?agent_id={id}` | GET | List all memories accessible to an agent |

### Access Control

- **Owner**: Namespace creator. Always has read/write access.
- **Read permission**: Can query memories in the namespace.
- **Write permission**: Can create/update memories in the namespace.
- **Automatic backfill**: Existing memories are placed in the `vex-main` namespace owned by `vex`.

### Database Functions

```sql
-- Check if agent can read a namespace
SELECT can_read_namespace('agent-123', 'namespace-uuid');

-- Check write access
SELECT can_write_namespace('agent-123', 'namespace-uuid');

-- Get all memories accessible to an agent
SELECT * FROM get_agent_memories('agent-123', NULL, 100);

-- Get memories from a specific namespace (with access check)
SELECT * FROM get_agent_memories('agent-123', 'namespace-uuid', 50);
```

### Python Client Example

```python
import httpx

class VexMemoryClient:
    def __init__(self, base_url="http://localhost:8000", agent_id="my-agent"):
        self.base_url = base_url
        self.agent_id = agent_id
    
    def get_shared_context(self, namespace_id):
        """Get all memories from a shared namespace."""
        resp = httpx.get(
            f"{self.base_url}/memories",
            params={"agent_id": self.agent_id, "namespace_id": namespace_id, "limit": 100}
        )
        return resp.json()
    
    def create_shared_memory(self, content, namespace_id, importance=0.5):
        """Create a memory in a shared namespace."""
        resp = httpx.post(
            f"{self.base_url}/memories",
            json={
                "content": content,
                "type": "semantic",
                "importance_score": importance,
                "namespace_id": namespace_id
            }
        )
        return resp.json()

# Usage
client = VexMemoryClient(agent_id="sub-agent-789")
memories = client.get_shared_context(namespace_id="vex-main-namespace-uuid")
```

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

## 🔄 Auto-Sync File Watcher

Vex Memory includes a file watcher daemon that monitors markdown files in `~/.openclaw/workspace/memory/` and automatically syncs new content to the graph database in real-time.

### Features

- **Real-time monitoring** — Detects changes to `.md` files within 1 second
- **Intelligent parsing** — Extracts headers, bullet points, and paragraphs as individual memories
- **Type inference** — Automatically categorizes memories as `semantic`, `episodic`, or `procedural` based on content
- **Importance scoring** — Analyzes keywords to assign appropriate importance scores
- **Deduplication** — Leverages the API's built-in duplicate detection to prevent redundant entries
- **Crash recovery** — Tracks sync state per file to avoid re-syncing content after restarts
- **Graceful error handling** — Logs failures without crashing, auto-restarts via systemd

### Installation

The file watcher is installed as a systemd user service that runs on boot:

```bash
# Install watchdog dependency
pip install watchdog

# Enable and start the service
systemctl --user enable --now vex-memory-sync.service

# Check status
systemctl --user status vex-memory-sync.service

# View logs
journalctl --user -u vex-memory-sync.service -f
```

### How It Works

1. **File Detection** — Monitors `~/.openclaw/workspace/memory/*.md` for modifications
2. **Debouncing** — Waits 500ms after last change to avoid partial writes
3. **Content Hashing** — Calculates SHA-256 hash to detect actual changes vs. metadata updates
4. **Parsing** — Splits markdown into sections based on headers (`#`) and bullet points (`-`, `*`, `•`)
5. **Metadata Enrichment** — Adds source file name and section context to each memory
6. **API Sync** — POSTs memories to `/memories` endpoint with appropriate type and importance
7. **State Tracking** — Saves sync state to `~/.config/vex-memory/sync-state.json`

### Example Workflow

```markdown
# ~/.openclaw/workspace/memory/2026-02-28.md

## Daily Log

- **09:00 AM** - Decided to migrate vex-memory to PostgreSQL 16
- **10:30 AM** - Deployed new API endpoint for graph traversal
- **14:00 PM** - User reported bug in consolidation logic

## Important Notes

- Always run `docker compose down` before schema migrations
- Production API uses 384-dim embeddings from all-minilm model
```

**Result:** Each bullet point is automatically extracted, classified (episodic for events, procedural for processes), scored for importance, and synced to the graph database within 1 second of saving the file.

### Configuration

Environment variables (set in systemd service or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VEX_MEMORY_API` | `http://localhost:8000` | API endpoint URL |

### Sync State

The watcher maintains state at `~/.config/vex-memory/sync-state.json`:

```json
{
  "/home/user/.openclaw/workspace/memory/2026-02-28.md": {
    "content_hash": "a3f2e8...",
    "line_count": 12,
    "last_sync": "2026-02-28T20:10:38.728Z"
  }
}
```

This prevents re-syncing unchanged files and enables crash recovery.

### Logs

- **systemd journal:** `journalctl --user -u vex-memory-sync.service -f`
- **File log:** `/tmp/vex-memory-sync.log`

### Service Management

```bash
# Start
systemctl --user start vex-memory-sync.service

# Stop
systemctl --user stop vex-memory-sync.service

# Restart
systemctl --user restart vex-memory-sync.service

# Disable (stop running on boot)
systemctl --user disable vex-memory-sync.service

# View resource usage
systemctl --user status vex-memory-sync.service
```

The service is resource-limited to 256MB RAM and 20% CPU to prevent runaway usage.

## 🤝 Multi-Agent Memory Namespaces

Vex Memory supports **namespace-based memory sharing** so sub-agents can access your context without cold starts.

### Concepts

- **Namespace:** A logical container for memories with an owner and access policy
- **Owner:** The agent who created the namespace (has full read/write access)
- **Access Policy:** JSONB object with `read` and `write` arrays of agent IDs
- **Default Namespace:** `vex-main` (all existing memories are backfilled here)

### Use Cases

1. **Sub-Agent Context Inheritance**
   - When you spawn a sub-agent for a task, grant it read access to your namespace
   - Sub-agent wakes up with your full context (no manual "read MEMORY.md" step)

2. **Team Memory**
   - Multiple humans + agents share a project namespace
   - Everyone sees the same memory graph

3. **Privacy Boundaries**
   - Work namespace (accessible to work-related agents only)
   - Personal namespace (private to you)

### API Usage

```bash
# Create a new namespace
curl -X POST http://localhost:8000/namespaces \
  -H "Content-Type: application/json" \
  -d '{
    "name": "project-apollo",
    "owner_agent": "vex",
    "access_policy": {"read": ["vex"], "write": ["vex"]}
  }'

# Response: {"namespace_id": "550e8400-...", "name": "project-apollo", ...}

# Grant read access to a sub-agent
curl -X POST http://localhost:8000/namespaces/550e8400-.../grant \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "sub-agent-123",
    "permission": "read",
    "grantor_agent": "vex"
  }'

# Create memory in specific namespace
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Apollo launch date confirmed: March 15, 2026",
    "type": "semantic",
    "namespace_id": "550e8400-..."
  }'

# Query memories filtered by namespace
curl "http://localhost:8000/memories?namespace=550e8400-...&limit=20"

# List all namespaces an agent can access
curl "http://localhost:8000/namespaces?agent_id=vex"

# Revoke access
curl -X POST http://localhost:8000/namespaces/550e8400-.../revoke \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "sub-agent-123",
    "permission": "read",
    "revoker_agent": "vex"
  }'
```

### Access Control Logic

- **Owner:** Full read + write access (always)
- **Read access:** Can query memories, cannot create/modify
- **Write access:** Can create and modify memories (includes read)
- **No access:** Namespace is invisible to the agent

Access checks happen at the database level via `can_read_namespace(agent_id, namespace_id)` and `can_write_namespace(agent_id, namespace_id)` functions.

### Database Functions

```sql
-- Check read access
SELECT can_read_namespace('agent-123', '550e8400-...') AS can_read;

-- Check write access
SELECT can_write_namespace('agent-123', '550e8400-...') AS can_write;

-- Get all memories an agent can access (respects namespaces)
SELECT * FROM get_agent_memories('agent-123', NULL, 100);

-- Get memories from specific namespace
SELECT * FROM get_agent_memories('agent-123', '550e8400-...', 50);
```

### Best Practices

1. **Default to vex-main:** If unsure, use the default namespace
2. **Grant least privilege:** Only grant write access when necessary
3. **Use descriptive names:** `project-apollo`, `personal-notes`, `work-context`
4. **Clean up:** Delete namespaces when projects complete
5. **Audit access:** Regularly review who has access to sensitive namespaces

## 🎲 Memory Confidence Scoring

Vex Memory distinguishes between **verified facts** and **uncertain assumptions** using confidence scores (0.0-1.0):

### Confidence Levels

| Range | Label | Examples |
|-------|-------|----------|
| **0.9-1.0** | High Confidence | "Ethan's birthday is December 20", "Server deployed at 3:00 PM on 2026-02-15" |
| **0.6-0.8** | Medium Confidence | "Ethan probably prefers dark mode", "The API seems to run on port 3000" |
| **0.3-0.5** | Low Confidence | "Maybe the database is on localhost", "Could be a network issue" |

### Auto-Tagging Logic

Confidence is automatically assigned based on:

1. **Linguistic Markers**
   - High: `is`, `are`, `was`, `confirmed`, `verified`, `definitely`
   - Medium: `probably`, `likely`, `seems`, `appears`, `usually`
   - Low: `maybe`, `possibly`, `might`, `could be`, `uncertain`

2. **Memory Type**
   - Episodic (witnessed events): 0.9 base
   - Semantic (facts): 0.8 base
   - Procedural (how-tos): 0.8 base
   - Emotional (interpretations): 0.7 base

3. **Content Quality**
   - Specific dates/numbers: +0.02 boost
   - Proper nouns (3+): +0.03 boost
   - Long detailed content (200+ chars): +0.05 boost
   - Questions or conditionals: -0.2 penalty

4. **Source Metadata**
   - Verified sources: +0.05 boost
   - High importance (0.9+): +0.05 boost
   - Auto-extraction: -0.1 penalty

### Retrieval Impact

Confidence affects retrieval ranking via the formula:

```
score = (relevance × 0.4) + (importance × 0.3) + (confidence × 0.2) + (recency × 0.1)
```

**High-confidence memories are prioritized** when multiple memories match a query, ensuring agents get verified facts before uncertain inferences.

### API Usage

```bash
# Create memory with explicit confidence
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Ethan'\''s birthday is December 20, 1995",
    "type": "semantic",
    "importance_score": 0.8,
    "confidence_score": 0.95
  }'

# Query with minimum confidence filter
curl "http://localhost:8000/memories?min_confidence=0.8&limit=20"

# Backfill confidence scores for existing memories
curl -X POST http://localhost:8000/memories/backfill-confidence
```

### Dashboard Visualization

The dashboard shows:
- **Confidence distribution chart** — histogram of memories across confidence ranges
- **Avg confidence metric** — overall system confidence
- **Per-memory confidence** — color-coded in inspector panel (green=high, yellow=medium, pink=low)

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
