# Vex Memory System - Deployment Log

**Date:** 2026-02-14  
**Status:** ✅ Deployed and verified

## What Was Done

### 1. Fixed Dockerfile.db — Missing Build Dependencies
Added `flex` and `bison` packages required by Apache AGE compilation:
```diff
+ flex \
+ bison \
```

### 2. Fixed Schema — AGE Extension Loading
Apache AGE requires `LOAD 'age'` and `SET search_path = ag_catalog` before calling `create_graph()`. Added to `schema.sql` before the graph creation:
```sql
LOAD 'age';
SET search_path = ag_catalog, public;
SELECT create_graph('memory_graph');
```

### 3. Simplified docker-initdb.sh
The original script tried to `pg_ctl restart` inside the entrypoint init context, which caused the container to exit. The `shared_preload_libraries = 'age'` setting was already handled by the Dockerfile's modification of `postgresql.conf.sample`, making the restart unnecessary.

### 4. Remapped DB Port
Local PostgreSQL was already running on port 5432, so the Docker DB was mapped to **5433:5432** in `docker-compose.yml`.

## Final State

| Service | Status | Port |
|---------|--------|------|
| PostgreSQL + AGE + pgvector | ✅ Healthy | localhost:5433 |
| FastAPI API | ✅ Running | localhost:8000 |

- **Health endpoint:** `GET http://localhost:8000/health` → `{"status":"ok","database":true,"memory_count":7}`
- **API docs:** `http://localhost:8000/docs`
- **Test suite:** 61/61 passed (9.55s)

---

## Split-Brain Bug Fix — 2026-02-14

### Problem
POST /memories wrote to PostgreSQL, but GET /memories, POST /query, and GET /stats read exclusively from the in-memory `MemoryRetriever` (loaded from `extracted_memories.json`). New memories were persisted in DB but invisible to semantic search and queries.

### Fix (`api.py`)
After successful DB insert in `create_memory()`, the new memory is now also appended to the in-memory retriever and indices are rebuilt:

```python
# Sync to in-memory retriever so queries/stats see it immediately
node = MemoryNode(...)
retriever = _get_retriever()
retriever.memories.append(node)
retriever._build_indices()
```

### Verification
1. POSTed test memory ("The capital of Freedonia is Glorptown")
2. Queried via POST /query ("What is the capital of Freedonia?") → returned the memory ✅
3. All 61 tests pass ✅

---

## Connection Details
- **Database URL:** `postgresql://vex:vex_memory_dev@localhost:5433/vex_memory`
- **API Base URL:** `http://localhost:8000`
