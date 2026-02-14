"""
Vex Memory REST API
===================

FastAPI service exposing memory retrieval, storage, and consolidation endpoints.
"""

import os
import uuid
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from extractor import MemoryExtractor, MemoryNode, MemoryType, EntityType
from consolidator import MemoryConsolidator
from retriever import MemoryRetriever, QueryContext, RetrievalStrategy, ContextWindow
import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vex Memory System",
    description="Graph + vector hybrid memory system for AI agents",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MemoryCreate(BaseModel):
    content: str
    type: str = "semantic"
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "api"
    event_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryOut(BaseModel):
    id: str
    type: str
    content: str
    importance_score: float
    decay_factor: float = 1.0
    access_count: int = 0
    source: Optional[str] = None
    event_time: Optional[datetime] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 4000
    strategy: Optional[str] = None
    conversation_history: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query: str
    memories: List[MemoryOut]
    total_tokens: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityOut(BaseModel):
    id: str
    name: str
    type: str
    canonical_name: Optional[str] = None
    mention_count: int = 0
    importance_score: float = 0.5


class ConsolidateRequest(BaseModel):
    target_date: date
    daily_file_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    database: bool
    memory_count: int = 0


class StatsResponse(BaseModel):
    total_memories: int
    total_entities: int
    memory_types: Dict[str, int]
    entity_types: Dict[str, int]


# ---------------------------------------------------------------------------
# In-memory fallback retriever (used when DB is unavailable)
# ---------------------------------------------------------------------------
_retriever: Optional[MemoryRetriever] = None


def _get_retriever() -> MemoryRetriever:
    global _retriever
    if _retriever is None:
        mem_file = os.environ.get("MEMORY_JSON", "extracted_memories.json")
        _retriever = MemoryRetriever(memory_file=mem_file if os.path.exists(mem_file) else None)
    return _retriever


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    db_ok = db.check_health()
    retriever = _get_retriever()
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database=db_ok,
        memory_count=len(retriever.memories),
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    retriever = _get_retriever()
    raw = retriever.get_retrieval_stats()
    return StatsResponse(
        total_memories=raw.get("total_memories", 0),
        total_entities=raw.get("index_sizes", {}).get("entity_terms", 0),
        memory_types=raw.get("memory_types", {}),
        entity_types={},
    )


@app.post("/memories", response_model=MemoryOut, status_code=201)
def create_memory(body: MemoryCreate):
    """Store a new memory (in-memory + DB if available)."""
    mem_id = str(uuid.uuid4())

    # Try DB first
    try:
        with db.get_cursor() as cur:
            cur.execute(
                """INSERT INTO memory_nodes
                   (id, type, content, importance_score, source, event_time, metadata)
                   VALUES (%s, %s::memory_type, %s, %s, %s, %s, %s::jsonb)
                   RETURNING id, type::text, content, importance_score, decay_factor,
                             access_count, source, event_time, created_at, metadata""",
                (
                    mem_id, body.type, body.content, body.importance_score,
                    body.source, body.event_time,
                    __import__("json").dumps(body.metadata),
                ),
            )
            row = cur.fetchone()

            # Sync to in-memory retriever so queries/stats see it immediately
            node = MemoryNode(
                id=mem_id,
                type=MemoryType(body.type),
                content=body.content,
                importance_score=body.importance_score,
                source=body.source,
                event_time=body.event_time,
                metadata=body.metadata,
            )
            retriever = _get_retriever()
            retriever.memories.append(node)
            retriever._build_indices()

            return MemoryOut(**row)
    except Exception as e:
        logger.warning(f"DB insert failed, using in-memory fallback: {e}")

    # Fallback: add to retriever's in-memory list
    node = MemoryNode(
        id=mem_id,
        type=MemoryType(body.type),
        content=body.content,
        importance_score=body.importance_score,
        source=body.source,
        event_time=body.event_time,
        metadata=body.metadata,
    )
    retriever = _get_retriever()
    retriever.memories.append(node)
    retriever._build_indices()

    return MemoryOut(
        id=mem_id,
        type=body.type,
        content=body.content,
        importance_score=body.importance_score,
        source=body.source,
        event_time=body.event_time,
        metadata=body.metadata,
    )


@app.get("/memories", response_model=List[MemoryOut])
def list_memories(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    type: Optional[str] = None,
    min_importance: float = Query(0.0, ge=0.0, le=1.0),
):
    """List memories with optional filters."""
    retriever = _get_retriever()
    mems = retriever.memories

    if type:
        try:
            mt = MemoryType(type)
            mems = [m for m in mems if m.type == mt]
        except ValueError:
            raise HTTPException(400, f"Invalid memory type: {type}")

    mems = [m for m in mems if m.importance_score >= min_importance]
    mems = sorted(mems, key=lambda m: m.importance_score, reverse=True)

    page = mems[offset: offset + limit]
    return [
        MemoryOut(
            id=m.id, type=m.type.value, content=m.content,
            importance_score=m.importance_score, source=m.source or "",
            event_time=m.event_time,
        )
        for m in page
    ]


@app.get("/memories/{memory_id}", response_model=MemoryOut)
def get_memory(memory_id: str):
    retriever = _get_retriever()
    for m in retriever.memories:
        if m.id == memory_id:
            return MemoryOut(
                id=m.id, type=m.type.value, content=m.content,
                importance_score=m.importance_score, source=m.source or "",
                event_time=m.event_time,
            )
    raise HTTPException(404, "Memory not found")


@app.post("/query", response_model=QueryResponse)
def query_memories(body: QueryRequest):
    """Query the memory system with natural language."""
    retriever = _get_retriever()

    qc = QueryContext(
        query=body.query,
        max_tokens=body.max_tokens,
        conversation_history=body.conversation_history,
    )
    if body.strategy:
        try:
            qc.strategies = [RetrievalStrategy(body.strategy)]
        except ValueError:
            raise HTTPException(400, f"Invalid strategy: {body.strategy}")

    ctx = retriever.query(qc)

    return QueryResponse(
        query=body.query,
        memories=[
            MemoryOut(
                id=m.id, type=m.type.value, content=m.content,
                importance_score=m.importance_score, source=m.source or "",
                event_time=m.event_time,
            )
            for m in ctx.memories
        ],
        total_tokens=ctx.total_tokens,
        metadata=ctx.assembly_metadata,
    )


@app.post("/extract")
def extract_from_text(content: str = Query(..., min_length=10)):
    """Extract structured memories from raw text."""
    extractor = MemoryExtractor()
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        memories = extractor.extract_from_file(tmp)
    finally:
        os.unlink(tmp)

    return {
        "count": len(memories),
        "memories": [
            {
                "id": m.id,
                "type": m.type.value,
                "content": m.content[:200],
                "importance_score": m.importance_score,
                "entities": [{"name": e.name, "type": e.type.value} for e in m.entities],
            }
            for m in memories
        ],
    }


@app.get("/entities", response_model=List[EntityOut])
def list_entities(limit: int = Query(50, ge=1, le=200)):
    """List known entities."""
    # Try DB
    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT id::text, name, type::text, canonical_name, mention_count, importance_score "
                "FROM entities ORDER BY importance_score DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            return [EntityOut(**r) for r in rows]
    except Exception:
        pass

    # Fallback: extract from in-memory data
    retriever = _get_retriever()
    entity_counts: Dict[str, int] = {}
    for m in retriever.memories:
        for e in getattr(m, "entities", []):
            key = getattr(e, "canonical_name", e.name if hasattr(e, "name") else str(e))
            entity_counts[key] = entity_counts.get(key, 0) + 1

    entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [
        EntityOut(id=str(i), name=name, type="other", canonical_name=name, mention_count=count)
        for i, (name, count) in enumerate(entities)
    ]
