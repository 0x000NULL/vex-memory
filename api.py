"""
Vex Memory REST API
===================

FastAPI service exposing memory retrieval, storage, and consolidation endpoints.
Supports optional Ollama embeddings for semantic search via pgvector.
"""

import os
import re
import uuid
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from extractor import MemoryExtractor, MemoryNode, MemoryType, EntityType
from auto_extract import extract_from_messages
from consolidator import MemoryConsolidator
from retriever import MemoryRetriever, QueryContext, RetrievalStrategy, ContextWindow
from emotions import analyze_sentiment, tag_memory_emotion, bulk_tag_emotions, get_emotional_memories
import db
import graph as graph_module
from dedup import find_duplicates, merge_memories, deduplicate_all
import decay
from temporal import parse_temporal_expression, temporal_search as temporal_search_fn, get_timeline as temporal_get_timeline, whats_changed_since
import feedback
import access_control

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-minilm")
AUTO_EXTRACT_ENABLED = os.environ.get("AUTO_EXTRACT_ENABLED", "false").lower() == "true"
AUTO_EXTRACT_THRESHOLD = float(os.environ.get("AUTO_EXTRACT_THRESHOLD", "0.5"))

app = FastAPI(
    title="Vex Memory System",
    description="Graph + vector hybrid memory system for AI agents",
    version="3.0.0",
)

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")

@app.get("/dashboard")
async def dashboard():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

app.mount("/dashboard-static", StaticFiles(directory=DASHBOARD_DIR), name="dashboard-static")

# Rewrite CSS/JS paths to use mounted static dir — alternatively serve them directly:
@app.get("/style.css")
async def dashboard_css():
    return FileResponse(os.path.join(DASHBOARD_DIR, "style.css"))

@app.get("/app.js")
async def dashboard_js():
    return FileResponse(os.path.join(DASHBOARD_DIR, "app.js"), media_type="application/javascript")


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

async def _get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Ollama. Returns None on failure (non-blocking)."""
    # Truncate if needed (Ollama model input limit)
    MAX_EMBED_LENGTH = 8000
    original_length = len(text)
    if original_length > MAX_EMBED_LENGTH:
        logger.warning(
            f"Content truncated from {original_length} to {MAX_EMBED_LENGTH} chars for embedding generation"
        )
        text = text[:MAX_EMBED_LENGTH]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/embeddings", json={
                "model": EMBED_MODEL,
                "prompt": text,
            })
            if r.status_code == 200:
                return r.json().get("embedding")
            logger.warning(f"Ollama returned {r.status_code}")
    except Exception as e:
        logger.warning(f"Embedding generation failed (Ollama may be unavailable): {e}")
    return None


def _get_embedding_sync(text: str) -> Optional[List[float]]:
    """Synchronous embedding helper for non-async contexts."""
    # Truncate if needed (Ollama model input limit)
    MAX_EMBED_LENGTH = 8000
    original_length = len(text)
    if original_length > MAX_EMBED_LENGTH:
        logger.warning(
            f"Content truncated from {original_length} to {MAX_EMBED_LENGTH} chars for embedding generation"
        )
        text = text[:MAX_EMBED_LENGTH]
    
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{OLLAMA_URL}/api/embeddings", json={
                "model": EMBED_MODEL,
                "prompt": text,
            })
            if r.status_code == 200:
                return r.json().get("embedding")
    except Exception as e:
        logger.warning(f"Sync embedding failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MemoryCreate(BaseModel):
    content: str
    type: str = "semantic"
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: str = "api"
    event_time: Optional[datetime] = None
    namespace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryOut(BaseModel):
    id: str
    type: str
    content: str
    importance_score: float
    confidence_score: float = 0.8
    decay_factor: float = 1.0
    access_count: int = 0
    source: Optional[str] = None
    event_time: Optional[datetime] = None
    created_at: Optional[datetime] = None
    namespace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 4000
    strategy: Optional[str] = None
    conversation_history: List[str] = Field(default_factory=list)


class BulkMemoryCreate(BaseModel):
    memories: List[MemoryCreate]


class BulkMemoryResponse(BaseModel):
    created: int
    failed: int
    ids: List[str]


class ConversationMessage(BaseModel):
    role: str = "user"
    content: str


class ConversationExtractRequest(BaseModel):
    messages: List[ConversationMessage]
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    source: str = "conversation-extraction"


class ConversationExtractResponse(BaseModel):
    extracted: int
    stored: int
    skipped_duplicates: int
    memories: List[MemoryOut]


class ExtractAndStoreRequest(BaseModel):
    text: str
    source: str = "auto-extraction"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractAndStoreResponse(BaseModel):
    extracted: int
    stored: int
    memories: List[MemoryOut]


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


# Namespace models
class NamespaceCreate(BaseModel):
    name: str
    owner_agent: str
    access_policy: Optional[Dict[str, List[str]]] = Field(default_factory=lambda: {"read": [], "write": []})


class NamespaceOut(BaseModel):
    namespace_id: str
    name: str
    owner_agent: Optional[str]
    access_policy: Dict[str, List[str]]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class GrantAccessRequest(BaseModel):
    agent_id: str
    permission: str  # 'read' or 'write'


class GrantAccessResponse(BaseModel):
    namespace_id: str
    name: str
    owner_agent: Optional[str]
    access_policy: Dict[str, List[str]]


# ---------------------------------------------------------------------------
# In-memory fallback retriever (used when DB is unavailable)
# ---------------------------------------------------------------------------
_retriever: Optional[MemoryRetriever] = None


def _get_retriever() -> MemoryRetriever:
    global _retriever
    if _retriever is None:
        mem_file = os.environ.get("MEMORY_JSON", "extracted_memories.json")
        _retriever = MemoryRetriever(memory_file=mem_file if os.path.exists(mem_file) else None)
        # If retriever is empty but DB has memories, load from DB
        if len(_retriever.memories) == 0 and db.check_health():
            _load_retriever_from_db(_retriever)
    return _retriever


def _load_retriever_from_db(retriever: MemoryRetriever):
    """Populate in-memory retriever from PostgreSQL."""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id::text, content, type::text, importance_score, source,
                           created_at
                    FROM memory_nodes ORDER BY created_at DESC
                """)
                rows = cur.fetchall()
        if not rows:
            return
        for row in rows:
            mid = str(row['id'])
            content = row['content']
            mtype = row['type']
            importance = row['importance_score']
            source = row['source']
            created = row['created_at']
            try:
                mt = MemoryType(mtype) if mtype else MemoryType.SEMANTIC
            except ValueError:
                mt = MemoryType.SEMANTIC
            mem = MemoryNode(
                id=mid,
                content=content or "",
                type=mt,
                importance_score=float(importance or 0.5),
                source=source or "database",
                event_time=created,
            )
            retriever.memories.append(mem)
        retriever._build_indices()
        logger.info(f"Loaded {len(rows)} memories from database into retriever")
    except Exception as e:
        logger.error(f"Failed to load memories from DB: {e}")


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
    """Store a new memory (in-memory + DB if available). Auto-generates embedding via Ollama."""
    
    from confidence import assign_confidence

    # --- Duplicate detection ---
    try:
        duplicates = find_duplicates(body.content, threshold=0.85)
        if duplicates:
            top = duplicates[0]
            # Merge any duplicate above threshold (was >0.9, now >0.85 to catch more cases)
            if top["similarity"] > 0.85:
                logger.info(
                    f"Duplicate detected (similarity={top['similarity']:.3f}), "
                    f"merging with existing memory {top['id']}"
                )
                # Update existing memory: keep higher importance, longer content, merge metadata
                import json as _json
                
                # Calculate merged confidence (average of old and new)
                from confidence import assign_confidence
                new_confidence = body.confidence_score
                if new_confidence is None:
                    metadata_for_scoring = body.metadata.copy()
                    metadata_for_scoring["importance_score"] = body.importance_score
                    metadata_for_scoring["source"] = body.source
                    new_confidence = assign_confidence(body.content, body.type, metadata_for_scoring)
                
                with db.get_cursor() as cur:
                    # Get existing memory's confidence and access count
                    cur.execute(
                        "SELECT confidence_score, access_count FROM memory_nodes WHERE id = %s",
                        (top["id"],)
                    )
                    existing = cur.fetchone()
                    if existing:
                        avg_confidence = (existing["confidence_score"] + new_confidence) / 2.0
                        new_access_count = existing["access_count"] + 1
                    else:
                        avg_confidence = new_confidence
                        new_access_count = 1
                    
                    cur.execute(
                        """UPDATE memory_nodes
                           SET importance_score = GREATEST(importance_score, %s),
                               confidence_score = %s,
                               access_count = %s,
                               content = CASE WHEN LENGTH(content) >= LENGTH(%s) THEN content ELSE %s END,
                               metadata = metadata || %s::jsonb
                           WHERE id = %s
                           RETURNING id, type::text, content, importance_score, confidence_score, decay_factor,
                                     access_count, source, event_time, created_at, namespace_id::text, metadata""",
                        (body.importance_score, avg_confidence, new_access_count,
                         body.content, body.content,
                         _json.dumps({"merge_source": body.source or "api", "merged_at": datetime.now().isoformat()}),
                         top["id"]),
                    )
                    row = cur.fetchone()
                    if row:
                        logger.info(f"Successfully merged into memory {top['id']}")
                        return MemoryOut(**row)
                    else:
                        logger.warning(f"Merge UPDATE returned no rows for {top['id']}, creating new memory instead")
    except Exception as e:
        logger.warning(f"Duplicate check failed, proceeding with insert: {e}")

    mem_id = str(uuid.uuid4())
    
    # Auto-assign confidence score if not provided
    if body.confidence_score is None:
        metadata_for_scoring = body.metadata.copy()
        metadata_for_scoring["importance_score"] = body.importance_score
        metadata_for_scoring["source"] = body.source
        confidence = assign_confidence(body.content, body.type, metadata_for_scoring)
    else:
        confidence = body.confidence_score

    # Track truncation for large content
    MAX_EMBED_LENGTH = 8000
    content_metadata = body.metadata.copy()
    if len(body.content) > MAX_EMBED_LENGTH:
        content_metadata["truncated"] = True
        content_metadata["original_length"] = len(body.content)

    # Try to generate embedding (best-effort, non-blocking on failure)
    embedding = _get_embedding_sync(body.content)

    # Try DB first
    try:
        import json as _json
        with db.get_cursor() as cur:
            if embedding:
                cur.execute(
                    """INSERT INTO memory_nodes
                       (id, type, content, importance_score, confidence_score, source, event_time, namespace_id, metadata, embedding)
                       VALUES (%s, %s::memory_type, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                       RETURNING id, type::text, content, importance_score, confidence_score, decay_factor,
                                 access_count, source, event_time, created_at, namespace_id::text, metadata""",
                    (
                        mem_id, body.type, body.content, body.importance_score, confidence,
                        body.source, body.event_time, body.namespace_id,
                        _json.dumps(content_metadata),
                        str(embedding),
                    ),
                )
            else:
                cur.execute(
                    """INSERT INTO memory_nodes
                       (id, type, content, importance_score, confidence_score, source, event_time, namespace_id, metadata)
                       VALUES (%s, %s::memory_type, %s, %s, %s, %s, %s, %s, %s::jsonb)
                       RETURNING id, type::text, content, importance_score, confidence_score, decay_factor,
                                 access_count, source, event_time, created_at, namespace_id::text, metadata""",
                    (
                        mem_id, body.type, body.content, body.importance_score, confidence,
                        body.source, body.event_time, body.namespace_id,
                        _json.dumps(content_metadata),
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
                metadata=content_metadata,
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
        metadata=content_metadata,
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
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    namespace_id: Optional[str] = Query(None, description="Filter by namespace"),
    agent_id: Optional[str] = Query(None, description="Filter by agent access")
):
    """List memories with optional filters including namespace and access control."""
    
    # Try DB query first for better filtering
    if min_confidence is not None or type or min_importance > 0.0 or namespace_id or agent_id:
        try:
            with db.get_cursor() as cur:
                filters = []
                params = []
                
                if type:
                    filters.append("type = %s::memory_type")
                    params.append(type)
                if min_importance > 0.0:
                    filters.append("importance_score >= %s")
                    params.append(min_importance)
                if min_confidence is not None:
                    # Use ROUND to avoid floating-point precision issues with REAL type
                    filters.append("ROUND(confidence_score::numeric, 2) >= %s")
                    params.append(min_confidence)
                if namespace_id:
                    filters.append("namespace_id = %s")
                    params.append(namespace_id)
                
                # Add access control filtering if agent_id is provided
                if agent_id:
                    filters.append("(namespace_id IS NULL OR can_read_namespace(%s, namespace_id))")
                    params.append(agent_id)
                
                where_clause = " AND ".join(filters) if filters else "TRUE"
                params.extend([limit, offset])
                
                cur.execute(f"""
                    SELECT id::text, type::text, content, importance_score, confidence_score,
                           decay_factor, access_count, source, event_time, created_at, namespace_id::text, metadata
                    FROM memory_nodes
                    WHERE {where_clause}
                    ORDER BY importance_score DESC, confidence_score DESC
                    LIMIT %s OFFSET %s
                """, tuple(params))
                rows = cur.fetchall()
                return [MemoryOut(**r) for r in rows]
        except Exception as e:
            logger.warning(f"DB query failed, using in-memory fallback: {e}")
    
    # Fallback to in-memory
    retriever = _get_retriever()
    mems = retriever.memories

    if type:
        try:
            mt = MemoryType(type)
            mems = [m for m in mems if m.type == mt]
        except ValueError:
            raise HTTPException(400, f"Invalid memory type: {type}")

    mems = [m for m in mems if m.importance_score >= min_importance]
    
    # Note: in-memory memories may not have confidence_score
    # Round to 2 decimal places to match DB precision and avoid floating-point issues
    if min_confidence is not None:
        mems = [m for m in mems if round(getattr(m, 'confidence_score', 0.8), 2) >= min_confidence]
    
    mems = sorted(mems, key=lambda m: m.importance_score, reverse=True)

    page = mems[offset: offset + limit]
    return [
        MemoryOut(
            id=m.id, type=m.type.value, content=m.content,
            importance_score=m.importance_score, 
            confidence_score=getattr(m, 'confidence_score', 0.8),
            source=m.source or "",
            event_time=m.event_time,
        )
        for m in page
    ]


@app.get("/memories/by-emotion/{emotion}", response_model=List[MemoryOut])
def api_get_memories_by_emotion(emotion: str, limit: int = Query(50, ge=1, le=200)):
    """Get memories tagged with a specific dominant emotion."""
    valid_emotions = {"joy", "pride", "frustration", "excitement", "concern", "relief", "curiosity", "satisfaction", "neutral"}
    if emotion not in valid_emotions:
        raise HTTPException(400, f"Invalid emotion: {emotion}. Valid: {', '.join(sorted(valid_emotions))}")
    rows = get_emotional_memories(emotion, limit=limit)
    return [MemoryOut(**r) for r in rows]


# ---------------------------------------------------------------------------
# Conversation Auto-Extraction
# ---------------------------------------------------------------------------

@app.post("/memories/extract-from-conversation", response_model=ConversationExtractResponse)
def extract_from_conversation(body: ConversationExtractRequest):
    """Extract and store memories from conversation messages.

    Uses spaCy NER + pattern matching to identify decisions, events,
    learnings, and facts. Each extracted memory is scored for importance.
    Duplicate detection via embedding similarity when available.
    """
    import json as _json

    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    candidates = extract_from_messages(messages, min_score=body.min_score)

    stored = []
    skipped = 0
    retriever = _get_retriever()

    for mem in candidates:
        # Duplicate detection via embedding similarity
        embedding = _get_embedding_sync(mem["content"])
        if embedding:
            try:
                with db.get_cursor() as cur:
                    cur.execute(
                        """SELECT id::text, content,
                                  1.0 - (embedding <=> %s::vector) AS similarity
                           FROM memory_nodes
                           WHERE embedding IS NOT NULL
                           ORDER BY embedding <=> %s::vector
                           LIMIT 1""",
                        (str(embedding), str(embedding)),
                    )
                    row = cur.fetchone()
                    if row and row.get("similarity", 0) > 0.92:
                        skipped += 1
                        logger.info(f"Skipped duplicate (sim={row['similarity']:.3f}): {mem['content'][:60]}")
                        continue
            except Exception as e:
                logger.warning(f"Duplicate check failed, storing anyway: {e}")

        mem_id = str(uuid.uuid4())
        source = body.source
        metadata = mem.get("metadata", {})

        try:
            with db.get_cursor() as cur:
                if embedding:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, metadata, embedding)
                           VALUES (%s, %s::memory_type, %s, %s, %s, %s::jsonb, %s::vector)
                           RETURNING id, type::text, content, importance_score, decay_factor,
                                     access_count, source, event_time, created_at, metadata""",
                        (mem_id, mem["type"], mem["content"], mem["importance_score"],
                         source, _json.dumps(metadata), str(embedding)),
                    )
                else:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, metadata)
                           VALUES (%s, %s::memory_type, %s, %s, %s, %s::jsonb)
                           RETURNING id, type::text, content, importance_score, decay_factor,
                                     access_count, source, event_time, created_at, metadata""",
                        (mem_id, mem["type"], mem["content"], mem["importance_score"],
                         source, _json.dumps(metadata)),
                    )
                row = cur.fetchone()
                stored.append(MemoryOut(**row))

                node = MemoryNode(
                    id=mem_id, type=MemoryType(mem["type"]), content=mem["content"],
                    importance_score=mem["importance_score"], source=source,
                    metadata=metadata,
                )
                retriever.memories.append(node)
        except Exception as e:
            logger.warning(f"Conversation extract insert failed, trying in-memory: {e}")
            node = MemoryNode(
                id=mem_id, type=MemoryType(mem["type"]), content=mem["content"],
                importance_score=mem["importance_score"], source=source,
                metadata=metadata,
            )
            retriever.memories.append(node)
            stored.append(MemoryOut(
                id=mem_id, type=mem["type"], content=mem["content"],
                importance_score=mem["importance_score"], source=source,
                metadata=metadata,
            ))

    if stored:
        retriever._build_indices()

    return ConversationExtractResponse(
        extracted=len(candidates),
        stored=len(stored),
        skipped_duplicates=skipped,
        memories=stored,
    )


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
    """Query the memory system with natural language. Uses semantic search via pgvector when available."""
    # Try semantic search via DB first
    semantic_results = []
    query_embedding = _get_embedding_sync(body.query)
    if query_embedding:
        try:
            with db.get_cursor() as cur:
                cur.execute(
                    """SELECT id::text, type::text, content, importance_score, decay_factor,
                              access_count, source, event_time, created_at, metadata,
                              1.0 - (embedding <=> %s::vector) AS similarity
                       FROM memory_nodes
                       WHERE embedding IS NOT NULL
                       ORDER BY embedding <=> %s::vector
                       LIMIT 20""",
                    (str(query_embedding), str(query_embedding)),
                )
                rows = cur.fetchall()
                for row in rows:
                    # Lower threshold to 0.2 to catch more edge cases
                    # (was 0.3, reduced to prevent 0-result queries)
                    if row.get("similarity", 0) >= 0.2:
                        semantic_results.append(row)
        except Exception as e:
            logger.warning(f"Semantic DB search failed: {e}")

    if semantic_results:
        # Use semantic results directly
        memories = [
            MemoryOut(
                id=r["id"], type=r["type"], content=r["content"],
                importance_score=r["importance_score"],
                decay_factor=r.get("decay_factor", 1.0),
                access_count=r.get("access_count", 0),
                source=r.get("source", ""),
                event_time=r.get("event_time"),
                created_at=r.get("created_at"),
                metadata=r.get("metadata", {}),
            )
            for r in semantic_results
        ]
        # Trim to token budget
        total_chars = 0
        trimmed = []
        for m in memories:
            total_chars += len(m.content)
            if total_chars // 4 > body.max_tokens:
                break
            trimmed.append(m)

        # If semantic search returned 0 results, log and fall back to keyword
        if not trimmed:
            logger.warning(
                f"Semantic search returned 0 results for query: '{body.query[:100]}...' "
                f"(found {len(semantic_results)} candidates but none met token budget). "
                f"Falling back to keyword search."
            )
        else:
            # Track access for returned memories
            for m in trimmed:
                decay.update_access(m.id)

            return QueryResponse(
                query=body.query,
                memories=trimmed,
                total_tokens=total_chars // 4,
                metadata={"search_type": "semantic", "total_candidates": len(semantic_results)},
            )

    # Fall back to keyword-based retrieval
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

    # Log warning if no results found
    if not ctx.memories:
        logger.warning(
            f"Query returned 0 results for: '{body.query[:100]}...' "
            f"(search_type: keyword, max_tokens: {body.max_tokens})"
        )

    # Track access for returned memories
    for m in ctx.memories:
        decay.update_access(m.id)

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
        metadata={**ctx.assembly_metadata, "search_type": "keyword"},
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


# ---------------------------------------------------------------------------
# Phase 2 Endpoints
# ---------------------------------------------------------------------------

@app.post("/memories/bulk", response_model=BulkMemoryResponse, status_code=201)
def bulk_create_memories(body: BulkMemoryCreate):
    """Bulk-create memories for efficient batch writes (e.g., pre-compaction dumps)."""
    import json as _json
    created = 0
    failed = 0
    ids = []

    for mem in body.memories:
        mem_id = str(uuid.uuid4())
        embedding = _get_embedding_sync(mem.content)
        try:
            with db.get_cursor() as cur:
                if embedding:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, event_time, metadata, embedding)
                           VALUES (%s, %s::memory_type, %s, %s, %s, %s, %s::jsonb, %s::vector)""",
                        (mem_id, mem.type, mem.content, mem.importance_score,
                         mem.source, mem.event_time, _json.dumps(mem.metadata), str(embedding)),
                    )
                else:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, event_time, metadata)
                           VALUES (%s, %s::memory_type, %s, %s, %s, %s, %s::jsonb)""",
                        (mem_id, mem.type, mem.content, mem.importance_score,
                         mem.source, mem.event_time, _json.dumps(mem.metadata)),
                    )
            # Sync to in-memory
            node = MemoryNode(
                id=mem_id, type=MemoryType(mem.type), content=mem.content,
                importance_score=mem.importance_score, source=mem.source,
                event_time=mem.event_time, metadata=mem.metadata,
            )
            retriever = _get_retriever()
            retriever.memories.append(node)
            created += 1
            ids.append(mem_id)
        except Exception as e:
            logger.warning(f"Bulk insert failed for one memory: {e}")
            failed += 1

    if created > 0:
        _get_retriever()._build_indices()

    return BulkMemoryResponse(created=created, failed=failed, ids=ids)


@app.post("/memories/extract-and-store", response_model=ExtractAndStoreResponse)
def extract_and_store(body: ExtractAndStoreRequest):
    """Extract key facts from raw text and store as memories.

    Uses heuristic extraction: sentences containing names, numbers, decisions,
    dates, or technical terms are kept. Controlled by threshold parameter.
    Source is tagged as 'auto-extraction' for quality tracking.

    Disabled by default (AUTO_EXTRACT_ENABLED env var). Returns empty when off
    unless memory count > 200 and env var is set.
    """
    # Check if auto-extraction is enabled
    retriever = _get_retriever()
    mem_count = len(retriever.memories)
    if not AUTO_EXTRACT_ENABLED and mem_count <= 200:
        return ExtractAndStoreResponse(extracted=0, stored=0, memories=[])

    threshold = body.threshold

    # Heuristic extraction: split into sentences, score each
    sentences = re.split(r'(?<=[.!?])\s+', body.text)
    candidates = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        score = _score_sentence(sent)
        if score >= threshold:
            candidates.append((sent, score))

    # Store qualifying sentences
    stored_memories = []
    for content, score in candidates:
        importance = min(1.0, score)
        mem_id = str(uuid.uuid4())
        embedding = _get_embedding_sync(content)
        try:
            import json as _json
            with db.get_cursor() as cur:
                if embedding:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, metadata, embedding)
                           VALUES (%s, 'semantic'::memory_type, %s, %s, %s, %s::jsonb, %s::vector)
                           RETURNING id, type::text, content, importance_score, decay_factor,
                                     access_count, source, event_time, created_at, metadata""",
                        (mem_id, content, importance, body.source,
                         _json.dumps({"extraction_score": score}), str(embedding)),
                    )
                else:
                    cur.execute(
                        """INSERT INTO memory_nodes
                           (id, type, content, importance_score, source, metadata)
                           VALUES (%s, 'semantic'::memory_type, %s, %s, %s, %s::jsonb)
                           RETURNING id, type::text, content, importance_score, decay_factor,
                                     access_count, source, event_time, created_at, metadata""",
                        (mem_id, content, importance, body.source,
                         _json.dumps({"extraction_score": score})),
                    )
                row = cur.fetchone()
                stored_memories.append(MemoryOut(**row))

                # Sync to retriever
                node = MemoryNode(
                    id=mem_id, type=MemoryType.SEMANTIC, content=content,
                    importance_score=importance, source=body.source,
                )
                retriever.memories.append(node)
        except Exception as e:
            logger.warning(f"Extract-and-store insert failed: {e}")

    if stored_memories:
        retriever._build_indices()

    return ExtractAndStoreResponse(
        extracted=len(candidates),
        stored=len(stored_memories),
        memories=stored_memories,
    )


def _score_sentence(sentence: str) -> float:
    """Score a sentence for memory-worthiness using heuristics."""
    score = 0.0
    s = sentence.lower()

    # Contains proper nouns (capitalized words not at start)
    proper_nouns = re.findall(r'(?<!^)(?<!\. )[A-Z][a-z]+', sentence)
    if proper_nouns:
        score += 0.3

    # Contains numbers/dates
    if re.search(r'\d{4}[-/]\d{2}[-/]\d{2}|\d+\.\d+|#\d+|\$[\d,]+', sentence):
        score += 0.2

    # Decision/action words
    decision_words = ['decided', 'chose', 'will', 'plan', 'deployed', 'shipped',
                      'completed', 'fixed', 'built', 'created', 'launched', 'migrated',
                      'implemented', 'configured', 'installed', 'updated']
    if any(w in s for w in decision_words):
        score += 0.3

    # Technical terms
    tech_terms = ['api', 'database', 'server', 'deploy', 'docker', 'git', 'python',
                  'postgresql', 'endpoint', 'config', 'cron', 'ssh', 'dns']
    if any(w in s for w in tech_terms):
        score += 0.1

    # Length bonus (longer = more info)
    if len(sentence) > 100:
        score += 0.1

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Context Endpoints
# ---------------------------------------------------------------------------

class ContextRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    max_tokens: int = 2000


class ContextResponse(BaseModel):
    context: str
    memories_used: List[MemoryOut]
    total_tokens: int


class SessionStartRequest(BaseModel):
    first_message: str
    max_tokens: int = 4000


@app.post("/context", response_model=ContextResponse)
def get_context(body: ContextRequest):
    """Get relevant memory context for a conversation message."""
    retriever = _get_retriever()

    # Convert history dicts to flat strings for the retriever
    history_strings = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in body.history]

    ctx = retriever.get_conversation_context(
        current_message=body.message,
        conversation_history=history_strings,
        max_tokens=body.max_tokens,
    )

    formatted = ctx.get_formatted_context(max_tokens=body.max_tokens)
    memories_out = [
        MemoryOut(
            id=m.id, type=m.type.value, content=m.content,
            importance_score=m.importance_score, source=m.source or "",
            event_time=m.event_time,
        )
        for m in ctx.memories
    ]

    return ContextResponse(
        context=formatted,
        memories_used=memories_out,
        total_tokens=ctx.total_tokens,
    )


@app.post("/context/session-start", response_model=ContextResponse)
def session_start_context(body: SessionStartRequest):
    """Broader context pull for session startup: recent important memories + query-relevant ones."""
    retriever = _get_retriever()

    # Get recent important memories
    recent_important = retriever.get_recent_important_memories(days=7, max_count=10)

    # Get query-relevant memories
    ctx = retriever.get_conversation_context(
        current_message=body.first_message,
        max_tokens=body.max_tokens,
    )

    # Merge: recent important first, then query results (deduplicated)
    seen_ids = set()
    merged_memories = []
    merged_scores = []

    for m in recent_important:
        if m.id not in seen_ids:
            merged_memories.append(m)
            merged_scores.append(m.importance_score)
            seen_ids.add(m.id)

    for i, m in enumerate(ctx.memories):
        if m.id not in seen_ids:
            merged_memories.append(m)
            score = ctx.relevance_scores[i] if i < len(ctx.relevance_scores) else 0.5
            merged_scores.append(score)
            seen_ids.add(m.id)

    # Truncate to token budget
    total_tokens = 0
    final_memories = []
    for m in merged_memories:
        t = len(m.content) // 4
        if total_tokens + t > body.max_tokens:
            break
        final_memories.append(m)
        total_tokens += t

    # Format context
    parts = []
    for m in final_memories:
        timestamp = ""
        if m.event_time:
            timestamp = f" ({m.event_time.strftime('%Y-%m-%d')})"
        parts.append(f"[{m.type.value.upper()}{timestamp}]\n{m.content}")

    memories_out = [
        MemoryOut(
            id=m.id, type=m.type.value, content=m.content,
            importance_score=m.importance_score, source=m.source or "",
            event_time=m.event_time,
        )
        for m in final_memories
    ]

    return ContextResponse(
        context="\n\n".join(parts),
        memories_used=memories_out,
        total_tokens=total_tokens,
    )


@app.get("/context/recent", response_model=ContextResponse)
def recent_context():
    """Return last 7 days of important memories formatted as context (no message needed)."""
    retriever = _get_retriever()
    recent = retriever.get_recent_important_memories(days=7, max_count=20)

    parts = []
    total_tokens = 0
    for m in recent:
        timestamp = ""
        if m.event_time:
            timestamp = f" ({m.event_time.strftime('%Y-%m-%d')})"
        parts.append(f"[{m.type.value.upper()}{timestamp}]\n{m.content}")
        total_tokens += len(m.content) // 4

    memories_out = [
        MemoryOut(
            id=m.id, type=m.type.value, content=m.content,
            importance_score=m.importance_score, source=m.source or "",
            event_time=m.event_time,
        )
        for m in recent
    ]

    return ContextResponse(
        context="\n\n".join(parts),
        memories_used=memories_out,
        total_tokens=total_tokens,
    )


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    type: Optional[str] = None
    importance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


@app.put("/memories/{memory_id}", response_model=MemoryOut)
def update_memory(memory_id: str, body: MemoryUpdate):
    """Update a memory's content, type, or importance."""
    retriever = _get_retriever()
    mem = None
    for m in retriever.memories:
        if m.id == memory_id:
            mem = m
            break
    if not mem:
        raise HTTPException(404, "Memory not found")

    # Apply updates to in-memory object
    if body.content is not None:
        mem.content = body.content
    if body.type is not None:
        mem.type = MemoryType(body.type)
    if body.importance_score is not None:
        mem.importance_score = body.importance_score

    # Try DB update
    import json as _json
    try:
        updates = []
        params = []
        if body.content is not None:
            updates.append("content = %s")
            params.append(body.content)
            # Re-embed
            embedding = _get_embedding_sync(body.content)
            if embedding:
                updates.append("embedding = %s::vector")
                params.append(str(embedding))
        if body.type is not None:
            updates.append("type = %s::memory_type")
            params.append(body.type)
        if body.importance_score is not None:
            updates.append("importance_score = %s")
            params.append(body.importance_score)
        if updates:
            params.append(memory_id)
            with db.get_cursor() as cur:
                cur.execute(
                    f"UPDATE memory_nodes SET {', '.join(updates)} WHERE id = %s",
                    tuple(params),
                )
    except Exception as e:
        logger.warning(f"DB update failed (in-memory updated): {e}")

    retriever._build_indices()

    return MemoryOut(
        id=mem.id, type=mem.type.value, content=mem.content,
        importance_score=mem.importance_score, source=mem.source or "",
        event_time=mem.event_time,
    )


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str):
    """Delete a memory by ID."""
    retriever = _get_retriever()
    idx = None
    for i, m in enumerate(retriever.memories):
        if m.id == memory_id:
            idx = i
            break
    if idx is None:
        raise HTTPException(404, "Memory not found")

    retriever.memories.pop(idx)
    retriever._build_indices()

    # Try DB delete
    try:
        with db.get_cursor() as cur:
            cur.execute("DELETE FROM memory_nodes WHERE id = %s", (memory_id,))
    except Exception as e:
        logger.warning(f"DB delete failed (removed from in-memory): {e}")

    return {"deleted": memory_id}


class ConsolidateResponse(BaseModel):
    related: Dict[str, Any] = Field(default_factory=dict)
    topics: Dict[str, Any] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0


@app.post("/memories/consolidate", response_model=ConsolidateResponse)
def consolidate_memories(
    similarity_threshold: float = Query(0.75, ge=0.5, le=1.0),
    min_cluster_size: int = Query(3, ge=2, le=20),
):
    """Trigger memory consolidation (similarity + topic-based). Returns summary."""
    from consolidator import PgVectorConsolidator
    from datetime import datetime as _dt

    consolidator = PgVectorConsolidator()
    start = _dt.now()

    try:
        related = consolidator.consolidate_related_memories(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )
    except Exception as e:
        logger.exception("Related consolidation failed")
        related = {"clusters_created": 0, "memories_affected": 0, "summaries": [], "error": str(e)}

    try:
        topics = consolidator.consolidate_by_topic()
    except Exception as e:
        logger.exception("Topic consolidation failed")
        topics = {"topics_summarized": 0, "summaries": [], "error": str(e)}

    elapsed = (_dt.now() - start).total_seconds()
    return ConsolidateResponse(related=related, topics=topics, elapsed_seconds=elapsed)


@app.post("/memories/backfill-embeddings")
def backfill_embeddings(limit: int = Query(50, ge=1, le=500)):
    """Backfill embeddings for existing memories that don't have them."""
    updated = 0
    failed = 0
    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT id::text, content FROM memory_nodes WHERE embedding IS NULL LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()

        for row in rows:
            embedding = _get_embedding_sync(row["content"])
            if embedding:
                try:
                    with db.get_cursor() as cur:
                        cur.execute(
                            "UPDATE memory_nodes SET embedding = %s::vector WHERE id = %s",
                            (str(embedding), row["id"]),
                        )
                    updated += 1
                except Exception as ex:
                    logger.error(f"Backfill update failed for {row['id']}: {ex}")
                    failed += 1
            else:
                logger.warning(f"Backfill embedding returned None for {row['id']}")
                failed += 1
    except Exception as e:
        logger.warning(f"Backfill query failed: {e}")

    return {"updated": updated, "failed": failed, "remaining": len(rows) - updated - failed if 'rows' in dir() else 0}


@app.post("/memories/deduplicate")
def deduplicate_memories(threshold: float = Query(0.9, ge=0.5, le=1.0)):
    """Run bulk deduplication across all memories. Returns count of merged pairs."""
    merged = deduplicate_all(threshold=threshold)
    # Rebuild in-memory index after dedup
    try:
        retriever = _get_retriever()
        with db.get_cursor() as cur:
            cur.execute(
                """SELECT id::text, type::text, content, importance_score, source, event_time, metadata
                   FROM memory_nodes ORDER BY created_at"""
            )
            rows = cur.fetchall()
        retriever.memories = [
            MemoryNode(
                id=r["id"], type=MemoryType(r["type"]), content=r["content"],
                importance_score=r["importance_score"], source=r.get("source"),
                event_time=r.get("event_time"), metadata=r.get("metadata", {}),
            )
            for r in rows
        ]
        retriever._build_indices()
    except Exception as e:
        logger.warning(f"Failed to rebuild in-memory index after dedup: {e}")
    return {"merged": merged, "threshold": threshold}


@app.post("/memories/decay-update")
def trigger_decay_update():
    """Recalculate decay factors for all memories. Intended for cron use."""
    result = decay.bulk_decay_update()
    return result


# ---------------------------------------------------------------------------
# Temporal / Timeline Endpoints
# ---------------------------------------------------------------------------

@app.get("/timeline", response_model=List[MemoryOut])
def timeline(
    start: str = Query(..., description="Start date (ISO format, e.g. 2026-02-11)"),
    end: str = Query(..., description="End date (ISO format, e.g. 2026-02-14)"),
):
    """Get chronological memory timeline for a date range."""
    rows = temporal_get_timeline(start, end)
    return [
        MemoryOut(
            id=r["id"], type=r["type"], content=r["content"],
            importance_score=r["importance_score"],
            decay_factor=r.get("decay_factor", 1.0),
            access_count=r.get("access_count", 0),
            source=r.get("source", ""),
            event_time=r.get("event_time"),
            created_at=r.get("created_at"),
            metadata=r.get("metadata", {}),
        )
        for r in rows
    ]


@app.get("/memories/since/{since_date}", response_model=List[MemoryOut])
def memories_since(since_date: str):
    """Get memories created or modified since a given date."""
    rows = whats_changed_since(since_date)
    return [
        MemoryOut(
            id=r["id"], type=r["type"], content=r["content"],
            importance_score=r["importance_score"],
            decay_factor=r.get("decay_factor", 1.0),
            access_count=r.get("access_count", 0),
            source=r.get("source", ""),
            event_time=r.get("event_time"),
            created_at=r.get("created_at"),
            metadata=r.get("metadata", {}),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Emotion Tagging Endpoints
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feedback / Learning Endpoints
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    memory_id: str
    feedback_type: str
    session_id: Optional[str] = None


@app.post("/feedback", status_code=201)
def api_record_feedback(body: FeedbackRequest):
    """Record feedback on a memory: used, ignored, or corrected."""
    try:
        result = feedback.record_feedback(body.memory_id, body.feedback_type, body.session_id)
        # Convert datetime for JSON
        if "created_at" in result and result["created_at"]:
            result["created_at"] = str(result["created_at"])
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/feedback/apply")
def api_apply_feedback():
    """Apply accumulated feedback to adjust memory importance scores."""
    return feedback.apply_feedback_scores()


@app.get("/feedback/stats")
def api_feedback_stats():
    """Get learning statistics on memory usefulness."""
    return feedback.get_learning_stats()


@app.post("/memories/tag-emotions")
def api_bulk_tag_emotions():
    """Bulk-tag all untagged memories with emotional context."""
    result = bulk_tag_emotions()
    return result


@app.post("/memories/backfill-confidence")
def api_backfill_confidence():
    """Backfill confidence scores for all memories using auto-tagging logic."""
    from confidence import backfill_all_confidence_scores
    result = backfill_all_confidence_scores()
    return result


# ---------------------------------------------------------------------------
# Graph Relationship Endpoints (Apache AGE)
# ---------------------------------------------------------------------------

class GraphLinkRequest(BaseModel):
    from_memory_id: str
    to_memory_id: str
    relationship_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.post("/graph/link")
def graph_link(body: GraphLinkRequest):
    """Manually create a relationship between two memories."""
    try:
        result = graph_module.add_relationship(
            body.from_memory_id, body.to_memory_id,
            body.relationship_type, body.metadata,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Graph link failed: {e}")
        raise HTTPException(500, f"Failed to create relationship: {e}")


@app.post("/graph/auto-link")
def graph_auto_link():
    """Automatically create RELATED_TO edges between similar memories (cosine > 0.7)."""
    try:
        result = graph_module.auto_link_memories()
        return result
    except Exception as e:
        logger.error(f"Auto-link failed: {e}")
        raise HTTPException(500, f"Auto-link failed: {e}")


@app.get("/graph/traverse/{memory_id}")
def graph_traverse(memory_id: str, depth: int = Query(2, ge=1, le=10)):
    """Get connected memories up to N hops from the given memory."""
    try:
        result = graph_module.traverse(memory_id, depth)
        return result
    except Exception as e:
        logger.error(f"Graph traverse failed: {e}")
        raise HTTPException(500, f"Traverse failed: {e}")


@app.get("/graph/path")
def graph_find_path(from_id: str = Query(...), to_id: str = Query(...)):
    """Find shortest path between two memories."""
    try:
        result = graph_module.find_path(from_id, to_id)
        return result
    except Exception as e:
        logger.error(f"Graph path failed: {e}")
        raise HTTPException(500, f"Path search failed: {e}")


@app.get("/graph/subgraph/{entity}")
def graph_subgraph(entity: str):
    """Get all memories and relationships involving an entity."""
    try:
        result = graph_module.get_subgraph(entity)
        return result
    except Exception as e:
        logger.error(f"Graph subgraph failed: {e}")
        raise HTTPException(500, f"Subgraph query failed: {e}")


# ---------------------------------------------------------------------------
# Namespace Endpoints (Multi-Agent Memory Sharing)
# ---------------------------------------------------------------------------

@app.post("/namespaces", response_model=NamespaceOut, status_code=201)
def create_namespace(body: NamespaceCreate):
    """Create a new memory namespace for multi-agent sharing."""
    import json as _json
    try:
        namespace_id = str(uuid.uuid4())
        with db.get_cursor() as cur:
            cur.execute("""
                INSERT INTO memory_namespaces (namespace_id, name, owner_agent, access_policy)
                VALUES (%s, %s, %s, %s::jsonb)
                RETURNING namespace_id::text, name, owner_agent, access_policy, created_at, updated_at
            """, (namespace_id, body.name, body.owner_agent, _json.dumps(body.access_policy)))
            row = cur.fetchone()
            return NamespaceOut(**row)
    except Exception as e:
        logger.error(f"Failed to create namespace: {e}")
        if "unique" in str(e).lower():
            raise HTTPException(409, f"Namespace '{body.name}' already exists")
        raise HTTPException(500, f"Failed to create namespace: {e}")


@app.get("/namespaces", response_model=List[NamespaceOut])
def list_namespaces(
    agent_id: Optional[str] = Query(None, description="Filter by agent access"),
    permission: str = Query("read", description="Filter by permission type: read or write")
):
    """List all namespaces, optionally filtered by agent access."""
    try:
        if agent_id:
            # Use access control to get only accessible namespaces
            namespaces = access_control.get_agent_namespaces(agent_id, permission)
            return [NamespaceOut(**ns) for ns in namespaces]
        else:
            # Return all namespaces (admin view)
            with db.get_cursor() as cur:
                cur.execute("""
                    SELECT namespace_id::text, name, owner_agent, access_policy, created_at, updated_at
                    FROM memory_namespaces
                    ORDER BY name
                """)
                rows = cur.fetchall()
                return [NamespaceOut(**row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to list namespaces: {e}")
        raise HTTPException(500, f"Failed to list namespaces: {e}")


@app.get("/namespaces/{namespace_id}", response_model=NamespaceOut)
def get_namespace(namespace_id: str):
    """Get namespace details."""
    try:
        with db.get_cursor() as cur:
            cur.execute("""
                SELECT namespace_id::text, name, owner_agent, access_policy, created_at, updated_at
                FROM memory_namespaces
                WHERE namespace_id = %s
            """, (namespace_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"Namespace {namespace_id} not found")
            return NamespaceOut(**row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get namespace: {e}")
        raise HTTPException(500, f"Failed to get namespace: {e}")


@app.post("/namespaces/{namespace_id}/grant", response_model=GrantAccessResponse)
def grant_namespace_access(
    namespace_id: str,
    body: GrantAccessRequest,
    grantor_agent: str = Query(..., description="Agent performing the grant")
):
    """Grant an agent read or write access to a namespace."""
    try:
        result = access_control.grant_access(
            namespace_id, body.agent_id, body.permission, grantor_agent
        )
        return GrantAccessResponse(**result)
    except access_control.AccessDeniedError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Failed to grant access: {e}")
        raise HTTPException(500, f"Failed to grant access: {e}")


@app.post("/namespaces/{namespace_id}/revoke")
def revoke_namespace_access(
    namespace_id: str,
    body: GrantAccessRequest,
    revoker_agent: str = Query(..., description="Agent performing the revocation")
):
    """Revoke an agent's access to a namespace."""
    try:
        result = access_control.revoke_access(
            namespace_id, body.agent_id, body.permission, revoker_agent
        )
        return result
    except access_control.AccessDeniedError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Failed to revoke access: {e}")
        raise HTTPException(500, f"Failed to revoke access: {e}")


@app.get("/namespaces/{namespace_id}/permissions")
def get_namespace_permissions(namespace_id: str):
    """Get full permission details for a namespace."""
    try:
        result = access_control.get_namespace_permissions(namespace_id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Failed to get permissions: {e}")
        raise HTTPException(500, f"Failed to get permissions: {e}")
