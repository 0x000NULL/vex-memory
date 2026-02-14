"""
Graph relationship management for Vex Memory System using Apache AGE.

Provides functions to create, traverse, and query relationships between
memories stored as a property graph in PostgreSQL via AGE (Cypher queries).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import db

logger = logging.getLogger(__name__)

GRAPH_NAME = "memory_graph"

VALID_RELATIONSHIP_TYPES = {
    "CAUSED_BY", "PART_OF", "RELATED_TO", "PRECEDED", "CONTRADICTS", "SUPPORTS"
}


def _age_cursor():
    """Get a DB cursor with AGE loaded and search_path set."""
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("LOAD 'age';")
    cur.execute('SET search_path = ag_catalog, "$user", public;')
    return conn, cur


def _cypher(query: str, cols: str = "v agtype") -> List[Any]:
    """Execute a Cypher query against memory_graph and return results."""
    conn, cur = _age_cursor()
    try:
        sql = f"SELECT * FROM cypher('{GRAPH_NAME}', $$ {query} $$) as ({cols});"
        cur.execute(sql)
        rows = cur.fetchall()
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def ensure_graph():
    """Create the memory_graph if it doesn't exist."""
    conn, cur = _age_cursor()
    try:
        cur.execute("SELECT * FROM ag_catalog.ag_graph WHERE name = %s;", (GRAPH_NAME,))
        if not cur.fetchone():
            cur.execute("SELECT ag_catalog.create_graph(%s);", (GRAPH_NAME,))
            logger.info(f"Created graph '{GRAPH_NAME}'")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def _ensure_memory_node(memory_id: str):
    """Create a Memory node if it doesn't already exist in the graph."""
    _cypher(f"MERGE (n:Memory {{id: '{memory_id}'}})")


def add_relationship(
    from_memory_id: str,
    to_memory_id: str,
    relationship_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a typed relationship between two memories.

    Args:
        from_memory_id: Source memory ID
        to_memory_id: Target memory ID
        relationship_type: One of CAUSED_BY, PART_OF, RELATED_TO, PRECEDED, CONTRADICTS, SUPPORTS
        metadata: Optional metadata dict attached to the edge

    Returns:
        Dict with from_id, to_id, type, metadata
    """
    if relationship_type not in VALID_RELATIONSHIP_TYPES:
        raise ValueError(
            f"Invalid relationship type '{relationship_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_RELATIONSHIP_TYPES))}"
        )

    metadata = metadata or {}
    meta_str = json.dumps(metadata).replace("'", "\\'")

    # Ensure both nodes exist, then create the edge
    _ensure_memory_node(from_memory_id)
    _ensure_memory_node(to_memory_id)

    _cypher(
        f"MATCH (a:Memory {{id: '{from_memory_id}'}}), (b:Memory {{id: '{to_memory_id}'}}) "
        f"CREATE (a)-[r:{relationship_type} {{metadata: '{meta_str}'}}]->(b)"
    )

    logger.info(f"Created {relationship_type} edge: {from_memory_id} -> {to_memory_id}")
    return {
        "from_id": from_memory_id,
        "to_id": to_memory_id,
        "type": relationship_type,
        "metadata": metadata,
    }


def auto_link_memories() -> Dict[str, Any]:
    """Automatically create RELATED_TO edges between memories with cosine similarity > 0.7.

    Queries pgvector for memory pairs with high embedding similarity and creates
    graph edges for any pair not already linked.

    Returns:
        Dict with 'links_created' count and 'pairs' list.
    """
    pairs_created = []

    try:
        with db.get_cursor() as cur:
            # Find pairs with cosine similarity > 0.7 that aren't the same memory
            cur.execute("""
                SELECT a.id::text AS id_a, b.id::text AS id_b,
                       1.0 - (a.embedding <=> b.embedding) AS similarity
                FROM memory_nodes a
                JOIN memory_nodes b ON a.id < b.id
                WHERE a.embedding IS NOT NULL
                  AND b.embedding IS NOT NULL
                  AND 1.0 - (a.embedding <=> b.embedding) > 0.7
                ORDER BY similarity DESC
                LIMIT 500
            """)
            candidates = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to query similarity pairs: {e}")
        return {"links_created": 0, "pairs": [], "error": str(e)}

    for row in candidates:
        id_a, id_b, sim = row["id_a"], row["id_b"], float(row["similarity"])
        try:
            # Check if edge already exists
            existing = _cypher(
                f"MATCH (a:Memory {{id: '{id_a}'}})-[r:RELATED_TO]-(b:Memory {{id: '{id_b}'}}) RETURN r",
                "r agtype",
            )
            if not existing:
                add_relationship(id_a, id_b, "RELATED_TO", {"auto": True, "similarity": round(sim, 4)})
                pairs_created.append({"from": id_a, "to": id_b, "similarity": round(sim, 4)})
        except Exception as e:
            logger.warning(f"Failed to auto-link {id_a} <-> {id_b}: {e}")

    logger.info(f"Auto-linked {len(pairs_created)} memory pairs")
    return {"links_created": len(pairs_created), "pairs": pairs_created}


def traverse(memory_id: str, depth: int = 2) -> Dict[str, Any]:
    """Return connected memories up to N hops from the given memory.

    Args:
        memory_id: Starting memory ID
        depth: Maximum traversal depth (default 2)

    Returns:
        Dict with 'origin', 'nodes' list, and 'edges' list.
    """
    depth = max(1, min(depth, 10))  # clamp

    rows = _cypher(
        f"MATCH path = (start:Memory {{id: '{memory_id}'}})-[*1..{depth}]-(connected:Memory) "
        f"RETURN connected.id, labels(connected)",
        "id agtype, labels agtype",
    )

    # Collect unique connected memory IDs
    connected_ids = set()
    for row in rows:
        mid = str(row["id"]).strip('"')
        if mid != memory_id:
            connected_ids.add(mid)

    # Get edges — query each pair individually since AGE doesn't support relationships() on VLE
    edge_rows = []
    if connected_ids:
        all_ids = connected_ids | {memory_id}
        for mid in all_ids:
            try:
                rows_e = _cypher(
                    f"MATCH (a:Memory {{id: '{mid}'}})-[r]->(b:Memory) "
                    f"RETURN a.id, type(r), b.id",
                    "from_id agtype, rel_type agtype, to_id agtype",
                )
                edge_rows.extend(rows_e)
            except Exception:
                pass

    edges = []
    seen_edges = set()
    for row in edge_rows:
        fid = str(row["from_id"]).strip('"')
        tid = str(row["to_id"]).strip('"')
        rt = str(row["rel_type"]).strip('"')
        key = (fid, tid, rt)
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append({"from": fid, "to": tid, "type": rt})

    # Enrich with memory content from DB
    nodes = []
    if connected_ids:
        try:
            with db.get_cursor() as cur:
                cur.execute(
                    "SELECT id::text, type::text, content, importance_score "
                    "FROM memory_nodes WHERE id = ANY(%s)",
                    (list(connected_ids),),
                )
                for row in cur.fetchall():
                    nodes.append(row)
        except Exception:
            nodes = [{"id": mid} for mid in connected_ids]

    return {"origin": memory_id, "depth": depth, "nodes": nodes, "edges": edges}


def find_path(from_id: str, to_id: str, max_depth: int = 6) -> Dict[str, Any]:
    """Find the shortest path between two memories using iterative deepening.

    AGE doesn't support shortestPath(), so we check increasing depths.

    Returns:
        Dict with 'from', 'to', 'path' (list of node IDs), 'edges', and 'length'.
    """
    if from_id == to_id:
        return {"from": from_id, "to": to_id, "path": [from_id], "edges": [], "length": 0}

    # Depth 1: direct connection
    try:
        rows = _cypher(
            f"MATCH (a:Memory {{id: '{from_id}'}})-[r]-(b:Memory {{id: '{to_id}'}}) "
            f"RETURN type(r)",
            "rel_type agtype",
        )
        if rows:
            rt = str(rows[0]["rel_type"]).strip('"')
            return {
                "from": from_id, "to": to_id,
                "path": [from_id, to_id],
                "edges": [{"from": from_id, "type": rt, "to": to_id}],
                "length": 1,
            }
    except Exception as e:
        logger.warning(f"Path depth-1 failed: {e}")

    # Depth 2+: BFS via intermediate nodes
    for depth in range(2, max_depth + 1):
        try:
            # Build match pattern: (a)-[]-(n1)-[]-(n2)-...-(b)
            intermediates = [f"(n{i}:Memory)" for i in range(1, depth)]
            rels = [f"[r{i}]" for i in range(depth)]
            nodes_list = ["a"] + [f"n{i}" for i in range(1, depth)] + ["b"]
            
            pattern_parts = []
            for i in range(depth):
                pattern_parts.append(f"({nodes_list[i]})-[{rels[i]}]-({nodes_list[i+1]})")
            
            # Use chained MATCH
            match_str = f"MATCH (a:Memory {{id: '{from_id}'}})-[*{depth}]-(b:Memory {{id: '{to_id}'}}) RETURN a.id, b.id"
            rows = _cypher(match_str, "a_id agtype, b_id agtype")
            if rows:
                # Found a path at this depth but can't easily extract intermediates from AGE VLE
                # Return just endpoints with length
                return {
                    "from": from_id, "to": to_id,
                    "path": [from_id, to_id],
                    "edges": [],
                    "length": depth,
                }
        except Exception as e:
            logger.warning(f"Path search at depth {depth} failed: {e}")

    return {"from": from_id, "to": to_id, "path": [], "edges": [], "length": -1}


def get_subgraph(entity_name: str) -> Dict[str, Any]:
    """Get all memories and relationships involving an entity.

    Searches memory_nodes for content/metadata mentioning the entity,
    then pulls their graph neighborhoods.

    Args:
        entity_name: Entity name to search for

    Returns:
        Dict with 'entity', 'memories' list, and 'relationships' list.
    """
    # Find memory IDs that reference this entity
    memory_ids = []
    try:
        with db.get_cursor() as cur:
            # Search in content and entities table
            cur.execute(
                """SELECT DISTINCT mn.id::text, mn.type::text, mn.content, mn.importance_score
                   FROM memory_nodes mn
                   LEFT JOIN memory_entity_links mel ON mn.id = mel.memory_id
                   LEFT JOIN entities e ON mel.entity_id = e.id
                   WHERE mn.content ILIKE %s
                      OR e.name ILIKE %s
                      OR e.canonical_name ILIKE %s
                   LIMIT 100""",
                (f"%{entity_name}%", f"%{entity_name}%", f"%{entity_name}%"),
            )
            rows = cur.fetchall()
            memory_ids = [r["id"] for r in rows]
    except Exception as e:
        logger.warning(f"Entity search failed, falling back to content search: {e}")
        try:
            with db.get_cursor() as cur:
                cur.execute(
                    "SELECT id::text, type::text, content, importance_score "
                    "FROM memory_nodes WHERE content ILIKE %s LIMIT 100",
                    (f"%{entity_name}%",),
                )
                rows = cur.fetchall()
                memory_ids = [r["id"] for r in rows]
        except Exception:
            rows = []

    if not memory_ids:
        return {"entity": entity_name, "memories": [], "relationships": []}

    # Get all relationships between these memories from the graph
    relationships = []
    for mid in memory_ids:
        try:
            edge_rows = _cypher(
                f"MATCH (a:Memory {{id: '{mid}'}})-[r]-(b:Memory) "
                f"RETURN a.id, type(r), b.id",
                "from_id agtype, rel_type agtype, to_id agtype",
            )
            for er in edge_rows:
                relationships.append({
                    "from": str(er["from_id"]).strip('"'),
                    "type": str(er["rel_type"]).strip('"'),
                    "to": str(er["to_id"]).strip('"'),
                })
        except Exception as e:
            logger.warning(f"Graph query failed for memory {mid}: {e}")

    # Deduplicate relationships
    seen = set()
    unique_rels = []
    for r in relationships:
        key = (r["from"], r["type"], r["to"])
        if key not in seen:
            seen.add(key)
            unique_rels.append(r)

    return {"entity": entity_name, "memories": rows, "relationships": unique_rels}
