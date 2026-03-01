"""
Tests for multi-agent memory namespace system.
"""

import pytest
import uuid
from datetime import datetime

import db
import access_control


@pytest.fixture
def clean_db():
    """Clean up test data before and after each test."""
    # Cleanup before test
    try:
        with db.get_cursor() as cur:
            cur.execute("DELETE FROM memory_nodes WHERE source = 'test-namespace'")
            cur.execute("DELETE FROM memory_namespaces WHERE name LIKE 'test-%'")
    except Exception:
        pass
    
    yield
    
    # Cleanup after test
    try:
        with db.get_cursor() as cur:
            cur.execute("DELETE FROM memory_nodes WHERE source = 'test-namespace'")
            cur.execute("DELETE FROM memory_namespaces WHERE name LIKE 'test-%'")
    except Exception:
        pass


def test_create_namespace(clean_db):
    """Test creating a new namespace."""
    namespace_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent, access_policy)
            VALUES (%s, %s, %s, %s::jsonb)
            RETURNING namespace_id::text, name, owner_agent, access_policy
        """, (namespace_id, name, 'test-agent', '{"read": [], "write": []}'))
        
        row = cur.fetchone()
        assert row['namespace_id'] == namespace_id
        assert row['name'] == name
        assert row['owner_agent'] == 'test-agent'


def test_owner_has_read_access(clean_db):
    """Test that namespace owner automatically has read access."""
    namespace_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s)
        """, (namespace_id, name, owner))
    
    assert access_control.can_read(owner, namespace_id) is True


def test_owner_has_write_access(clean_db):
    """Test that namespace owner automatically has write access."""
    namespace_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s)
        """, (namespace_id, name, owner))
    
    assert access_control.can_write(owner, namespace_id) is True


def test_grant_read_access(clean_db):
    """Test granting read access to another agent."""
    namespace_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    other_agent = 'test-reader'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s)
        """, (namespace_id, name, owner))
    
    # Other agent should not have access initially
    assert access_control.can_read(other_agent, namespace_id) is False
    
    # Grant read access
    result = access_control.grant_access(namespace_id, other_agent, 'read', owner)
    
    # Now should have read access
    assert access_control.can_read(other_agent, namespace_id) is True
    # But not write access
    assert access_control.can_write(other_agent, namespace_id) is False


def test_grant_write_access(clean_db):
    """Test granting write access to another agent."""
    namespace_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    other_agent = 'test-writer'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s)
        """, (namespace_id, name, owner))
    
    # Grant write access
    access_control.grant_access(namespace_id, other_agent, 'write', owner)
    
    # Should have write access
    assert access_control.can_write(other_agent, namespace_id) is True


def test_revoke_access(clean_db):
    """Test revoking access from an agent."""
    namespace_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    other_agent = 'test-reader'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s)
        """, (namespace_id, name, owner))
    
    # Grant and then revoke read access
    access_control.grant_access(namespace_id, other_agent, 'read', owner)
    assert access_control.can_read(other_agent, namespace_id) is True
    
    access_control.revoke_access(namespace_id, other_agent, 'read', owner)
    assert access_control.can_read(other_agent, namespace_id) is False


def test_access_denied_for_non_owner(clean_db):
    """Test that non-owner cannot grant access without permission."""
    namespace_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    other_agent = 'test-unauthorized'
    third_agent = 'test-third'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s)
        """, (namespace_id, name, owner))
    
    # Non-owner trying to grant access should fail
    with pytest.raises(access_control.AccessDeniedError):
        access_control.grant_access(namespace_id, third_agent, 'read', other_agent)


def test_namespace_filtered_memory_query(clean_db):
    """Test querying memories filtered by namespace."""
    # Create two namespaces
    ns1_id = str(uuid.uuid4())
    ns2_id = str(uuid.uuid4())
    name1 = f"test-{uuid.uuid4().hex[:8]}"
    name2 = f"test-{uuid.uuid4().hex[:8]}"
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, 'agent1'), (%s, %s, 'agent2')
        """, (ns1_id, name1, ns2_id, name2))
        
        # Insert memories into different namespaces
        mem1_id = str(uuid.uuid4())
        mem2_id = str(uuid.uuid4())
        
        cur.execute("""
            INSERT INTO memory_nodes (id, type, content, namespace_id, source)
            VALUES 
                (%s, 'semantic', 'Memory in namespace 1', %s, 'test-namespace'),
                (%s, 'semantic', 'Memory in namespace 2', %s, 'test-namespace')
        """, (mem1_id, ns1_id, mem2_id, ns2_id))
        
        # Query namespace 1
        cur.execute("""
            SELECT id::text, content
            FROM memory_nodes
            WHERE namespace_id = %s
        """, (ns1_id,))
        
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0]['content'] == 'Memory in namespace 1'


def test_get_agent_namespaces(clean_db):
    """Test getting all namespaces an agent has access to."""
    ns1_id = str(uuid.uuid4())
    ns2_id = str(uuid.uuid4())
    name1 = f"test-{uuid.uuid4().hex[:8]}"
    name2 = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    reader = 'test-reader'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s), (%s, %s, %s)
        """, (ns1_id, name1, owner, ns2_id, name2, owner))
    
    # Grant read access to second namespace
    access_control.grant_access(ns2_id, reader, 'read', owner)
    
    # Reader should see only ns2
    namespaces = access_control.get_agent_namespaces(reader, 'read')
    assert len(namespaces) == 1
    assert namespaces[0]['namespace_id'] == ns2_id
    
    # Owner should see both
    owner_namespaces = access_control.get_agent_namespaces(owner, 'read')
    assert len(owner_namespaces) >= 2
    namespace_ids = [ns['namespace_id'] for ns in owner_namespaces]
    assert ns1_id in namespace_ids
    assert ns2_id in namespace_ids


def test_memory_with_namespace(clean_db):
    """Test creating a memory in a specific namespace."""
    namespace_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, 'test-agent')
        """, (namespace_id, name))
        
        mem_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO memory_nodes (id, type, content, namespace_id, source)
            VALUES (%s, 'semantic', 'Test memory in namespace', %s, 'test-namespace')
            RETURNING id::text, namespace_id::text
        """, (mem_id, namespace_id))
        
        row = cur.fetchone()
        assert row['namespace_id'] == namespace_id


def test_filter_memories_by_access(clean_db):
    """Test filtering memory IDs by agent access."""
    # Create namespace
    ns_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    reader = 'test-reader'
    unauthorized = 'test-unauthorized'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s)
        """, (ns_id, name, owner))
        
        # Create memories
        mem1_id = str(uuid.uuid4())
        mem2_id = str(uuid.uuid4())
        
        cur.execute("""
            INSERT INTO memory_nodes (id, type, content, namespace_id, source)
            VALUES 
                (%s, 'semantic', 'Memory 1', %s, 'test-namespace'),
                (%s, 'semantic', 'Memory 2', %s, 'test-namespace')
        """, (mem1_id, ns_id, mem2_id, ns_id))
    
    # Grant read access to reader
    access_control.grant_access(ns_id, reader, 'read', owner)
    
    # Test filtering
    all_mem_ids = [mem1_id, mem2_id]
    
    # Owner should see all
    owner_accessible = access_control.filter_memories_by_access(owner, all_mem_ids)
    assert len(owner_accessible) == 2
    
    # Reader should see all (has read access)
    reader_accessible = access_control.filter_memories_by_access(reader, all_mem_ids)
    assert len(reader_accessible) == 2
    
    # Unauthorized should see none
    unauth_accessible = access_control.filter_memories_by_access(unauthorized, all_mem_ids)
    assert len(unauth_accessible) == 0


def test_get_agent_memories_function(clean_db):
    """Test the PostgreSQL get_agent_memories function."""
    # Create namespace and grant access
    ns_id = str(uuid.uuid4())
    name = f"test-{uuid.uuid4().hex[:8]}"
    owner = 'test-owner'
    reader = 'test-reader'
    
    with db.get_cursor() as cur:
        cur.execute("""
            INSERT INTO memory_namespaces (namespace_id, name, owner_agent)
            VALUES (%s, %s, %s)
        """, (ns_id, name, owner))
        
        # Create memories
        mem1_id = str(uuid.uuid4())
        mem2_id = str(uuid.uuid4())
        
        cur.execute("""
            INSERT INTO memory_nodes (id, type, content, namespace_id, source, importance_score)
            VALUES 
                (%s, 'semantic', 'Important memory', %s, 'test-namespace', 0.9),
                (%s, 'semantic', 'Less important memory', %s, 'test-namespace', 0.3)
        """, (mem1_id, ns_id, mem2_id, ns_id))
    
    # Grant read access to reader (after transaction is committed)
    access_control.grant_access(ns_id, reader, 'read', owner)
    
    # Query as reader
    with db.get_cursor() as cur:
        cur.execute("SELECT * FROM get_agent_memories(%s, %s, 10)", (reader, ns_id))
        rows = cur.fetchall()
    
    assert len(rows) == 2
    # Should be ordered by importance
    assert rows[0]['content'] == 'Important memory'
    assert rows[1]['content'] == 'Less important memory'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
