"""
Access Control for Multi-Agent Memory Namespaces
=================================================

Handles permissions for namespace-based memory sharing between agents.
Each namespace has an owner and a JSONB access_policy with read/write arrays.
"""

import logging
from typing import List, Optional, Dict, Any
import json

import db

logger = logging.getLogger(__name__)


class AccessDeniedError(Exception):
    """Raised when an agent attempts unauthorized access to a namespace."""
    pass


def can_read(agent_id: str, namespace_id: str) -> bool:
    """
    Check if an agent has read access to a namespace.
    
    Args:
        agent_id: Identifier of the requesting agent
        namespace_id: UUID of the namespace to check
        
    Returns:
        True if agent can read from namespace, False otherwise
    """
    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT can_read_namespace(%s, %s) AS can_read",
                (agent_id, namespace_id)
            )
            row = cur.fetchone()
            return row['can_read'] if row else False
    except Exception as e:
        logger.error(f"Error checking read access for {agent_id} on {namespace_id}: {e}")
        return False


def can_write(agent_id: str, namespace_id: str) -> bool:
    """
    Check if an agent has write access to a namespace.
    
    Args:
        agent_id: Identifier of the requesting agent
        namespace_id: UUID of the namespace to check
        
    Returns:
        True if agent can write to namespace, False otherwise
    """
    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT can_write_namespace(%s, %s) AS can_write",
                (agent_id, namespace_id)
            )
            row = cur.fetchone()
            return row['can_write'] if row else False
    except Exception as e:
        logger.error(f"Error checking write access for {agent_id} on {namespace_id}: {e}")
        return False


def get_agent_namespaces(agent_id: str, permission: str = 'read') -> List[Dict[str, Any]]:
    """
    Get all namespaces an agent has access to.
    
    Args:
        agent_id: Identifier of the agent
        permission: 'read' or 'write' - filter by permission type
        
    Returns:
        List of namespace dicts with id, name, owner, and access level
    """
    try:
        with db.get_cursor() as cur:
            if permission == 'read':
                cur.execute("""
                    SELECT 
                        namespace_id::text, 
                        name, 
                        owner_agent,
                        access_policy,
                        CASE 
                            WHEN owner_agent = %s THEN 'owner'
                            WHEN access_policy->'write' @> to_jsonb(%s::text) THEN 'write'
                            ELSE 'read'
                        END as access_level
                    FROM memory_namespaces
                    WHERE owner_agent = %s 
                       OR access_policy->'read' @> to_jsonb(%s::text)
                       OR access_policy->'write' @> to_jsonb(%s::text)
                    ORDER BY name
                """, (agent_id, agent_id, agent_id, agent_id, agent_id))
            else:  # write
                cur.execute("""
                    SELECT 
                        namespace_id::text, 
                        name, 
                        owner_agent,
                        access_policy,
                        CASE 
                            WHEN owner_agent = %s THEN 'owner'
                            ELSE 'write'
                        END as access_level
                    FROM memory_namespaces
                    WHERE owner_agent = %s 
                       OR access_policy->'write' @> to_jsonb(%s::text)
                    ORDER BY name
                """, (agent_id, agent_id, agent_id))
            
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting namespaces for {agent_id}: {e}")
        return []


def grant_access(namespace_id: str, agent_id: str, permission: str, grantor_agent: str) -> Dict[str, Any]:
    """
    Grant an agent access to a namespace.
    
    Args:
        namespace_id: UUID of the namespace
        agent_id: Agent to grant access to
        permission: 'read' or 'write'
        grantor_agent: Agent performing the grant (must have write access or be owner)
        
    Returns:
        Updated namespace info
        
    Raises:
        AccessDeniedError: If grantor doesn't have permission to grant access
        ValueError: If permission type is invalid
    """
    if permission not in ('read', 'write'):
        raise ValueError(f"Invalid permission type: {permission}. Must be 'read' or 'write'")
    
    # Check if grantor has permission to grant access (must be owner or have write access)
    if not can_write(grantor_agent, namespace_id):
        raise AccessDeniedError(
            f"Agent {grantor_agent} does not have permission to grant access to namespace {namespace_id}"
        )
    
    try:
        with db.get_cursor() as cur:
            # Get current access policy
            cur.execute(
                "SELECT access_policy FROM memory_namespaces WHERE namespace_id = %s",
                (namespace_id,)
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Namespace {namespace_id} not found")
            
            policy = row['access_policy']
            
            # Add agent to the appropriate permission array if not already there
            if agent_id not in policy.get(permission, []):
                # Update the JSONB policy
                cur.execute(f"""
                    UPDATE memory_namespaces
                    SET access_policy = jsonb_set(
                        access_policy,
                        '{{{permission}}}',
                        COALESCE(access_policy->'{permission}', '[]'::jsonb) || to_jsonb(%s::text)
                    )
                    WHERE namespace_id = %s
                    RETURNING namespace_id::text, name, owner_agent, access_policy
                """, (agent_id, namespace_id))
                
                updated = cur.fetchone()
                logger.info(f"Granted {permission} access to {agent_id} on namespace {namespace_id}")
                return dict(updated)
            else:
                logger.info(f"Agent {agent_id} already has {permission} access to {namespace_id}")
                cur.execute(
                    "SELECT namespace_id::text, name, owner_agent, access_policy FROM memory_namespaces WHERE namespace_id = %s",
                    (namespace_id,)
                )
                return dict(cur.fetchone())
                
    except AccessDeniedError:
        raise
    except Exception as e:
        logger.error(f"Error granting access: {e}")
        raise


def revoke_access(namespace_id: str, agent_id: str, permission: str, revoker_agent: str) -> Dict[str, Any]:
    """
    Revoke an agent's access to a namespace.
    
    Args:
        namespace_id: UUID of the namespace
        agent_id: Agent to revoke access from
        permission: 'read' or 'write'
        revoker_agent: Agent performing the revocation (must be owner or have write access)
        
    Returns:
        Updated namespace info
        
    Raises:
        AccessDeniedError: If revoker doesn't have permission
        ValueError: If permission type is invalid or namespace not found
    """
    if permission not in ('read', 'write'):
        raise ValueError(f"Invalid permission type: {permission}. Must be 'read' or 'write'")
    
    # Check if revoker has permission (must be owner or have write access)
    if not can_write(revoker_agent, namespace_id):
        raise AccessDeniedError(
            f"Agent {revoker_agent} does not have permission to revoke access on namespace {namespace_id}"
        )
    
    try:
        with db.get_cursor() as cur:
            # Get current access policy
            cur.execute(
                "SELECT access_policy FROM memory_namespaces WHERE namespace_id = %s",
                (namespace_id,)
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Namespace {namespace_id} not found")
            
            policy = row['access_policy']
            
            # Remove agent from the permission array
            current_list = policy.get(permission, [])
            if agent_id in current_list:
                updated_list = [a for a in current_list if a != agent_id]
                
                cur.execute(f"""
                    UPDATE memory_namespaces
                    SET access_policy = jsonb_set(
                        access_policy,
                        '{{{permission}}}',
                        to_jsonb(%s::text[])
                    )
                    WHERE namespace_id = %s
                    RETURNING namespace_id::text, name, owner_agent, access_policy
                """, (updated_list, namespace_id))
                
                updated = cur.fetchone()
                logger.info(f"Revoked {permission} access from {agent_id} on namespace {namespace_id}")
                return dict(updated)
            else:
                logger.info(f"Agent {agent_id} doesn't have {permission} access to {namespace_id}")
                cur.execute(
                    "SELECT namespace_id::text, name, owner_agent, access_policy FROM memory_namespaces WHERE namespace_id = %s",
                    (namespace_id,)
                )
                return dict(cur.fetchone())
                
    except AccessDeniedError:
        raise
    except Exception as e:
        logger.error(f"Error revoking access: {e}")
        raise


def get_namespace_permissions(namespace_id: str) -> Dict[str, Any]:
    """
    Get full permission details for a namespace.
    
    Args:
        namespace_id: UUID of the namespace
        
    Returns:
        Dict with owner, read_agents, and write_agents lists
    """
    try:
        with db.get_cursor() as cur:
            cur.execute("""
                SELECT 
                    namespace_id::text,
                    name,
                    owner_agent,
                    access_policy->'read' as read_agents,
                    access_policy->'write' as write_agents,
                    created_at,
                    updated_at
                FROM memory_namespaces
                WHERE namespace_id = %s
            """, (namespace_id,))
            
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Namespace {namespace_id} not found")
            
            return {
                'namespace_id': row['namespace_id'],
                'name': row['name'],
                'owner': row['owner_agent'],
                'read_agents': row['read_agents'] or [],
                'write_agents': row['write_agents'] or [],
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            }
    except Exception as e:
        logger.error(f"Error getting namespace permissions: {e}")
        raise


def filter_memories_by_access(agent_id: str, memory_ids: List[str]) -> List[str]:
    """
    Filter a list of memory IDs to only those the agent can access.
    
    Args:
        agent_id: Agent requesting access
        memory_ids: List of memory UUIDs to check
        
    Returns:
        Filtered list of accessible memory IDs
    """
    if not memory_ids:
        return []
    
    try:
        with db.get_cursor() as cur:
            cur.execute("""
                SELECT m.id::text
                FROM memory_nodes m
                WHERE m.id = ANY(%s::uuid[])
                  AND can_read_namespace(%s, m.namespace_id)
            """, (memory_ids, agent_id))
            
            rows = cur.fetchall()
            return [row['id'] for row in rows]
    except Exception as e:
        logger.error(f"Error filtering memories by access: {e}")
        return []
