#!/usr/bin/env python3
"""
Neo4j Graph Validation Script for Ubik Hippocampal Node

Validates the Neo4j identity graph by:
- Listing all node types and counts
- Listing all relationship types
- Showing Self (CoreIdentity) connections
- Testing path traversal queries

Usage:
    python scripts/validate_neo4j.py

Returns:
    Exit code 0 if validation passes, 1 otherwise.
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

# Add parent directory to path for imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exceptions import DatabaseConnectionError, ConfigurationError

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ubik.validate_neo4j")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Neo4jValidationConfig:
    """Configuration for Neo4j validation."""
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))


# =============================================================================
# Validation Functions
# =============================================================================

def get_node_counts(session: Any) -> List[Dict[str, Any]]:
    """
    Get counts of all node types in the graph.

    Args:
        session: Neo4j session.

    Returns:
        List of dicts with 'type' and 'count' keys.
    """
    result = session.run("""
        MATCH (n)
        RETURN labels(n)[0] as type, count(n) as count
        ORDER BY count DESC
    """)
    return [{"type": r["type"], "count": r["count"]} for r in result]


def get_relationship_counts(session: Any) -> List[Dict[str, Any]]:
    """
    Get counts of all relationship types in the graph.

    Args:
        session: Neo4j session.

    Returns:
        List of dicts with 'type' and 'count' keys.
    """
    result = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
    """)
    return [{"type": r["type"], "count": r["count"]} for r in result]


def get_self_connections(session: Any) -> List[Dict[str, Any]]:
    """
    Get all connections from the Self (CoreIdentity) node.

    Args:
        session: Neo4j session.

    Returns:
        List of connection dicts.
    """
    result = session.run("""
        MATCH (self:CoreIdentity {name: 'Self'})-[r]->(connected)
        RETURN type(r) as rel, connected.name as name, labels(connected)[0] as type
        ORDER BY type(r), connected.name
    """)
    return [
        {"rel": r["rel"], "name": r["name"], "type": r["type"]}
        for r in result
    ]


def test_path_traversal(session: Any, start_node: str = "Family Legacy") -> List[List[str]]:
    """
    Test path traversal from a starting node.

    Args:
        session: Neo4j session.
        start_node: Name of the starting node.

    Returns:
        List of path node lists.
    """
    result = session.run("""
        MATCH path = (start:Value {name: $start_node})-[*1..2]-(related)
        RETURN [n in nodes(path) | n.name] as path_nodes
        LIMIT 10
    """, start_node=start_node)
    return [r["path_nodes"] for r in result]


def get_concept_nodes(session: Any) -> List[str]:
    """
    Get all Concept-labeled nodes.

    Args:
        session: Neo4j session.

    Returns:
        List of concept names.
    """
    result = session.run("MATCH (c:Concept) RETURN c.name as name LIMIT 20")
    return [r["name"] for r in result]


# =============================================================================
# Main Validation
# =============================================================================

def validate_graph(config: Optional[Neo4jValidationConfig] = None) -> bool:
    """
    Run all validation checks on the Neo4j graph.

    Args:
        config: Optional configuration override.

    Returns:
        True if all validations pass.
    """
    if config is None:
        config = Neo4jValidationConfig()

    print("\n" + "=" * 60)
    print(" NEO4J GRAPH VALIDATION")
    print("=" * 60)

    driver: Optional[Driver] = None

    try:
        driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password)
        )
        driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {config.uri}")

        with driver.session() as session:
            # 1. Node types
            print("\n[1] Node Types:")
            node_counts = get_node_counts(session)
            if node_counts:
                for item in node_counts:
                    print(f"    • {item['type']}: {item['count']}")
                    logger.info(f"Node type {item['type']}: {item['count']}")
            else:
                print("    ⚠ No nodes found")
                logger.warning("No nodes found in graph")

            # 2. Relationship types
            print("\n[2] Relationship Types:")
            rel_counts = get_relationship_counts(session)
            if rel_counts:
                for item in rel_counts:
                    print(f"    • {item['type']}: {item['count']}")
                    logger.info(f"Relationship type {item['type']}: {item['count']}")
            else:
                print("    ⚠ No relationships found")
                logger.warning("No relationships found in graph")

            # 3. Self connections
            print("\n[3] Self (CoreIdentity) Connections:")
            connections = get_self_connections(session)
            if connections:
                for conn in connections:
                    print(f"    • Self -[{conn['rel']}]-> {conn['name']} ({conn['type']})")
            else:
                print("    ⚠ No Self connections found")
                logger.warning("CoreIdentity 'Self' node not found or has no connections")

            # 4. Path traversal test
            print("\n[4] Testing Path Traversal from 'Family Legacy':")
            paths = test_path_traversal(session)
            if paths:
                for path in paths:
                    print(f"    • {' -> '.join(str(n) for n in path)}")
            else:
                print("    ⚠ No paths found (check node labels)")
                logger.warning("No paths found from 'Family Legacy'")

            # 5. Concept nodes
            print("\n[5] Checking for Concept-labeled nodes:")
            concepts = get_concept_nodes(session)
            if concepts:
                for name in concepts:
                    print(f"    • {name}")
            else:
                print("    ⚠ No Concept nodes found")
                print("    Note: Identity graph queries look for :Concept nodes")
                print("    Your schema may use :Value, :Trait, :Relationship labels")
                logger.warning("No Concept nodes found - may need schema update")

        print("\n" + "=" * 60)
        print(" VALIDATION COMPLETE")
        print("=" * 60 + "\n")

        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n✗ Validation failed: {e}")
        return False

    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    success = validate_graph()
    sys.exit(0 if success else 1)
