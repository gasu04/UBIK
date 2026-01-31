#!/usr/bin/env python3
"""
Neo4j Schema Initialization for Ubik Parfitian Identity Model

This schema implements a graph structure based on Derek Parfit's
theory of personal identity, emphasizing psychological continuity
and the "Relation R" (psychological connectedness + continuity).

Node Types:
- Concept: Abstract concepts, beliefs, values
- Memory: Episodic memory anchors (links to ChromaDB)
- Trait: Personality traits and characteristics
- Relationship: Interpersonal relationships
- TimeSlice: Temporal snapshots of identity state

Relationship Types:
- INFLUENCES: One concept affects another
- DERIVES_FROM: Concept originates from another
- SUPPORTS: Provides evidence/support
- CONFLICTS_WITH: Creates tension/contradiction
- CONNECTS_TO: Parfitian psychological connection
- CONTINUES_AS: Temporal continuity link
- REMEMBERS: Memory connection
- VALUES: Association with value
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

from exceptions import DatabaseConnectionError, ConfigurationError

# Setup
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ubik.schema")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Neo4jSchemaConfig:
    """Configuration for Neo4j schema initialization."""
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))

    def validate(self) -> bool:
        """Validate configuration values."""
        if not self.password:
            raise ConfigurationError("NEO4J_PASSWORD is required", "NEO4J_PASSWORD")
        return True


def init_schema(driver: Driver) -> None:
    """
    Initialize the Neo4j schema with constraints and indexes.

    Args:
        driver: Neo4j driver instance.
    """

    with driver.session() as session:

        # =====================================================================
        # Constraints (ensure uniqueness)
        # =====================================================================

        constraints = [
            # Concept nodes must have unique names
            ("concept_name", "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE"),

            # Memory references must be unique
            ("memory_id", "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.chromadb_id IS UNIQUE"),

            # Traits must be unique
            ("trait_name", "CREATE CONSTRAINT trait_name IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE"),

            # TimeSlices must have unique identifiers
            ("timeslice_id", "CREATE CONSTRAINT timeslice_id IF NOT EXISTS FOR (ts:TimeSlice) REQUIRE ts.timestamp IS UNIQUE"),
        ]

        logger.info("Creating constraints...")
        for name, query in constraints:
            try:
                session.run(query)
                logger.info(f"  ✓ Created constraint: {name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"  • Constraint exists: {name}")
                else:
                    logger.error(f"  ✗ Error creating {name}: {e}")

        # =====================================================================
        # Indexes (improve query performance)
        # =====================================================================

        indexes = [
            # Full-text search on concept descriptions
            ("concept_fulltext", """
                CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS
                FOR (c:Concept) ON EACH [c.name, c.description]
            """),

            # Index on concept categories
            ("concept_category", "CREATE INDEX concept_category IF NOT EXISTS FOR (c:Concept) ON (c.category)"),

            # Index on stability (for frozen vs mutable queries)
            ("concept_stability", "CREATE INDEX concept_stability IF NOT EXISTS FOR (c:Concept) ON (c.stability)"),

            # Index on trait type
            ("trait_type", "CREATE INDEX trait_type IF NOT EXISTS FOR (t:Trait) ON (t.trait_type)"),

            # Index on memory type
            ("memory_type", "CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)"),

            # Index on relationship weight for traversal optimization
            ("rel_weight", "CREATE INDEX rel_weight IF NOT EXISTS FOR ()-[r:INFLUENCES]-() ON (r.weight)"),
        ]

        logger.info("\nCreating indexes...")
        for name, query in indexes:
            try:
                session.run(query)
                logger.info(f"  ✓ Created index: {name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"  • Index exists: {name}")
                else:
                    logger.error(f"  ✗ Error creating {name}: {e}")


def create_core_identity_nodes(driver: Driver) -> None:
    """
    Create the foundational identity structure.

    Creates the core Self node, values, traits, and relationships
    that form the basis of the Parfitian identity model.

    Args:
        driver: Neo4j driver instance.
    """

    with driver.session() as session:
        logger.info("\nCreating core identity nodes...")

        # =====================================================================
        # Core Identity Node (The "Self")
        # =====================================================================

        session.run("""
            MERGE (self:Concept:CoreIdentity {name: 'Self'})
            SET self.description = 'The central identity node representing Gines',
                self.stability = 'core',
                self.frozen = false,
                self.created_at = datetime(),
                self.parfitian_note = 'The Self is not a simple, unified entity but a bundle of psychological connections'
        """)
        logger.info("  ✓ Created Self node")

        # =====================================================================
        # Core Value Nodes
        # =====================================================================

        core_values = [
            {
                "name": "Authenticity",
                "description": "Being genuine and true to oneself in all interactions",
                "category": "personal_virtue",
                "importance": 0.95
            },
            {
                "name": "Family Legacy",
                "description": "Preserving and transmitting wisdom, values, and love to future generations",
                "category": "life_purpose",
                "importance": 0.98
            },
            {
                "name": "Intellectual Curiosity",
                "description": "Deep drive to understand, learn, and explore ideas",
                "category": "personality",
                "importance": 0.85
            },
            {
                "name": "Meaningful Connection",
                "description": "Valuing deep, authentic relationships over superficial interactions",
                "category": "relationships",
                "importance": 0.90
            },
            {
                "name": "Psychological Continuity",
                "description": "Identity persists through connected chains of memory, intention, and character",
                "category": "philosophy",
                "importance": 0.88
            }
        ]

        for value in core_values:
            session.run("""
                MERGE (v:Concept:Value {name: $name})
                SET v.description = $description,
                    v.category = $category,
                    v.importance = $importance,
                    v.stability = 'core',
                    v.frozen = false,
                    v.created_at = datetime()
                WITH v
                MATCH (self:CoreIdentity {name: 'Self'})
                MERGE (self)-[r:VALUES]->(v)
                SET r.weight = $importance,
                    r.established = datetime()
            """, **value)
            logger.info(f"  ✓ Created value: {value['name']}")

        # =====================================================================
        # Personality Trait Nodes
        # =====================================================================

        traits = [
            {
                "name": "Reflective",
                "description": "Tendency toward deep introspection and thoughtful analysis",
                "trait_type": "cognitive_style",
                "strength": 0.85
            },
            {
                "name": "Warm",
                "description": "Natural warmth and genuine care in interpersonal interactions",
                "trait_type": "interpersonal",
                "strength": 0.80
            },
            {
                "name": "Methodical",
                "description": "Preference for systematic, organized approaches to problems",
                "trait_type": "behavioral",
                "strength": 0.75
            },
            {
                "name": "Creative",
                "description": "Ability to generate novel ideas and see unusual connections",
                "trait_type": "cognitive_style",
                "strength": 0.70
            }
        ]

        for trait in traits:
            session.run("""
                MERGE (t:Trait {name: $name})
                SET t.description = $description,
                    t.trait_type = $trait_type,
                    t.strength = $strength,
                    t.stability = 'stable',
                    t.created_at = datetime()
                WITH t
                MATCH (self:CoreIdentity {name: 'Self'})
                MERGE (self)-[r:HAS_TRAIT]->(t)
                SET r.strength = $strength
            """, **trait)
            logger.info(f"  ✓ Created trait: {trait['name']}")

        # =====================================================================
        # Relationship Nodes (Key People)
        # =====================================================================

        relationships = [
            {
                "name": "Grandchildren",
                "relationship_type": "family",
                "description": "Future grandchildren - the intended recipients of Ubik's wisdom",
                "importance": 0.99
            },
            {
                "name": "Spouse",
                "relationship_type": "family",
                "description": "Life partner and closest confidant",
                "importance": 0.98
            },
            {
                "name": "Children",
                "relationship_type": "family",
                "description": "Children - bridge between self and grandchildren",
                "importance": 0.97
            }
        ]

        for rel in relationships:
            session.run("""
                MERGE (r:Relationship {name: $name})
                SET r.relationship_type = $relationship_type,
                    r.description = $description,
                    r.importance = $importance,
                    r.created_at = datetime()
                WITH r
                MATCH (self:CoreIdentity {name: 'Self'})
                MERGE (self)-[c:CONNECTED_TO]->(r)
                SET c.connection_type = $relationship_type,
                    c.strength = $importance
            """, **rel)
            logger.info(f"  ✓ Created relationship: {rel['name']}")


def create_parfitian_structure(driver: Driver) -> None:
    """
    Create the Parfitian psychological continuity structure.

    Establishes TimeSlice nodes and psychological connections
    between values and traits as per Parfit's theory.

    Args:
        driver: Neo4j driver instance.
    """

    with driver.session() as session:
        logger.info("\nCreating Parfitian continuity structure...")

        # =====================================================================
        # Create TimeSlice Template
        # =====================================================================

        session.run("""
            MERGE (ts:TimeSlice {timestamp: 'TEMPLATE'})
            SET ts.description = 'Template for identity time slices',
                ts.is_template = true,
                ts.note = 'Each TimeSlice captures identity state at a moment, linked by CONTINUES_AS relationships'
        """)
        logger.info("  ✓ Created TimeSlice template")

        # =====================================================================
        # Create Initial TimeSlice (Project Start)
        # =====================================================================

        session.run("""
            MERGE (ts:TimeSlice {timestamp: '2024-01-01T00:00:00Z'})
            SET ts.description = 'Ubik project inception - initial identity capture',
                ts.is_template = false,
                ts.phase = 'initial_capture',
                ts.notes = 'Beginning of systematic identity preservation effort'
            WITH ts
            MATCH (self:CoreIdentity {name: 'Self'})
            MERGE (self)-[r:AT_TIME]->(ts)
            SET r.note = 'Identity state at project start'
        """)
        logger.info("  ✓ Created initial TimeSlice")

        # =====================================================================
        # Create Psychological Connection Structure
        # =====================================================================

        # Connect values to traits (psychological bundles)
        connections = [
            ("Authenticity", "Reflective", "Authenticity requires self-reflection"),
            ("Authenticity", "Warm", "Genuine warmth flows from authentic self"),
            ("Family Legacy", "Meaningful Connection", "Legacy transmitted through deep connection"),
            ("Intellectual Curiosity", "Reflective", "Curiosity drives reflection"),
            ("Intellectual Curiosity", "Creative", "Curiosity fuels creative exploration"),
            ("Meaningful Connection", "Warm", "Warmth enables meaningful connection"),
            ("Psychological Continuity", "Reflective", "Understanding continuity requires reflection")
        ]

        for source, target, context in connections:
            session.run("""
                MATCH (s {name: $source})
                MATCH (t {name: $target})
                MERGE (s)-[r:CONNECTS_TO]->(t)
                SET r.context = $context,
                    r.weight = 0.8,
                    r.parfitian_type = 'psychological_connection'
            """, source=source, target=target, context=context)

        logger.info(f"  ✓ Created {len(connections)} psychological connections")


def create_memory_anchors(driver: Driver) -> None:
    """
    Create memory anchor nodes that link to ChromaDB entries.

    Memory anchors in Neo4j link to actual memory content stored
    in ChromaDB, enabling graph-based memory traversal.

    Args:
        driver: Neo4j driver instance.
    """

    with driver.session() as session:
        logger.info("\nCreating memory anchors...")

        # Sample memory anchors (these would link to actual ChromaDB entries)
        memories = [
            {
                "chromadb_id": "ep_sample_001",
                "memory_type": "letter",
                "summary": "First letter to grandchildren about Ubik project purpose",
                "emotional_valence": "positive"
            },
            {
                "chromadb_id": "ep_sample_002",
                "memory_type": "therapy_session",
                "summary": "Discussion of authenticity in career decisions",
                "emotional_valence": "reflective"
            },
            {
                "chromadb_id": "ep_sample_003",
                "memory_type": "family_meeting",
                "summary": "Summer vacation planning with family",
                "emotional_valence": "positive"
            }
        ]

        for mem in memories:
            session.run("""
                MERGE (m:Memory {chromadb_id: $chromadb_id})
                SET m.memory_type = $memory_type,
                    m.summary = $summary,
                    m.emotional_valence = $emotional_valence,
                    m.created_at = datetime()
                WITH m
                MATCH (self:CoreIdentity {name: 'Self'})
                MERGE (self)-[r:REMEMBERS]->(m)
                SET r.importance = 0.7
            """, **mem)
            logger.info(f"  ✓ Created memory anchor: {mem['chromadb_id']}")

        # Connect memories to relevant concepts
        session.run("""
            MATCH (m:Memory {chromadb_id: 'ep_sample_001'})
            MATCH (v:Value {name: 'Family Legacy'})
            MERGE (m)-[r:SUPPORTS]->(v)
            SET r.note = 'Letter demonstrates commitment to family legacy'
        """)

        session.run("""
            MATCH (m:Memory {chromadb_id: 'ep_sample_002'})
            MATCH (v:Value {name: 'Authenticity'})
            MERGE (m)-[r:SUPPORTS]->(v)
            SET r.note = 'Session explores authentic self-expression'
        """)

        logger.info("  ✓ Connected memories to concepts")


def verify_schema(driver: Driver) -> bool:
    """
    Verify the schema is correctly initialized.

    Outputs node counts, relationship counts, and Self connections.

    Args:
        driver: Neo4j driver instance.

    Returns:
        True if verification completed successfully.
    """

    with driver.session() as session:
        logger.info("\n" + "=" * 60)
        logger.info("Schema Verification")
        logger.info("=" * 60)

        # Count nodes by type
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC
        """)

        logger.info("\nNode counts:")
        for record in result:
            logger.info(f"  • {record['type']}: {record['count']}")

        # Count relationships
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
        """)

        logger.info("\nRelationship counts:")
        for record in result:
            logger.info(f"  • {record['type']}: {record['count']}")

        # Test query: Find all connections from Self
        result = session.run("""
            MATCH (self:CoreIdentity {name: 'Self'})-[r]->(connected)
            RETURN type(r) as relationship, connected.name as name, labels(connected)[0] as type
            ORDER BY type(r)
        """)

        logger.info("\nSelf connections:")
        for record in result:
            logger.info(f"  • -{record['relationship']}-> {record['name']} ({record['type']})")

    return True


def main(config: Optional[Neo4jSchemaConfig] = None) -> bool:
    """
    Main entry point for schema initialization.

    Args:
        config: Optional configuration override.

    Returns:
        True if initialization completed successfully.
    """
    if config is None:
        config = Neo4jSchemaConfig()

    print("=" * 60)
    print("Ubik Neo4j Schema Initialization")
    print("Parfitian Identity Model")
    print("=" * 60)

    try:
        config.validate()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return False

    driver: Optional[Driver] = None

    try:
        driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password)
        )

        # Verify connection
        driver.verify_connectivity()
        logger.info(f"✓ Connected to Neo4j at {config.uri}")

        # Initialize schema
        init_schema(driver)
        create_core_identity_nodes(driver)
        create_parfitian_structure(driver)
        create_memory_anchors(driver)
        verify_schema(driver)

        print("\n" + "=" * 60)
        print("✓ Schema initialization complete!")
        print("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Error during schema initialization: {e}")
        return False
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
