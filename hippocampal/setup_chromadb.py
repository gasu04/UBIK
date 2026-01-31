#!/usr/bin/env python3
"""
ChromaDB Collection Setup for Ubik Hippocampal Node

Creates two primary collections:
1. ubik_episodic - Personal experiences, conversations, time-bound memories
2. ubik_semantic - Conceptual knowledge, beliefs, values, preferences

Usage:
    python setup_chromadb.py

The script will:
- Connect to ChromaDB with retry logic
- Create/recreate the episodic and semantic collections
- Add sample data for testing
- Verify the setup with a test query
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from exceptions import DatabaseConnectionError, DatabaseError

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ubik.setup_chromadb")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ChromaDBSetupConfig:
    """Configuration for ChromaDB setup."""
    host: str = field(default_factory=lambda: os.getenv("CHROMADB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("CHROMADB_PORT", "8001")))
    token: str = field(default_factory=lambda: os.getenv("CHROMADB_TOKEN", ""))
    max_retries: int = 10
    retry_delay: float = 3.0
    embedding_model: str = "all-MiniLM-L6-v2"

    # HNSW index parameters
    hnsw_space: str = "cosine"
    hnsw_construction_ef: int = 128
    hnsw_search_ef: int = 64
    hnsw_m: int = 16


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_EPISODIC: List[Dict[str, Any]] = [
    {
        "id": "ep_sample_001",
        "document": "Today I wrote my first letter to my future grandchildren about why I started the Ubik project.",
        "metadata": {
            "type": "letter",
            "timestamp": "2024-01-15T10:30:00Z",
            "emotional_valence": "positive",
            "importance": 0.9,
            "participants": "gines,grandchildren",
            "themes": "legacy,purpose,family"
        }
    },
    {
        "id": "ep_sample_002",
        "document": "In therapy today, we discussed how my values around authenticity shaped my career decisions.",
        "metadata": {
            "type": "therapy_session",
            "timestamp": "2024-01-14T14:00:00Z",
            "emotional_valence": "reflective",
            "importance": 0.8,
            "participants": "gines,therapist",
            "themes": "values,career,authenticity"
        }
    },
    {
        "id": "ep_sample_003",
        "document": "Family meeting: We talked about summer vacation plans and everyone's school achievements.",
        "metadata": {
            "type": "family_meeting",
            "timestamp": "2024-01-13T18:00:00Z",
            "emotional_valence": "positive",
            "importance": 0.7,
            "participants": "family",
            "themes": "planning,celebration,connection"
        }
    }
]

SAMPLE_SEMANTIC: List[Dict[str, Any]] = [
    {
        "id": "sem_sample_001",
        "document": "I believe authenticity is the foundation of meaningful relationships. Being genuine matters more than being perfect.",
        "metadata": {
            "type": "belief",
            "category": "relationships",
            "confidence": 0.95,
            "stability": "core",
            "source": "life_experience",
            "frozen": False
        }
    },
    {
        "id": "sem_sample_002",
        "document": "My core value: Family legacy transcends material possessions - it's about values, wisdom, and love passed down.",
        "metadata": {
            "type": "value",
            "category": "family",
            "confidence": 0.98,
            "stability": "core",
            "source": "reflection",
            "frozen": False
        }
    },
    {
        "id": "sem_sample_003",
        "document": "Communication preference: I prefer thoughtful, reflective conversations over quick exchanges. Depth over breadth.",
        "metadata": {
            "type": "preference",
            "category": "communication",
            "confidence": 0.85,
            "stability": "stable",
            "source": "self_observation",
            "frozen": False
        }
    },
    {
        "id": "sem_sample_004",
        "document": "Philosophical stance: Identity persists through psychological continuity - memories, values, and intentions form the self.",
        "metadata": {
            "type": "belief",
            "category": "philosophy",
            "confidence": 0.90,
            "stability": "stable",
            "source": "reading_parfitian_identity",
            "frozen": False
        }
    }
]


# =============================================================================
# Setup Functions
# =============================================================================

def connect_with_retry(config: ChromaDBSetupConfig) -> chromadb.HttpClient:
    """
    Connect to ChromaDB with retry logic.

    Args:
        config: Setup configuration.

    Returns:
        Connected ChromaDB client.

    Raises:
        DatabaseConnectionError: If connection fails after all retries.
    """
    client: Optional[chromadb.HttpClient] = None

    for attempt in range(1, config.max_retries + 1):
        try:
            client = chromadb.HttpClient(
                host=config.host,
                port=config.port,
                headers={"Authorization": f"Bearer {config.token}"}
            )
            client.heartbeat()
            logger.info(f"Connected to ChromaDB at {config.host}:{config.port}")
            return client
        except Exception as e:
            logger.warning(
                f"Connection attempt {attempt}/{config.max_retries} failed: {e}"
            )
            if attempt < config.max_retries:
                time.sleep(config.retry_delay)

    raise DatabaseConnectionError(
        service="chromadb",
        host=config.host,
        port=config.port,
        reason=f"Failed after {config.max_retries} attempts"
    )


def get_embedding_function(config: ChromaDBSetupConfig) -> Any:
    """
    Get the embedding function for ChromaDB collections.

    Args:
        config: Setup configuration.

    Returns:
        Embedding function instance.
    """
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        logger.info(f"Using SentenceTransformer embeddings ({config.embedding_model})")
        return embedding_fn
    except Exception as e:
        logger.warning(f"SentenceTransformer not available: {e}")
        logger.info("Using default embeddings")
        return embedding_functions.DefaultEmbeddingFunction()


def create_collection(
    client: chromadb.HttpClient,
    name: str,
    description: str,
    embedding_fn: Any,
    config: ChromaDBSetupConfig
) -> Any:
    """
    Create a ChromaDB collection with standard configuration.

    Args:
        client: ChromaDB client.
        name: Collection name.
        description: Collection description.
        embedding_fn: Embedding function to use.
        config: Setup configuration.

    Returns:
        Created collection.
    """
    # Delete existing collection if present
    try:
        client.delete_collection(name)
        logger.info(f"Deleted existing collection: {name}")
    except Exception:
        pass  # Collection doesn't exist

    collection = client.create_collection(
        name=name,
        embedding_function=embedding_fn,
        metadata={
            "description": description,
            "hnsw:space": config.hnsw_space,
            "hnsw:construction_ef": config.hnsw_construction_ef,
            "hnsw:search_ef": config.hnsw_search_ef,
            "hnsw:M": config.hnsw_m,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "version": "1.0.0",
            "node": "hippocampal"
        }
    )
    logger.info(f"Created collection: {name}")
    return collection


def add_sample_data(
    collection: Any,
    samples: List[Dict[str, Any]]
) -> int:
    """
    Add sample data to a collection.

    Args:
        collection: ChromaDB collection.
        samples: List of sample documents with metadata.

    Returns:
        Number of documents added.
    """
    collection.add(
        documents=[s["document"] for s in samples],
        metadatas=[s["metadata"] for s in samples],
        ids=[s["id"] for s in samples]
    )
    logger.info(f"Added {len(samples)} sample documents to {collection.name}")
    return len(samples)


def verify_setup(client: chromadb.HttpClient) -> bool:
    """
    Verify the ChromaDB setup with test queries.

    Args:
        client: ChromaDB client.

    Returns:
        True if verification passed.
    """
    logger.info("Verifying setup...")

    collections = client.list_collections()
    logger.info(f"Active collections: {len(collections)}")

    for col in collections:
        count = col.count()
        logger.info(f"  {col.name}: {count} documents")

    # Test semantic search
    try:
        semantic = client.get_collection("ubik_semantic")
        results = semantic.query(
            query_texts=["family values and legacy"],
            n_results=2
        )

        if results['documents'][0]:
            logger.info("Test query successful")
            for i, (doc, meta) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0]
            )):
                logger.debug(f"Result {i+1}: {meta.get('type')} - {doc[:50]}...")
            return True
        else:
            logger.warning("Test query returned no results")
            return False

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def create_collections(config: Optional[ChromaDBSetupConfig] = None) -> bool:
    """
    Initialize ChromaDB collections for Ubik memory system.

    Args:
        config: Optional configuration override.

    Returns:
        True if setup completed successfully.
    """
    if config is None:
        config = ChromaDBSetupConfig()

    print("=" * 60)
    print("Ubik ChromaDB Collection Setup")
    print("=" * 60)

    try:
        # Connect to ChromaDB
        client = connect_with_retry(config)
        print(f"\n✓ Connected to ChromaDB")

        # Get embedding function
        embedding_fn = get_embedding_function(config)
        print(f"✓ Embedding function ready")

        # Create episodic collection
        print("\n" + "-" * 40)
        print("Creating ubik_episodic collection...")
        episodic = create_collection(
            client=client,
            name="ubik_episodic",
            description="Episodic memories - personal experiences, conversations, events",
            embedding_fn=embedding_fn,
            config=config
        )
        add_sample_data(episodic, SAMPLE_EPISODIC)
        print(f"  ✓ Created ubik_episodic with {len(SAMPLE_EPISODIC)} samples")

        # Create semantic collection
        print("\n" + "-" * 40)
        print("Creating ubik_semantic collection...")
        semantic = create_collection(
            client=client,
            name="ubik_semantic",
            description="Semantic knowledge - concepts, beliefs, values, preferences",
            embedding_fn=embedding_fn,
            config=config
        )
        add_sample_data(semantic, SAMPLE_SEMANTIC)
        print(f"  ✓ Created ubik_semantic with {len(SAMPLE_SEMANTIC)} samples")

        # Verification
        print("\n" + "=" * 60)
        print("Verification")
        print("=" * 60)

        collections = client.list_collections()
        print(f"\nActive collections: {len(collections)}")
        for col in collections:
            count = col.count()
            print(f"  • {col.name}: {count} documents")

        # Test query
        print("\n" + "-" * 40)
        print("Test Query: 'family values and legacy'")
        print("-" * 40)

        results = client.get_collection("ubik_semantic").query(
            query_texts=["family values and legacy"],
            n_results=2
        )

        for i, (doc, meta) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        )):
            print(f"\n  Result {i+1}:")
            print(f"    Type: {meta.get('type')}")
            print(f"    Category: {meta.get('category')}")
            print(f"    Text: {doc[:80]}...")

        print("\n" + "=" * 60)
        print("✓ ChromaDB setup complete!")
        print("=" * 60)

        return True

    except DatabaseConnectionError as e:
        logger.error(f"Failed to connect: {e}")
        print(f"\n✗ {e}")
        return False
    except Exception as e:
        logger.exception(f"Setup failed: {e}")
        print(f"\n✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = create_collections()
    sys.exit(0 if success else 1)
