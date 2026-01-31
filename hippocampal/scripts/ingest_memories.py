#!/usr/bin/env python3
"""
Memory Ingestion Script for Ubik Hippocampal Node

Loads episodic and semantic memories from JSON files into ChromaDB.

Usage:
    python ingest_memories.py --episodic data/episodic.json
    python ingest_memories.py --semantic data/semantic.json
    python ingest_memories.py --all data/  # Loads all JSON files from directory
    python ingest_memories.py --stats

Returns:
    Exit code 0 on success, 1 on failure.
"""

import os
import sys
import json
import argparse
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exceptions import DatabaseConnectionError

# Load environment from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ubik.ingest_memories")


def get_chromadb_client() -> chromadb.HttpClient:
    """Connect to ChromaDB with authentication."""
    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8001"))
    token = os.getenv("CHROMADB_TOKEN", "ubik_chroma_token_2024")

    client = chromadb.HttpClient(
        host=host,
        port=port,
        headers={"Authorization": f"Bearer {token}"}
    )
    client.heartbeat()  # Verify connection
    return client


def get_embedding_function():
    """Get the embedding function for ChromaDB."""
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        return embedding_functions.DefaultEmbeddingFunction()


def ingest_episodic(client: chromadb.HttpClient, memories: List[Dict[str, Any]]) -> int:
    """
    Ingest episodic memories into ChromaDB.

    Expected format per memory:
    {
        "content": "Memory content text",
        "type": "letter|therapy_session|family_meeting|conversation|event",
        "timestamp": "2024-01-15T10:30:00Z",
        "emotional_valence": "positive|negative|neutral|reflective|mixed",
        "importance": 0.0-1.0,
        "participants": "person1,person2",
        "themes": "theme1,theme2",
        "source_file": "optional_source.md"
    }
    """
    collection = client.get_or_create_collection(
        name="ubik_episodic",
        embedding_function=get_embedding_function()
    )

    documents = []
    metadatas = []
    ids = []

    for memory in memories:
        doc_id = memory.get("id", f"ep_{uuid.uuid4().hex[:12]}")

        documents.append(memory["content"])
        metadatas.append({
            "type": memory.get("type", "event"),
            "timestamp": memory.get("timestamp", datetime.now().isoformat() + "Z"),
            "emotional_valence": memory.get("emotional_valence", "neutral"),
            "importance": float(memory.get("importance", 0.5)),
            "participants": memory.get("participants", "gines"),
            "themes": memory.get("themes", ""),
            "source_file": memory.get("source_file", ""),
            "ingested_at": datetime.now().isoformat() + "Z"
        })
        ids.append(doc_id)

    # Upsert to handle duplicates
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return len(documents)


def ingest_semantic(client: chromadb.HttpClient, knowledge: List[Dict[str, Any]]) -> int:
    """
    Ingest semantic knowledge into ChromaDB.

    Expected format per entry:
    {
        "content": "Knowledge/belief/value statement",
        "type": "belief|value|preference|fact|opinion",
        "category": "family|relationships|philosophy|communication|career|health",
        "confidence": 0.0-1.0,
        "stability": "core|stable|evolving",
        "source": "reflection|therapy|reading|life_experience",
        "frozen": false
    }
    """
    collection = client.get_or_create_collection(
        name="ubik_semantic",
        embedding_function=get_embedding_function()
    )

    documents = []
    metadatas = []
    ids = []

    for entry in knowledge:
        doc_id = entry.get("id", f"sem_{uuid.uuid4().hex[:12]}")

        documents.append(entry["content"])
        metadatas.append({
            "type": entry.get("type", "belief"),
            "category": entry.get("category", "general"),
            "confidence": float(entry.get("confidence", 0.8)),
            "stability": entry.get("stability", "stable"),
            "source": entry.get("source", "reflection"),
            "frozen": entry.get("frozen", False),
            "ingested_at": datetime.now().isoformat()
        })
        ids.append(doc_id)

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return len(documents)


def load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load memories from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and object with "memories" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("memories", data.get("entries", [data]))
    return []


def main() -> bool:
    """
    Main entry point for memory ingestion.

    Returns:
        True if ingestion completed successfully.
    """
    parser = argparse.ArgumentParser(description="Ingest memories into Hippocampal Node")
    parser.add_argument("--episodic", type=str, help="Path to episodic memories JSON")
    parser.add_argument("--semantic", type=str, help="Path to semantic knowledge JSON")
    parser.add_argument("--all", type=str, help="Directory containing memory JSON files")
    parser.add_argument("--stats", action="store_true", help="Show current collection stats")
    args = parser.parse_args()

    print("=" * 60)
    print("Ubik Memory Ingestion")
    print("=" * 60)

    try:
        client = get_chromadb_client()
        logger.info("Connected to ChromaDB")
        print("Connected to ChromaDB")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        print(f"Failed to connect to ChromaDB: {e}")
        return False

    if args.stats:
        print("\nCollection Statistics:")
        for col in client.list_collections():
            count = col.count()
            logger.info(f"Collection {col.name}: {count} documents")
            print(f"  {col.name}: {count} documents")
        return True

    total_episodic = 0
    total_semantic = 0

    if args.episodic:
        path = Path(args.episodic)
        if path.exists():
            memories = load_json_file(path)
            count = ingest_episodic(client, memories)
            total_episodic += count
            logger.info(f"Ingested {count} episodic memories from {path}")
            print(f"Ingested {count} episodic memories from {path}")
        else:
            logger.warning(f"File not found: {path}")
            print(f"File not found: {path}")

    if args.semantic:
        path = Path(args.semantic)
        if path.exists():
            knowledge = load_json_file(path)
            count = ingest_semantic(client, knowledge)
            total_semantic += count
            logger.info(f"Ingested {count} semantic entries from {path}")
            print(f"Ingested {count} semantic entries from {path}")
        else:
            logger.warning(f"File not found: {path}")
            print(f"File not found: {path}")

    if args.all:
        dir_path = Path(args.all)
        if dir_path.is_dir():
            for json_file in dir_path.glob("*.json"):
                data = load_json_file(json_file)
                if not data:
                    continue

                # Determine type from filename or content
                filename = json_file.stem.lower()
                if "episodic" in filename or "event" in filename or "memory" in filename:
                    count = ingest_episodic(client, data)
                    total_episodic += count
                    logger.info(f"Ingested {count} episodic memories from {json_file.name}")
                    print(f"Ingested {count} episodic memories from {json_file.name}")
                elif "semantic" in filename or "knowledge" in filename or "belief" in filename:
                    count = ingest_semantic(client, data)
                    total_semantic += count
                    logger.info(f"Ingested {count} semantic entries from {json_file.name}")
                    print(f"Ingested {count} semantic entries from {json_file.name}")
                else:
                    # Try to infer from content
                    sample = data[0] if data else {}
                    if "emotional_valence" in sample or "participants" in sample:
                        count = ingest_episodic(client, data)
                        total_episodic += count
                        logger.info(f"Ingested {count} episodic memories from {json_file.name}")
                        print(f"Ingested {count} episodic memories from {json_file.name}")
                    else:
                        count = ingest_semantic(client, data)
                        total_semantic += count
                        logger.info(f"Ingested {count} semantic entries from {json_file.name}")
                        print(f"Ingested {count} semantic entries from {json_file.name}")
        else:
            logger.warning(f"Directory not found: {dir_path}")
            print(f"Directory not found: {dir_path}")

    print("\n" + "=" * 60)
    logger.info(f"Total ingested: {total_episodic} episodic, {total_semantic} semantic")
    print(f"Total ingested: {total_episodic} episodic, {total_semantic} semantic")

    # Show final stats
    print("\nFinal Collection Statistics:")
    for col in client.list_collections():
        count = col.count()
        logger.info(f"Final count - {col.name}: {count} documents")
        print(f"  {col.name}: {count} documents")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
