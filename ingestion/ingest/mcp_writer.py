#!/usr/bin/env python3
"""
UBIK Ingestion — Local Memory Writer

Direct ChromaDB + Neo4j writer for the Hippocampal Node.
Replaces the remote HippocampalClient MCP round-trip with local I/O
so ingestion never leaves the node that hosts the storage layer.

Drop-in replacement for HippocampalClient.  Implements the same
store_episodic / store_semantic / update_identity_graph interface so
IngestPipeline works without modification.

Collections targeted:
    ubik_episodic     — personal experiences, conversations, events
    ubik_semantic     — beliefs, values, preferences
    ubik_intellectual — curated external knowledge / highlights (created on demand)

Embeddings:
    sentence-transformers all-MiniLM-L6-v2 (384 dimensions, cosine space)
    Matches the model used to build the existing ubik_* collections.

Version: 1.0.0
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EPISODIC_COLLECTION = "ubik_episodic"
_SEMANTIC_COLLECTION = "ubik_semantic"
_INTELLECTUAL_COLLECTION = "ubik_intellectual"

# HNSW settings matching the existing ubik_episodic / ubik_semantic collections.
_HNSW_METADATA: Dict[str, Any] = {
    "hnsw:space": "cosine",
    "hnsw:M": 16,
    "hnsw:construction_ef": 128,
    "hnsw:search_ef": 64,
    "node": "hippocampal",
    "version": "1.0.0",
}


class FrozenVoiceError(Exception):
    """Raised when semantic memory is frozen and ``force=False``."""


class LocalMemoryWriter:
    """
    Writes MemoryCandidates directly to local ChromaDB and Neo4j.

    Designed to run on the Hippocampal Node where ChromaDB, Neo4j, and the
    MCP server all live.  By writing directly to the storage layer this class
    eliminates the MCP network round-trip and the cross-node Tailscale hop
    that the old HippocampalClient required.

    Interface contract:
        Implements the same async ``store_episodic``, ``store_semantic``,
        and ``update_identity_graph`` signatures as ``HippocampalClient`` so
        ``IngestPipeline`` needs no changes.

    Args:
        chroma_host: ChromaDB host (default: ``CHROMADB_HOST`` env or ``localhost``).
        chroma_port: ChromaDB port (default: ``CHROMADB_PORT`` env or ``8001``).
        chroma_token: ChromaDB auth token (default: ``CHROMADB_TOKEN`` env).
        neo4j_uri: Neo4j Bolt URI (default: ``NEO4J_URI`` env or
            ``bolt://localhost:7687``).
        neo4j_user: Neo4j username (default: ``NEO4J_USER`` env or ``neo4j``).
        neo4j_password: Neo4j password (default: ``NEO4J_PASSWORD`` env).

    Example:
        async with LocalMemoryWriter() as writer:
            await writer.store_episodic(
                content="Met Elena at the park today...",
                memory_type="event",
                importance=0.8,
            )
            stats = await writer.get_stats()
            print(stats)  # {'episodic': 84, 'semantic': 8, 'intellectual': 0}
    """

    def __init__(
        self,
        *,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        chroma_token: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ) -> None:
        self._chroma_host = chroma_host or os.getenv("CHROMADB_HOST", "localhost")
        self._chroma_port = chroma_port or int(os.getenv("CHROMADB_PORT", "8001"))
        self._chroma_token = chroma_token or os.getenv("CHROMADB_TOKEN", "")
        self._neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self._neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")

        self._chroma: Any = None          # chromadb.HttpClient — lazy
        self._neo4j_driver: Any = None    # neo4j.AsyncGraphDatabase.driver — lazy
        self._embed_fn: Any = None        # SentenceTransformer — lazy

        self._episodic: Any = None
        self._semantic: Any = None
        self._intellectual: Any = None

        self._initialized: bool = False
        self.is_connected: bool = False

    # -------------------------------------------------------------------------
    # Context manager (matches HippocampalClient protocol)
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "LocalMemoryWriter":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Initialise ChromaDB client, embedding model, and verify connectivity."""
        await asyncio.get_event_loop().run_in_executor(None, self._connect_sync)
        self._initialized = True
        self.is_connected = True
        stats = await self.get_stats()
        logger.info(
            "LocalMemoryWriter ready — episodic=%d semantic=%d intellectual=%d",
            stats["episodic"],
            stats["semantic"],
            stats["intellectual"],
        )

    def _connect_sync(self) -> None:
        """Blocking ChromaDB + embedding setup — runs in executor thread."""
        import chromadb
        from sentence_transformers import SentenceTransformer

        settings = chromadb.Settings(anonymized_telemetry=False)
        if self._chroma_token:
            settings = chromadb.Settings(
                anonymized_telemetry=False,
                chroma_client_auth_provider=(
                    "chromadb.auth.token_authn.TokenAuthClientProvider"
                ),
                chroma_client_auth_credentials=self._chroma_token,
            )

        self._chroma = chromadb.HttpClient(
            host=self._chroma_host,
            port=self._chroma_port,
            settings=settings,
        )
        self._chroma.heartbeat()

        self._episodic = self._chroma.get_collection(_EPISODIC_COLLECTION)
        self._semantic = self._chroma.get_collection(_SEMANTIC_COLLECTION)
        self._intellectual = self._get_or_create_intellectual()

        # Load embedding model — cached in memory after first load
        self._embed_fn = SentenceTransformer(_EMBEDDING_MODEL)

    def _get_or_create_intellectual(self) -> Any:
        """Return ``ubik_intellectual``, creating it with correct HNSW settings if absent."""
        try:
            return self._chroma.get_collection(_INTELLECTUAL_COLLECTION)
        except Exception:
            logger.info("Creating %s collection", _INTELLECTUAL_COLLECTION)
            return self._chroma.create_collection(
                name=_INTELLECTUAL_COLLECTION,
                metadata={
                    **_HNSW_METADATA,
                    "description": (
                        "Curated external knowledge — book highlights, curated references"
                    ),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    # -------------------------------------------------------------------------
    # Neo4j driver (lazy, optional — skipped gracefully if unavailable)
    # -------------------------------------------------------------------------

    def _neo4j(self) -> Optional[Any]:
        """Return or lazily create the Neo4j async driver.

        Returns ``None`` if neo4j is not installed or credentials are absent
        so all callers can treat it as best-effort without try/except.
        """
        if self._neo4j_driver is not None:
            return self._neo4j_driver
        if not self._neo4j_password:
            logger.debug("NEO4J_PASSWORD not set — graph ops will be skipped")
            return None
        try:
            from neo4j import AsyncGraphDatabase
            self._neo4j_driver = AsyncGraphDatabase.driver(
                self._neo4j_uri,
                auth=(self._neo4j_user, self._neo4j_password),
            )
            return self._neo4j_driver
        except ImportError:
            logger.warning("neo4j package not installed — graph ops skipped")
            return None
        except Exception as exc:
            logger.warning("Neo4j driver init failed: %s — graph ops skipped", exc)
            return None

    # -------------------------------------------------------------------------
    # Embedding helpers
    # -------------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """Return a normalised 384-dim embedding for *text*."""
        return self._embed_fn.encode(text, normalize_embeddings=True).tolist()

    async def _embed_async(self, text: str) -> List[float]:
        """Non-blocking wrapper around :meth:`_embed`."""
        return await asyncio.get_event_loop().run_in_executor(None, self._embed, text)

    # -------------------------------------------------------------------------
    # ID generators (prefix + timestamp + 8-char uuid)
    # -------------------------------------------------------------------------

    @staticmethod
    def _ep_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"ep_{ts}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _sem_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"sem_{ts}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _int_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"int_{ts}_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------------------
    # Core storage API (same signatures as HippocampalClient)
    # -------------------------------------------------------------------------

    async def store_episodic(
        self,
        content: str,
        memory_type: str,
        timestamp: Optional[str] = None,
        emotional_valence: str = "neutral",
        importance: float = 0.5,
        participants: str = "gines",
        themes: str = "",
        source_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write an episodic memory directly to ``ubik_episodic``.

        Args:
            content: Memory text.
            memory_type: Event type (letter, therapy_session, family_meeting, …).
            timestamp: ISO timestamp string; defaults to now.
            emotional_valence: Tone (positive, negative, neutral, reflective, mixed).
            importance: Score 0–1.
            participants: Comma-separated participant names.
            themes: Comma-separated theme tags.
            source_file: Original source filename for provenance.

        Returns:
            ``{"status": "success", "memory_id": "<id>"}``
        """
        memory_id = self._ep_id()
        embedding = await self._embed_async(content)
        metadata: Dict[str, Any] = {
            "type": memory_type,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat() + "Z",
            "emotional_valence": emotional_valence,
            "importance": importance,
            "participants": participants,
            "themes": themes,
            "source_file": source_file or "",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._episodic.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
            ),
        )
        await self._create_memory_node(memory_id, memory_type, metadata)
        logger.debug("Stored episodic %s (source=%s)", memory_id, source_file)
        return {"status": "success", "memory_id": memory_id}

    async def store_semantic(
        self,
        content: str,
        knowledge_type: str,
        category: str,
        confidence: float = 0.8,
        stability: str = "stable",
        source: str = "reflection",
        force: bool = False,
    ) -> Dict[str, Any]:
        """Write semantic knowledge directly to ``ubik_semantic``.

        Respects Frozen Voice mode: raises :class:`FrozenVoiceError` when
        the MCP server reports ``semantic_frozen=True`` unless *force* is set.

        Args:
            content: Belief, value, or preference statement.
            knowledge_type: Type (belief, value, preference, fact, opinion).
            category: Category (family, relationships, philosophy, career, …).
            confidence: Confidence level 0–1.
            stability: Memory stability (core, stable, evolving).
            source: Origin of knowledge.
            force: Override freeze protection.

        Returns:
            ``{"status": "success", "knowledge_id": "<id>"}``

        Raises:
            FrozenVoiceError: If semantic memory is frozen and *force* is False.
        """
        if not force and await self._is_frozen():
            raise FrozenVoiceError(
                "Semantic memory is frozen (Frozen Voice mode active). "
                "Pass force=True to override."
            )
        knowledge_id = self._sem_id()
        embedding = await self._embed_async(content)
        metadata: Dict[str, Any] = {
            "knowledge_type": knowledge_type,
            "category": category,
            "confidence": confidence,
            "stability": stability,
            "source": source,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._semantic.add(
                ids=[knowledge_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
            ),
        )
        logger.debug("Stored semantic %s (category=%s)", knowledge_id, category)
        return {"status": "success", "knowledge_id": knowledge_id}

    async def store_intellectual(
        self,
        content: str,
        external_author: Optional[str] = None,
        book_title: Optional[str] = None,
        resonance_note: Optional[str] = None,
        importance: float = 0.5,
        themes: str = "",
        source_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write curated external knowledge directly to ``ubik_intellectual``.

        External highlights are never connected to the Self node in Neo4j
        because they represent someone else's voice, not Gines's.

        Args:
            content: The highlight or passage text.
            external_author: Author of the source material.
            book_title: Title of the source book/article.
            resonance_note: Gines's personal annotation or reaction.
            importance: Score 0–1 (use ReadwiseHighlightTier mapping upstream).
            themes: Comma-separated theme tags.
            source_file: Original source filename.

        Returns:
            ``{"status": "success", "memory_id": "<id>"}``
        """
        memory_id = self._int_id()
        embedding = await self._embed_async(content)
        metadata: Dict[str, Any] = {
            "external_author": external_author or "",
            "book_title": book_title or "",
            "resonance_note": resonance_note or "",
            "importance": importance,
            "themes": themes,
            "source_file": source_file or "",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._intellectual.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
            ),
        )
        logger.debug("Stored intellectual %s (author=%s)", memory_id, external_author)
        return {"status": "success", "memory_id": memory_id}

    async def update_identity_graph(
        self,
        from_concept: str,
        relation_type: str,
        to_concept: str,
        weight: float = 1.0,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """MERGE a relationship in the Neo4j identity graph.

        Mirrors ``HippocampalClient.update_identity_graph`` so transcript
        Neo4j operations work unchanged.

        Args:
            from_concept: Source concept or entity name.
            relation_type: Relationship type label.
            to_concept: Target concept or entity name.
            weight: Relationship strength 0–1.
            context: Optional context string.

        Returns:
            ``{"status": "success"}`` or ``{"status": "skipped/error", …}``
        """
        driver = self._neo4j()
        if driver is None:
            return {"status": "skipped", "reason": "Neo4j unavailable"}
        try:
            async with driver.session() as session:
                await session.run(
                    """
                    MERGE (a:Concept {name: $from_concept})
                    MERGE (b:Concept {name: $to_concept})
                    MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
                    ON CREATE SET r.weight = $weight,
                                  r.context = $context,
                                  r.created_at = datetime()
                    ON MATCH  SET r.weight = $weight,
                                  r.context = $context,
                                  r.updated_at = datetime()
                    """,
                    from_concept=from_concept,
                    to_concept=to_concept,
                    rel_type=relation_type,
                    weight=weight,
                    context=context or "",
                )
            return {"status": "success"}
        except Exception as exc:
            logger.warning("Neo4j graph update failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, int]:
        """Return document counts for all three collections.

        Returns:
            ``{"episodic": N, "semantic": N, "intellectual": N}``
        """
        def _counts() -> Dict[str, int]:
            return {
                "episodic": self._episodic.count(),
                "semantic": self._semantic.count(),
                "intellectual": self._intellectual.count(),
            }
        return await asyncio.get_event_loop().run_in_executor(None, _counts)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _is_frozen(self) -> bool:
        """Check Frozen Voice mode via a local MCP health request (best-effort).

        Returns ``False`` if the MCP server is unreachable (don't block writes).
        """
        import httpx
        mcp_host = os.getenv("MCP_HOST", "localhost")
        mcp_port = int(os.getenv("MCP_PORT", "8080"))
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"http://{mcp_host}:{mcp_port}/health")
                return bool(resp.json().get("semantic_frozen", False))
        except Exception:
            return False

    async def _create_memory_node(
        self,
        memory_id: str,
        memory_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Create a Memory node anchored to the Self CoreIdentity node (best-effort)."""
        driver = self._neo4j()
        if driver is None:
            return
        try:
            async with driver.session() as session:
                await session.run(
                    """
                    MATCH (s:CoreIdentity {name: 'Self'})
                    MERGE (m:Memory {id: $memory_id})
                    SET m.type      = $memory_type,
                        m.ingested_at = $ingested_at
                    MERGE (s)-[:HAS_MEMORY]->(m)
                    """,
                    memory_id=memory_id,
                    memory_type=memory_type,
                    ingested_at=metadata.get("ingested_at", ""),
                )
        except Exception as exc:
            # Non-fatal — ChromaDB already committed, graph is additive
            logger.debug("Neo4j memory anchor skipped for %s: %s", memory_id, exc)

    async def close(self) -> None:
        """Close Neo4j driver. ChromaDB HTTP client is stateless."""
        if self._neo4j_driver is not None:
            try:
                await self._neo4j_driver.close()
            except Exception:
                pass
            self._neo4j_driver = None
        self._initialized = False
        self.is_connected = False
