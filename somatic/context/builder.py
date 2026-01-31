"""Unified context builder for RAG generation.

Builds context with token budget constraints, balancing:
- Long-term retrieval (semantic + episodic memories)
- Identity graph anchors

Phase 3.5 will add:
- Short-term conversation buffer (3-5 turns)
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings

logger = logging.getLogger("ubik.context.builder")


@dataclass
class ContextBudget:
    """Token budget allocation for context building.

    Default allocation (2048 tokens):
    - Retrieval: 60% (~1229 tokens) - semantic + episodic
    - Identity: 20% (~410 tokens) - core values from graph
    - Reserved: 20% (~410 tokens) - Phase 3.5 conversation history
    """

    total_tokens: int = 2048
    retrieval_pct: float = 0.60
    identity_pct: float = 0.20
    # Note: remaining 20% reserved for Phase 3.5 conversation history

    @property
    def retrieval_tokens(self) -> int:
        """Token budget for semantic + episodic retrieval."""
        return int(self.total_tokens * self.retrieval_pct)

    @property
    def identity_tokens(self) -> int:
        """Token budget for identity graph context."""
        return int(self.total_tokens * self.identity_pct)

    @property
    def reserved_tokens(self) -> int:
        """Reserved tokens for Phase 3.5 features."""
        remaining = 1.0 - self.retrieval_pct - self.identity_pct
        return int(self.total_tokens * remaining)

    @classmethod
    def from_settings(cls) -> "ContextBudget":
        """Create ContextBudget from application settings."""
        settings = get_settings()
        return cls(
            total_tokens=settings.rag.context_budget_total,
            retrieval_pct=settings.rag.context_budget_retrieval_pct / 100.0,
            identity_pct=settings.rag.context_budget_identity_pct / 100.0,
        )


@dataclass
class RetrievalLimits:
    """Hard caps on retrieval counts.

    These limits prevent context explosion regardless of budget.
    """

    max_semantic: int = 12  # Core beliefs, values, preferences
    max_episodic: int = 10  # Relevant experiences, conversations
    max_identity_concepts: int = 5  # Anchoring values from Neo4j

    @classmethod
    def from_settings(cls) -> "RetrievalLimits":
        """Create RetrievalLimits from application settings."""
        settings = get_settings()
        return cls(
            max_semantic=settings.rag.max_semantic_memories,
            max_episodic=settings.rag.max_episodic_memories,
            max_identity_concepts=settings.rag.max_identity_concepts,
        )


@dataclass
class ContextMetrics:
    """Metrics from context building for observability."""

    semantic_count: int = 0
    episodic_count: int = 0
    identity_concepts: int = 0
    total_chars: int = 0
    estimated_tokens: int = 0
    retrieval_time_ms: float = 0.0
    identity_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "semantic_count": self.semantic_count,
            "episodic_count": self.episodic_count,
            "identity_concepts": self.identity_concepts,
            "total_chars": self.total_chars,
            "estimated_tokens": self.estimated_tokens,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "identity_time_ms": round(self.identity_time_ms, 2),
        }


@dataclass
class BuiltContext:
    """Result of context building."""

    context: str
    metrics: ContextMetrics
    query: str

    def __str__(self) -> str:
        return self.context


class UnifiedContextBuilder:
    """Builds context for RAG generation with budget constraints.

    Phase 3 Scope:
    - Long-term retrieval (semantic + episodic)
    - Identity graph anchors

    Phase 3.5 Addition (TODO):
    - Short-term conversation buffer (3-5 turns)

    Usage:
        async with HippocampalClient() as client:
            builder = UnifiedContextBuilder(client)
            result = await builder.build_context("family values")
            print(result.context)
            print(result.metrics.to_dict())
    """

    # Approximate tokens per character (conservative estimate)
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        mcp_client: Any,
        budget: Optional[ContextBudget] = None,
        limits: Optional[RetrievalLimits] = None,
    ):
        """Initialize context builder.

        Args:
            mcp_client: Connected MCP client (HippocampalClient or V2).
            budget: Token budget configuration (defaults to settings).
            limits: Retrieval limits (defaults to settings).
        """
        self.mcp = mcp_client
        self.budget = budget or ContextBudget.from_settings()
        self.limits = limits or RetrievalLimits.from_settings()

    async def build_context(
        self,
        query: str,
        include_identity: bool = True,
        # Phase 3.5: conversation_history: List[Turn] = None
    ) -> BuiltContext:
        """Build context string for generation.

        Args:
            query: The user's query to retrieve context for.
            include_identity: Whether to include identity graph context.

        Returns:
            BuiltContext with formatted context and metrics.
        """
        import time

        context_parts: list[str] = []
        metrics = ContextMetrics()

        # 1. Retrieve semantic memories (beliefs, values, preferences)
        retrieval_start = time.perf_counter()

        semantic = await self._retrieve_semantic(query)
        if semantic:
            metrics.semantic_count = len(semantic)
            context_parts.append("## Core Beliefs & Values")
            for mem in semantic:
                category = mem.metadata.get("category", "general")
                context_parts.append(f"- [{category}] {mem.content}")
            logger.debug(f"Retrieved {len(semantic)} semantic memories")

        # 2. Retrieve episodic memories (experiences, conversations)
        episodic = await self._retrieve_episodic(query)
        if episodic:
            metrics.episodic_count = len(episodic)
            context_parts.append("\n## Relevant Experiences")
            for mem in episodic:
                timestamp = mem.metadata.get("timestamp", "")[:10]
                mem_type = mem.metadata.get("type", "memory")
                # Truncate long memories to conserve tokens
                content = self._truncate(mem.content, max_chars=200)
                context_parts.append(f"- [{timestamp}] ({mem_type}) {content}")
            logger.debug(f"Retrieved {len(episodic)} episodic memories")

        retrieval_end = time.perf_counter()
        metrics.retrieval_time_ms = (retrieval_end - retrieval_start) * 1000

        # 3. Identity graph anchors (optional, non-blocking on failure)
        if include_identity:
            identity_start = time.perf_counter()
            identity_context = await self._retrieve_identity(query)
            if identity_context:
                metrics.identity_concepts = identity_context["concept_count"]
                context_parts.append("\n## Identity Anchors")
                context_parts.append(identity_context["formatted"])
            identity_end = time.perf_counter()
            metrics.identity_time_ms = (identity_end - identity_start) * 1000

        # Build final context
        context = "\n".join(context_parts)
        metrics.total_chars = len(context)
        metrics.estimated_tokens = self._estimate_tokens(context)

        # Log metrics (never log actual content for privacy)
        logger.info(
            f"Built context for query (len={len(query)}): "
            f"semantic={metrics.semantic_count}, "
            f"episodic={metrics.episodic_count}, "
            f"identity={metrics.identity_concepts}, "
            f"~{metrics.estimated_tokens} tokens"
        )

        return BuiltContext(
            context=context,
            metrics=metrics,
            query=query,
        )

    async def _retrieve_semantic(self, query: str) -> list[Any]:
        """Retrieve semantic memories with limit enforcement."""
        try:
            results = await self.mcp.query_semantic(
                query=query,
                n_results=self.limits.max_semantic,
            )
            return results[: self.limits.max_semantic]
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return []

    async def _retrieve_episodic(self, query: str) -> list[Any]:
        """Retrieve episodic memories with limit enforcement."""
        try:
            results = await self.mcp.query_episodic(
                query=query,
                n_results=self.limits.max_episodic,
            )
            return results[: self.limits.max_episodic]
        except Exception as e:
            logger.error(f"Episodic retrieval failed: {e}")
            return []

    async def _retrieve_identity(self, query: str) -> Optional[dict]:
        """Retrieve identity graph context.

        Handles both V1 (dict) and V2 (IdentityContext) return types.

        Returns:
            Dict with 'formatted' string and 'concept_count', or None on failure.
        """
        try:
            # Query identity around core concepts
            concepts_found: list[str] = []

            for concept in ["Self", "Family Legacy", "Core Values"]:
                try:
                    identity = await self.mcp.get_identity_context(
                        concept=concept,
                        depth=1,
                    )

                    # Handle V2 IdentityContext (has .paths attribute)
                    if hasattr(identity, "paths"):
                        paths = identity.paths
                    # Handle V1 dict return
                    elif isinstance(identity, dict):
                        if identity.get("status") != "success":
                            continue
                        paths = identity.get("context", [])
                    else:
                        continue

                    for path in paths:
                        nodes = path.get("nodes", [])
                        for node in nodes:
                            name = node.get("name", "")
                            if name and name not in concepts_found:
                                concepts_found.append(name)

                    # Respect limit
                    if len(concepts_found) >= self.limits.max_identity_concepts:
                        break

                except Exception as e:
                    logger.debug(f"Identity lookup for '{concept}' failed: {e}")
                    continue

            if not concepts_found:
                return None

            # Truncate to limit
            concepts_found = concepts_found[: self.limits.max_identity_concepts]

            return {
                "formatted": f"Core connected values: {', '.join(concepts_found)}",
                "concept_count": len(concepts_found),
            }

        except Exception as e:
            logger.warning(f"Identity retrieval failed: {e}")
            return None

    def _truncate(self, text: str, max_chars: int) -> str:
        """Truncate text to max characters with ellipsis."""
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count.

        Uses conservative estimate of ~4 chars per token.
        For precise counting, use the tokenizer directly.
        """
        return len(text) // self.CHARS_PER_TOKEN
