"""RAG service for Gines voice generation.

CRITICAL: Uses AsyncOpenAI for non-blocking inference.
Never use synchronous OpenAI client in async context.

This module supports the Golden Set evaluation framework (Section 7.3).
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from context import BuiltContext, UnifiedContextBuilder
from mcp_client import CircuitBreaker, CircuitOpenError, HippocampalClientV2
from mcp_client.resilience import RetryConfig, retry_async
from prompts import (
    VOICE_SYSTEM_PROMPT,
    format_voice_prompt,
    get_stop_tokens,
    parse_response,
)

logger = logging.getLogger("ubik.rag.service")


# =============================================================================
# Template Types (for Golden Set evaluation)
# =============================================================================

class TemplateType(Enum):
    """Voice template types for different conversation contexts."""
    DEFAULT = "default"
    FAMILY = "family"
    REFLECTIVE = "reflective"
    ADVISORY = "advisory"
    SIMPLE = "simple"


def select_template(query: str, recipient: str = "family") -> TemplateType:
    """
    Select appropriate voice template based on query content.

    Args:
        query: User's query text
        recipient: Who is asking (family, friend, etc.)

    Returns:
        Appropriate template type
    """
    query_lower = query.lower()

    # Family-related queries
    if any(word in query_lower for word in ["grandchild", "grandson", "granddaughter", "family", "children", "kids"]):
        return TemplateType.FAMILY

    # Reflective/philosophical queries
    if any(word in query_lower for word in ["believe", "think about", "feel about", "meaning", "purpose", "life"]):
        return TemplateType.REFLECTIVE

    # Advice-seeking queries
    if any(word in query_lower for word in ["advice", "should i", "what would you", "help me", "guide"]):
        return TemplateType.ADVISORY

    # Simple/direct queries
    if len(query.split()) < 8 or query.endswith("?") and len(query) < 50:
        return TemplateType.SIMPLE

    return TemplateType.DEFAULT


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    @classmethod
    def from_settings(cls) -> "GenerationConfig":
        """Create GenerationConfig from application settings."""
        settings = get_settings()
        return cls(
            temperature=settings.rag.default_temperature,
            max_tokens=settings.rag.default_max_tokens,
        )


@dataclass
class RAGResult:
    """Result from RAG generation.

    Supports the Golden Set evaluation framework (Section 7.3).

    Attributes:
        response: Clean response with reasoning stripped
        raw_response: Original model output including reasoning blocks
        reasoning: Extracted reasoning (for debugging)
        context: Context used for generation
        template_name: Voice template used for generation
        retrieval_time_ms: Time spent retrieving memories
        generation_time_ms: Time spent in model inference
        total_time_ms: Total pipeline time
        model: Model used for generation
    """

    response: str  # Clean response (reasoning stripped)
    raw_response: str  # Original model output
    reasoning: Optional[str]  # Extracted reasoning (for debugging)
    context: BuiltContext  # Context used for generation
    template_name: str = "default"  # Voice template used
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def had_reasoning(self) -> bool:
        """Check if model used reasoning."""
        return self.reasoning is not None

    @property
    def reasoning_stripped(self) -> bool:
        """Alias for had_reasoning (Golden Set compatibility)."""
        return self.reasoning is not None

    @property
    def context_used(self) -> str:
        """Get context string (Golden Set compatibility)."""
        return self.context.context if self.context else ""


@dataclass
class RAGServiceStats:
    """Statistics for RAG service operations."""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    circuit_rejections: int = 0
    total_generation_time_ms: float = 0.0

    def record_success(self, generation_time_ms: float) -> None:
        """Record a successful generation."""
        self.requests += 1
        self.successes += 1
        self.total_generation_time_ms += generation_time_ms

    def record_failure(self) -> None:
        """Record a failed generation."""
        self.requests += 1
        self.failures += 1

    def record_circuit_rejection(self) -> None:
        """Record a circuit breaker rejection."""
        self.requests += 1
        self.circuit_rejections += 1

    @property
    def avg_generation_time_ms(self) -> float:
        """Average generation time in milliseconds."""
        if self.successes == 0:
            return 0.0
        return self.total_generation_time_ms / self.successes

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "circuit_rejections": self.circuit_rejections,
            "avg_generation_time_ms": round(self.avg_generation_time_ms, 2),
        }


class RAGService:
    """RAG service for Gines voice generation.

    Uses:
    - AsyncOpenAI for non-blocking vLLM inference
    - UnifiedContextBuilder for memory retrieval
    - CircuitBreaker for fault tolerance
    - Retry logic for transient failures

    Usage:
        async with RAGService() as service:
            result = await service.generate(
                query="What does family mean to you?",
                recipient="grandchildren",
            )
            print(result.response)
    """

    def __init__(
        self,
        vllm_base_url: Optional[str] = None,
        model: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """Initialize RAG service.

        Args:
            vllm_base_url: vLLM API base URL (defaults to settings).
            model: Model path/name (defaults to settings).
            generation_config: Generation parameters (defaults to settings).
        """
        settings = get_settings()

        self._vllm_base_url = vllm_base_url or settings.vllm.base_url
        self._model = model or settings.vllm.model_path
        self._config = generation_config or GenerationConfig.from_settings()

        # AsyncOpenAI client (CRITICAL: must be async)
        self._client: Optional[AsyncOpenAI] = None

        # MCP client for memory retrieval (V2: retry + circuit breaker + connection mgmt)
        self._mcp_client: Optional[HippocampalClientV2] = None
        self._context_builder: Optional[UnifiedContextBuilder] = None

        # Circuit breakers for resilience
        self._vllm_circuit = CircuitBreaker("vllm")
        self._mcp_circuit = CircuitBreaker("hippocampal-mcp")

        # Statistics
        self.stats = RAGServiceStats()

        # Stop tokens for generation
        self._stop_tokens = get_stop_tokens()

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create AsyncOpenAI client."""
        if self._client is None:
            logger.debug(f"Creating AsyncOpenAI client for {self._vllm_base_url}")
            self._client = AsyncOpenAI(
                base_url=self._vllm_base_url,
                api_key="not-needed",  # vLLM doesn't require API key
            )
        return self._client

    async def _get_context_builder(self) -> UnifiedContextBuilder:
        """Get or create context builder with MCP client."""
        if self._context_builder is None:
            if self._mcp_client is None:
                self._mcp_client = HippocampalClientV2()
                await self._mcp_client.connect()
            self._context_builder = UnifiedContextBuilder(self._mcp_client)
        return self._context_builder

    async def generate(
        self,
        query: str,
        recipient: str = "family",
        conversation_history: Optional[str] = None,
        include_identity: bool = True,
        config_override: Optional[GenerationConfig] = None,
    ) -> RAGResult:
        """Generate a response using RAG.

        Args:
            query: The user's question or prompt.
            recipient: Who is asking (e.g., "family", "grandchildren").
            conversation_history: Optional conversation context.
            include_identity: Include identity graph context.
            config_override: Override generation parameters.

        Returns:
            RAGResult with response and metadata.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            Exception: On generation failure after retries.
        """
        total_start = time.perf_counter()
        config = config_override or self._config

        # 1. Check vLLM circuit breaker
        if not await self._vllm_circuit.allow_request():
            self.stats.record_circuit_rejection()
            raise CircuitOpenError("vllm")

        try:
            # 2. Select template based on query
            template_type = select_template(query, recipient)

            # 3. Build context (with MCP circuit breaker)
            retrieval_start = time.perf_counter()
            context = await self._build_context_safe(
                query=query,
                include_identity=include_identity,
            )
            retrieval_time = (time.perf_counter() - retrieval_start) * 1000

            # 4. Format prompt
            prompt = format_voice_prompt(
                context=context.context,
                query=query,
                recipient=recipient,
                history=conversation_history,
            )

            # 5. Generate with retry
            gen_start = time.perf_counter()
            raw_response = await self._generate_with_retry(
                prompt=prompt,
                config=config,
            )
            gen_time = (time.perf_counter() - gen_start) * 1000

            # 6. Parse response (strip reasoning)
            parsed = parse_response(raw_response)

            # 7. Record success
            await self._vllm_circuit.record_success()
            self.stats.record_success(gen_time)

            total_time = (time.perf_counter() - total_start) * 1000

            logger.info(
                f"RAG generation complete: "
                f"template={template_type.value}, "
                f"context={context.metrics.estimated_tokens} tokens, "
                f"retrieval={retrieval_time:.0f}ms, "
                f"gen_time={gen_time:.0f}ms, "
                f"had_reasoning={parsed.has_reasoning}"
            )

            return RAGResult(
                response=parsed.clean_response,
                raw_response=parsed.raw_response,
                reasoning=parsed.reasoning,
                context=context,
                template_name=template_type.value,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=gen_time,
                total_time_ms=total_time,
                model=self._model,
                metadata={
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "recipient": recipient,
                },
            )

        except CircuitOpenError:
            raise
        except Exception as e:
            await self._vllm_circuit.record_failure()
            self.stats.record_failure()
            logger.error(f"RAG generation failed: {e}")
            raise

    async def ask(
        self,
        query: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        recipient: str = "family",
    ) -> RAGResult:
        """Generate a response in Gines's authentic voice.

        Convenience wrapper around ``generate()`` with a simpler interface.

        Args:
            query: The question or prompt.
            temperature: Generation temperature (0.0-2.0).
            max_tokens: Maximum response tokens.
            recipient: Who is asking (affects tone).

        Returns:
            RAGResult with clean output and metadata.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            Exception: On generation failure after retries.
        """
        config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await self.generate(
            query=query,
            recipient=recipient,
            config_override=config,
        )

    async def _build_context_safe(
        self,
        query: str,
        include_identity: bool,
    ) -> BuiltContext:
        """Build context with circuit breaker protection."""
        if not await self._mcp_circuit.allow_request():
            logger.warning("MCP circuit open, using empty context")
            from context import BuiltContext, ContextMetrics

            return BuiltContext(
                context="",
                metrics=ContextMetrics(),
                query=query,
            )

        try:
            builder = await self._get_context_builder()
            context = await builder.build_context(
                query=query,
                include_identity=include_identity,
            )
            await self._mcp_circuit.record_success()
            return context

        except Exception as e:
            await self._mcp_circuit.record_failure()
            logger.warning(f"Context retrieval failed: {e}, using empty context")
            from context import BuiltContext, ContextMetrics

            return BuiltContext(
                context="",
                metrics=ContextMetrics(),
                query=query,
            )

    async def _generate_with_retry(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text with retry logic."""

        async def _do_generate() -> str:
            client = await self._get_client()

            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": VOICE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                # Note: vLLM handles stop tokens via chat template
            )

            return response.choices[0].message.content or ""

        # Use retry logic
        retry_config = RetryConfig.from_settings()
        return await retry_async(_do_generate, config=retry_config)

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all service dependencies.

        Returns:
            Dict with "status" key ("healthy", "degraded", "unhealthy")
            and component-level status details.

        Required by Golden Set evaluation framework (Section 7.3).
        """
        status: Dict[str, Any] = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "stats": self.stats.to_dict(),
        }

        # Check Hippocampal Node / MCP connection
        try:
            builder = await self._get_context_builder()
            if builder.mcp:
                health = await builder.mcp.health_check()
                if health.get("status") == "success":
                    status["components"]["hippocampal"] = {
                        "status": "healthy",
                        "circuit": self._mcp_circuit.state.value,
                    }
                else:
                    status["status"] = "degraded"
                    status["components"]["hippocampal"] = {
                        "status": "error",
                        "message": health.get("message", "Unknown error"),
                        "circuit": self._mcp_circuit.state.value,
                    }
        except Exception as e:
            status["status"] = "degraded"
            status["components"]["hippocampal"] = {
                "status": "error",
                "message": str(e),
                "circuit": self._mcp_circuit.state.value,
            }

        # Check vLLM (critical - failure makes service unhealthy)
        try:
            client = await self._get_client()
            await client.models.list()
            status["components"]["vllm"] = {
                "status": "healthy",
                "model": self._model,
                "circuit": self._vllm_circuit.state.value,
            }
        except Exception as e:
            status["status"] = "unhealthy"
            status["components"]["vllm"] = {
                "status": "error",
                "message": str(e),
                "circuit": self._vllm_circuit.state.value,
            }

        return status

    async def _health_check_legacy(self) -> dict:
        """Legacy health check format (deprecated).

        Returns:
            Dict with status of vLLM and MCP services.
        """
        result = {
            "vllm": {"status": "unknown", "circuit": self._vllm_circuit.state.value},
            "mcp": {"status": "unknown", "circuit": self._mcp_circuit.state.value},
            "stats": self.stats.to_dict(),
        }

        # Check vLLM
        try:
            client = await self._get_client()
            await client.models.list()
            result["vllm"]["status"] = "healthy"
            result["vllm"]["model"] = self._model
        except Exception as e:
            result["vllm"]["status"] = "unhealthy"
            result["vllm"]["error"] = str(e)

        # Check MCP
        try:
            builder = await self._get_context_builder()
            if builder.mcp:
                health = await builder.mcp.health_check()
                if health.get("status") == "success":
                    result["mcp"]["status"] = "healthy"
                else:
                    result["mcp"]["status"] = "unhealthy"
                    result["mcp"]["error"] = health.get("message", "Unknown error")
        except Exception as e:
            result["mcp"]["status"] = "unhealthy"
            result["mcp"]["error"] = str(e)

        return result

    async def close(self) -> None:
        """Clean shutdown of RAG service."""
        if self._mcp_client:
            await self._mcp_client.disconnect()
            self._mcp_client = None
            self._context_builder = None

        if self._client:
            await self._client.close()
            self._client = None

        logger.info(f"RAG service closed. Stats: {self.stats.to_dict()}")

    async def __aenter__(self) -> "RAGService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
