"""Configuration loader for UBIK Somatic Node.

All configuration values are loaded from environment variables.
No hardcoded network addresses, ports, or credentials allowed.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Path to .env file (relative to working directory)
_ENV_FILE = Path(__file__).parent.parent / ".env"


class HippocampalSettings(BaseSettings):
    """Hippocampal node connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="HIPPOCAMPAL_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="ubik-hippocampal",
        description="Hippocampal node hostname (Tailscale or IP)",
    )
    mcp_port: int = Field(default=8080, description="MCP server port")
    chroma_port: int = Field(default=8001, description="ChromaDB port")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mcp_url(self) -> str:
        """Compute MCP endpoint URL."""
        return f"http://{self.host}:{self.mcp_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chroma_url(self) -> str:
        """Compute ChromaDB endpoint URL."""
        return f"http://{self.host}:{self.chroma_port}"


class VLLMSettings(BaseSettings):
    """vLLM inference server settings."""

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="localhost", description="vLLM server hostname")
    port: int = Field(default=8002, description="vLLM server port")
    model_path: str = Field(description="Path to the model weights")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_url(self) -> str:
        """Compute vLLM OpenAI-compatible API base URL."""
        return f"http://{self.host}:{self.port}/v1"


class RAGSettings(BaseSettings):
    """RAG service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Context budget (tokens)
    context_budget_total: int = Field(
        default=2048, description="Total context budget in tokens"
    )
    context_budget_retrieval_pct: int = Field(
        default=60, description="Percentage of budget for retrieval"
    )
    context_budget_identity_pct: int = Field(
        default=20, description="Percentage of budget for identity"
    )

    # Retrieval limits (hard caps)
    max_semantic_memories: int = Field(
        default=12, description="Maximum semantic memories to retrieve"
    )
    max_episodic_memories: int = Field(
        default=10, description="Maximum episodic memories to retrieve"
    )
    max_identity_concepts: int = Field(
        default=5, description="Maximum identity concepts to retrieve"
    )

    # Generation defaults
    default_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Default generation temperature"
    )
    default_max_tokens: int = Field(
        default=512, gt=0, description="Default max tokens for generation"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def retrieval_token_budget(self) -> int:
        """Compute token budget for retrieval."""
        return int(self.context_budget_total * self.context_budget_retrieval_pct / 100)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def identity_token_budget(self) -> int:
        """Compute token budget for identity context."""
        return int(self.context_budget_total * self.context_budget_identity_pct / 100)


class ResilienceSettings(BaseSettings):
    """Resilience configuration for retry and circuit breaker patterns."""

    # Retry logic
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        alias="RETRY_MAX_ATTEMPTS",
        description="Maximum retry attempts",
    )
    retry_base_delay_ms: int = Field(
        default=1000,
        ge=0,
        alias="RETRY_BASE_DELAY_MS",
        description="Base delay between retries in milliseconds",
    )
    retry_max_delay_ms: int = Field(
        default=30000,
        ge=0,
        alias="RETRY_MAX_DELAY_MS",
        description="Maximum delay between retries in milliseconds",
    )
    retry_jitter_max_ms: int = Field(
        default=500,
        ge=0,
        alias="RETRY_JITTER_MAX_MS",
        description="Maximum jitter to add to delays in milliseconds",
    )

    # Circuit breaker
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        description="Number of failures before circuit opens",
    )
    circuit_breaker_recovery_timeout_s: int = Field(
        default=60,
        ge=1,
        alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S",
        description="Seconds before attempting recovery",
    )

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def retry_base_delay_s(self) -> float:
        """Compute base delay in seconds."""
        return self.retry_base_delay_ms / 1000.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def retry_max_delay_s(self) -> float:
        """Compute max delay in seconds."""
        return self.retry_max_delay_ms / 1000.0


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    format: Literal["json", "text"] = Field(
        default="json", description="Log output format"
    )
    dir: Path = Field(
        default=Path("/home/gasu/ubik/logs"), description="Directory for log files"
    )
    memory_content: bool = Field(
        default=False,
        description="Whether to log raw memory content (SECURITY: keep False)",
    )


class Settings(BaseSettings):
    """Root settings aggregating all configuration sections.

    Usage:
        from config.settings import get_settings

        settings = get_settings()
        print(settings.vllm.base_url)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    hippocampal: HippocampalSettings = Field(
        default_factory=HippocampalSettings  # type: ignore[arg-type]
    )
    vllm: VLLMSettings = Field(
        default_factory=VLLMSettings  # type: ignore[arg-type]
    )
    rag: RAGSettings = Field(
        default_factory=RAGSettings  # type: ignore[arg-type]
    )
    resilience: ResilienceSettings = Field(
        default_factory=ResilienceSettings  # type: ignore[arg-type]
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings  # type: ignore[arg-type]
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings loaded from environment variables.

    Note:
        Settings are cached for performance. Call `get_settings.cache_clear()`
        if you need to reload settings (e.g., in tests).
    """
    return Settings()
