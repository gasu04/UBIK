#!/usr/bin/env python3
"""
Maestro Configuration — Typed, Validated Settings for UBIK Orchestrator

Loads all configuration from environment variables (and an optional .env file)
into a hierarchy of pydantic-settings models.  No IP addresses or credentials
are hardcoded here; every sensitive value comes from the environment.

Architecture:
    MaestroConfig      — orchestrator operational parameters (MAESTRO_* prefix)
    HippocampalConfig  — Mac Mini M4 Pro node + hosted services
    SomaticConfig      — PowerSpec WSL2 node + hosted services
    AppConfig          — immutable composite container returned by get_config()
    get_config()       — @lru_cache factory; call this everywhere

    Key design choices:
    • Fields that share the HIPPOCAMPAL_/SOMATIC_ prefix use env_prefix so
      pydantic-settings resolves them automatically.
    • Fields whose env-var names have no node prefix (NEO4J_*, CHROMADB_*,
      MCP_PORT, VLLM_*) use validation_alias to override the prefix lookup.
    • Derived URLs are @computed_field properties — never stored, always fresh.

Usage:
    from maestro.config import get_config

    cfg = get_config()
    print(cfg.hippocampal.neo4j_http_url)   # http://100.103.242.91:7474
    print(cfg.somatic.vllm_url)             # http://100.79.166.114:8002
    print(cfg.maestro.log_level)            # INFO

    # Invalidate cache (e.g. in tests):
    get_config.cache_clear()

Dependencies:
    pydantic>=2.10.0
    pydantic-settings>=2.1.0

Author: UBIK Project
Version: 0.1.0
"""

import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# UBIK Root auto-detection
# ---------------------------------------------------------------------------

def _detect_ubik_root() -> Path:
    """Resolve the UBIK project root directory.

    Priority:
        1. ``UBIK_ROOT`` environment variable (always wins).
        2. Platform default:
           - macOS  → ``/Volumes/990PRO 4T/UBIK``
           - Linux  → ``/home/gasu/ubik``

    Returns:
        Absolute path to the UBIK root directory.

    Raises:
        ValueError: If ``UBIK_ROOT`` is unset and the platform default does
            not exist on disk.
    """
    import os

    env_root = os.getenv("UBIK_ROOT")
    if env_root:
        return Path(env_root)

    system = platform.system()
    if system == "Darwin":
        default = Path("/Volumes/990PRO 4T/UBIK")
    else:  # Linux / WSL
        default = Path("/home/gasu/ubik")

    if not default.exists():
        raise ValueError(
            f"UBIK_ROOT is not set and the platform default '{default}' does not "
            "exist.  Set the UBIK_ROOT environment variable to your UBIK root."
        )
    return default


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

class MaestroConfig(BaseSettings):
    """Maestro orchestrator operational parameters.

    All fields are read from environment variables with the ``MAESTRO_`` prefix.

    Attributes:
        log_dir: Override for the log output directory.  When ``None`` (default),
            ``AppConfig.log_dir`` resolves to ``{UBIK_ROOT}/logs/maestro/``.
        check_interval_s: Seconds between consecutive service health checks.
        log_level: Python logging level name (case-insensitive).

    Example:
        >>> cfg = MaestroConfig()
        >>> cfg.log_level
        'INFO'
    """

    log_dir: Optional[str] = None
    check_interval_s: int = 300
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="MAESTRO_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Normalise and validate the logging level string.

        Args:
            v: Raw log level string from environment.

        Returns:
            Upper-cased validated level string.

        Raises:
            ValueError: If *v* is not a recognised Python logging level.
        """
        valid: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = str(v).upper()
        if upper not in valid:
            raise ValueError(
                f"MAESTRO_LOG_LEVEL must be one of {sorted(valid)}, got: {v!r}"
            )
        return upper


class HippocampalConfig(BaseSettings):
    """Hippocampal Node (Mac Mini M4 Pro, macOS) configuration.

    Services hosted on this node:
        - Neo4j graph database
        - ChromaDB vector store
        - MCP server

    Field resolution:
        Fields prefixed ``HIPPOCAMPAL_`` in the environment are picked up via
        ``env_prefix``.  Service-level fields (``NEO4J_*``, ``CHROMADB_*``,
        ``MCP_PORT``) use ``validation_alias`` so the prefix is bypassed and
        the canonical env-var name is used directly.

    Attributes:
        tailscale_ip: Tailscale IP of this node.
        tailscale_hostname: Tailscale hostname of this node.
        neo4j_http_port: Neo4j HTTP interface port.
        neo4j_bolt_port: Neo4j Bolt protocol port.
        neo4j_user: Neo4j database user.
        neo4j_password: Neo4j database password (required in production).
        chromadb_port: ChromaDB HTTP API port.
        chromadb_token: ChromaDB bearer token (required in production).
        mcp_port: MCP server HTTP port.

    Example:
        >>> cfg = HippocampalConfig()
        >>> cfg.neo4j_http_url
        'http://100.103.242.91:7474'
    """

    # Node identity — resolved via env_prefix="HIPPOCAMPAL_"
    tailscale_ip: str = "100.103.242.91"
    tailscale_hostname: str = "ubik-hippocampal"

    # Service ports — validation_alias bypasses the HIPPOCAMPAL_ prefix
    neo4j_http_port: int = Field(7474, validation_alias="NEO4J_HTTP_PORT")
    neo4j_bolt_port: int = Field(7687, validation_alias="NEO4J_BOLT_PORT")
    neo4j_user: str = Field("neo4j", validation_alias="NEO4J_USER")
    neo4j_password: Optional[str] = Field(None, validation_alias="NEO4J_PASSWORD")
    chromadb_port: int = Field(8001, validation_alias="CHROMADB_PORT")
    chromadb_token: Optional[str] = Field(None, validation_alias="CHROMADB_TOKEN")
    mcp_port: int = Field(8080, validation_alias="MCP_PORT")

    model_config = SettingsConfigDict(
        env_prefix="HIPPOCAMPAL_",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # allow Python field names in addition to aliases
    )

    # ------------------------------------------------------------------
    # Derived URLs — computed from validated fields, never hardcoded
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def neo4j_http_url(self) -> str:
        """Full Neo4j HTTP URL (``http://<ip>:<port>``)."""
        return f"http://{self.tailscale_ip}:{self.neo4j_http_port}"

    @computed_field  # type: ignore[misc]
    @property
    def neo4j_bolt_url(self) -> str:
        """Full Neo4j Bolt URL (``bolt://<ip>:<port>``)."""
        return f"bolt://{self.tailscale_ip}:{self.neo4j_bolt_port}"

    @computed_field  # type: ignore[misc]
    @property
    def chromadb_url(self) -> str:
        """Full ChromaDB HTTP URL (``http://<ip>:<port>``)."""
        return f"http://{self.tailscale_ip}:{self.chromadb_port}"

    @computed_field  # type: ignore[misc]
    @property
    def mcp_url(self) -> str:
        """Full MCP server URL (``http://<ip>:<port>``)."""
        return f"http://{self.tailscale_ip}:{self.mcp_port}"


class SomaticConfig(BaseSettings):
    """Somatic Node (PowerSpec RTX 5090, WSL2 Linux) configuration.

    Services hosted on this node:
        - vLLM inference server

    Field resolution:
        ``SOMATIC_TAILSCALE_*`` fields use ``env_prefix``.  ``VLLM_*`` fields
        use ``validation_alias`` to bypass the prefix.

    Attributes:
        tailscale_ip: Tailscale IP of this node.
        tailscale_hostname: Tailscale hostname of this node.
        vllm_port: vLLM HTTP server port.
        vllm_model_path: Absolute path to the loaded model *on this node*.

    Example:
        >>> cfg = SomaticConfig()
        >>> cfg.vllm_url
        'http://100.79.166.114:8002'
    """

    # Node identity — resolved via env_prefix="SOMATIC_"
    tailscale_ip: str = "100.79.166.114"
    tailscale_hostname: str = "ubik-somatic"

    # Service ports — validation_alias bypasses the SOMATIC_ prefix
    vllm_port: int = Field(8002, validation_alias="VLLM_PORT")
    vllm_model_path: str = Field(
        "/home/gasu/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ",
        validation_alias="VLLM_MODEL_PATH",
    )

    model_config = SettingsConfigDict(
        env_prefix="SOMATIC_",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------
    # Derived URLs
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def vllm_url(self) -> str:
        """Base vLLM HTTP URL (``http://<ip>:<port>``)."""
        return f"http://{self.tailscale_ip}:{self.vllm_port}"

    @computed_field  # type: ignore[misc]
    @property
    def vllm_openai_url(self) -> str:
        """vLLM OpenAI-compatible API endpoint (``/v1``)."""
        return f"http://{self.tailscale_ip}:{self.vllm_port}/v1"


# ---------------------------------------------------------------------------
# Composite container
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Composite, read-only configuration container for UBIK Maestro.

    Aggregates all node and service configurations into a single object
    suitable for dependency injection throughout the orchestrator.

    Attributes:
        ubik_root: Detected UBIK project root directory.
        maestro: Maestro orchestrator operational settings.
        hippocampal: Hippocampal node (Mac Mini M4 Pro) settings.
        somatic: Somatic node (PowerSpec WSL2) settings.

    Example:
        >>> from maestro.config import get_config
        >>> cfg = get_config()
        >>> cfg.hippocampal.neo4j_http_url
        'http://100.103.242.91:7474'
        >>> cfg.log_dir
        PosixPath('/Volumes/990PRO 4T/UBIK/logs/maestro')

    See Also:
        get_config: Cached factory that returns this object.
    """

    ubik_root: Path
    maestro: MaestroConfig
    hippocampal: HippocampalConfig
    somatic: SomaticConfig

    @property
    def log_dir(self) -> Path:
        """Resolved log directory.

        Returns ``MAESTRO_LOG_DIR`` when set; otherwise defaults to
        ``{ubik_root}/logs/maestro/``.

        Returns:
            Absolute path to the Maestro log directory.
        """
        if self.maestro.log_dir:
            return Path(self.maestro.log_dir)
        return self.ubik_root / "logs" / "maestro"


# ---------------------------------------------------------------------------
# Cached factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the cached application configuration.

    Resolves UBIK_ROOT, locates the ``.env`` file, and constructs all
    sub-configurations on the first call.  Every subsequent call returns the
    same ``AppConfig`` instance (zero I/O overhead).

    Returns:
        Fully initialised :class:`AppConfig` instance.

    Raises:
        ValueError: If ``UBIK_ROOT`` cannot be detected.
        pydantic.ValidationError: If a required environment variable is
            missing or fails validation.

    Example:
        >>> from maestro.config import get_config
        >>> cfg = get_config()
        >>> print(cfg.maestro.log_level)
        INFO

    Note:
        Call ``get_config.cache_clear()`` to force a reload — useful in tests
        or after mutating environment variables at runtime.
    """
    ubik_root = _detect_ubik_root()
    env_file = ubik_root / "maestro" / ".env"
    env_file_arg: Optional[str] = str(env_file) if env_file.exists() else None

    return AppConfig(
        ubik_root=ubik_root,
        maestro=MaestroConfig(_env_file=env_file_arg),
        hippocampal=HippocampalConfig(_env_file=env_file_arg),
        somatic=SomaticConfig(_env_file=env_file_arg),
    )
