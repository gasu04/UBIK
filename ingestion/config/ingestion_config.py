"""
UBIK Ingestion System - Configuration

Typed configuration for the ingestion pipeline: LLM endpoints (per
privacy tier), directory layout, Gate 1 thresholds, and prompt
versioning. All values come from environment variables (the
``ingestion/.env`` file is loaded first without overriding the process
environment) — no network addresses or thresholds are hardcoded here.

How it fits in: enrichment and Gate 1 import ``load_config()`` and
read endpoints/paths/thresholds from the returned ``IngestionConfig``.

Usage:
    from config.ingestion_config import load_config

    config = load_config()
    url = config.endpoints.sensitive_endpoint   # Somatic vLLM, per tier
    if score < config.gates.quarantine_below:
        ...

Dependencies: stdlib only.

Tier classification: Tier 2 (standard, 80% coverage). Misconfiguration
fails loud at load time or at first connection attempt.

Version: 0.1.0
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    'EndpointConfig',
    'PathsConfig',
    'GateThresholds',
    'IngestionConfig',
    'load_config',
    'DEFAULT_PROMPT_VERSION',
    'ALL_PRIVACY_TIERS',
]

DEFAULT_PROMPT_VERSION = "v1.0.0"

# Privacy tiers from the Known_Persons registry.
# Keep in sync with ingest/registry.py PRIVACY_TIERS.
ALL_PRIVACY_TIERS = frozenset({"private", "therapy", "family", "business"})

# ingestion/ directory (this file lives in ingestion/config/)
_INGESTION_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding set values."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _somatic_vllm_url() -> str:
    """
    Compose the default Somatic vLLM base URL from environment.

    Resolution order:
        1. SOMATIC_VLLM_URL (full URL, wins outright)
        2. http://{SOMATIC_TAILSCALE_IP}:{VLLM_PORT}/v1
        3. http://{SOMATIC_LAN_IP}:{VLLM_PORT}/v1 (LAN fallback)
        4. http://{SOMATIC_TAILSCALE_HOSTNAME or 'ubik-somatic'}:{VLLM_PORT}/v1
    """
    explicit = os.environ.get("SOMATIC_VLLM_URL")
    if explicit:
        return explicit.rstrip("/")

    port = os.environ.get("VLLM_PORT", "8002")
    host = (
        os.environ.get("SOMATIC_TAILSCALE_IP")
        or os.environ.get("SOMATIC_LAN_IP")
        or os.environ.get("SOMATIC_TAILSCALE_HOSTNAME", "ubik-somatic")
    )
    return f"http://{host}:{port}/v1"


@dataclass(frozen=True)
class EndpointConfig:
    """
    LLM endpoints by privacy tier.

    Privacy tiers come from the Known_Persons registry
    (private, therapy, family, business). Routing is fail-safe: every
    tier goes to the sensitive endpoint — by default the self-hosted
    Somatic vLLM, never a cloud API — unless the tier is explicitly
    listed in UBIK_STANDARD_PRIVACY_TIERS.

    Attributes:
        sensitive_endpoint: OpenAI-compatible base URL for sensitive content.
        standard_endpoint: Base URL for tiers explicitly relaxed to standard.
        lan_fallback_endpoint: Optional LAN URL to try if the primary
            (Tailscale) route is unreachable.
        model: Model name/path served by the endpoints.
        request_timeout_seconds: Per-request timeout for enrichment calls.
        standard_tiers: Privacy tiers allowed on the standard endpoint
            (default: none — everything routes sensitive).
    """
    sensitive_endpoint: str
    standard_endpoint: str
    lan_fallback_endpoint: Optional[str] = None
    model: str = ""
    request_timeout_seconds: float = 120.0
    standard_tiers: frozenset = frozenset()

    @classmethod
    def from_env(cls) -> "EndpointConfig":
        """Build endpoint config from environment variables."""
        default_url = _somatic_vllm_url()
        lan_ip = os.environ.get("SOMATIC_LAN_IP")
        port = os.environ.get("VLLM_PORT", "8002")
        lan_url = os.environ.get(
            "SOMATIC_VLLM_LAN_URL",
            f"http://{lan_ip}:{port}/v1" if lan_ip else None,
        )
        standard_tiers = frozenset(
            t.strip().lower()
            for t in os.environ.get("UBIK_STANDARD_PRIVACY_TIERS", "").split(",")
            if t.strip()
        )
        unknown = standard_tiers - ALL_PRIVACY_TIERS
        if unknown:
            raise ValueError(
                f"UBIK_STANDARD_PRIVACY_TIERS contains unknown tiers {sorted(unknown)}; "
                f"valid: {sorted(ALL_PRIVACY_TIERS)}"
            )
        return cls(
            sensitive_endpoint=os.environ.get("UBIK_SENSITIVE_LLM_URL", default_url),
            standard_endpoint=os.environ.get("UBIK_STANDARD_LLM_URL", default_url),
            lan_fallback_endpoint=lan_url,
            model=os.environ.get("UBIK_ENRICHMENT_MODEL", ""),
            request_timeout_seconds=float(
                os.environ.get("UBIK_LLM_TIMEOUT_SECONDS", "120")
            ),
            standard_tiers=standard_tiers,
        )

    def for_tier(self, privacy_tier: str) -> str:
        """
        Return the endpoint URL for a Known_Persons privacy tier.

        Args:
            privacy_tier: One of "private", "therapy", "family", "business".

        Returns:
            The standard endpoint if the tier is explicitly relaxed via
            UBIK_STANDARD_PRIVACY_TIERS; the sensitive endpoint otherwise.

        Raises:
            ValueError: On a tier not in ALL_PRIVACY_TIERS — a typo must
                never silently route anywhere.
        """
        tier = privacy_tier.strip().lower()
        if tier not in ALL_PRIVACY_TIERS:
            raise ValueError(
                f"Unknown privacy tier: {privacy_tier!r}; "
                f"valid: {sorted(ALL_PRIVACY_TIERS)}"
            )
        if tier in self.standard_tiers:
            return self.standard_endpoint
        return self.sensitive_endpoint


@dataclass(frozen=True)
class PathsConfig:
    """
    Directory layout of the ingestion workspace.

    All paths derive from ``UBIK_INGESTION_ROOT`` (default: the
    ingestion/ directory containing this package).

    Attributes:
        root: Ingestion workspace root.
        sources_dir: Raw, immutable originals organized by source type.
        enriched_dir: Enrichment output, regenerable, versioned in filename.
        pending_review_dir: Gate 1 queue.
        approved_dir: Passed Gate 1, ready for IngestPipeline.
        quarantine_source_dir: Quarantined source files.
        quarantine_enrichment_dir: Quarantined enrichment output.
        registry_dir: YAML registries (known_persons, content_types).
        qa_dir: Learned rules, schema, and rubric docs.
        quality_log: JSONL quality/audit log.
    """
    root: Path
    sources_dir: Path
    enriched_dir: Path
    pending_review_dir: Path
    approved_dir: Path
    quarantine_source_dir: Path
    quarantine_enrichment_dir: Path
    registry_dir: Path
    qa_dir: Path
    quality_log: Path

    @classmethod
    def from_env(cls) -> "PathsConfig":
        """Build the path layout from UBIK_INGESTION_ROOT (or default)."""
        root = Path(
            os.environ.get("UBIK_INGESTION_ROOT", str(_INGESTION_ROOT))
        ).expanduser().resolve()
        return cls(
            root=root,
            sources_dir=root / "sources",
            enriched_dir=root / "enriched",
            pending_review_dir=root / "pending_review",
            approved_dir=root / "approved",
            quarantine_source_dir=root / "quarantine" / "source",
            quarantine_enrichment_dir=root / "quarantine" / "enrichment",
            registry_dir=root / "registry",
            qa_dir=root / "qa",
            quality_log=root / "logs" / "ingestion_quality_log.jsonl",
        )


@dataclass(frozen=True)
class GateThresholds:
    """
    Gate 1 confidence thresholds.

    Attributes:
        auto_approve_confidence: At or above this, enrichment output is
            auto-approved into approved/.
        quarantine_below: Below this, output goes to quarantine/; the
            band in between lands in pending_review/.

    Raises:
        ValueError: If thresholds are out of [0, 1] or inverted.
    """
    auto_approve_confidence: float = 0.85
    quarantine_below: float = 0.60

    def __post_init__(self) -> None:
        if not (0.0 <= self.quarantine_below <= self.auto_approve_confidence <= 1.0):
            raise ValueError(
                "Gate thresholds must satisfy "
                "0 <= quarantine_below <= auto_approve_confidence <= 1, got "
                f"quarantine_below={self.quarantine_below}, "
                f"auto_approve_confidence={self.auto_approve_confidence}"
            )

    @classmethod
    def from_env(cls) -> "GateThresholds":
        """Build thresholds from environment variables."""
        return cls(
            auto_approve_confidence=float(
                os.environ.get("UBIK_GATE_AUTO_APPROVE", "0.85")
            ),
            quarantine_below=float(
                os.environ.get("UBIK_GATE_QUARANTINE_BELOW", "0.60")
            ),
        )


@dataclass(frozen=True)
class IngestionConfig:
    """
    Top-level ingestion configuration.

    Attributes:
        endpoints: LLM endpoints per privacy tier.
        paths: Workspace directory layout.
        gates: Gate 1 thresholds.
        prompt_version: Version tag stamped on every enrichment output
            so regenerated enrichments are distinguishable.

    Example:
        >>> config = load_config()
        >>> config.gates.auto_approve_confidence
        0.85
    """
    endpoints: EndpointConfig
    paths: PathsConfig
    gates: GateThresholds
    prompt_version: str = DEFAULT_PROMPT_VERSION

    @classmethod
    def from_env(cls) -> "IngestionConfig":
        """Build the full configuration from environment variables."""
        return cls(
            endpoints=EndpointConfig.from_env(),
            paths=PathsConfig.from_env(),
            gates=GateThresholds.from_env(),
            prompt_version=os.environ.get(
                "UBIK_PROMPT_VERSION", DEFAULT_PROMPT_VERSION
            ),
        )


def load_config(env_file: Optional[Path] = None) -> IngestionConfig:
    """
    Load ingestion configuration, reading .env files first.

    Args:
        env_file: Optional explicit .env path. Defaults to
            ``ingestion/.env``, then ``$UBIK_ROOT/maestro/.env`` for the
            Somatic/Tailscale variables.

    Returns:
        Fully populated IngestionConfig.

    Raises:
        ValueError: If gate thresholds are invalid.

    Note:
        Values already present in os.environ always win over .env files.
    """
    if env_file is not None:
        _load_env_file(Path(env_file))
    else:
        _load_env_file(_INGESTION_ROOT / ".env")
        ubik_root = os.environ.get("UBIK_ROOT")
        if ubik_root:
            _load_env_file(Path(ubik_root) / "maestro" / ".env")
    return IngestionConfig.from_env()
