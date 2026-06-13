"""
Unit tests for config.ingestion_config — env-driven configuration.

Covers: default values, env-var overrides, the per-tier endpoint
resolver (including loud failure on unknown tiers), threshold
validation, and path derivation from UBIK_INGESTION_ROOT.
"""

import pytest

from config.ingestion_config import (
    EndpointConfig,
    GateThresholds,
    IngestionConfig,
    PathsConfig,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip all config-relevant env vars so each test starts clean."""
    for var in [
        "SOMATIC_VLLM_URL", "SOMATIC_TAILSCALE_IP", "SOMATIC_LAN_IP",
        "SOMATIC_TAILSCALE_HOSTNAME", "SOMATIC_VLLM_LAN_URL", "VLLM_PORT",
        "UBIK_SENSITIVE_LLM_URL", "UBIK_STANDARD_LLM_URL",
        "UBIK_ENRICHMENT_MODEL", "UBIK_LLM_TIMEOUT_SECONDS",
        "UBIK_INGESTION_ROOT", "UBIK_GATE_AUTO_APPROVE",
        "UBIK_GATE_QUARANTINE_BELOW", "UBIK_PROMPT_VERSION",
        "UBIK_STANDARD_PRIVACY_TIERS",
    ]:
        monkeypatch.delenv(var, raising=False)


# =============================================================================
# Endpoints
# =============================================================================

def test_endpoints_default_to_somatic_hostname():
    config = EndpointConfig.from_env()
    assert config.sensitive_endpoint == "http://ubik-somatic:8002/v1"
    assert config.standard_endpoint == "http://ubik-somatic:8002/v1"


def test_endpoints_compose_from_tailscale_ip(monkeypatch):
    monkeypatch.setenv("SOMATIC_TAILSCALE_IP", "203.0.113.7")
    monkeypatch.setenv("VLLM_PORT", "9999")
    config = EndpointConfig.from_env()
    assert config.sensitive_endpoint == "http://203.0.113.7:9999/v1"


def test_explicit_vllm_url_wins(monkeypatch):
    monkeypatch.setenv("SOMATIC_TAILSCALE_IP", "203.0.113.7")
    monkeypatch.setenv("SOMATIC_VLLM_URL", "http://example.test:8002/v1/")
    config = EndpointConfig.from_env()
    assert config.sensitive_endpoint == "http://example.test:8002/v1"


def test_all_tiers_route_sensitive_by_default(monkeypatch):
    """Fail-safe: without explicit relaxation, every tier is sensitive."""
    monkeypatch.setenv("UBIK_SENSITIVE_LLM_URL", "http://sensitive.test/v1")
    monkeypatch.setenv("UBIK_STANDARD_LLM_URL", "http://standard.test/v1")
    config = EndpointConfig.from_env()
    for tier in ("private", "therapy", "family", "business"):
        assert config.for_tier(tier) == "http://sensitive.test/v1"


def test_relaxed_tier_routes_standard(monkeypatch):
    monkeypatch.setenv("UBIK_SENSITIVE_LLM_URL", "http://sensitive.test/v1")
    monkeypatch.setenv("UBIK_STANDARD_LLM_URL", "http://standard.test/v1")
    monkeypatch.setenv("UBIK_STANDARD_PRIVACY_TIERS", "business")
    config = EndpointConfig.from_env()
    assert config.for_tier("business") == "http://standard.test/v1"
    assert config.for_tier("therapy") == "http://sensitive.test/v1"


def test_unknown_standard_tier_env_raises(monkeypatch):
    monkeypatch.setenv("UBIK_STANDARD_PRIVACY_TIERS", "busines")
    with pytest.raises(ValueError, match="unknown tiers"):
        EndpointConfig.from_env()


def test_lan_fallback_from_lan_ip(monkeypatch):
    monkeypatch.setenv("SOMATIC_LAN_IP", "192.0.2.10")
    config = EndpointConfig.from_env()
    assert config.lan_fallback_endpoint == "http://192.0.2.10:8002/v1"


def test_lan_fallback_absent_without_lan_config():
    assert EndpointConfig.from_env().lan_fallback_endpoint is None


def test_for_tier_unknown_tier_raises():
    config = EndpointConfig.from_env()
    with pytest.raises(ValueError, match="privacy tier"):
        config.for_tier("public")


# =============================================================================
# Gate thresholds
# =============================================================================

def test_gate_thresholds_defaults():
    gates = GateThresholds.from_env()
    assert gates.auto_approve_confidence == 0.85
    assert gates.quarantine_below == 0.60


def test_gate_thresholds_env_override(monkeypatch):
    monkeypatch.setenv("UBIK_GATE_AUTO_APPROVE", "0.9")
    monkeypatch.setenv("UBIK_GATE_QUARANTINE_BELOW", "0.5")
    gates = GateThresholds.from_env()
    assert gates.auto_approve_confidence == 0.9
    assert gates.quarantine_below == 0.5


def test_inverted_thresholds_raise():
    with pytest.raises(ValueError, match="thresholds"):
        GateThresholds(auto_approve_confidence=0.5, quarantine_below=0.8)


def test_out_of_range_threshold_raises():
    with pytest.raises(ValueError, match="thresholds"):
        GateThresholds(auto_approve_confidence=1.5, quarantine_below=0.6)


# =============================================================================
# Paths
# =============================================================================

def test_paths_derive_from_ingestion_root(monkeypatch, tmp_path):
    monkeypatch.setenv("UBIK_INGESTION_ROOT", str(tmp_path))
    paths = PathsConfig.from_env()
    assert paths.root == tmp_path
    assert paths.sources_dir == tmp_path / "sources"
    assert paths.quarantine_enrichment_dir == tmp_path / "quarantine" / "enrichment"
    assert paths.quality_log == tmp_path / "logs" / "ingestion_quality_log.jsonl"


def test_paths_default_root_is_ingestion_dir():
    paths = PathsConfig.from_env()
    assert (paths.registry_dir / "known_persons.yaml").exists()


# =============================================================================
# Top-level config
# =============================================================================

def test_full_config_from_env_defaults():
    config = IngestionConfig.from_env()
    assert config.prompt_version == "v1.0.0"
    assert config.gates.quarantine_below == 0.60
    assert config.endpoints.sensitive_endpoint.startswith("http://")


def test_prompt_version_env_override(monkeypatch):
    monkeypatch.setenv("UBIK_PROMPT_VERSION", "v2.3.0")
    assert IngestionConfig.from_env().prompt_version == "v2.3.0"
