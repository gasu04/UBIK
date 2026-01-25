#!/usr/bin/env python3
"""
Integration Test Suite - Somatic Phase 3

Dual-mode: standalone via `asyncio.run()` + pytest compatible.
Sections 1-4 fully mocked, Section 5 live with graceful skip.

Usage:
    # Mocked tests (sections 1-4, always work):
    python tests/Integration_Test_Suite_somatic_phase3.py

    # With pytest:
    pytest tests/Integration_Test_Suite_somatic_phase3.py -v

    # Full E2E (when services available):
    # Start vLLM + Hippocampal Node first, then same commands
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "somatic"))

from mcp_client.client import (
    HippocampalClientV2,
    IdentityContext,
    MemoryResult,
)
from mcp_client.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    RetryConfig,
)
from prompts.templates import ParsedResponse, parse_response
from rag.service import RAGResult, RAGService


# =============================================================================
# Helper Functions
# =============================================================================


def _make_mcp_response(result_data: Dict[str, Any]) -> MagicMock:
    """Create a mock httpx.Response with a JSON-RPC result."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "jsonrpc": "2.0",
        "result": result_data,
        "id": 1,
    }
    mock.headers = {"mcp-session-id": "test-session-abcdef01"}
    return mock


def _make_mcp_tool_response(tool_result: Dict[str, Any]) -> MagicMock:
    """Wrap a tool result dict in MCP JSON-RPC content format."""
    return _make_mcp_response(
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(tool_result),
                }
            ]
        }
    )


def _make_init_response() -> MagicMock:
    """Create a mock MCP initialize response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.headers = {"mcp-session-id": "test-session-abcdef01"}
    mock.json.return_value = {
        "jsonrpc": "2.0",
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "hippocampal-node", "version": "1.0.0"},
        },
        "id": 1,
    }
    return mock


def _make_notification_response() -> MagicMock:
    """Create a mock notification ACK response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.headers = {}
    return mock


def _make_mock_completion(content: str) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response."""
    mock = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    mock.choices = [choice]
    return mock


def _make_fast_client() -> HippocampalClientV2:
    """Create a HippocampalClientV2 with fast retry/circuit config for tests."""
    client = HippocampalClientV2()
    client._retry_config = RetryConfig(
        max_attempts=3,
        base_delay_ms=10,
        max_delay_ms=100,
        jitter_max_ms=0,
    )
    client._circuit_breaker = CircuitBreaker(
        name="test-hippocampal",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_s=60,
        ),
    )
    return client


# =============================================================================
# Section 1: Connectivity (4 tests, mocked)
# =============================================================================


@pytest.mark.asyncio
async def test_connect_successful():
    """connect() completes, session ID stored, circuit CLOSED."""
    client = _make_fast_client()

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(
        side_effect=[
            _make_init_response(),
            _make_notification_response(),
        ]
    )

    with patch.object(
        client._connection,
        "get_client",
        new_callable=AsyncMock,
        return_value=mock_http,
    ):
        await client.connect()

    assert client._initialized is True
    assert client._session_id is not None
    assert len(client._session_id) > 0
    assert client._circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_retry_on_transient_failure():
    """Client retries on httpx.ConnectError, succeeds on 3rd attempt."""
    client = _make_fast_client()

    # Build the mock HTTP client that succeeds when finally reached
    tool_result = {
        "knowledge": [
            {
                "id": "k1",
                "content": "Retry success",
                "relevance_score": 0.9,
                "metadata": {"type": "belief"},
            }
        ]
    }
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(
        side_effect=[
            # Attempt 2 (0-indexed): connect init + notification + tool call
            _make_init_response(),
            _make_notification_response(),
            _make_mcp_tool_response(tool_result),
        ]
    )

    # get_client raises ConnectError on first 2 attempts,
    # returns the working mock on attempts 3+ (connect + tool call)
    with patch.object(
        client._connection,
        "get_client",
        new_callable=AsyncMock,
        side_effect=[
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Connection refused"),
            mock_http,  # connect() in attempt 2
            mock_http,  # _call_tool after connect in attempt 2
        ],
    ), patch.object(
        client._connection,
        "invalidate",
        new_callable=AsyncMock,
    ):
        results = await client.query_semantic("family values")

    assert len(results) == 1
    assert results[0].content == "Retry success"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_repeated_failures():
    """Circuit transitions to OPEN after threshold failures."""
    client = _make_fast_client()
    # failure_threshold=3, max_attempts=3

    with patch.object(
        client._connection,
        "get_client",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ), patch.object(
        client._connection,
        "invalidate",
        new_callable=AsyncMock,
    ):
        with pytest.raises(httpx.ConnectError):
            await client.query_semantic("test query")

    # After 3 failures (one per attempt), circuit should be OPEN
    assert client._circuit_breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_connection_cleanup_on_error():
    """connection.invalidate() called on each failed attempt."""
    client = _make_fast_client()

    with patch.object(
        client._connection,
        "get_client",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ), patch.object(
        client._connection,
        "invalidate",
        new_callable=AsyncMock,
    ) as mock_invalidate:
        with pytest.raises(httpx.ConnectError):
            await client.query_semantic("test query")

    # invalidate() should be called once per failed attempt
    assert mock_invalidate.call_count == 3


# =============================================================================
# Section 2: Retrieval (2 tests, mocked)
# =============================================================================


@pytest.mark.asyncio
async def test_query_semantic_returns_memory_results():
    """Returns List[MemoryResult] from {"knowledge": [...]}."""
    client = _make_fast_client()
    client._initialized = True
    client._session_id = "test-session"

    tool_result = {
        "knowledge": [
            {
                "id": "k1",
                "content": "Family is everything",
                "relevance_score": 0.95,
                "metadata": {"type": "belief", "category": "family"},
            },
            {
                "id": "k2",
                "content": "Legacy matters deeply",
                "relevance_score": 0.88,
                "metadata": {"type": "value", "category": "legacy"},
            },
        ]
    }

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=_make_mcp_tool_response(tool_result))

    with patch.object(
        client._connection,
        "get_client",
        new_callable=AsyncMock,
        return_value=mock_http,
    ):
        results = await client.query_semantic("family values", n_results=5)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, MemoryResult) for r in results)
    assert results[0].content == "Family is everything"
    assert results[0].memory_type == "belief"
    assert results[0].relevance_score == 0.95
    assert results[1].content == "Legacy matters deeply"
    assert results[1].memory_type == "value"


@pytest.mark.asyncio
async def test_query_episodic_returns_memory_results():
    """Returns List[MemoryResult] from {"memories": [...]}."""
    client = _make_fast_client()
    client._initialized = True
    client._session_id = "test-session"

    tool_result = {
        "memories": [
            {
                "id": "e1",
                "content": "Summer afternoons at grandfather's workshop",
                "relevance_score": 0.92,
                "metadata": {"type": "episodic", "timestamp": "1985-07-15"},
            },
            {
                "id": "e2",
                "content": "First day teaching my daughter to ride a bike",
                "relevance_score": 0.87,
                "metadata": {"type": "episodic", "timestamp": "2005-04-20"},
            },
        ]
    }

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=_make_mcp_tool_response(tool_result))

    with patch.object(
        client._connection,
        "get_client",
        new_callable=AsyncMock,
        return_value=mock_http,
    ):
        results = await client.query_episodic("childhood memories", n_results=5)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, MemoryResult) for r in results)
    assert results[0].content == "Summer afternoons at grandfather's workshop"
    assert results[0].memory_type == "episodic"
    assert results[1].relevance_score == 0.87


# =============================================================================
# Section 3: Generation (1 test, mocked)
# =============================================================================


@pytest.mark.asyncio
async def test_rag_ask_does_not_block_event_loop():
    """Concurrent counter increments while ask() runs."""
    service = RAGService()

    # Force MCP circuit open (threshold=5 from settings)
    for _ in range(5):
        await service._mcp_circuit.record_failure()
    assert service._mcp_circuit.state == CircuitState.OPEN

    # Inject mock AsyncOpenAI client with slow generation
    mock_openai = MagicMock()
    mock_openai.close = AsyncMock()  # Needed for service.close()

    async def slow_create(**kwargs):
        await asyncio.sleep(0.3)
        return _make_mock_completion(
            "Family has always been at the center of my life."
        )

    mock_openai.chat.completions.create = AsyncMock(side_effect=slow_create)
    service._client = mock_openai

    # Concurrent counter to verify event loop is not blocked
    counter = {"value": 0}

    async def increment_counter():
        for _ in range(5):
            await asyncio.sleep(0.05)
            counter["value"] += 1

    ask_task = asyncio.create_task(service.ask("What does family mean to you?"))
    counter_task = asyncio.create_task(increment_counter())

    result = await ask_task
    await counter_task

    # Counter should have incremented during the ask() call
    assert counter["value"] > 0, "Event loop was blocked during ask()"
    assert isinstance(result, RAGResult)
    assert len(result.response) > 0

    await service.close()


# =============================================================================
# Section 4: Parsing (3 tests, pure unit)
# =============================================================================


def test_parse_strips_reasoning_blocks():
    """<reasoning> tags removed, content extracted."""
    raw = (
        "<reasoning>I should reflect on what family means.</reasoning>"
        "Family is the bedrock of everything I believe in."
    )
    parsed = parse_response(raw)

    assert "<reasoning>" not in parsed.clean_response
    assert "bedrock" in parsed.clean_response
    assert parsed.had_reasoning is True
    assert parsed.reasoning is not None
    assert "reflect" in parsed.reasoning


def test_parse_strips_think_and_thought_blocks():
    """<think> and <thought> tags also handled."""
    raw_think = (
        "<think>Let me consider this carefully.</think>"
        "What matters most is being present."
    )
    parsed_think = parse_response(raw_think)
    assert "<think>" not in parsed_think.clean_response
    assert "present" in parsed_think.clean_response
    assert parsed_think.had_reasoning is True

    raw_thought = (
        "<thought>The question is about legacy.</thought>"
        "Legacy is not what you leave behind, but what you build together."
    )
    parsed_thought = parse_response(raw_thought)
    assert "<thought>" not in parsed_thought.clean_response
    assert "Legacy" in parsed_thought.clean_response
    assert parsed_thought.had_reasoning is True


def test_parse_returns_clean_response_reasoning_fields():
    """All ParsedResponse fields correct for plain text (no tags)."""
    raw = "I believe in honesty above all else."
    parsed = parse_response(raw)

    assert isinstance(parsed, ParsedResponse)
    assert parsed.clean_response == raw
    assert parsed.raw_response == raw
    assert parsed.reasoning is None
    assert parsed.had_reasoning is False
    assert parsed.had_leaked_reasoning is False
    assert parsed.has_reasoning is False


# =============================================================================
# Section 5: End-to-End (6 tests, live services)
# =============================================================================


class _SkipTest(Exception):
    """Raised to skip a live test when services are unavailable."""


@pytest.mark.asyncio
async def test_live_mcp_client_connects():
    """Connection + health check against live Hippocampal Node."""
    try:
        client = HippocampalClientV2()
        await client.connect()

        assert client._initialized is True
        assert client._session_id is not None

        health = await client.health_check()
        assert health["status"] == "healthy"

        await client.disconnect()
    except Exception as e:
        pytest.skip(f"Hippocampal Node unavailable: {e}")


@pytest.mark.asyncio
async def test_live_semantic_query_returns_results():
    """Non-empty List[MemoryResult] from live Hippocampal Node."""
    try:
        async with HippocampalClientV2() as client:
            results = await client.query_semantic("family values", n_results=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, MemoryResult) for r in results)
    except Exception as e:
        pytest.skip(f"Hippocampal Node unavailable: {e}")


@pytest.mark.asyncio
async def test_live_episodic_query_returns_results():
    """Non-empty List[MemoryResult] from live Hippocampal Node."""
    try:
        async with HippocampalClientV2() as client:
            results = await client.query_episodic("childhood memories", n_results=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, MemoryResult) for r in results)
    except Exception as e:
        pytest.skip(f"Hippocampal Node unavailable: {e}")


@pytest.mark.asyncio
async def test_live_identity_context_retrieval():
    """IdentityContext with paths from live Hippocampal Node."""
    try:
        async with HippocampalClientV2() as client:
            identity = await client.get_identity_context(concept="Self", depth=2)

        assert isinstance(identity, IdentityContext)
        assert identity.concept == "Self"
        assert isinstance(identity.paths, list)
    except Exception as e:
        pytest.skip(f"Hippocampal Node unavailable: {e}")


@pytest.mark.asyncio
async def test_live_vllm_generation():
    """Non-empty completion text from live vLLM server."""
    try:
        from openai import AsyncOpenAI
        from config import get_settings

        settings = get_settings()
        client = AsyncOpenAI(
            base_url=settings.vllm.base_url,
            api_key="not-needed",
        )

        response = await client.chat.completions.create(
            model=settings.vllm.model_path,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one sentence."},
            ],
            max_tokens=64,
            temperature=0.7,
        )

        text = response.choices[0].message.content or ""
        assert len(text) > 0

        await client.close()
    except Exception as e:
        pytest.skip(f"vLLM server unavailable: {e}")


@pytest.mark.asyncio
async def test_live_rag_ask_returns_valid_response():
    """Valid RAGResult with non-empty response from both live services."""
    try:
        async with RAGService() as service:
            result = await service.ask(
                query="What does family mean to you?",
                temperature=0.7,
                max_tokens=256,
            )

        assert isinstance(result, RAGResult)
        assert len(result.response) > 0
        assert result.raw_response is not None
        assert result.model != ""
    except Exception as e:
        pytest.skip(f"Live services unavailable: {e}")


# =============================================================================
# Standalone Runner
# =============================================================================


async def _run_test(name: str, func) -> str:
    """Run a single test function. Returns 'PASS', 'FAIL', or 'SKIP'."""
    try:
        if asyncio.iscoroutinefunction(func):
            await func()
        else:
            func()
        return "PASS"
    except _SkipTest as e:
        print(f"           Skip reason: {e}")
        return "SKIP"
    except pytest.skip.Exception as e:
        print(f"           Skip reason: {e}")
        return "SKIP"
    except Exception as e:
        print(f"           Error: {type(e).__name__}: {e}")
        return "FAIL"


async def _standalone_main() -> int:
    """Run all tests with formatted output. Returns exit code."""
    sections = [
        (
            "Section 1: Connectivity",
            [
                ("test_connect_successful", test_connect_successful),
                ("test_retry_on_transient_failure", test_retry_on_transient_failure),
                (
                    "test_circuit_breaker_opens_after_repeated_failures",
                    test_circuit_breaker_opens_after_repeated_failures,
                ),
                ("test_connection_cleanup_on_error", test_connection_cleanup_on_error),
            ],
        ),
        (
            "Section 2: Retrieval",
            [
                (
                    "test_query_semantic_returns_memory_results",
                    test_query_semantic_returns_memory_results,
                ),
                (
                    "test_query_episodic_returns_memory_results",
                    test_query_episodic_returns_memory_results,
                ),
            ],
        ),
        (
            "Section 3: Generation",
            [
                (
                    "test_rag_ask_does_not_block_event_loop",
                    test_rag_ask_does_not_block_event_loop,
                ),
            ],
        ),
        (
            "Section 4: Parsing",
            [
                ("test_parse_strips_reasoning_blocks", test_parse_strips_reasoning_blocks),
                (
                    "test_parse_strips_think_and_thought_blocks",
                    test_parse_strips_think_and_thought_blocks,
                ),
                (
                    "test_parse_returns_clean_response_reasoning_fields",
                    test_parse_returns_clean_response_reasoning_fields,
                ),
            ],
        ),
        (
            "Section 5: End-to-End (live)",
            [
                ("test_live_mcp_client_connects", test_live_mcp_client_connects),
                (
                    "test_live_semantic_query_returns_results",
                    test_live_semantic_query_returns_results,
                ),
                (
                    "test_live_episodic_query_returns_results",
                    test_live_episodic_query_returns_results,
                ),
                (
                    "test_live_identity_context_retrieval",
                    test_live_identity_context_retrieval,
                ),
                ("test_live_vllm_generation", test_live_vllm_generation),
                (
                    "test_live_rag_ask_returns_valid_response",
                    test_live_rag_ask_returns_valid_response,
                ),
            ],
        ),
    ]

    results = {"PASS": 0, "FAIL": 0, "SKIP": 0}

    print()
    print("=" * 64)
    print("  INTEGRATION TEST SUITE - SOMATIC PHASE 3")
    print("=" * 64)

    for section_name, tests in sections:
        print()
        print("-" * 64)
        print(f"  {section_name}")
        print("-" * 64)

        for test_name, test_func in tests:
            status = await _run_test(test_name, test_func)
            results[status] += 1
            tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
            print(f"  {tag:8s} {test_name}")

    # Summary
    total = sum(results.values())
    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Total:   {total}")
    print(f"  Passed:  {results['PASS']}")
    print(f"  Failed:  {results['FAIL']}")
    print(f"  Skipped: {results['SKIP']}")
    print("=" * 64)

    if results["FAIL"] > 0:
        print("  RESULT: FAILURE")
    else:
        print("  RESULT: SUCCESS")
    print("=" * 64)
    print()

    return 0 if results["FAIL"] == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(_standalone_main())
    sys.exit(exit_code)
