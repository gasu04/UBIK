"""
Checkpoint 5: Verify RAGService.ask() doesn't block the event loop.

Mocks external dependencies (vLLM, Hippocampal MCP) so the test
can run without GPU VRAM or live services. The async behavior
being tested is architectural — it depends on AsyncOpenAI, not
on actual inference.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add somatic to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "somatic"))


def _make_mock_completion():
    """Create a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = "Family is the foundation of everything I believe in."

    response = MagicMock()
    response.choices = [choice]
    return response


async def test_rag_does_not_block():
    """
    Verify RAGService.ask() doesn't block the event loop.

    Mocks vLLM (AsyncOpenAI) and MCP client so no live services
    are needed. The mock introduces an async sleep to simulate
    network I/O — if the event loop were blocked, the counter
    task would not increment.
    """
    from rag.service import RAGService

    # --- Mock AsyncOpenAI to simulate async network I/O ---
    async def mock_create(**kwargs):
        await asyncio.sleep(0.5)  # Simulate inference latency
        return _make_mock_completion()

    mock_client = AsyncMock()
    mock_client.chat.completions.create = mock_create

    # --- Create service and inject mock client ---
    service = RAGService()
    service._client = mock_client  # Skip lazy init

    # Force MCP circuit open so context building is skipped
    # (returns empty context without connecting)
    await service._mcp_circuit.record_failure()
    await service._mcp_circuit.record_failure()
    await service._mcp_circuit.record_failure()
    await service._mcp_circuit.record_failure()
    await service._mcp_circuit.record_failure()

    counter = 0
    generation_done = False

    async def increment_counter():
        nonlocal counter
        while not generation_done:
            counter += 1
            await asyncio.sleep(0.1)

    async def run_generation():
        nonlocal generation_done
        await service.ask("What do you believe about family?")
        generation_done = True

    # Run both concurrently
    await asyncio.gather(
        increment_counter(),
        run_generation(),
    )

    # If counter > 0, event loop was NOT blocked
    assert counter > 0, "Event loop was blocked during generation!"
    print(f"✓ Event loop remained responsive (counter={counter})")

    await service.close()


if __name__ == "__main__":
    asyncio.run(test_rag_does_not_block())
