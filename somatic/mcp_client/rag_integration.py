#!/usr/bin/env python3
"""
Ubik Somatic Node - RAG Integration Module

Integrates the Hippocampal memory system with vLLM inference
to provide context-aware responses based on personal memories
and identity.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from .hippocampal_client import HippocampalClient, get_context_for_inference

# Load environment
load_dotenv()

logger = logging.getLogger("ubik.rag")


@dataclass
class RAGConfig:
    """Configuration for RAG context building."""

    episodic_results: int = 3
    semantic_results: int = 3
    include_identity: bool = True
    max_context_tokens: int = 2048
    context_template: str = """<personal_context>
{context}
</personal_context>

Based on the above personal context about Gines, respond to the following:

User: {query}
Assistant:"""


class RAGContextBuilder:
    """
    Builds RAG-augmented prompts by combining user queries
    with relevant context from the Hippocampal memory system.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._client: Optional[HippocampalClient] = None

    async def get_client(self) -> HippocampalClient:
        """Get or create the Hippocampal client."""
        if self._client is None:
            self._client = HippocampalClient()
        return self._client

    async def close(self):
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def build_prompt(self, query: str) -> str:
        """
        Build a RAG-augmented prompt for the given query.

        Args:
            query: The user's question or request

        Returns:
            Complete prompt with personal context injected
        """
        client = await self.get_client()

        # Get context from Hippocampal node
        context = await client.get_rag_context(
            query=query,
            episodic_results=self.config.episodic_results,
            semantic_results=self.config.semantic_results,
            include_identity=self.config.include_identity,
        )

        # Build the complete prompt
        if context:
            prompt = self.config.context_template.format(context=context, query=query)
        else:
            # Fallback without context
            prompt = f"User: {query}\nAssistant:"

        logger.info(f"Built RAG prompt: {len(prompt)} chars")
        return prompt

    async def query_vllm(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        vllm_url: Optional[str] = None,
    ) -> str:
        """
        Send a prompt to the local vLLM server.

        Args:
            prompt: The complete prompt to send
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            vllm_url: vLLM server URL (default from env)

        Returns:
            Generated response text
        """
        url = (
            vllm_url
            or f"http://{os.getenv('VLLM_HOST', 'localhost')}:{os.getenv('VLLM_PORT', '8080')}"
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{url}/v1/completions",
                json={
                    "model": os.getenv(
                        "VLLM_MODEL", "DeepSeek-R1-Distill-Qwen-14B-AWQ"
                    ),
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": ["User:", "\n\nUser:"],
                },
            )
            response.raise_for_status()
            result = response.json()

            return result["choices"][0]["text"].strip()

    async def ask(
        self, query: str, max_tokens: int = 512, temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: build context, query vLLM, return response.

        Args:
            query: User's question
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            Dict with 'response', 'context_used', and 'prompt'
        """
        # Build the augmented prompt
        prompt = await self.build_prompt(query)

        # Query vLLM
        response = await self.query_vllm(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )

        return {
            "query": query,
            "response": response,
            "prompt_length": len(prompt),
            "context_injected": "<personal_context>" in prompt,
        }


# =============================================================================
# CLI Testing
# =============================================================================


async def main():
    """Test the RAG integration."""
    print("=" * 60)
    print("UBIK RAG Integration Test")
    print("=" * 60)

    rag = RAGContextBuilder()

    try:
        # Test prompt building
        print("\n[1/2] Building RAG prompt...")
        prompt = await rag.build_prompt("What are my core values?")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Preview:\n{prompt[:500]}...")

        # Test full pipeline (if vLLM is running)
        print("\n[2/2] Testing full RAG pipeline...")
        try:
            result = await rag.ask("What matters most to me about family?")
            print(f"  Response: {result['response'][:200]}...")
            print(f"  Context injected: {result['context_injected']}")
        except Exception as e:
            print(f"  ⚠ vLLM query failed (server may not be running): {e}")

    finally:
        await rag.close()

    print("\n" + "=" * 60)
    print("RAG integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
