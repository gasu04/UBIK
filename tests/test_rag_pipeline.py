#!/usr/bin/env python3
"""End-to-end RAG test - Memory retrieval + Inference"""

import asyncio
from openai import OpenAI
import sys
sys.path.insert(0, '/home/gasu/ubik/somatic')

from mcp_client import HippocampalClient

async def test_rag_pipeline():
    print("\n" + "=" * 60)
    print(" END-TO-END RAG PIPELINE TEST")
    print("=" * 60)

    query = "What do I believe about family and legacy?"

    # Step 1: Get RAG context from Hippocampal Node
    print(f"\n[1] Query: '{query}'")
    print("[2] Retrieving context from Hippocampal Node...")

    async with HippocampalClient() as client:
        context = await client.get_rag_context(query, episodic_results=3, semantic_results=3)

    print(f"    Retrieved {len(context)} characters of context")
    print(f"\n--- Context Preview ---\n{context[:500]}...\n")

    # Step 2: Build prompt with context
    print("[3] Building augmented prompt...")

    system_prompt = """You are Gines, speaking authentically based on your memories and values.
Use the context provided to inform your response, speaking in first person as yourself."""

    full_prompt = f"""<context>
{context}
</context>

<question>
{query}
</question>

Respond as Gines, drawing on the context above:"""

    # Step 3: Generate response via vLLM
    print("[4] Generating response via vLLM...")

    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="not-needed"
    )

    response = client.completions.create(
        model="/home/gasu/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ",
        prompt=full_prompt,
        max_tokens=300,
        temperature=0.7,
        stop=["</response>", "\n\n\n"]
    )

    generated = response.choices[0].text.strip()

    print("\n" + "=" * 60)
    print(" GENERATED RESPONSE (as Gines)")
    print("=" * 60)
    print(f"\n{generated}\n")
    print("=" * 60)
    print(" ✓ END-TO-END RAG TEST COMPLETE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())
