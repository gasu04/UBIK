# Ubik Project - Python Scripts Documentation

**Version:** 1.0.0
**Generated:** 2026-01-23

---

This document lists all Python scripts in the Ubik project with their paths and descriptions.

---

## MCP Client Package (`somatic/mcp_client/`)

| Script | Path | Description |
|--------|------|-------------|
| `__init__.py` | `somatic/mcp_client/__init__.py` | Package initializer that exports `HippocampalClient`, `MemoryResult`, `get_context_for_inference`, and `RAGContextBuilder` |
| `hippocampal_client.py` | `somatic/mcp_client/hippocampal_client.py` | Main MCP client for connecting to the Hippocampal Node. Provides access to episodic memory, semantic memory, and identity graph services via HTTP/MCP protocol |
| `rag_integration.py` | `somatic/mcp_client/rag_integration.py` | RAG integration module that connects the Hippocampal memory system with vLLM inference to provide context-aware responses based on personal memories |
| `utils.py` | `somatic/mcp_client/utils.py` | Utility functions including logging setup, memory formatting, content truncation, timestamp parsing, and validation helpers for memory/knowledge types |

---

## Inference Scripts (`somatic/inference/`)

| Script | Path | Description |
|--------|------|-------------|
| `vllm_server.py` | `somatic/inference/vllm_server.py` | vLLM inference server launcher for AWQ-quantized DeepSeek model. Includes system validation, GPU memory checking, and automatic retry logic |
| `query.py` | `somatic/inference/query.py` | CLI client for querying the local vLLM DeepSeek model. Supports single queries, interactive mode, streaming responses, and thinking tag extraction |
| `rag_status.py` | `somatic/inference/rag_status.py` | Status checker that reports on ChromaDB collections, MCP server connectivity, and RAG readiness for the DeepSeek model |

---

## Test Scripts (`somatic/tests/`)

| Script | Path | Description |
|--------|------|-------------|
| `test_connectivity.py` | `somatic/tests/test_connectivity.py` | Comprehensive connectivity tests for the MCP client including health checks, semantic/episodic queries, identity graph queries, and RAG context generation |
| `test_rag.py` | `somatic/tests/test_rag.py` | Tests for RAG context building and prompt integration using pytest. Tests `RAGContextBuilder` and `RAGContext` classes |

---

## Root Test Scripts (`tests/`)

| Script | Path | Description |
|--------|------|-------------|
| `test_write_operations.py` | `tests/test_write_operations.py` | Tests write operations to the Hippocampal Node including storing episodic memories and semantic knowledge |
| `test_rag_pipeline.py` | `tests/test_rag_pipeline.py` | End-to-end RAG pipeline test that retrieves context from the Hippocampal Node and generates responses via vLLM |

---

## Cache/Data (`data/`)

| Script | Path | Description |
|--------|------|-------------|
| `__init__.py` | `data/cache/huggingface/modules/__init__.py` | Empty package initializer for Hugging Face cache modules |

---

## Summary

**Total Scripts:** 12

| Category | Count |
|----------|-------|
| MCP Client | 4 |
| Inference | 3 |
| Tests (somatic) | 2 |
| Tests (root) | 2 |
| Cache/Data | 1 |
