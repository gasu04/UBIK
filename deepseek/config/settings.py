#!/usr/bin/env python3
"""
Settings - DeepSeek RAG System Configuration

Centralised, environment-driven configuration following the dataclass pattern.
All tuneable values live here; override via environment variables or a .env file.

Usage:
    from config.settings import settings

    llm = OllamaLLM(model=settings.deepseek_model, base_url=settings.ollama_base_url)

Dependencies:
    - python-dotenv (optional, for .env file loading)

Version: 1.0.0
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env if present (optional dependency — silently skipped if not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(Path(__file__).parents[1] / ".env")
except ImportError:
    pass

# Project root is one level above this config/ directory
_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class DeepSeekSettings:
    """
    Runtime configuration for the DeepSeek RAG system.

    All fields read from environment variables with safe defaults so the
    system works out-of-the-box without any .env file.

    Attributes:
        ollama_base_url: Base URL for the local Ollama service.
        deepseek_model: Ollama model tag to use for generation.
        jina_api_key: Jina AI API key for cloud embeddings.
        jina_model: Jina embedding model identifier.
        embeddings_base_dir: Directory where ChromaDB collections are stored.
        chunk_size: Character count per document chunk.
        chunk_overlap: Character overlap between adjacent chunks.
        embedding_batch_size: Number of chunks per Jina API batch.
        embedding_batch_delay: Seconds to wait between batches.
        rag_k: Number of chunks retrieved per query.
        llm_temperature: LLM sampling temperature (0.0 = deterministic).
        log_level: Python logging level name.
    """

    # Ollama
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-r1:14b")
    )

    # Jina AI Embeddings
    jina_api_key: str = field(
        default_factory=lambda: os.getenv("JINA_API_KEY", "")
    )
    jina_model: str = field(
        default_factory=lambda: os.getenv("JINA_MODEL", "jina-embeddings-v2-base-en")
    )

    # Storage
    embeddings_base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("EMBEDDINGS_BASE_DIR", str(_PROJECT_ROOT / "chroma_embeddings"))
        )
    )

    # RAG tuning
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "3000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "500"))
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    )
    embedding_batch_delay: float = field(
        default_factory=lambda: float(os.getenv("EMBEDDING_BATCH_DELAY", "1.0"))
    )
    rag_k: int = field(
        default_factory=lambda: int(os.getenv("RAG_K", "5"))
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )

    # Logging
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def validate(self) -> None:
        """
        Validate critical configuration values.

        Raises:
            ValueError: If a required value is missing or out of range.
        """
        if not self.jina_api_key:
            raise ValueError(
                "JINA_API_KEY is required. Set it in your environment or config/.env"
            )
        if not (1 <= self.rag_k <= 50):
            raise ValueError(f"RAG_K must be between 1 and 50, got {self.rag_k}")
        if not (0.0 <= self.llm_temperature <= 2.0):
            raise ValueError(
                f"LLM_TEMPERATURE must be between 0.0 and 2.0, got {self.llm_temperature}"
            )


# Module-level singleton — import and use directly
settings = DeepSeekSettings()
