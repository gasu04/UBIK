"""RAG service for Gines voice generation.

Unified RAG service for authentic voice generation.
Supports the Golden Set evaluation framework (Section 7.3).

Usage:
    from somatic.rag import RAGService, RAGResult

    async with RAGService() as service:
        result = await service.ask("What do you believe about family?")
        print(result.response)
"""

from .service import (
    GenerationConfig,
    RAGResult,
    RAGService,
    RAGServiceStats,
    TemplateType,
    select_template,
)

__all__ = [
    "GenerationConfig",
    "RAGResult",
    "RAGService",
    "RAGServiceStats",
    "TemplateType",
    "select_template",
]

__version__ = "2.1.0"
