"""
config - DeepSeek RAG System Configuration Package

Exports the shared settings singleton used across all modules.
"""

from .settings import settings, DeepSeekSettings

__all__ = ["settings", "DeepSeekSettings"]
