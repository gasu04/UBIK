"""
Ubik Somatic Node - MCP Client Utilities

Helper functions for logging, formatting, and common operations.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import settings from config module
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings


def setup_logging(
    name: str = "mcp_client",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the MCP client.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file name
        log_dir: Optional log directory (default: /home/gasu/ubik/somatic/logs)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file or log_dir:
        log_directory = Path(log_dir or "/home/gasu/ubik/somatic/logs")
        log_directory.mkdir(parents=True, exist_ok=True)

        file_name = log_file or "mcp_client.log"
        file_path = log_directory / file_name

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def format_memory_context(
    memories: List[Dict[str, Any]],
    memory_type: str = "memory",
    max_items: int = 5,
    include_scores: bool = False,
) -> str:
    """
    Format a list of memories for display or prompt injection.

    Args:
        memories: List of memory dictionaries
        memory_type: Type label (memory, belief, experience)
        max_items: Maximum items to include
        include_scores: Include relevance scores

    Returns:
        Formatted string
    """
    if not memories:
        return f"No {memory_type} entries found."

    lines = [f"## {memory_type.title()} ({len(memories[:max_items])} items)"]

    for i, mem in enumerate(memories[:max_items], 1):
        content = mem.get("content", "")
        score = mem.get("relevance_score", mem.get("score"))

        if include_scores and score is not None:
            lines.append(f"{i}. [{score:.2f}] {content}")
        else:
            lines.append(f"{i}. {content}")

        # Add metadata if present
        metadata = mem.get("metadata", {})
        if metadata.get("category"):
            lines.append(f"   Category: {metadata['category']}")
        if metadata.get("timestamp"):
            lines.append(f"   Date: {metadata['timestamp'][:10]}")

    return "\n".join(lines)


def truncate_content(content: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate content to a maximum length.

    Args:
        content: Text to truncate
        max_length: Maximum character length
        suffix: Suffix to append if truncated

    Returns:
        Truncated string
    """
    if len(content) <= max_length:
        return content
    return content[: max_length - len(suffix)] + suffix


def parse_timestamp(timestamp: str) -> Optional[datetime]:
    """
    Parse various timestamp formats.

    Args:
        timestamp: Timestamp string (ISO format or common variants)

    Returns:
        datetime object or None if parsing fails
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue

    return None


def get_env_config() -> Dict[str, Any]:
    """
    Get environment configuration for the MCP client.

    Returns:
        Dictionary of configuration values
    """
    settings = get_settings()
    return {
        "hippocampal_host": settings.hippocampal.host,
        "hippocampal_port": settings.hippocampal.mcp_port,
        "log_level": settings.logging.level,
        "timeout": 30.0,  # TODO: Add to ResilienceSettings if needed
        "max_retries": settings.resilience.retry_max_attempts,
    }


def validate_memory_type(memory_type: str) -> bool:
    """
    Validate episodic memory type.

    Args:
        memory_type: The type to validate

    Returns:
        True if valid
    """
    valid_types = {
        "letter",
        "therapy_session",
        "family_meeting",
        "conversation",
        "event",
        "training_log",
        "reflection",
    }
    return memory_type.lower() in valid_types


def validate_knowledge_type(knowledge_type: str) -> bool:
    """
    Validate semantic knowledge type.

    Args:
        knowledge_type: The type to validate

    Returns:
        True if valid
    """
    valid_types = {
        "belief",
        "value",
        "preference",
        "fact",
        "opinion",
    }
    return knowledge_type.lower() in valid_types


def validate_emotional_valence(valence: str) -> bool:
    """
    Validate emotional valence value.

    Args:
        valence: The valence to validate

    Returns:
        True if valid
    """
    valid_valences = {
        "positive",
        "negative",
        "neutral",
        "reflective",
        "mixed",
    }
    return valence.lower() in valid_valences
