"""Prompt templates and parsing for Gines voice generation."""

from .stop_tokens import (
    DEEPSEEK_R1_EOS_TOKEN_ID,
    DEEPSEEK_R1_STOP_TOKENS,
    DEEPSEEK_R1_STOP_TOKENS_ASCII,
    VLLM_STOP_SEQUENCES,
    get_stop_tokens,
)
from .templates import (
    VOICE_REFLECTION_TEMPLATE,
    VOICE_SYSTEM_PROMPT,
    VOICE_SYSTEM_PROMPT_CASUAL,
    VOICE_USER_TEMPLATE,
    VOICE_USER_TEMPLATE_SIMPLE,
    ParsedResponse,
    extract_reasoning,
    format_reflection_prompt,
    format_voice_prompt,
    parse_response,
    strip_reasoning,
)

__all__ = [
    # System prompts
    "VOICE_SYSTEM_PROMPT",
    "VOICE_SYSTEM_PROMPT_CASUAL",
    # User templates
    "VOICE_USER_TEMPLATE",
    "VOICE_USER_TEMPLATE_SIMPLE",
    "VOICE_REFLECTION_TEMPLATE",
    # Parsing
    "ParsedResponse",
    "extract_reasoning",
    "parse_response",
    "strip_reasoning",
    # Formatting
    "format_reflection_prompt",
    "format_voice_prompt",
    # Stop tokens
    "DEEPSEEK_R1_STOP_TOKENS",
    "DEEPSEEK_R1_STOP_TOKENS_ASCII",
    "DEEPSEEK_R1_EOS_TOKEN_ID",
    "VLLM_STOP_SEQUENCES",
    "get_stop_tokens",
]
