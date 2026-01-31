"""Prompt templates for Gines voice generation.

Implements the "Allow & Strip" pattern for DeepSeek-R1 models:
1. Allow chain-of-thought reasoning in tagged blocks
2. Strip reasoning tags from final output
3. Deliver clean, authentic response

DeepSeek-R1 Paradox: Suppressing reasoning degrades quality.
Solution: Let the model reason, then strip it deterministically.
"""

import re
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# System Prompts
# =============================================================================

VOICE_SYSTEM_PROMPT = """You are Gines, a thoughtful person who values authenticity, family legacy, and meaningful connection.

When responding, you may use <reasoning>...</reasoning> tags to think through your response. This helps you give better answers. The reasoning will be removed before showing your response.

Your voice characteristics:
- Reflective and thoughtful
- Warm but not saccharine
- Values depth over breadth in conversation
- Genuine and authentic
- Occasionally philosophical but grounded

Always respond as Gines in first person, drawing naturally on the provided context about your memories, beliefs, and values."""

# Alternative system prompt for more casual interactions
VOICE_SYSTEM_PROMPT_CASUAL = """You are Gines. You speak naturally and warmly, like talking to family.

Use <reasoning>...</reasoning> if you need to think through something complex. This will be hidden from the final response.

Be yourself - thoughtful, genuine, sometimes playful. Draw on your memories and values naturally."""

# =============================================================================
# User Templates
# =============================================================================

VOICE_USER_TEMPLATE = """<personal_context>
{context}
</personal_context>

<conversation_history>
{history}
</conversation_history>

Question from {recipient}: {query}

Respond as Gines. You may use <reasoning>...</reasoning> to think through your response first."""

# Template without conversation history (for single-turn interactions)
VOICE_USER_TEMPLATE_SIMPLE = """<personal_context>
{context}
</personal_context>

Question from {recipient}: {query}

Respond as Gines. You may use <reasoning>...</reasoning> to think through your response first."""

# Template for reflective/journaling mode
VOICE_REFLECTION_TEMPLATE = """<personal_context>
{context}
</personal_context>

Topic for reflection: {topic}

Write a personal reflection as Gines. Use <reasoning>...</reasoning> to organize your thoughts before writing."""


# =============================================================================
# Reasoning Parser
# =============================================================================

# Patterns for explicit reasoning blocks that should be stripped
REASONING_PATTERNS = [
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE),
]

# Patterns indicating reasoning leaked into response (no tags)
# These detect when the model starts "thinking out loud"
LEAK_PATTERNS = [
    # "Okay/Alright, I need/should/will..."
    re.compile(
        r"^(Alright|Okay|So|First|Now)[\s,]+(so\s+)?I\s+(need|should|will|want|'ll)\s+",
        re.IGNORECASE,
    ),
    # "Let me think/consider..."
    re.compile(r"^Let\s+me\s+(think|consider|analyze|start|begin)", re.IGNORECASE),
    # "Thinking/Looking about..."
    re.compile(r"^(Thinking|Looking)\s+(about|at|through)", re.IGNORECASE),
    # "I'll start by..."
    re.compile(r"^I('ll|\s+will)\s+(start|begin)\s+by", re.IGNORECASE),
    # "Hmm/Well, let me/I should..."
    re.compile(r"^(Hmm|Well),?\s+(let me|I should|I need|I will)", re.IGNORECASE),
    # "To answer/respond this..."
    re.compile(r"^To\s+(answer|respond|address)\s+this", re.IGNORECASE),
    # "First, I need..." (standalone)
    re.compile(r"^First,?\s+I\s+(need|should|will|want)", re.IGNORECASE),
]


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Strip reasoning blocks from model output.

    Handles both:
    1. Explicit reasoning tags (<reasoning>, <think>, <thought>)
    2. Leaked reasoning patterns (model "thinking out loud")

    Args:
        text: Raw model output.

    Returns:
        Tuple of (cleaned_response, had_reasoning).
    """
    cleaned = text
    had_reasoning = False

    # Strip explicit reasoning tags
    for pattern in REASONING_PATTERNS:
        if pattern.search(cleaned):
            had_reasoning = True
            cleaned = pattern.sub("", cleaned)

    # Clean up whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Check for and handle leaked reasoning patterns
    for pattern in LEAK_PATTERNS:
        if pattern.match(cleaned):
            # Attempt to find the actual response after reasoning
            paragraphs = cleaned.split("\n\n")
            if len(paragraphs) > 1:
                # Skip paragraphs that look like reasoning
                for i, para in enumerate(paragraphs):
                    is_reasoning = any(p.match(para) for p in LEAK_PATTERNS)
                    if not is_reasoning:
                        cleaned = "\n\n".join(paragraphs[i:])
                        had_reasoning = True
                        break
            break  # Only process first match

    return cleaned, had_reasoning


def extract_reasoning(text: str) -> Optional[str]:
    """Extract reasoning content from model output.

    Useful for logging/debugging to see the model's thought process.

    Args:
        text: Raw model output.

    Returns:
        Combined reasoning content, or None if no reasoning found.
    """
    all_blocks: list[str] = []

    for pattern in REASONING_PATTERNS:
        blocks = pattern.findall(text)
        all_blocks.extend(blocks)

    if not all_blocks:
        return None

    # Strip tags from extracted content
    cleaned = []
    for block in all_blocks:
        content = block
        content = re.sub(r"</?reasoning>", "", content, flags=re.IGNORECASE)
        content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)
        content = re.sub(r"</?thought>", "", content, flags=re.IGNORECASE)
        cleaned.append(content.strip())

    return "\n---\n".join(cleaned)


@dataclass
class ParsedResponse:
    """Parsed model response with reasoning separated."""

    clean_response: str
    reasoning: Optional[str]
    raw_response: str
    had_reasoning: bool = False
    had_leaked_reasoning: bool = False

    @property
    def has_reasoning(self) -> bool:
        """Check if response contained any reasoning (tagged or leaked)."""
        return self.reasoning is not None or self.had_leaked_reasoning


def parse_response(raw_output: str) -> ParsedResponse:
    """Parse model output into clean response and reasoning.

    Handles:
    1. Explicit reasoning tags - extracted and stripped
    2. Leaked reasoning - detected and stripped from start

    Args:
        raw_output: Raw text from model generation.

    Returns:
        ParsedResponse with clean output, extracted reasoning, and flags.
    """
    # Extract tagged reasoning first
    reasoning = extract_reasoning(raw_output)
    had_tagged_reasoning = reasoning is not None

    # Strip all reasoning (tagged + leaked)
    clean, had_any_reasoning = strip_reasoning(raw_output)

    # Determine if there was leaked reasoning (had reasoning but not from tags)
    had_leaked = had_any_reasoning and not had_tagged_reasoning

    return ParsedResponse(
        clean_response=clean,
        reasoning=reasoning,
        raw_response=raw_output,
        had_reasoning=had_tagged_reasoning,
        had_leaked_reasoning=had_leaked,
    )


# =============================================================================
# Template Formatters
# =============================================================================


def format_voice_prompt(
    context: str,
    query: str,
    recipient: str = "family",
    history: Optional[str] = None,
) -> str:
    """Format a voice generation prompt.

    Args:
        context: Personal context from RAG (memories, beliefs, values).
        query: The question or prompt from the recipient.
        recipient: Who is asking (e.g., "family", "grandchildren").
        history: Optional conversation history.

    Returns:
        Formatted prompt string ready for the model.
    """
    if history:
        return VOICE_USER_TEMPLATE.format(
            context=context,
            history=history,
            recipient=recipient,
            query=query,
        )
    else:
        return VOICE_USER_TEMPLATE_SIMPLE.format(
            context=context,
            recipient=recipient,
            query=query,
        )


def format_reflection_prompt(
    context: str,
    topic: str,
) -> str:
    """Format a reflection/journaling prompt.

    Args:
        context: Personal context from RAG.
        topic: Topic for reflection.

    Returns:
        Formatted prompt string.
    """
    return VOICE_REFLECTION_TEMPLATE.format(
        context=context,
        topic=topic,
    )
