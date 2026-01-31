"""Stop tokens for DeepSeek-R1-Distill-Qwen models.

These tokens signal the end of generation and should be used
with vLLM/OpenAI-compatible APIs.
"""

# DeepSeek-R1-Distill-Qwen-14B stop tokens
# The model uses fullwidth characters (｜) in special tokens
DEEPSEEK_R1_STOP_TOKENS: list[str] = [
    "<｜end▁of▁sentence｜>",  # Main EOS token (fullwidth pipes, subscript underscore)
    "<｜User｜>",  # Start of next user turn
]

# Alternative representations (ASCII approximation for compatibility)
DEEPSEEK_R1_STOP_TOKENS_ASCII: list[str] = [
    "<|end_of_sentence|>",
    "<|User|>",
]

# Token IDs (from tokenizer_config.json)
DEEPSEEK_R1_EOS_TOKEN_ID = 151643

# For use with vLLM chat completions
# Note: vLLM handles stop tokens automatically with chat templates
# These are only needed for raw completions API
VLLM_STOP_SEQUENCES: list[str] = [
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
]


def get_stop_tokens() -> list[str]:
    """Get stop tokens for the current model.

    Returns:
        List of stop token strings.
    """
    return DEEPSEEK_R1_STOP_TOKENS.copy()
