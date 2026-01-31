#!/usr/bin/env python3
"""
Ubik DeepSeek Query Client

Send prompts to the local vLLM DeepSeek model.
"""

import argparse
import json
import sys
import re
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8001
DEFAULT_MODEL = "/home/gasu/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ"
DEFAULT_MAX_TOKENS = 2048


def extract_response(content: str, show_thinking: bool = False) -> str:
    """Extract the response, optionally removing thinking tags."""
    if show_thinking:
        return content

    # Remove <think>...</think> blocks
    pattern = r'<think>.*?</think>\s*'
    cleaned = re.sub(pattern, '', content, flags=re.DOTALL)
    return cleaned.strip()


def query_model(
    prompt: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    show_thinking: bool = False,
    stream: bool = False,
) -> str:
    """Send a query to the vLLM server and return the response."""

    url = f"http://{host}:{port}/v1/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    try:
        if stream:
            return stream_response(url, payload, show_thinking)
        else:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return extract_response(content, show_thinking)
    except requests.exceptions.ConnectionError:
        return f"Error: Cannot connect to vLLM server at {host}:{port}. Is it running?"
    except requests.exceptions.Timeout:
        return "Error: Request timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing response: {e}"


def stream_response(url: str, payload: dict, show_thinking: bool = False) -> str:
    """Stream the response token by token."""
    full_response = ""
    in_thinking = False

    try:
        response = requests.post(url, json=payload, stream=True, timeout=300)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        full_response += content

                        # Handle thinking tag display
                        if show_thinking:
                            print(content, end='', flush=True)
                        else:
                            # Track thinking state and only print non-thinking content
                            if '<think>' in content:
                                in_thinking = True
                                content = content.split('<think>')[0]
                                if content:
                                    print(content, end='', flush=True)
                            elif '</think>' in content:
                                in_thinking = False
                                content = content.split('</think>')[-1]
                                if content:
                                    print(content, end='', flush=True)
                            elif not in_thinking:
                                print(content, end='', flush=True)
                except json.JSONDecodeError:
                    continue

        print()  # Final newline
        return extract_response(full_response, show_thinking)

    except requests.exceptions.RequestException as e:
        return f"\nError during streaming: {e}"


def interactive_mode(args):
    """Run in interactive chat mode."""
    print("=" * 60)
    print("Ubik DeepSeek Interactive Query")
    print("=" * 60)
    print(f"Server: {args.host}:{args.port}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Show thinking: {args.show_thinking}")
    print(f"Streaming: {args.stream}")
    print("=" * 60)
    print("Type 'quit' or 'exit' to quit, 'clear' to reset.")
    print("=" * 60)
    print()

    conversation_history = []

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        if prompt.lower() == 'clear':
            conversation_history = []
            print("Conversation cleared.\n")
            continue

        print("\nDeepSeek: ", end='')

        response = query_model(
            prompt=prompt,
            host=args.host,
            port=args.port,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=args.system,
            show_thinking=args.show_thinking,
            stream=args.stream,
        )

        if not args.stream:
            print(response)

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Query the local DeepSeek model via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python query.py "What is the capital of France?"

  # Interactive mode
  python query.py -i

  # With custom settings
  python query.py -t 0.5 --max-tokens 1000 "Explain quantum computing"

  # Show model's thinking process
  python query.py --show-thinking "Solve: 2x + 5 = 15"

  # Stream response
  python query.py --stream "Write a short poem"
"""
    )

    parser.add_argument("prompt", nargs="?", help="The prompt to send (omit for interactive mode)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name/path")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("-s", "--system", type=str, help="System prompt")
    parser.add_argument("--show-thinking", action="store_true", help="Show model's <think> reasoning")
    parser.add_argument("--stream", action="store_true", help="Stream response tokens")

    args = parser.parse_args()

    if args.interactive or args.prompt is None:
        interactive_mode(args)
    else:
        response = query_model(
            prompt=args.prompt,
            host=args.host,
            port=args.port,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=args.system,
            show_thinking=args.show_thinking,
            stream=args.stream,
        )

        if not args.stream:
            print(response)


if __name__ == "__main__":
    main()
