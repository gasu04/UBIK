#!/usr/bin/env python3
"""
start_ollama - Ensure Ollama Service Is Running with DeepSeek Model

Checks whether the local Ollama service is running, starts it if not,
pulls the configured DeepSeek model if missing, and pre-loads it into
memory so the first query is fast.

Usage:
    python start_ollama.py
    # or via launcher.py which calls this automatically

Dependencies:
    - requests>=2.0.0
    - ollama (binary, must be installed separately)

Version: 1.0.0
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings


def is_ollama_running() -> bool:
    """Check if Ollama service is running by pinging the API."""
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_service() -> Optional[subprocess.Popen]:
    """Start Ollama service in the background."""
    print("🚀 Starting Ollama service...")
    try:
        # Start ollama serve in background
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        # Wait for service to be ready
        print("⏳ Waiting for Ollama to initialize...")
        max_retries = 30
        for i in range(max_retries):
            if is_ollama_running():
                print("✅ Ollama service is running!")
                return process
            time.sleep(1)
            print(f"   Attempt {i+1}/{max_retries}...")

        print("❌ Failed to start Ollama service within timeout")
        return None

    except FileNotFoundError:
        print("❌ Error: 'ollama' command not found. Please install Ollama first.")
        return None
    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        return None


def check_model_exists(model_name: str) -> bool:
    """Check if the specified model is already pulled."""
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if model.get("name") == model_name:
                    return True
        return False
    except Exception as e:
        print(f"⚠️  Warning: Could not check models: {e}")
        return False


def pull_model(model_name: str) -> bool:
    """Pull the specified Ollama model."""
    print(f"📥 Pulling model: {model_name}")
    print("   This may take a while for large models...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✅ Model {model_name} pulled successfully!")
            return True
        else:
            print(f"❌ Failed to pull model: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False


def load_model(model_name: str) -> bool:
    """Load the model into memory by making a test API call."""
    print(f"\n🔄 Loading model into memory: {model_name}")
    print("   This will preload the model for faster queries...")

    try:
        # Make a test generation request to load the model
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "test",
                "stream": False
            },
            timeout=120  # 2 minute timeout for model loading
        )

        if response.status_code == 200:
            print(f"✅ Model {model_name} is now loaded and ready!")
            return True
        else:
            print(f"⚠️  Warning: Could not preload model (status {response.status_code})")
            print(f"   The model is still available, but first query may be slower.")
            return True  # Don't fail, model is still usable

    except requests.exceptions.Timeout:
        print(f"⚠️  Warning: Model loading timed out")
        print(f"   The model is still available, but first query may be slower.")
        return True  # Don't fail, model is still usable
    except Exception as e:
        print(f"⚠️  Warning: Could not preload model: {e}")
        print(f"   The model is still available, but first query may be slower.")
        return True  # Don't fail, model is still usable


def main() -> None:
    """Main function to ensure Ollama is running with the specified model."""
    model_name = settings.deepseek_model

    print("=" * 60)
    print("🔍 Checking Ollama Status")
    print("=" * 60)

    # Check if Ollama is already running
    if is_ollama_running():
        print("✅ Ollama is already running!")
    else:
        print("⚠️  Ollama is not running.")
        process = start_ollama_service()
        if not process:
            print("❌ Could not start Ollama. Exiting.")
            sys.exit(1)

    # Check if model exists, if not pull it
    print(f"\n🔍 Checking for model: {model_name}")
    if check_model_exists(model_name):
        print(f"✅ Model {model_name} is already available!")
    else:
        print(f"⚠️  Model {model_name} not found locally.")
        if not pull_model(model_name):
            print("❌ Could not pull model. Exiting.")
            sys.exit(1)

    # Load the model into memory
    if not load_model(model_name):
        print("❌ Could not load model. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Ollama is ready with model:", model_name)
    print("=" * 60)
    print("\n🎉 The model is loaded and ready for queries!")
    print(f"   You can now run your RAG script or use: ollama run {model_name}")


if __name__ == "__main__":
    main()
