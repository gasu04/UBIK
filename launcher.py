#!/usr/bin/env python3
"""
launcher - DeepSeek Workflow Launcher

Entry point for all DeepSeek workflows. Checks and starts the Ollama
service, then presents a menu to choose between:
  1. Google Drive RAG — query your documents
  2. Direct Chat — talk directly to DeepSeek
  3. Free Memory — close apps and clear system cache

Usage:
    python launcher.py

Dependencies:
    - requests>=2.0.0
    - ollama (binary, must be installed separately)

Version: 1.0.0
"""

import subprocess
import sys
import requests
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings


def check_and_cleanup_ollama():
    """Check if Ollama is running and healthy, only cleanup if needed"""
    print("🔍 Checking Ollama status...")

    try:
        # First, check if Ollama API is responding
        try:
            response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print("   ✓ Ollama is running and healthy\n")
                return True
        except requests.exceptions.RequestException:
            pass  # Ollama is not responding, need to check processes

        # Check for any ollama processes (both serve and runner)
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )

        # Count ollama processes (excluding grep itself)
        ollama_lines = [
            line for line in result.stdout.split('\n')
            if 'ollama' in line.lower() and 'grep' not in line.lower()
        ]

        # Filter for actual ollama processes (serve or runner)
        ollama_processes = [
            line for line in ollama_lines
            if '/ollama' in line or 'ollama serve' in line or 'ollama runner' in line
        ]

        if not ollama_processes:
            print("   ✓ No existing Ollama processes found\n")
            return True

        num_processes = len(ollama_processes)
        print(f"   ⚠️  Found {num_processes} Ollama process(es) but API not responding")

        # Show what processes were found
        for line in ollama_processes[:3]:  # Show first 3
            # Extract just the command part
            parts = line.split()
            if len(parts) > 10:
                cmd = ' '.join(parts[10:12])
                print(f"      • {cmd}...")

        if num_processes > 3:
            print(f"      ... and {num_processes - 3} more")

        # Kill all ollama processes to ensure clean state
        print("\n   🧹 Cleaning up stuck Ollama processes...")
        subprocess.run(["pkill", "-9", "ollama"], check=False)

        # Wait a moment for processes to die
        import time
        time.sleep(2)

        print("   ✓ Cleanup complete\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Warning: Could not check processes: {e}")
        return True  # Continue anyway
    except Exception as e:
        print(f"   ⚠️  Warning: Error during cleanup: {e}")
        return True  # Continue anyway


def free_memory():
    """Run the free_memory.sh script to clear system memory"""
    print("\n" + "=" * 60)
    print("🧹 FREEING UP SYSTEM MEMORY")
    print("=" * 60)
    print()

    script_path = Path(__file__).parent / "scripts" / "free_memory.sh"
    # Fallback to root location for backwards compatibility
    if not script_path.exists():
        script_path = Path(__file__).parent / "free_memory.sh"

    if not script_path.exists():
        print(f"❌ Error: free_memory.sh not found at {script_path}")
        return False

    try:
        # Run the cleanup script
        result = subprocess.run(
            ["bash", str(script_path)],
            check=False  # Don't fail on errors
        )

        if result.returncode == 0:
            print("\n✅ Memory cleanup completed successfully!")
        else:
            print("\n⚠️  Cleanup completed with some warnings (this is normal)")

        return True

    except Exception as e:
        print(f"\n❌ Error running cleanup: {e}")
        return False


def run_start_ollama():
    """Run the start_ollama.py script to ensure Ollama is ready"""
    print("=" * 60)
    print("🚀 STEP 1: Starting Ollama Service")
    print("=" * 60)
    print()

    script_path = Path(__file__).parent / "start_ollama.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting Ollama: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Error: start_ollama.py not found at {script_path}")
        return False


def chat_with_deepseek(model_name: str = "") -> None:
    model_name = model_name or settings.deepseek_model
    """Direct chat interface with DeepSeek model"""
    print("\n" + "=" * 60)
    print("💬 Direct Chat with DeepSeek R1")
    print("=" * 60)
    print("Commands:")
    print("  - Type your prompt and press Enter")
    print("  - Type 'quit' or 'exit' to return to menu")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("💭 Your prompt: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Returning to menu...")
                break

            print("\n🤖 DeepSeek is thinking...\n")

            # Make API call to Ollama
            response = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300  # 5 minute timeout
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "No response received")
                print(f"🤖 DeepSeek:\n{answer}\n")
            else:
                print(f"❌ Error: Received status code {response.status_code}\n")

        except requests.exceptions.Timeout:
            print("❌ Error: Request timed out. The model might be too slow.\n")
        except requests.exceptions.ConnectionError:
            print("❌ Error: Cannot connect to Ollama. Is it still running?\n")
            print("   Try restarting with option 0 from the main menu.\n")
            break
        except KeyboardInterrupt:
            print("\n\n👋 Returning to menu...")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def run_gdrive_rag():
    """Run the Google Drive RAG script"""
    print("\n" + "=" * 60)
    print("📚 Starting Google Drive RAG System")
    print("=" * 60)
    print()

    script_path = Path(__file__).parent / "gdrive_rag_deepseek.py"

    try:
        subprocess.run([sys.executable, str(script_path)])
    except FileNotFoundError:
        print(f"\n❌ Error: gdrive_rag_deepseek.py not found at {script_path}")
    except KeyboardInterrupt:
        print("\n\n👋 RAG session ended")


def main_menu():
    """Display main menu and handle user choice"""
    while True:
        print("\n" + "=" * 60)
        print("🎯 DEEPSEEK LAUNCHER - MAIN MENU")
        print("=" * 60)
        print("\nChoose your workflow:")
        print("  1. 📚 Google Drive RAG (Query your documents)")
        print("  2. 💬 Direct Chat (Talk directly to DeepSeek)")
        print("  3. 🧹 Free Up Memory (Close apps & clear cache)")
        print("  0. 🔄 Restart Ollama Service")
        print("  q. 🚪 Quit")
        print("=" * 60)

        choice = input("\nYour choice: ").strip().lower()

        if choice == '1':
            run_gdrive_rag()
        elif choice == '2':
            chat_with_deepseek()
        elif choice == '3':
            free_memory()
        elif choice == '0':
            print("\n🔄 Restarting Ollama...\n")
            check_and_cleanup_ollama()
            if run_start_ollama():
                print("\n✅ Ollama restarted successfully!")
            else:
                print("\n❌ Failed to restart Ollama")
        elif choice in ['q', 'quit', 'exit']:
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, 3, 0, or q")


def main():
    """Main function"""
    print("=" * 60)
    print("🚀 DEEPSEEK LAUNCHER")
    print("=" * 60)
    print()

    # Step 0: Check and cleanup any existing Ollama processes
    check_and_cleanup_ollama()

    # Optional: Ask if user wants to free up memory
    print("💡 Tip: For best performance with Qwen-Distill 14B, free memory is recommended")
    free_mem = input("   Free up memory now? (y/n, default=n): ").strip().lower()
    if free_mem in ['y', 'yes']:
        free_memory()

    # Step 1: Ensure Ollama is running
    if not run_start_ollama():
        print("\n❌ Failed to start Ollama. Please check the errors above.")
        print("   You may need to:")
        print("   1. Install Ollama if not installed")
        print("   2. Check your internet connection for model download")
        print("   3. Free up system memory")
        sys.exit(1)

    # Step 2: Show main menu
    main_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
