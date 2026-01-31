#!/usr/bin/env python3
"""
Ubik vLLM Inference Server - AWQ Quantized Model

High-performance inference serving for the Somatic Node.

Features:
    - Graceful shutdown with GPU memory cleanup
    - RTX 5080/Blackwell GPU compatibility
    - Interactive VRAM availability waiting
    - Signal handling for clean termination
"""

import os
import sys
import argparse
import yaml
import time
import subprocess
import select
import signal
import gc
import atexit
from pathlib import Path
from typing import Optional, List

# Estimated VRAM requirements (in GB) - adjust based on your model
MINIMUM_VRAM_GB = 16.0
RETRY_INTERVAL_MINUTES = 20

# Global reference for cleanup
_server_process: Optional[subprocess.Popen] = None


def setup_rtx5080_environment() -> None:
    """
    Configure environment for RTX 5080/Blackwell GPU compatibility.

    Sets environment variables to work around known vLLM issues with
    newer NVIDIA GPUs (SM120 architecture).

    See: https://github.com/vllm-project/vllm/issues/14452
    """
    # Use Flash Attention 2 (FA3 doesn't work with Blackwell yet)
    os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "2")

    # Help with CUDA memory fragmentation (PYTORCH_CUDA_ALLOC_CONF is deprecated)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Reduce memory reservation aggressiveness
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    # Disable NCCL P2P if causing issues (uncomment if needed)
    # os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    print("RTX 5080/Blackwell environment configured:")
    print(f"  VLLM_FLASH_ATTN_VERSION={os.environ.get('VLLM_FLASH_ATTN_VERSION')}")
    print(f"  PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF')}")
    print(f"  CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING')}")


def cleanup_gpu_memory(verbose: bool = True) -> None:
    """
    Attempt to release GPU memory.

    Calls PyTorch CUDA cleanup functions and Python garbage collection
    to free GPU resources. This helps mitigate vLLM's known memory
    leak issues.

    See: https://github.com/vllm-project/vllm/issues/1908
    See: https://github.com/vllm-project/vllm/issues/23793
    """
    if verbose:
        print("\nCleaning up GPU memory...")

    try:
        # Try vLLM-specific cleanup first
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                cleanup_dist_env_and_memory,
            )
            destroy_model_parallel()
            cleanup_dist_env_and_memory()
            if verbose:
                print("  - vLLM distributed state cleaned up")
        except (ImportError, Exception) as e:
            if verbose:
                print(f"  - vLLM cleanup skipped: {e}")

        # PyTorch CUDA cleanup
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Try to reset peak memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except Exception:
                pass

            if verbose:
                print("  - PyTorch CUDA cache cleared")

        # Force garbage collection
        gc.collect()
        if verbose:
            print("  - Python garbage collection completed")

        # Try to destroy process group if it exists
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                if verbose:
                    print("  - Distributed process group destroyed")
        except Exception:
            pass

    except ImportError:
        if verbose:
            print("  - PyTorch not available, skipping GPU cleanup")
    except Exception as e:
        if verbose:
            print(f"  - GPU cleanup warning: {e}")


def force_gpu_reset(gpu_id: int = 0) -> bool:
    """
    Force reset GPU state using nvidia-smi.

    WARNING: This will terminate ALL processes using the GPU.
    Use only as a last resort when normal cleanup fails.

    Args:
        gpu_id: GPU device index to reset (default: 0)

    Returns:
        True if reset succeeded, False otherwise
    """
    print(f"\nForce resetting GPU {gpu_id}...")
    print("WARNING: This will terminate ALL GPU processes!")

    try:
        # First, try to query what's using the GPU
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name,used_memory",
             "--format=csv,noheader", "-i", str(gpu_id)],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"Processes using GPU {gpu_id}:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")

        # Attempt GPU reset
        result = subprocess.run(
            ["nvidia-smi", "--gpu-reset", "-i", str(gpu_id)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"GPU {gpu_id} reset successful")
            return True
        else:
            print(f"GPU reset failed: {result.stderr}")
            print("Note: GPU reset may require root/admin privileges")
            return False

    except FileNotFoundError:
        print("nvidia-smi not found")
        return False
    except Exception as e:
        print(f"GPU reset error: {e}")
        return False


def signal_handler(signum: int, frame) -> None:
    """
    Handle shutdown signals gracefully.

    Terminates the vLLM server process and attempts to clean up
    GPU memory before exiting.
    """
    global _server_process

    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n\nReceived {sig_name}, initiating graceful shutdown...")

    if _server_process is not None and _server_process.poll() is None:
        print("Stopping vLLM server (this may take 30+ seconds for CUDA cleanup)...")

        # Send SIGTERM first for graceful shutdown
        _server_process.terminate()

        try:
            # vLLM needs significant time to clean up CUDA resources and process groups
            # 60 seconds is usually sufficient for large models
            for elapsed in range(60):
                if _server_process.poll() is not None:
                    print(f"Server stopped gracefully after {elapsed}s")
                    break
                if elapsed % 10 == 0 and elapsed > 0:
                    print(f"  Still waiting for graceful shutdown... ({elapsed}s)")
                time.sleep(1)
            else:
                # 60 seconds elapsed without exit
                print("Server didn't stop after 60s, force killing...")
                _server_process.kill()
                try:
                    _server_process.wait(timeout=10)
                    print("Server force-killed successfully")
                except subprocess.TimeoutExpired:
                    print("Warning: Could not confirm process termination")
        except subprocess.TimeoutExpired:
            print("Timeout waiting for server, force killing...")
            _server_process.kill()

    # Clean up GPU memory
    cleanup_gpu_memory(verbose=True)

    # Give CUDA a moment to release resources
    time.sleep(1)

    # Show final memory state
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"\nFinal GPU memory state:")
        print(f"  Used: {mem_info['used_gb']:.1f} GB / {mem_info['total_gb']:.1f} GB")
        print(f"  Free: {mem_info['free_gb']:.1f} GB")

    print("\nShutdown complete.")
    sys.exit(0)


def run_server(cmd: List[str], force_reset_on_exit: bool = False) -> int:
    """
    Run vLLM server with proper signal handling and cleanup.

    Args:
        cmd: Command list to execute
        force_reset_on_exit: If True, force GPU reset on exit

    Returns:
        Server process return code
    """
    global _server_process

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Windows-specific signal
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

    # Register cleanup on normal exit
    def exit_cleanup():
        cleanup_gpu_memory(verbose=False)
        if force_reset_on_exit:
            force_gpu_reset()

    atexit.register(exit_cleanup)

    print(f"\nStarting vLLM server...")
    print(f"Command: {' '.join(cmd)}\n")
    print("=" * 60)
    print("Server output follows. Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    return_code = 1

    try:
        # Start server as subprocess (not exec) so we can handle signals
        _server_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            # Create new process group for clean signal handling
            start_new_session=False,
        )

        # Wait for server to exit
        return_code = _server_process.wait()

        if return_code != 0:
            print(f"\nServer exited with code {return_code}")

    except KeyboardInterrupt:
        # This shouldn't be reached due to signal handler, but just in case
        print("\nInterrupted by user")
        if _server_process and _server_process.poll() is None:
            _server_process.terminate()
            try:
                _server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _server_process.kill()
        return_code = 130  # Standard exit code for SIGINT

    except Exception as e:
        print(f"\nServer error: {e}")
        return_code = 1

    finally:
        # Ensure cleanup runs
        cleanup_gpu_memory(verbose=True)

        if force_reset_on_exit:
            force_gpu_reset()

    return return_code


def check_nvidia_smi() -> bool:
    """Check if nvidia-smi is available."""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_gpu_memory_info() -> dict:
    """Get GPU memory information using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if lines:
            # Take first GPU
            total, used, free = map(float, lines[0].split(','))
            return {
                "total_mb": total,
                "used_mb": used,
                "free_mb": free,
                "total_gb": total / 1024,
                "used_gb": used / 1024,
                "free_gb": free / 1024,
            }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not get GPU memory info: {e}")
    return None


def check_cuda_available() -> bool:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_vllm_installed() -> bool:
    """Check if vLLM is installed."""
    try:
        import vllm
        return True
    except ImportError:
        return False


def check_model_exists(model_path: str) -> bool:
    """Check if the model directory/files exist."""
    path = Path(model_path)
    if path.exists():
        # Check for common model files
        if path.is_dir():
            model_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin")) + list(path.glob("config.json"))
            return len(model_files) > 0
        return True
    return False


def prompt_user_with_timeout(prompt: str, timeout_seconds: int = 30) -> str:
    """Prompt user for input with a timeout. Returns empty string if no input."""
    print(prompt, end=' ', flush=True)

    # Check if we're in an interactive terminal
    if not sys.stdin.isatty():
        print("(non-interactive mode, skipping prompt)")
        return ""

    try:
        # Unix-like systems
        if hasattr(select, 'select'):
            ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
            if ready:
                return sys.stdin.readline().strip().lower()
            else:
                print(f"\n(No response after {timeout_seconds}s)")
                return ""
    except Exception:
        pass

    # Fallback: just try to read (will block on Windows)
    try:
        import threading
        result = [None]

        def read_input():
            try:
                result[0] = input().strip().lower()
            except EOFError:
                result[0] = ""

        thread = threading.Thread(target=read_input, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if result[0] is not None:
            return result[0]
        else:
            print(f"\n(No response after {timeout_seconds}s)")
            return ""
    except Exception:
        return ""


def validate_system_requirements(config: dict) -> tuple[bool, list[str]]:
    """
    Validate all system requirements.
    Returns (success, list of error messages).
    """
    errors = []
    warnings = []

    print("\n" + "=" * 60)
    print("System Validation")
    print("=" * 60)

    # Check nvidia-smi
    print("Checking nvidia-smi...", end=" ")
    if check_nvidia_smi():
        print("✓ OK")
    else:
        print("✗ FAILED")
        errors.append("nvidia-smi not available. NVIDIA drivers may not be installed.")

    # Check CUDA
    print("Checking CUDA availability...", end=" ")
    if check_cuda_available():
        import torch
        print(f"✓ OK (CUDA {torch.version.cuda})")
    else:
        print("✗ FAILED")
        errors.append("CUDA not available. PyTorch may not have CUDA support.")

    # Check vLLM
    print("Checking vLLM installation...", end=" ")
    if check_vllm_installed():
        import vllm
        print(f"✓ OK (v{vllm.__version__})")
    else:
        print("✗ FAILED")
        errors.append("vLLM not installed. Install with: pip install vllm")

    # Check model exists
    model_path = config["model"]["name"]
    print(f"Checking model path...", end=" ")
    if check_model_exists(model_path):
        print(f"✓ OK")
    else:
        print("✗ FAILED")
        errors.append(f"Model not found at: {model_path}")

    print("=" * 60)

    return len(errors) == 0, errors


def check_gpu_memory_sufficient(config: dict) -> tuple[bool, dict]:
    """
    Check if there's enough GPU memory available.
    Returns (sufficient, memory_info).
    """
    memory_info = get_gpu_memory_info()

    if memory_info is None:
        return False, None

    # Calculate required memory based on config
    gpu_util = config["model"].get("gpu_memory_utilization", 0.90)
    required_gb = MINIMUM_VRAM_GB

    # Available = free memory
    available_gb = memory_info["free_gb"]

    return available_gb >= required_gb, memory_info


def wait_for_gpu_memory(config: dict, max_retries: int = None) -> bool:
    """
    Wait for sufficient GPU memory, prompting user on each retry.
    Returns True if memory became available, False if user aborted.
    """
    retry_count = 0

    while True:
        sufficient, memory_info = check_gpu_memory_sufficient(config)

        if sufficient:
            return True

        if memory_info:
            print(f"\n{'=' * 60}")
            print("INSUFFICIENT GPU MEMORY")
            print(f"{'=' * 60}")
            print(f"  Total VRAM:     {memory_info['total_gb']:.1f} GB")
            print(f"  Used VRAM:      {memory_info['used_gb']:.1f} GB")
            print(f"  Free VRAM:      {memory_info['free_gb']:.1f} GB")
            print(f"  Required VRAM:  {MINIMUM_VRAM_GB:.1f} GB (minimum)")
            print(f"{'=' * 60}")
        else:
            print("\nCould not determine GPU memory status.")

        # Ask user what to do
        print("\nOptions:")
        print("  [a]bort  - Exit the server")
        print("  [r]etry  - Check GPU memory again now")
        print("  [w]ait   - Wait 20 minutes and retry automatically")
        print("")

        response = prompt_user_with_timeout(
            "Do you want to abort, retry, or wait? [a/r/w] (default: wait after 30s):",
            timeout_seconds=30
        )

        if response in ('a', 'abort'):
            print("Aborted by user.")
            return False
        elif response in ('r', 'retry'):
            print("Retrying immediately...")
            continue
        else:
            # Wait (default behavior)
            retry_count += 1
            if max_retries and retry_count >= max_retries:
                print(f"Max retries ({max_retries}) reached. Aborting.")
                return False

            print(f"\nWaiting {RETRY_INTERVAL_MINUTES} minutes before retry #{retry_count}...")
            print("(Press Ctrl+C to abort)\n")

            try:
                # Show countdown every minute
                for remaining in range(RETRY_INTERVAL_MINUTES, 0, -1):
                    print(f"  Retry in {remaining} minute(s)...", end='\r')
                    time.sleep(60)
                print(" " * 40, end='\r')  # Clear line
            except KeyboardInterrupt:
                print("\n\nAborted by user (Ctrl+C).")
                return False

    return False

def load_config(config_path: str = None) -> dict:
    """Load server configuration."""
    default_config = {
        "model": {
            "name": "~/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ",
            "dtype": "float16",
            "quantization": "awq_marlin",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 98304,
            "trust_remote_code": True,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8002,  # Match .env VLLM_PORT for Somatic Node
        },
        "engine": {
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_num_seqs": 128,
        }
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            if user_config:
                # Deep merge
                for key in user_config:
                    if key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
    
    # Expand paths
    default_config["model"]["name"] = os.path.expanduser(default_config["model"]["name"])
    
    return default_config

def main():
    global MINIMUM_VRAM_GB

    parser = argparse.ArgumentParser(
        description="Ubik vLLM Server with graceful shutdown and RTX 5080 support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start with defaults
  %(prog)s --rtx5080                # Enable RTX 5080 compatibility mode
  %(prog)s --model /path/to/model   # Use custom model path
  %(prog)s --force-gpu-reset        # Force GPU reset on exit (dangerous)

Environment Variables:
  VLLM_FLASH_ATTN_VERSION    Flash Attention version (set to 2 for Blackwell)
  PYTORCH_ALLOC_CONF         CUDA allocator config
  CUDA_MODULE_LOADING        CUDA module loading mode
        """
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model", type=str, help="Model path override")
    parser.add_argument("--port", type=int, help="Port override")
    parser.add_argument("--host", type=str, help="Host override")
    parser.add_argument("--skip-checks", action="store_true", help="Skip system validation")
    parser.add_argument("--min-vram", type=float, default=None,
                        help=f"Minimum required VRAM in GB (default: {MINIMUM_VRAM_GB})")
    parser.add_argument("--rtx5080", action="store_true",
                        help="Enable RTX 5080/Blackwell compatibility mode")
    parser.add_argument("--force-gpu-reset", action="store_true",
                        help="Force GPU reset on exit (WARNING: kills all GPU processes)")
    parser.add_argument("--cleanup-only", action="store_true",
                        help="Only run GPU cleanup without starting server")
    args = parser.parse_args()

    # Handle cleanup-only mode
    if args.cleanup_only:
        print("Running GPU cleanup only...")
        cleanup_gpu_memory(verbose=True)
        if args.force_gpu_reset:
            force_gpu_reset()
        mem_info = get_gpu_memory_info()
        if mem_info:
            print(f"\nGPU memory after cleanup:")
            print(f"  Used: {mem_info['used_gb']:.1f} GB / {mem_info['total_gb']:.1f} GB")
            print(f"  Free: {mem_info['free_gb']:.1f} GB")
        sys.exit(0)

    # Update minimum VRAM if specified
    if args.min_vram is not None:
        MINIMUM_VRAM_GB = args.min_vram

    # Setup RTX 5080 environment if requested
    if args.rtx5080:
        setup_rtx5080_environment()

    config = load_config(args.config)

    # Apply overrides
    if args.model:
        config["model"]["name"] = os.path.expanduser(args.model)
    if args.port:
        config["server"]["port"] = args.port
    if args.host:
        config["server"]["host"] = args.host

    print("=" * 60)
    print("Ubik vLLM Server (AWQ Quantized)")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"Quantization: {config['model'].get('quantization', 'none')}")
    print(f"Host: {config['server']['host']}:{config['server']['port']}")
    print(f"GPU Memory Utilization: {config['model']['gpu_memory_utilization']}")
    print(f"Max Context Length: {config['model']['max_model_len']}")
    print("=" * 60)

    # System validation
    if not args.skip_checks:
        # Validate system requirements
        system_ok, errors = validate_system_requirements(config)
        if not system_ok:
            print("\nSystem validation failed:")
            for err in errors:
                print(f"  - {err}")
            print("\nUse --skip-checks to bypass validation (not recommended).")
            sys.exit(1)

        # Check GPU memory
        if not wait_for_gpu_memory(config):
            print("\nExiting due to insufficient GPU memory.")
            sys.exit(1)

        print("\nAll system checks passed!")
    else:
        print("\nSkipping system validation (--skip-checks)")

    print("")
    
    # Build vLLM command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["model"]["name"],
        "--host", config["server"]["host"],
        "--port", str(config["server"]["port"]),
        "--dtype", config["model"]["dtype"],
        "--tensor-parallel-size", str(config["model"]["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(config["model"]["gpu_memory_utilization"]),
        "--max-model-len", str(config["model"]["max_model_len"]),
        "--max-num-seqs", str(config["engine"]["max_num_seqs"]),
    ]
    
    # Add quantization
    if config["model"].get("quantization"):
        cmd.extend(["--quantization", config["model"]["quantization"]])
    
    # Add trust remote code
    if config["model"].get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    
    # Add engine options
    if config["engine"].get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")
    
    if config["engine"].get("enable_chunked_prefill"):
        cmd.append("--enable-chunked-prefill")

    # Run server with proper signal handling and cleanup
    return_code = run_server(cmd, force_reset_on_exit=args.force_gpu_reset)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
