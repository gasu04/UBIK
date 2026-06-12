#!/usr/bin/env python3
"""
DeepSeek Performance Benchmark Tool
Measures performance metrics when running DeepSeek model locally via Ollama
"""

import time
import psutil
import os
import statistics
from datetime import datetime
from langchain_ollama import OllamaLLM

# Benchmark Configuration
MODEL_NAME = "deepseek-r1:14b"
WARMUP_QUERIES = 2
TEST_QUERIES = 5

# Test prompts of varying complexity
TEST_PROMPTS = {
    "simple": [
        "What is 2+2?",
        "Say hello.",
        "What color is the sky?",
        "Name a fruit.",
        "What is Python?",
    ],
    "medium": [
        "Explain what machine learning is in 2-3 sentences.",
        "Write a Python function to calculate factorial.",
        "What are the benefits of using virtual environments?",
        "Describe the difference between a list and a tuple in Python.",
        "Explain what a REST API is.",
    ],
    "complex": [
        "Explain the concept of neural networks and how backpropagation works.",
        "Write a Python class that implements a binary search tree with insert and search methods.",
        "Compare and contrast different sorting algorithms and their time complexities.",
        "Explain the CAP theorem in distributed systems.",
        "Describe how transformers work in natural language processing.",
    ],
    "reasoning": [
        "If a train leaves Chicago at 3pm traveling 60mph, and another leaves New York at 4pm traveling 80mph, when do they meet? The cities are 790 miles apart.",
        "You have 12 balls, one is slightly heavier. Using a balance scale only 3 times, how do you find the heavy ball?",
        "Explain step by step: If all A are B, and all B are C, what can we conclude about A and C?",
        "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
        "If you're running a race and pass the person in 2nd place, what place are you in now?",
    ]
}

def get_system_info():
    """Get system information"""
    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "platform": os.uname().sysname,
        "machine": os.uname().machine,
    }

def get_process_stats():
    """Get current process resource usage"""
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(interval=0.1),
        "memory_mb": round(process.memory_info().rss / (1024**1024), 2),
        "threads": process.num_threads(),
    }

def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

def benchmark_query(llm, prompt, warmup=False):
    """
    Benchmark a single query

    Returns dict with timing and resource metrics
    """
    # Get baseline stats
    start_memory = psutil.Process().memory_info().rss / (1024**2)
    start_cpu = psutil.cpu_percent(interval=None)

    # Run query
    start_time = time.time()

    try:
        response = llm.invoke(prompt)
        end_time = time.time()
        success = True
        error = None
    except Exception as e:
        end_time = time.time()
        response = ""
        success = False
        error = str(e)

    # Get end stats
    end_memory = psutil.Process().memory_info().rss / (1024**2)
    end_cpu = psutil.cpu_percent(interval=None)

    # Calculate metrics
    duration = end_time - start_time
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(response)
    total_tokens = input_tokens + output_tokens
    tokens_per_second = output_tokens / duration if duration > 0 else 0

    return {
        "success": success,
        "error": error,
        "duration_seconds": round(duration, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "memory_used_mb": round(end_memory - start_memory, 2),
        "cpu_percent": round((start_cpu + end_cpu) / 2, 1),
        "response_length": len(response),
        "warmup": warmup,
    }

def run_benchmark_suite(llm, prompt_type, prompts):
    """Run a suite of benchmarks for a given prompt type"""
    print(f"\n{'='*70}")
    print(f"Testing: {prompt_type.upper()} queries")
    print(f"{'='*70}")

    results = []

    # Warmup
    if WARMUP_QUERIES > 0:
        print(f"\n🔥 Warming up ({WARMUP_QUERIES} queries)...")
        for i in range(min(WARMUP_QUERIES, len(prompts))):
            print(f"   Warmup {i+1}/{WARMUP_QUERIES}...", end="\r")
            result = benchmark_query(llm, prompts[i], warmup=True)
            results.append(result)
        print(f"   Warmup complete!{' '*30}")

    # Actual tests
    print(f"\n📊 Running tests ({min(TEST_QUERIES, len(prompts))} queries)...")
    for i in range(min(TEST_QUERIES, len(prompts))):
        prompt = prompts[i]
        print(f"\n   Query {i+1}/{min(TEST_QUERIES, len(prompts))}")
        print(f"   Prompt: {prompt[:60]}...")

        result = benchmark_query(llm, prompt, warmup=False)
        results.append(result)

        if result["success"]:
            print(f"   ✅ Duration: {result['duration_seconds']}s")
            print(f"   ⚡ Tokens/sec: {result['tokens_per_second']}")
            print(f"   💾 Memory: {result['memory_used_mb']}MB")
        else:
            print(f"   ❌ Error: {result['error']}")

    return results

def calculate_statistics(results):
    """Calculate statistics from benchmark results"""
    # Filter out warmup and failed queries
    valid_results = [r for r in results if not r["warmup"] and r["success"]]

    if not valid_results:
        return None

    durations = [r["duration_seconds"] for r in valid_results]
    tokens_per_sec = [r["tokens_per_second"] for r in valid_results]
    memory_used = [r["memory_used_mb"] for r in valid_results]

    return {
        "count": len(valid_results),
        "duration": {
            "min": round(min(durations), 2),
            "max": round(max(durations), 2),
            "mean": round(statistics.mean(durations), 2),
            "median": round(statistics.median(durations), 2),
        },
        "tokens_per_second": {
            "min": round(min(tokens_per_sec), 2),
            "max": round(max(tokens_per_sec), 2),
            "mean": round(statistics.mean(tokens_per_sec), 2),
            "median": round(statistics.median(tokens_per_sec), 2),
        },
        "memory_mb": {
            "min": round(min(memory_used), 2),
            "max": round(max(memory_used), 2),
            "mean": round(statistics.mean(memory_used), 2),
        }
    }

def print_summary(all_results, system_info):
    """Print comprehensive summary"""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    print("\n📱 System Information:")
    print(f"   Platform: {system_info['platform']} ({system_info['machine']})")
    print(f"   CPU Cores: {system_info['cpu_count']}")
    print(f"   CPU Frequency: {system_info['cpu_freq']} MHz")
    print(f"   RAM Total: {system_info['ram_total_gb']} GB")
    print(f"   RAM Available: {system_info['ram_available_gb']} GB")

    print(f"\n🤖 Model: {MODEL_NAME}")

    for prompt_type, results in all_results.items():
        stats = calculate_statistics(results)

        if not stats:
            print(f"\n❌ {prompt_type.upper()}: No valid results")
            continue

        print(f"\n📊 {prompt_type.upper()} Queries ({stats['count']} tests):")
        print(f"   Response Time:")
        print(f"      Mean:   {stats['duration']['mean']}s")
        print(f"      Median: {stats['duration']['median']}s")
        print(f"      Range:  {stats['duration']['min']}s - {stats['duration']['max']}s")
        print(f"   Throughput:")
        print(f"      Mean:   {stats['tokens_per_second']['mean']} tokens/sec")
        print(f"      Median: {stats['tokens_per_second']['median']} tokens/sec")
        print(f"      Range:  {stats['tokens_per_second']['min']} - {stats['tokens_per_second']['max']} tokens/sec")
        print(f"   Memory Usage:")
        print(f"      Mean:   {stats['memory_mb']['mean']} MB")
        print(f"      Range:  {stats['memory_mb']['min']} - {stats['memory_mb']['max']} MB")

def save_results(all_results, system_info, filename="benchmark_results.txt"):
    """Save results to file"""
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DEEPSEEK PERFORMANCE BENCHMARK RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_NAME}\n")

        f.write("\n" + "-"*70 + "\n")
        f.write("SYSTEM INFORMATION\n")
        f.write("-"*70 + "\n")
        for key, value in system_info.items():
            f.write(f"{key}: {value}\n")

        for prompt_type, results in all_results.items():
            f.write("\n" + "-"*70 + "\n")
            f.write(f"{prompt_type.upper()} QUERIES\n")
            f.write("-"*70 + "\n")

            stats = calculate_statistics(results)
            if stats:
                f.write(f"\nTests: {stats['count']}\n")
                f.write(f"\nResponse Time:\n")
                f.write(f"  Mean:   {stats['duration']['mean']}s\n")
                f.write(f"  Median: {stats['duration']['median']}s\n")
                f.write(f"  Min:    {stats['duration']['min']}s\n")
                f.write(f"  Max:    {stats['duration']['max']}s\n")
                f.write(f"\nThroughput:\n")
                f.write(f"  Mean:   {stats['tokens_per_second']['mean']} tokens/sec\n")
                f.write(f"  Median: {stats['tokens_per_second']['median']} tokens/sec\n")
                f.write(f"  Min:    {stats['tokens_per_second']['min']} tokens/sec\n")
                f.write(f"  Max:    {stats['tokens_per_second']['max']} tokens/sec\n")
                f.write(f"\nMemory Usage:\n")
                f.write(f"  Mean:   {stats['memory_mb']['mean']} MB\n")
                f.write(f"  Min:    {stats['memory_mb']['min']} MB\n")
                f.write(f"  Max:    {stats['memory_mb']['max']} MB\n")

    print(f"\n💾 Results saved to: {filename}")

def main():
    """Main benchmark function"""
    print("="*70)
    print("DEEPSEEK PERFORMANCE BENCHMARK")
    print("="*70)

    # Get system info
    system_info = get_system_info()

    print("\n📱 System Information:")
    for key, value in system_info.items():
        print(f"   {key}: {value}")

    print(f"\n🤖 Model: {MODEL_NAME}")
    print(f"🔥 Warmup queries: {WARMUP_QUERIES}")
    print(f"📊 Test queries per category: {TEST_QUERIES}")

    # Initialize LLM
    print("\n🔌 Connecting to Ollama...")
    try:
        llm = OllamaLLM(model=MODEL_NAME, temperature=0.7)
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("\nMake sure:")
        print("1. Ollama is running")
        print(f"2. Model is installed: ollama pull {MODEL_NAME}")
        return

    # Run benchmarks
    all_results = {}

    for prompt_type, prompts in TEST_PROMPTS.items():
        results = run_benchmark_suite(llm, prompt_type, prompts)
        all_results[prompt_type] = results

    # Print summary
    print_summary(all_results, system_info)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.txt"
    save_results(all_results, system_info, filename)

    print("\n" + "="*70)
    print("✅ Benchmark complete!")
    print("="*70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
