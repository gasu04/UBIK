# DeepSeek Performance Benchmark Tool

A comprehensive benchmarking tool to measure DeepSeek's performance on your system.

## What It Measures

### Performance Metrics:
- ⏱️ **Response Time** - How long it takes to generate responses
- ⚡ **Throughput** - Tokens generated per second
- 💾 **Memory Usage** - RAM consumed during inference
- 🖥️ **CPU Usage** - Processor utilization
- 📊 **Statistics** - Min, max, mean, and median across multiple runs

### Test Categories:
1. **Simple Queries** - Basic questions (e.g., "What is 2+2?")
2. **Medium Queries** - Moderate complexity (e.g., "Explain machine learning")
3. **Complex Queries** - In-depth explanations (e.g., "How do neural networks work?")
4. **Reasoning Queries** - Logic problems and multi-step reasoning

## Installation

```bash
cd "/Volumes/990PRO 4T/DeepSeek"
source venv/bin/activate

# Install required package
pip install psutil
```

## Usage

### Basic Run

```bash
python benchmark_deepseek.py
```

This will:
1. Display your system information
2. Run warmup queries (2 by default)
3. Test each category (5 queries per category)
4. Display comprehensive statistics
5. Save results to `benchmark_results_TIMESTAMP.txt`

### Customization

Edit the configuration at the top of `benchmark_deepseek.py`:

```python
MODEL_NAME = "deepseek-r1:14b"  # Change model
WARMUP_QUERIES = 2              # Number of warmup runs
TEST_QUERIES = 5                # Queries per category
```

## Sample Output

```
======================================================================
DEEPSEEK PERFORMANCE BENCHMARK
======================================================================

📱 System Information:
   cpu_count: 8
   cpu_freq: 2400.0 MHz
   ram_total_gb: 16.0
   ram_available_gb: 8.5
   platform: Darwin
   machine: arm64

🤖 Model: deepseek-r1:14b
🔥 Warmup queries: 2
📊 Test queries per category: 5

======================================================================
Testing: SIMPLE queries
======================================================================

🔥 Warming up (2 queries)...
   Warmup complete!

📊 Running tests (5 queries)...

   Query 1/5
   Prompt: What is 2+2?...
   ✅ Duration: 0.85s
   ⚡ Tokens/sec: 23.5
   💾 Memory: 12.3MB

[...]

======================================================================
BENCHMARK SUMMARY
======================================================================

📊 SIMPLE Queries (5 tests):
   Response Time:
      Mean:   0.92s
      Median: 0.88s
      Range:  0.85s - 1.12s
   Throughput:
      Mean:   22.3 tokens/sec
      Median: 23.1 tokens/sec
      Range:  18.5 - 24.8 tokens/sec
   Memory Usage:
      Mean:   11.8 MB
      Range:  10.2 - 13.5 MB

[... more categories ...]

💾 Results saved to: benchmark_results_20231201_143022.txt

✅ Benchmark complete!
```

## Understanding Results

### Response Time
- **Lower is better**
- Simple queries: ~0.5-2s expected
- Complex queries: ~3-10s expected
- Depends on: CPU, RAM, model size

### Throughput (Tokens/sec)
- **Higher is better**
- Typical range: 10-50 tokens/sec
- Depends on: Hardware, model size, complexity
- Comparison:
  - 10-20 tok/s: Slower, acceptable for basic use
  - 20-40 tok/s: Good performance
  - 40+ tok/s: Excellent performance

### Memory Usage
- Shows additional RAM used per query
- DeepSeek-14B typically uses ~8-10GB base + query overhead
- Monitor if you're running low on RAM

## Troubleshooting

### "ModuleNotFoundError: No module named 'psutil'"
```bash
pip install psutil
```

### "Error connecting to Ollama"
1. Start Ollama: `ollama serve`
2. Verify model is installed: `ollama list`
3. Pull model if needed: `ollama pull deepseek-r1:14b`

### Slow Performance
- Check available RAM: Close other applications
- Check CPU usage: Ensure no background processes
- Try a smaller model: `ollama pull deepseek-r1:7b`

## Comparing Results

Run benchmarks at different times to compare:
- After system restart (cold start)
- With different RAM availability
- With different models
- With different temperature settings

All results are saved with timestamps for easy comparison.

## Tips for Best Results

1. **Close other applications** - Free up RAM and CPU
2. **Run multiple times** - Results may vary, average them
3. **Let system warm up** - First run may be slower
4. **Monitor during run** - Use Activity Monitor to see resource usage
5. **Compare models** - Benchmark 7B vs 14B vs other sizes

## Advanced: Monitoring System Resources

While benchmark runs, open another terminal:

```bash
# Watch CPU and Memory
watch -n 1 'ps aux | grep ollama'

# Or use Activity Monitor (macOS GUI)
open -a "Activity Monitor"
```

## Files Generated

- `benchmark_results_YYYYMMDD_HHMMSS.txt` - Detailed results
- Saved in the same directory as the script

## Next Steps

After benchmarking:
1. Compare your results with different models
2. Optimize system settings if needed
3. Decide which model size works best for your needs
4. Share results in the community for comparison

---

**Happy benchmarking! 🚀**
