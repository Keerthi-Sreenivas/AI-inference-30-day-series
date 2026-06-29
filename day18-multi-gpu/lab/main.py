"""
Day 18: Multi-GPU Inference — Measuring Scaling Efficiency

This script benchmarks a vLLM server running with tensor parallelism
and calculates scaling efficiency compared to a single-GPU baseline.

Tensor parallelism splits model weight matrices across multiple GPUs.
Each GPU handles a slice of the attention heads and feed-forward layers,
then results are combined via an all-reduce operation (NCCL).

Before running:
    # Multi-GPU (2 GPUs):
    vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 2 --port 8000

    # Or single GPU baseline:
    vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 1 --port 8000

    # Ungated alternative:
    vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --port 8000

Usage:
    python main.py                        # Report raw throughput
    python main.py --baseline-tps 150     # Compare against single-GPU baseline
    python main.py --num-requests 50      # Send more concurrent requests
"""

import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ───────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{VLLM_BASE_URL}/health"
MODELS_URL = f"{VLLM_BASE_URL}/v1/models"
COMPLETIONS_URL = f"{VLLM_BASE_URL}/v1/chat/completions"

# Fixed prompt for consistent benchmarking
BENCHMARK_PROMPT = "Explain the concept of distributed computing in detail."
MAX_TOKENS = 256


# ─── Service availability check ─────────────────────────────────────────────

def check_server():
    """Verify vLLM is running before we start sending requests."""
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        resp.raise_for_status()
    except (requests.ConnectionError, requests.Timeout):
        print("ERROR: vLLM server is not running.")
        print()
        print("Start it with:")
        print("  vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 2 --port 8000")
        print()
        print("Or for single-GPU baseline:")
        print("  vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 1 --port 8000")
        raise SystemExit(1)


def get_model_info():
    """Fetch the loaded model name from vLLM."""
    resp = requests.get(MODELS_URL, timeout=10)
    data = resp.json()
    models = data.get("data", [])
    if models:
        return models[0].get("id", "unknown")
    return "unknown"


# ─── Benchmark logic ─────────────────────────────────────────────────────────

def send_request(request_id):
    """
    Send a single chat completion request and return tokens generated.

    Each request goes through vLLM's scheduler, which batches it with
    others using continuous batching. With tensor parallelism, the
    computation for each request is split across all GPUs.
    """
    payload = {
        "model": get_model_info(),
        "messages": [
            {"role": "user", "content": BENCHMARK_PROMPT}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }

    resp = requests.post(COMPLETIONS_URL, json=payload, timeout=120)
    result = resp.json()

    # Extract token count from usage stats
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return completion_tokens


def run_benchmark(num_requests, num_workers=10):
    """
    Send concurrent requests and measure aggregate throughput.

    Why concurrent requests? Tensor parallelism shines under load.
    A single request may not saturate all GPUs, but many concurrent
    requests fill the batch and keep all GPUs busy.
    """
    print(f"Sending {num_requests} concurrent requests...")
    print(f"  Prompt: \"{BENCHMARK_PROMPT[:50]}...\"")
    print(f"  Max tokens per request: {MAX_TOKENS}")
    print()

    total_tokens = 0
    completed = 0

    start_time = time.perf_counter()

    # ThreadPoolExecutor sends requests in parallel
    # vLLM's continuous batching handles them efficiently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(send_request, i): i
            for i in range(num_requests)
        }

        for future in as_completed(futures):
            try:
                tokens = future.result()
                total_tokens += tokens
                completed += 1
            except Exception as e:
                print(f"  Request failed: {e}")

    elapsed = time.perf_counter() - start_time
    return total_tokens, elapsed, completed


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-GPU vLLM scaling efficiency"
    )
    parser.add_argument(
        "--baseline-tps", type=float, default=None,
        help="Single-GPU throughput (tokens/sec) for efficiency calculation"
    )
    parser.add_argument(
        "--num-requests", type=int, default=20,
        help="Number of concurrent requests to send (default: 20)"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=2,
        help="Number of GPUs the server is using (default: 2)"
    )
    args = parser.parse_args()

    # Step 1: Check server is available
    check_server()

    # Step 2: Get model info
    model_name = get_model_info()
    print(f"Model: {model_name}")
    print(f"GPUs (reported): {args.num_gpus}")
    print()

    # Step 3: Run the benchmark
    total_tokens, elapsed, completed = run_benchmark(args.num_requests)

    # Step 4: Calculate throughput
    # Throughput = total tokens generated / wall-clock time
    # This measures aggregate output across all concurrent requests
    throughput = total_tokens / elapsed if elapsed > 0 else 0

    # Step 5: Print results
    print("=" * 60)
    print("MULTI-GPU SCALING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Requests sent:          {args.num_requests}")
    print(f"  Requests completed:     {completed}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Elapsed time:           {elapsed:.2f} s")
    print(f"  Throughput:             {throughput:.1f} tokens/sec")

    # Step 6: Calculate scaling efficiency if baseline provided
    # Scaling efficiency = actual / (baseline × num_gpus) × 100
    # Perfect scaling (100%) means doubling GPUs doubles throughput
    # In practice, all-reduce communication overhead reduces efficiency
    if args.baseline_tps:
        theoretical_max = args.baseline_tps * args.num_gpus
        efficiency = (throughput / theoretical_max) * 100
        print(f"  Baseline (1 GPU):       {args.baseline_tps:.1f} tokens/sec")
        print(f"  Theoretical max:        {theoretical_max:.1f} tokens/sec")
        print(f"  Scaling efficiency:     {efficiency:.1f}%")
        print()

        # Interpret the result
        if efficiency >= 90:
            print("  Excellent! Near-linear scaling (likely NVLink).")
        elif efficiency >= 70:
            print("  Good scaling. Typical for PCIe interconnect.")
        elif efficiency >= 50:
            print("  Moderate scaling. Consider larger batch sizes.")
        else:
            print("  Poor scaling. Check interconnect or increase load.")
    else:
        # Single-GPU fallback: report baseline
        print()
        print("  No baseline provided. This measurement can serve as baseline.")
        print(f"  Re-run with: python main.py --baseline-tps {throughput:.1f} --num-gpus 2")
        print()
        print("  TIP: Run with TP=1 first to get baseline, then TP=2 to measure scaling.")

    print("=" * 60)


if __name__ == "__main__":
    main()
