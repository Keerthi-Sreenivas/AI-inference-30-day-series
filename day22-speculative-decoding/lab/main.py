"""
Day 22: Speculative Decoding — Draft Models for Speed

Measures speculative decoding performance by comparing vLLM generation speed
with and without a draft model. Reports acceptance rate (%) as the key metric.

Target model: meta-llama/Llama-3.1-8B (served via vLLM)
Draft model: HuggingFaceTB/SmolLM2-1.7B-Instruct

Requirements:
- vLLM server with speculation on port 8000
- vLLM baseline server on port 8001
"""

import time
import requests

# --- Configuration ---
# Two vLLM servers: one with speculative decoding, one without.
# This lets us measure the exact speedup from draft model speculation.
SPECULATIVE_URL = "http://localhost:8000"
BASELINE_URL = "http://localhost:8001"
METRICS_URL = f"{SPECULATIVE_URL}/metrics"

# --- Check server availability ---
# Both servers must be running before we can benchmark. The speculative server
# runs Llama-3.1-8B with SmolLM2-1.7B-Instruct as the draft model.
print("Checking server availability...")
print("=" * 70)

try:
    spec_health = requests.get(f"{SPECULATIVE_URL}/health", timeout=5)
    spec_health.raise_for_status()
    print(f"  Speculative server ({SPECULATIVE_URL}): ONLINE")
except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
    print(f"ERROR: Speculative decoding server not reachable at {SPECULATIVE_URL}")
    print("\nStart it with:")
    print("  vllm serve meta-llama/Llama-3.1-8B \\")
    print("      --speculative-model HuggingFaceTB/SmolLM2-1.7B-Instruct \\")
    print("      --num-speculative-tokens 5 \\")
    print("      --port 8000")
    raise SystemExit(1)

try:
    base_health = requests.get(f"{BASELINE_URL}/health", timeout=5)
    base_health.raise_for_status()
    print(f"  Baseline server ({BASELINE_URL}): ONLINE")
except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
    print(f"ERROR: Baseline server not reachable at {BASELINE_URL}")
    print("\nStart it with:")
    print("  vllm serve meta-llama/Llama-3.1-8B --port 8001")
    raise SystemExit(1)

print("\nBoth servers online. Starting benchmark...\n")

# --- Define test prompts ---
# Different prompt types test different acceptance rate scenarios.
# Factual prompts are predictable (high acceptance), creative prompts are
# diverse (lower acceptance), code prompts are structured (medium acceptance).
test_prompts = [
    {
        "name": "Factual",
        "prompt": "What is the capital of France? Provide a brief one-paragraph answer.",
        "description": "Predictable tokens → high acceptance rate expected",
    },
    {
        "name": "Creative",
        "prompt": "Write a short poem about a robot learning to feel emotions for the first time.",
        "description": "Creative tokens → lower acceptance rate expected",
    },
    {
        "name": "Code",
        "prompt": "Write a Python function that sorts a list using merge sort. Include comments.",
        "description": "Structured tokens → medium acceptance rate expected",
    },
]

# --- Benchmark helper function ---
# Sends a completion request to a vLLM server and measures wall-clock time.
# Returns the elapsed time and number of tokens generated.
def benchmark_request(base_url, prompt, max_tokens=200, temperature=0.7):
    """Send a completion request and measure time + tokens generated."""
    payload = {
        "model": "meta-llama/Llama-3.1-8B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    elapsed = time.perf_counter() - start

    response.raise_for_status()
    data = response.json()

    # Extract token count from usage stats returned by vLLM
    completion_tokens = data["usage"]["completion_tokens"]
    return elapsed, completion_tokens


# --- Benchmark baseline (no speculation) ---
# The baseline server generates tokens one at a time, standard autoregressive.
# This gives us the reference speed to measure speculation against.
print("=" * 70)
print("BASELINE (standard autoregressive decoding)")
print("=" * 70)

baseline_results = []
for prompt_info in test_prompts:
    elapsed, tokens = benchmark_request(BASELINE_URL, prompt_info["prompt"])
    tokens_per_sec = tokens / elapsed
    baseline_results.append({
        "name": prompt_info["name"],
        "elapsed": elapsed,
        "tokens": tokens,
        "tokens_per_sec": tokens_per_sec,
    })
    print(f"  {prompt_info['name']:<10} | {tokens:>3} tokens | {elapsed:.2f}s | {tokens_per_sec:.1f} tok/s")

print()

# --- Benchmark speculative decoding ---
# The speculative server uses SmolLM2-1.7B-Instruct as a draft model to guess
# ahead, then Llama-3.1-8B verifies. Correct guesses are "free" tokens.
print("=" * 70)
print("SPECULATIVE DECODING (draft model: SmolLM2-1.7B-Instruct)")
print("=" * 70)

speculative_results = []
for prompt_info in test_prompts:
    elapsed, tokens = benchmark_request(SPECULATIVE_URL, prompt_info["prompt"])
    tokens_per_sec = tokens / elapsed
    speculative_results.append({
        "name": prompt_info["name"],
        "elapsed": elapsed,
        "tokens": tokens,
        "tokens_per_sec": tokens_per_sec,
    })
    print(f"  {prompt_info['name']:<10} | {tokens:>3} tokens | {elapsed:.2f}s | {tokens_per_sec:.1f} tok/s")

print()

# --- Fetch acceptance rate from metrics ---
# vLLM exposes Prometheus metrics including speculative decoding acceptance rate.
# This tells us what percentage of draft tokens the target model agreed with.
print("=" * 70)
print("SPECULATIVE DECODING METRICS")
print("=" * 70)

acceptance_rate = None
try:
    metrics_response = requests.get(METRICS_URL, timeout=10)
    metrics_text = metrics_response.text

    # Parse acceptance rate from Prometheus metrics format
    for line in metrics_text.split("\n"):
        if "spec_decode_draft_acceptance_rate" in line and not line.startswith("#"):
            # Extract the value from the metric line
            parts = line.strip().split()
            if len(parts) >= 2:
                acceptance_rate = float(parts[-1]) * 100  # Convert to percentage
                break

    if acceptance_rate is not None:
        print(f"  Draft token acceptance rate: {acceptance_rate:.1f}%")
        print(f"  This means {acceptance_rate:.0f}% of draft model guesses were accepted by the target.")
    else:
        print("  Acceptance rate metric not found in /metrics endpoint.")
        print("  (May require more requests to populate metrics)")
except requests.RequestException as e:
    print(f"  Could not fetch metrics: {e}")

print()

# --- Calculate speedup and print summary ---
# Speedup = baseline_time / speculative_time for the same number of tokens.
# A speedup > 1.0 means speculation is faster than standard decoding.
print("=" * 70)
print("SUMMARY: Speculative Decoding Speedup")
print("=" * 70)
print(f"{'Prompt Type':<12} {'Baseline tok/s':<16} {'Speculative tok/s':<19} {'Speedup':<10}")
print("-" * 57)

total_speedup = 0
for base, spec in zip(baseline_results, speculative_results):
    speedup = spec["tokens_per_sec"] / base["tokens_per_sec"] if base["tokens_per_sec"] > 0 else 0
    total_speedup += speedup
    print(f"{base['name']:<12} {base['tokens_per_sec']:<16.1f} {spec['tokens_per_sec']:<19.1f} {speedup:<10.2f}×")

avg_speedup = total_speedup / len(baseline_results)
print("-" * 57)
print(f"{'Average':<12} {'':16} {'':19} {avg_speedup:.2f}×")

print()
if acceptance_rate is not None:
    print(f"Acceptance Rate: {acceptance_rate:.1f}%")
print(f"Average Speedup: {avg_speedup:.2f}×")
print()
print("Key insight: Speculative decoding gives IDENTICAL output quality to")
print("standard decoding — it only changes speed. The acceptance rate determines")
print("how much speedup you get. Factual/structured tasks typically have higher")
print("acceptance rates than creative/open-ended generation.")
