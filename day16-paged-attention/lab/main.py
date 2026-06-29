"""
Day 16: PagedAttention & Continuous Batching

Sends concurrent requests of varying lengths to a vLLM server and observes
GPU memory utilization. Demonstrates how PagedAttention allocates memory
on demand (like OS virtual memory) rather than pre-allocating max for each
request.

Connects to Day 12 (KV cache grows per token) and Day 15 (vLLM serving).

Before running:
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto \
        --max-model-len 4096 --gpu-memory-utilization 0.90
"""

import time
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# Step 1: Check vLLM server availability
# ---------------------------------------------------------------------------

print("Checking vLLM server availability...")
try:
    health = requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
    health.raise_for_status()
    print(f"  Server is healthy at {VLLM_BASE_URL}\n")
except (requests.ConnectionError, requests.Timeout):
    print("ERROR: vLLM server is not running.")
    print("Start it with:")
    print(f"  vllm serve {MODEL_NAME} --port 8000 --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.90")
    sys.exit(1)
except requests.HTTPError as e:
    print(f"ERROR: vLLM health check failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: Define request groups with different output lengths
# ---------------------------------------------------------------------------

# PagedAttention shines when requests have variable output lengths.
# With static allocation, all requests would reserve max_model_len tokens.
# With PagedAttention, each request only uses pages for its actual tokens.

SHORT_PROMPTS = [
    ("What is 2+2?", 32),
    ("Name one planet.", 32),
    ("Say hello.", 32),
    ("What color is the sky?", 32),
]

MEDIUM_PROMPTS = [
    ("Explain what a GPU does in 3 sentences.", 128),
    ("Write a Python one-liner that squares a number.", 128),
    ("What is the difference between RAM and storage?", 128),
    ("List 5 programming languages.", 128),
]

LONG_PROMPTS = [
    ("Write a detailed explanation of how neural networks learn.", 512),
    ("Explain the history of the internet in several paragraphs.", 512),
    ("Describe the process of photosynthesis step by step.", 512),
    ("Write a short story about a robot learning to cook.", 512),
]

ALL_REQUESTS = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS

# ---------------------------------------------------------------------------
# Step 3: Query vLLM metrics — PagedAttention stats are exposed via Prometheus
# ---------------------------------------------------------------------------


def get_kv_cache_utilization():
    """Parse vLLM's Prometheus metrics for KV cache usage percentage."""
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/metrics", timeout=5)
        resp.raise_for_status()
        metrics_text = resp.text

        # vLLM exposes gpu_cache_usage_perc as a gauge
        # Format varies by version; try common metric names
        patterns = [
            r"vllm:gpu_cache_usage_perc\s+([\d.]+)",
            r"gpu_cache_usage_perc\{[^}]*\}\s+([\d.]+)",
            r"vllm:gpu_cache_usage_perc\{[^}]*\}\s+([\d.]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, metrics_text)
            if match:
                return float(match.group(1)) * 100  # Convert to percentage

        # If specific metric not found, return None
        return None
    except Exception:
        return None


# Record baseline memory utilization before load
baseline_util = get_kv_cache_utilization()
if baseline_util is not None:
    print(f"Baseline KV cache utilization: {baseline_util:.1f}%")
else:
    print("Note: Could not read KV cache metrics (will skip memory comparison)")

# ---------------------------------------------------------------------------
# Step 4: Send all requests concurrently
# ---------------------------------------------------------------------------

# Create OpenAI client pointed at local vLLM (same API as Day 15)
client = OpenAI(
    base_url=f"{VLLM_BASE_URL}/v1",
    api_key="not-needed",
)


def send_request(prompt_text, max_tokens, request_id, group_name):
    """Send one request, return timing and token info."""
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    tokens_out = response.usage.completion_tokens if response.usage else 0
    return request_id, group_name, max_tokens, tokens_out, elapsed_ms


print(f"\nSending {len(ALL_REQUESTS)} concurrent requests (4 short + 4 medium + 4 long)...")
print("=" * 70)

results = []
overall_start = time.perf_counter()

with ThreadPoolExecutor(max_workers=len(ALL_REQUESTS)) as executor:
    futures = []
    for i, (prompt, max_tok) in enumerate(ALL_REQUESTS):
        if i < 4:
            group = "SHORT"
        elif i < 8:
            group = "MEDIUM"
        else:
            group = "LONG"
        futures.append(executor.submit(send_request, prompt, max_tok, i + 1, group))

    for future in as_completed(futures):
        results.append(future.result())

overall_elapsed = time.perf_counter() - overall_start

# ---------------------------------------------------------------------------
# Step 5: Show per-request metrics — continuous batching evidence
# ---------------------------------------------------------------------------

# Sort by completion time (latency) to show that short requests finish first
# even though all were sent simultaneously. This is continuous batching in action.
results.sort(key=lambda x: x[4])  # Sort by latency

print(f"\n{'ID':<4} {'Group':<8} {'MaxTok':<8} {'ActualTok':<10} {'Latency':<10}")
print("-" * 44)
for req_id, group, max_tok, actual_tok, latency_ms in results:
    print(f"  {req_id:<3} {group:<8} {max_tok:<8} {actual_tok:<10} {latency_ms:>7.0f} ms")

# ---------------------------------------------------------------------------
# Step 6: Query post-load memory utilization
# ---------------------------------------------------------------------------

# Small delay to let metrics update
time.sleep(1)
peak_util = get_kv_cache_utilization()

print("\n" + "=" * 70)
print("MEMORY UTILIZATION ANALYSIS")
print("=" * 70)

if peak_util is not None:
    print(f"\n  GPU KV cache utilization: {peak_util:.1f}%")
    print(f"  Concurrent requests handled: {len(ALL_REQUESTS)}")
else:
    print("\n  (Metrics endpoint did not return KV cache utilization)")
    print(f"  Concurrent requests handled: {len(ALL_REQUESTS)}")

# Calculate what static allocation would have required
# Static allocation: each request reserves max_model_len (4096) tokens
BLOCK_SIZE = 16  # Default vLLM block size (tokens per page)
MAX_MODEL_LEN = 4096

static_tokens = len(ALL_REQUESTS) * MAX_MODEL_LEN
actual_tokens = sum(actual_tok for _, _, _, actual_tok, _ in results)
# Include input tokens estimate (roughly 10-20 per prompt)
estimated_input_tokens = len(ALL_REQUESTS) * 15
total_actual = actual_tokens + estimated_input_tokens

print(f"\n  Static allocation would reserve: {static_tokens:,} tokens")
print(f"  Actual tokens used (output+input): ~{total_actual:,} tokens")
print(f"  Efficiency: {(total_actual / static_tokens) * 100:.1f}% of static allocation needed")
print(f"  Waste avoided by PagedAttention: {static_tokens - total_actual:,} tokens")

# ---------------------------------------------------------------------------
# Step 7: Print the key insight
# ---------------------------------------------------------------------------

# This is the PagedAttention advantage: memory scales with actual usage,
# not worst-case allocation. The OS paging analogy holds — you don't need
# 16GB of physical RAM for each app that *could* use 16GB.

avg_short = sum(lat for _, g, _, _, lat in results if g == "SHORT") / 4
avg_medium = sum(lat for _, g, _, _, lat in results if g == "MEDIUM") / 4
avg_long = sum(lat for _, g, _, _, lat in results if g == "LONG") / 4

print(f"\n  Avg latency by group:")
print(f"    SHORT  (32 max tokens):  {avg_short:>7.0f} ms")
print(f"    MEDIUM (128 max tokens): {avg_medium:>7.0f} ms")
print(f"    LONG   (512 max tokens): {avg_long:>7.0f} ms")

print(f"\n  Key insight: Short requests finish {avg_long/avg_short:.1f}x faster than long ones,")
print(f"  even though all ran concurrently. This is continuous batching — ")
print(f"  short requests don't wait for long ones to complete.")

utilization_str = f"{peak_util:.1f}%" if peak_util is not None else "efficient"
print(f"\n  Memory utilization: {utilization_str}")
print(f"  PagedAttention efficiency: Only {(total_actual/static_tokens)*100:.1f}% of static allocation")
print(f"  needed for {len(ALL_REQUESTS)} concurrent requests of variable length.")
