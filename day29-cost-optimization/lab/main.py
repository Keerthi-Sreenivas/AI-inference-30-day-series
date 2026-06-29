"""
Day 29: Cost Optimization — Cloud vs Local, Engine Comparison

Benchmark inference throughput, calculate cost per 1K tokens for different
deployment scenarios, and determine break-even points between cloud APIs
and self-hosted inference.

Requires: vLLM running on localhost:8000
  Start with: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
Optional: SGLang on localhost:8001
"""

import time
import requests
import psutil

# --- Check engine availability ---
# We need at least vLLM running. SGLang is optional for comparison.
# This pattern (check before proceed) prevents confusing errors later.
VLLM_URL = "http://localhost:8000"
SGLANG_URL = "http://localhost:8001"

print("=" * 70)
print("COST OPTIMIZATION: Cloud vs Local, Engine Comparison")
print("=" * 70)

# Check vLLM (required)
print(f"\nChecking vLLM at {VLLM_URL}...")
try:
    r = requests.get(f"{VLLM_URL}/health", timeout=5)
    r.raise_for_status()
    vllm_available = True
    print("  ✓ vLLM is running")
except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
    print("  ✗ ERROR: vLLM is not running!")
    print(f"    Start it with: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000")
    raise SystemExit(1)

# Check SGLang (optional)
print(f"Checking SGLang at {SGLANG_URL}...")
try:
    r = requests.get(f"{SGLANG_URL}/health", timeout=5)
    r.raise_for_status()
    sglang_available = True
    print("  ✓ SGLang is running")
except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
    sglang_available = False
    print("  ○ SGLang not available — will skip comparison")

print()

# --- Benchmark vLLM throughput ---
# Send multiple requests and measure aggregate throughput. This is the
# critical number for cost calculations: tokens generated per second
# determines how much you can amortize hardware costs.
print("=" * 70)
print("BENCHMARK: Measuring Throughput")
print("=" * 70)

BENCHMARK_PROMPTS = [
    "Explain what Python is in two sentences.",
    "What is machine learning?",
    "Define neural network briefly.",
    "What is a GPU used for?",
    "Explain API in simple terms.",
    "What is Docker?",
    "Define latency.",
    "What is throughput?",
    "Explain cloud computing briefly.",
    "What is quantization in AI?",
]
MAX_TOKENS = 50


def benchmark_engine(url, engine_name, prompts, max_tokens):
    """Send requests to an engine and measure throughput."""
    total_tokens = 0
    total_time = 0

    print(f"\n  Benchmarking {engine_name} ({len(prompts)} requests, {max_tokens} max tokens each)...")

    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        response = requests.post(
            f"{url}/v1/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=60,
        )
        elapsed = time.perf_counter() - start

        if response.status_code == 200:
            result = response.json()
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            total_tokens += tokens
            total_time += elapsed
            print(f"    Request {i+1}: {tokens} tokens in {elapsed:.2f}s")
        else:
            print(f"    Request {i+1}: FAILED ({response.status_code})")

    throughput = total_tokens / total_time if total_time > 0 else 0
    return throughput, total_tokens, total_time


# Benchmark vLLM
vllm_throughput, vllm_tokens, vllm_time = benchmark_engine(
    VLLM_URL, "vLLM", BENCHMARK_PROMPTS, MAX_TOKENS
)
print(f"\n  vLLM Result: {vllm_throughput:.1f} tokens/sec "
      f"({vllm_tokens} tokens in {vllm_time:.1f}s)")

# Benchmark SGLang if available
sglang_throughput = 0
if sglang_available:
    sglang_throughput, sglang_tokens, sglang_time = benchmark_engine(
        SGLANG_URL, "SGLang", BENCHMARK_PROMPTS, MAX_TOKENS
    )
    print(f"\n  SGLang Result: {sglang_throughput:.1f} tokens/sec "
          f"({sglang_tokens} tokens in {sglang_time:.1f}s)")

# --- Define cost scenarios ---
# Each scenario has a fixed hourly cost and a method to calculate throughput.
# Cloud APIs charge per token directly. Self-hosted charges per hour regardless
# of utilization — that's the key difference.
print("\n" + "=" * 70)
print("COST ANALYSIS: Deployment Scenarios")
print("=" * 70)

# Use measured throughput for the "local GPU" scenario
# Estimate throughput for other hardware tiers based on ratios
measured_throughput = vllm_throughput  # tokens/sec on current hardware

# Hardware cost estimates (hourly, based on major cloud providers)
scenarios = [
    {
        "name": "Cloud API (GPT-4o output)",
        "type": "cloud",
        "cost_per_1k": 0.015,
        "throughput": None,  # Pay per token, throughput is "infinite"
    },
    {
        "name": "Cloud API (Llama-3.1-8B via Together)",
        "type": "cloud",
        "cost_per_1k": 0.0002,
        "throughput": None,
    },
    {
        "name": "Self-hosted: A100 80GB (vLLM)",
        "type": "self_hosted",
        "hourly_cost": 2.50,
        "throughput": measured_throughput * 3.0,  # A100 estimate ~3x L4
    },
    {
        "name": "Self-hosted: L4 24GB (vLLM)",
        "type": "self_hosted",
        "hourly_cost": 0.80,
        "throughput": measured_throughput,  # Use measured value
    },
    {
        "name": "Self-hosted: T4 16GB (quantized)",
        "type": "self_hosted",
        "hourly_cost": 0.35,
        "throughput": measured_throughput * 0.4,  # T4 estimate ~40% of L4
    },
    {
        "name": "CPU-only (llama.cpp)",
        "type": "self_hosted",
        "hourly_cost": 0.05,
        "throughput": 15.0,  # Typical CPU llama.cpp for 7B model
    },
]

# --- Calculate cost per 1K tokens ---
# For self-hosted: cost_per_1k = (hourly_cost / tokens_per_hour) * 1000
# For cloud: cost_per_1k is given directly (they charge per token)
print(f"\n{'Scenario':<40} {'Tok/s':<10} {'$/1K tok':<12} {'$/1M tok/day':<15} {'Best For'}")
print("-" * 100)

cost_results = []

for s in scenarios:
    if s["type"] == "cloud":
        cost_per_1k = s["cost_per_1k"]
        throughput_str = "N/A"
        # Monthly cost at 1M tokens/day = 1M/1K * cost_per_1k * 30
        monthly_1m = (1_000_000 / 1000) * cost_per_1k * 30
    else:
        throughput = s["throughput"]
        tokens_per_hour = throughput * 3600
        cost_per_1k = (s["hourly_cost"] / tokens_per_hour) * 1000 if tokens_per_hour > 0 else 999
        throughput_str = f"{throughput:.0f}"
        # Monthly cost = hourly * 24 * 30 (always running)
        monthly_1m = s["hourly_cost"] * 24 * 30

    # Determine best use case
    if cost_per_1k > 0.01:
        best_for = "Low volume, zero ops"
    elif cost_per_1k > 0.001:
        best_for = "Medium volume"
    elif cost_per_1k > 0.0003:
        best_for = "High volume production"
    else:
        best_for = "Maximum throughput"

    cost_results.append({
        "name": s["name"],
        "throughput": throughput_str,
        "cost_per_1k": cost_per_1k,
        "monthly": monthly_1m,
        "best_for": best_for,
    })

    print(f"{s['name']:<40} {throughput_str:<10} ${cost_per_1k:<11.5f} ${monthly_1m:<14.0f} {best_for}")

# --- Break-even analysis ---
# At what daily volume does self-hosted become cheaper than cloud API?
# Cloud cost = tokens_per_day / 1000 * cloud_price
# Self-hosted cost = hourly_rate * 24 (fixed, always running)
# Break-even: hourly_rate * 24 = tokens_per_day / 1000 * cloud_price
# tokens_per_day = (hourly_rate * 24 * 1000) / cloud_price
print("\n" + "=" * 70)
print("BREAK-EVEN ANALYSIS")
print("=" * 70)

cloud_price = 0.015  # GPT-4o output price
a100_hourly = 2.50
l4_hourly = 0.80

breakeven_a100 = (a100_hourly * 24 * 1000) / cloud_price
breakeven_l4 = (l4_hourly * 24 * 1000) / cloud_price

print(f"\n  Comparing against Cloud API at ${cloud_price}/1K tokens:")
print(f"  ─────────────────────────────────────────────────────")
print(f"  A100 break-even: {breakeven_a100:,.0f} tokens/day ({breakeven_a100/1e6:.1f}M)")
print(f"  L4 break-even:   {breakeven_l4:,.0f} tokens/day ({breakeven_l4/1e6:.1f}M)")
print(f"\n  Below these thresholds: Cloud API is cheaper")
print(f"  Above these thresholds: Self-hosted saves money")

# Against cheaper cloud API (Together/Fireworks for open models)
cheap_cloud = 0.0002
breakeven_cheap_a100 = (a100_hourly * 24 * 1000) / cheap_cloud

print(f"\n  Comparing against cheap cloud API at ${cheap_cloud}/1K tokens:")
print(f"  ─────────────────────────────────────────────────────")
print(f"  A100 break-even: {breakeven_cheap_a100:,.0f} tokens/day ({breakeven_cheap_a100/1e6:.0f}M)")
print(f"  (Much harder to beat cheap cloud APIs — need very high volume)")

# --- Optimization recommendations ---
print("\n" + "=" * 70)
print("OPTIMIZATION RECOMMENDATIONS BY TIER")
print("=" * 70)

print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │ LOW VOLUME (< 100K tokens/day)                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │ • Use cloud APIs — no hardware cost, no ops overhead            │
  │ • Choose based on quality needs: GPT-4o for best, open models   │
  │   via Together/Fireworks for cheap                              │
  │ • Monthly cost: < $5                                            │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ MEDIUM VOLUME (100K – 5M tokens/day)                            │
  ├─────────────────────────────────────────────────────────────────┤
  │ • Consider self-hosted on L4/T4 for open models                 │
  │ • Use quantization (INT4) to fit on smaller/cheaper GPUs        │
  │ • Enable continuous batching to maximize throughput              │
  │ • Monthly cost: $20 – $600                                      │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ HIGH VOLUME (> 5M tokens/day)                                   │
  ├─────────────────────────────────────────────────────────────────┤
  │ • Self-hosted is clearly cheaper — invest in optimization       │
  │ • Use A100/H100 with vLLM or SGLang for maximum throughput      │
  │ • Enable prefix caching (SGLang) if prompts share prefixes      │
  │ • Consider spot instances (60-70% savings, handle interruptions)│
  │ • Monthly cost: $600 – $2000+                                   │
  └─────────────────────────────────────────────────────────────────┘
""")

# --- Summary metrics ---
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Metric':<40} {'Value':<25}")
print("-" * 65)
print(f"{'Measured vLLM throughput':<40} {vllm_throughput:.1f} tokens/sec")
if sglang_available:
    print(f"{'Measured SGLang throughput':<40} {sglang_throughput:.1f} tokens/sec")
print(f"{'Cheapest self-hosted (per 1K tok)':<40} ${min(r['cost_per_1k'] for r in cost_results if 'Self' in r['name']):.5f}")
print(f"{'Cloud API GPT-4o (per 1K tok)':<40} $0.01500")
print(f"{'Break-even vs GPT-4o (L4)':<40} {breakeven_l4/1e6:.1f}M tokens/day")
print(f"{'System RAM':<40} {psutil.virtual_memory().total / 1e9:.0f} GB")
print(f"{'CPU cores':<40} {psutil.cpu_count()}")
print("-" * 65)
print("\nKey insight: The cheapest option depends on your VOLUME. Cloud APIs")
print("win at low volume (zero idle cost). Self-hosted wins at high volume")
print("(fixed hardware cost amortized over many tokens). Always benchmark")
print("YOUR workload — these numbers change with prompt length, batch size,")
print("and model choice.")
