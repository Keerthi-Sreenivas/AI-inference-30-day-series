"""
Day 15: vLLM — Production Inference Serving

Measures throughput (requests/second) by sending concurrent requests to a
running vLLM server. Demonstrates how vLLM handles parallelism compared to
sequential generation (Day 4).

Before running:
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto --max-model-len 4096
"""

import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Sampling parameters (recall Day 4: temperature controls randomness)
TEMPERATURE = 0.7
MAX_TOKENS = 128

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
    print(f"  vllm serve {MODEL_NAME} --port 8000 --dtype auto --max-model-len 4096")
    sys.exit(1)
except requests.HTTPError as e:
    print(f"ERROR: vLLM health check failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: Define test prompts — diverse lengths simulate realistic load
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    "What is machine learning in one sentence?",
    "Explain the difference between CPU and GPU for AI workloads.",
    "Write a Python function that reverses a string.",
    "What are the three laws of thermodynamics?",
    "Describe how a transformer model processes text step by step.",
    "Give me 5 tips for writing clean Python code.",
    "What is the capital of France?",
    "Explain gradient descent to a 10-year-old.",
    "Compare REST and GraphQL APIs in 3 bullet points.",
    "Write a haiku about programming.",
    "What causes a segmentation fault in C?",
    "Explain what a KV cache is and why it matters for inference.",
]

# ---------------------------------------------------------------------------
# Step 3: Create OpenAI client pointed at local vLLM server
# ---------------------------------------------------------------------------

# vLLM exposes an OpenAI-compatible API — same client, different base URL
client = OpenAI(
    base_url=f"{VLLM_BASE_URL}/v1",
    api_key="not-needed",  # vLLM doesn't require authentication by default
)


def send_request(prompt_text, request_id):
    """Send a single chat completion request and return timing info."""
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    tokens_generated = response.usage.completion_tokens if response.usage else 0
    return request_id, elapsed_ms, tokens_generated


# ---------------------------------------------------------------------------
# Step 4: Send requests concurrently — this is where vLLM shines
# ---------------------------------------------------------------------------

# A sequential approach would process one prompt at a time (slow).
# ThreadPoolExecutor fires all requests at once; vLLM's continuous batching
# processes them in parallel on the GPU.

NUM_CONCURRENT = len(TEST_PROMPTS)
print(f"Sending {NUM_CONCURRENT} concurrent requests to vLLM...")
print("-" * 60)

results = []
overall_start = time.perf_counter()

with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as executor:
    futures = {
        executor.submit(send_request, prompt, i + 1): i
        for i, prompt in enumerate(TEST_PROMPTS)
    }

    for future in as_completed(futures):
        req_id, latency_ms, tokens = future.result()
        results.append((req_id, latency_ms, tokens))

overall_elapsed = time.perf_counter() - overall_start

# ---------------------------------------------------------------------------
# Step 5: Print individual request latencies
# ---------------------------------------------------------------------------

# Sort by request ID for readable output
results.sort(key=lambda x: x[0])

print(f"\n{'Request':<10} {'Latency':<12} {'Tokens':<8}")
print("-" * 30)
for req_id, latency_ms, tokens in results:
    print(f"  {req_id:<8} {latency_ms:>7.0f} ms   {tokens:>4}")

# ---------------------------------------------------------------------------
# Step 6: Print aggregate throughput — the key production metric
# ---------------------------------------------------------------------------

total_requests = len(results)
throughput = total_requests / overall_elapsed
total_tokens = sum(t for _, _, t in results)
avg_latency = sum(lat for _, lat, _ in results) / total_requests

print("-" * 60)
print(f"\nResults ({NUM_CONCURRENT} concurrent requests):")
print(f"  Total wall-clock time: {overall_elapsed:.2f}s")
print(f"  Throughput:            {throughput:.2f} req/sec")
print(f"  Total tokens:          {total_tokens}")
print(f"  Avg latency/request:   {avg_latency:.0f} ms")
print(f"\nCompare: sequential at ~2s each would take ~{NUM_CONCURRENT * 2}s")
print(f"  vLLM completed all {NUM_CONCURRENT} in {overall_elapsed:.2f}s")
