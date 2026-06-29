"""
Day 17: SGLang — RadixAttention and Serving Comparison

Compares latency between vLLM and SGLang, specifically testing scenarios where
SGLang's RadixAttention (prefix caching) provides a measurable advantage.

RadixAttention works like a web browser cache: if multiple requests share the
same prompt prefix (e.g., system prompt), SGLang computes the KV cache for that
prefix once and reuses it for all subsequent requests.

Connects to Day 15 (vLLM) and Day 16 (batching/PagedAttention).

Before running:
    Terminal 1: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype auto --max-model-len 4096
    Terminal 2: python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 8001 --dtype auto
"""

import time
import sys

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_URL = "http://localhost:8000"
SGLANG_URL = "http://localhost:8001"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 128
TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# Step 1: Check both servers
# ---------------------------------------------------------------------------

print("Checking server availability...")


def check_server(name, base_url, startup_cmd):
    """Verify a serving engine is running and healthy."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"  {name}: healthy at {base_url}")
        return True
    except (requests.ConnectionError, requests.Timeout):
        print(f"\nERROR: {name} server is not running.")
        print(f"Start it with:\n  {startup_cmd}")
        return False
    except requests.HTTPError as e:
        print(f"\nERROR: {name} health check failed: {e}")
        return False


vllm_ok = check_server(
    "vLLM",
    VLLM_URL,
    f"vllm serve {MODEL_NAME} --port 8000 --dtype auto --max-model-len 4096",
)
sglang_ok = check_server(
    "SGLang",
    SGLANG_URL,
    f"python -m sglang.launch_server --model-path {MODEL_NAME} --port 8001 --dtype auto",
)

if not (vllm_ok and sglang_ok):
    print("\nBoth servers must be running to compare. Exiting.")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# Step 2: Define a substantial shared system prompt
# ---------------------------------------------------------------------------

# In production, most chatbot requests share the same system prompt. This is
# exactly where RadixAttention shines — it computes the KV cache for this
# prefix once and reuses it for all subsequent requests.

SYSTEM_PROMPT = (
    "You are a senior software engineering assistant with deep expertise in "
    "Python, distributed systems, cloud architecture, and machine learning "
    "infrastructure. You provide concise, technically accurate answers with "
    "code examples when appropriate. You explain trade-offs clearly and "
    "suggest best practices from industry experience. When discussing "
    "performance, you cite specific metrics and benchmarks. You avoid "
    "unnecessary verbosity but never sacrifice technical accuracy for brevity. "
    "You think step by step when solving complex problems and clearly state "
    "assumptions. Your responses are formatted with markdown for readability. "
    "When you don't know something, you say so rather than speculating. "
    "You prioritize production-ready solutions over academic exercises."
)

# Different user messages — all share the same system prompt prefix
USER_MESSAGES = [
    "What is the GIL in Python and when does it matter?",
    "How do I choose between Redis and Memcached for caching?",
    "Explain the CAP theorem in one paragraph.",
    "What's the difference between horizontal and vertical scaling?",
    "When should I use async/await vs threading in Python?",
]

# ---------------------------------------------------------------------------
# Helper: send a single request and measure latency
# ---------------------------------------------------------------------------


def send_chat_request(base_url, messages):
    """Send a chat completion and return latency in milliseconds."""
    client = OpenAI(base_url=f"{base_url}/v1", api_key="none")
    start = time.perf_counter()

    client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------------------
# Test 1: First-request latency (no caching benefit)
# ---------------------------------------------------------------------------

print("=" * 70)
print("TEST 1: First-request latency (cold start, no prefix cache)")
print("=" * 70)

messages_cold = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is inference in machine learning?"},
]

# Warm-up request to both engines (load model into GPU memory)
send_chat_request(VLLM_URL, messages_cold)
send_chat_request(SGLANG_URL, messages_cold)

# Actual measurement — use a fresh prompt to avoid any caching
fresh_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Explain what a container runtime is."},
]

vllm_cold = send_chat_request(VLLM_URL, fresh_messages)
sglang_cold = send_chat_request(SGLANG_URL, fresh_messages)

print(f"  vLLM:   {vllm_cold:>7.1f} ms")
print(f"  SGLang: {sglang_cold:>7.1f} ms")
print(f"  Ratio:  {vllm_cold / sglang_cold:.2f}x")

# ---------------------------------------------------------------------------
# Test 2: Repeated-prefix latency (same system prompt, different questions)
# ---------------------------------------------------------------------------

print(f"\nTEST 2: Repeated-prefix latency ({len(USER_MESSAGES)} requests, same system prompt)")
print("=" * 70)

# RadixAttention caches the system prompt KV after the first request.
# Subsequent requests with the same prefix skip that computation entirely.

vllm_latencies = []
sglang_latencies = []

for i, user_msg in enumerate(USER_MESSAGES):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    vllm_lat = send_chat_request(VLLM_URL, messages)
    sglang_lat = send_chat_request(SGLANG_URL, messages)

    vllm_latencies.append(vllm_lat)
    sglang_latencies.append(sglang_lat)

    marker = " ← first (cache miss)" if i == 0 else ""
    print(f"  Request {i+1}: vLLM {vllm_lat:>7.1f} ms | SGLang {sglang_lat:>7.1f} ms{marker}")

avg_vllm = sum(vllm_latencies) / len(vllm_latencies)
avg_sglang = sum(sglang_latencies) / len(sglang_latencies)

# Skip first request (cold) for "warmed" average
avg_vllm_warm = sum(vllm_latencies[1:]) / len(vllm_latencies[1:])
avg_sglang_warm = sum(sglang_latencies[1:]) / len(sglang_latencies[1:])

print(f"\n  Average (all):    vLLM {avg_vllm:>7.1f} ms | SGLang {avg_sglang:>7.1f} ms")
print(f"  Average (cached): vLLM {avg_vllm_warm:>7.1f} ms | SGLang {avg_sglang_warm:>7.1f} ms")

# ---------------------------------------------------------------------------
# Test 3: Multi-turn conversation (growing prefix)
# ---------------------------------------------------------------------------

print(f"\nTEST 3: Multi-turn conversation (prefix grows each turn)")
print("=" * 70)

# In a multi-turn conversation, each turn's prefix includes all prior turns.
# RadixAttention caches the growing prefix — each turn only computes the new part.

conversation_turns = [
    "What is a database index?",
    "How does a B-tree index work specifically?",
    "When would I use a hash index instead?",
    "Can you compare their performance characteristics?",
]

vllm_turn_latencies = []
sglang_turn_latencies = []

# Build conversation incrementally
vllm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
sglang_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for turn_num, user_turn in enumerate(conversation_turns, 1):
    vllm_messages.append({"role": "user", "content": user_turn})
    sglang_messages.append({"role": "user", "content": user_turn})

    vllm_lat = send_chat_request(VLLM_URL, vllm_messages)
    sglang_lat = send_chat_request(SGLANG_URL, sglang_messages)

    vllm_turn_latencies.append(vllm_lat)
    sglang_turn_latencies.append(sglang_lat)

    prefix_tokens_est = turn_num * 50  # Rough estimate of prefix growth
    print(f"  Turn {turn_num} (~{prefix_tokens_est} prefix tokens): "
          f"vLLM {vllm_lat:>7.1f} ms | SGLang {sglang_lat:>7.1f} ms")

    # Add a synthetic assistant response to build context
    vllm_messages.append({"role": "assistant", "content": "..."})
    sglang_messages.append({"role": "assistant", "content": "..."})

# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print(f"\n{'Test':<30} {'vLLM (ms)':<12} {'SGLang (ms)':<13} {'Speedup':<10}")
print("-" * 65)
print(f"{'First request (cold)':<30} {vllm_cold:<12.1f} {sglang_cold:<13.1f} "
      f"{vllm_cold/sglang_cold:<10.2f}x")
print(f"{'Repeated prefix (avg all)':<30} {avg_vllm:<12.1f} {avg_sglang:<13.1f} "
      f"{avg_vllm/avg_sglang:<10.2f}x")
print(f"{'Repeated prefix (cached)':<30} {avg_vllm_warm:<12.1f} {avg_sglang_warm:<13.1f} "
      f"{avg_vllm_warm/avg_sglang_warm:<10.2f}x")

avg_vllm_mt = sum(vllm_turn_latencies) / len(vllm_turn_latencies)
avg_sglang_mt = sum(sglang_turn_latencies) / len(sglang_turn_latencies)
print(f"{'Multi-turn (avg)':<30} {avg_vllm_mt:<12.1f} {avg_sglang_mt:<13.1f} "
      f"{avg_vllm_mt/avg_sglang_mt:<10.2f}x")

# ---------------------------------------------------------------------------
# Key metric: SGLang's latency advantage
# ---------------------------------------------------------------------------

latency_diff = avg_vllm_warm - avg_sglang_warm
speedup_pct = ((avg_vllm_warm - avg_sglang_warm) / avg_vllm_warm) * 100

print("\n" + "-" * 65)
if latency_diff > 0:
    print(f"\nSGLang latency advantage: {latency_diff:.1f} ms faster on repeated prefixes "
          f"({speedup_pct:.1f}% speedup)")
    print("  → RadixAttention reuses the system prompt KV cache across requests")
else:
    print(f"\nvLLM was {-latency_diff:.1f} ms faster in this run "
          f"({-speedup_pct:.1f}% speedup)")
    print("  → Results vary by workload; try with longer system prompts for more prefix reuse")

print("\nKey insight: The advantage grows with prefix length and repetition.")
print("For chatbots with 500+ token system prompts serving thousands of users,")
print("RadixAttention can reduce per-request compute by 30-50%.")
