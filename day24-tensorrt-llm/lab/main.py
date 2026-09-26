"""
Day 24: TensorRT-LLM, Triton & NVIDIA NIMs

Benchmarks inference throughput using TensorRT-LLM/NIM or vLLM endpoint.
Measures tokens/sec, requests/sec, and calculates throughput gain (%) vs
a PyTorch baseline estimate.

Endpoints:
- Port 8000: TensorRT-LLM / NVIDIA NIM (preferred)
- Port 8001: vLLM baseline (fallback)
"""

import time
import requests
import statistics

# --- Configuration ---
# TensorRT-LLM (via NVIDIA NIM) typically serves on port 8000.
# vLLM baseline serves on port 8001 for comparison.
NIM_URL = "http://localhost:8000"
VLLM_URL = "http://localhost:8001"
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Published PyTorch baseline for 8B model on single A100: ~30-40 tokens/sec.
# This serves as our reference when only one endpoint is available.
PYTORCH_BASELINE_TPS = 35.0

# --- Detect available endpoint ---
# Check which inference server is available. TensorRT-LLM / NIM gives maximum
# throughput but requires NVIDIA GPU + Docker. vLLM is the accessible fallback.
print("=" * 70)
print("ENDPOINT DETECTION")
print("=" * 70)

active_url = None
active_engine = None

try:
    resp = requests.get(f"{NIM_URL}/health", timeout=5)
    resp.raise_for_status()
    active_url = NIM_URL
    active_engine = "TensorRT-LLM / NVIDIA NIM"
    print(f"  TensorRT-LLM / NIM endpoint detected at {NIM_URL}")
except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
    print(f"  No TensorRT-LLM / NIM endpoint at {NIM_URL}")

try:
    resp = requests.get(f"{VLLM_URL}/health", timeout=5)
    resp.raise_for_status()
    if active_url is None:
        active_url = VLLM_URL
        active_engine = "vLLM"
    print(f"  vLLM endpoint detected at {VLLM_URL}")
except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
    print(f"  No vLLM endpoint at {VLLM_URL}")

if active_url is None:
    print("\nERROR: No inference endpoint available.")
    print("\nOption A — Start NVIDIA NIM (requires NVIDIA GPU + Docker):")
    print("  docker run --gpus all -p 8000:8000 \\")
    print("      nvcr.io/nim/meta/llama-3.1-8b-instruct:latest")
    print("\nOption B — Start vLLM (requires NVIDIA GPU):")
    print("  vllm serve meta-llama/Llama-3.1-8B --port 8001")
    raise SystemExit(1)

print(f"\n  Active engine: {active_engine}")
print(f"  Endpoint URL: {active_url}")
print()

# --- Detect GPU hardware ---
# Print GPU info to contextualize performance numbers. TensorRT-LLM performance
# varies significantly by GPU architecture (Ampere vs Hopper vs Blackwell).
print("=" * 70)
print("HARDWARE DETECTION")
print("=" * 70)

gpu_name = "Unknown"
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {gpu_vram:.1f} GB")
        print(f"  CUDA: {cuda_version}")
    else:
        print("  No CUDA GPU detected (running on CPU)")
        print("  Note: TensorRT-LLM requires NVIDIA GPU for meaningful benchmarks")
except ImportError:
    print("  PyTorch not available for GPU detection")

print()

# --- Define throughput test prompts ---
# Five prompts of similar complexity simulate realistic concurrent workload.
# Varied topics test whether the engine handles different token distributions equally.
test_prompts = [
    "Explain the concept of supply and demand in economics in a brief paragraph.",
    "What are the three laws of thermodynamics? Summarize each in one sentence.",
    "Describe how photosynthesis works in plants using simple language.",
    "Write a brief explanation of how WiFi signals transmit data wirelessly.",
    "Explain the water cycle and its four main stages in a short paragraph.",
]

# --- Measure single-request latency ---
# Sequential requests measure per-request performance without batching effects.
# This shows the raw inference speed of the engine for individual users.
print("=" * 70)
print(f"SINGLE-REQUEST LATENCY ({active_engine})")
print("=" * 70)

single_results = []
for i, prompt in enumerate(test_prompts):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7,
    }

    start = time.perf_counter()
    response = requests.post(f"{active_url}/v1/chat/completions", json=payload, timeout=60)
    elapsed = time.perf_counter() - start

    response.raise_for_status()
    data = response.json()
    tokens_generated = data["usage"]["completion_tokens"]
    tokens_per_sec = tokens_generated / elapsed

    single_results.append({
        "elapsed": elapsed,
        "tokens": tokens_generated,
        "tokens_per_sec": tokens_per_sec,
    })

    print(f"  Prompt {i+1}: {tokens_generated:>3} tokens in {elapsed:.2f}s = {tokens_per_sec:.1f} tok/s")

median_tps = statistics.median([r["tokens_per_sec"] for r in single_results])
print(f"\n  Median single-request throughput: {median_tps:.1f} tokens/sec")
print()

# --- Measure batch throughput ---
# Send all prompts in rapid succession to simulate concurrent load. This measures
# how well the engine handles back-to-back requests (batching efficiency).
print("=" * 70)
print(f"BATCH THROUGHPUT ({active_engine}, 5 sequential requests)")
print("=" * 70)

batch_start = time.perf_counter()
batch_tokens = 0

for prompt in test_prompts:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7,
    }
    response = requests.post(f"{active_url}/v1/chat/completions", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    batch_tokens += data["usage"]["completion_tokens"]

batch_elapsed = time.perf_counter() - batch_start
batch_tps = batch_tokens / batch_elapsed
batch_rps = len(test_prompts) / batch_elapsed

print(f"  Total tokens generated: {batch_tokens}")
print(f"  Total time: {batch_elapsed:.2f}s")
print(f"  Effective throughput: {batch_tps:.1f} tokens/sec")
print(f"  Requests/sec: {batch_rps:.2f}")
print()

# --- Calculate throughput gain ---
# Compare measured throughput against the PyTorch baseline estimate.
# TensorRT-LLM typically achieves 3-5× improvement over vanilla PyTorch.
# vLLM typically achieves 2-3× improvement over vanilla PyTorch.
throughput_gain = ((median_tps - PYTORCH_BASELINE_TPS) / PYTORCH_BASELINE_TPS) * 100

print("=" * 70)
print("THROUGHPUT GAIN vs PyTorch Baseline")
print("=" * 70)
print(f"  PyTorch baseline (estimated): {PYTORCH_BASELINE_TPS:.0f} tokens/sec")
print(f"  {active_engine} (measured): {median_tps:.1f} tokens/sec")
print(f"  Throughput gain: {throughput_gain:+.1f}%")
print()

if throughput_gain > 200:
    print("  Excellent! 3×+ speedup indicates TensorRT-LLM kernel optimizations")
    print("  (layer fusion, quantization-aware kernels, GPU-specific tuning).")
elif throughput_gain > 100:
    print("  Good! 2×+ speedup indicates effective batching and kernel optimization")
    print("  typical of vLLM or partially optimized TensorRT-LLM.")
elif throughput_gain > 0:
    print("  Moderate gain. Consider checking GPU utilization and batch size settings.")
else:
    print("  Below baseline — possible issues: cold start, small batch, or CPU fallback.")

print()

# --- Final summary ---
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Metric':<30} {'Value':<25}")
print("-" * 55)
print(f"{'Engine':<30} {active_engine:<25}")
print(f"{'GPU':<30} {gpu_name:<25}")
print(f"{'Median tokens/sec':<30} {median_tps:<25.1f}")
print(f"{'Batch requests/sec':<30} {batch_rps:<25.2f}")
print(f"{'Throughput gain vs PyTorch':<30} {throughput_gain:+.1f}%{'':20}")
print("-" * 55)
print()
print(f"Throughput Gain: {throughput_gain:+.1f}%")
print()
print("Key insight: TensorRT-LLM achieves peak throughput by compiling models into")
print("GPU-specific execution plans with kernel auto-tuning and layer fusion. The")
print("trade-off is portability — compiled engines only work on the exact GPU")
print("architecture they were built for. ONNX Runtime (Day 23) is the portable")
print("alternative when you need cross-platform deployment.")
