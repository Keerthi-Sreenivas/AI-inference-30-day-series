"""
Day 23: ONNX Runtime — Cross-Platform Optimization

Exports SmolLM2-1.7B-Instruct to ONNX format and benchmarks text generation
speed against standard PyTorch inference. Prints speedup factor (×) as the
key metric.

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache 2.0, no login required)
"""

import time
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# --- Configuration ---
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MAX_NEW_TOKENS = 100
NUM_RUNS = 3  # Number of runs per prompt to get stable measurements

# --- Load tokenizer (shared between both paths) ---
# The tokenizer is the same regardless of runtime — only the model execution differs.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded: vocab size = {tokenizer.vocab_size:,}\n")

# --- Load PyTorch model ---
# Standard Transformers loading — this is our baseline for comparison.
# The model runs in eager mode with Python dispatch overhead.
print("=" * 70)
print("Loading PyTorch model (baseline)")
print("=" * 70)

try:
    pytorch_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

param_count = sum(p.numel() for p in pytorch_model.parameters())
print(f"PyTorch model loaded: {param_count / 1e9:.2f}B parameters")
print(f"Device: {pytorch_model.device}\n")

# --- Load ONNX model ---
# The optimum library handles PyTorch → ONNX export automatically with export=True.
# This traces the model into a static computation graph that ONNX Runtime optimizes.
# First run exports the model (takes 1-3 minutes), subsequent runs use cached export.
print("=" * 70)
print("Loading ONNX model (with export)")
print("=" * 70)
print("Exporting to ONNX format (first run takes 1-3 minutes)...")

onnx_model = ORTModelForCausalLM.from_pretrained(
    MODEL_NAME,
    export=True,
)
print("ONNX export complete!")
print("ONNX Runtime applies graph optimizations: operator fusion, constant folding,")
print("memory planning — all happening in C++ with no Python GIL overhead.\n")

# --- Define benchmark prompts ---
# Three prompts of varying input length test how both runtimes handle different
# workloads. Longer inputs mean more time in the prefill phase before generation.
benchmark_prompts = [
    {
        "name": "Short input",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    },
    {
        "name": "Medium input",
        "messages": [{"role": "user", "content": "Explain how a car engine works in simple terms. Cover the four-stroke cycle."}],
    },
    {
        "name": "Long input",
        "messages": [{"role": "user", "content": "You are a computer science professor. Explain the difference between stack and heap memory allocation, including when each is used, their performance characteristics, and common pitfalls developers encounter with each approach. Give concrete examples in Python."}],
    },
]

# --- Helper: benchmark a single model ---
# Runs generation NUM_RUNS times and returns the median tokens/sec.
# Median is more stable than mean — it ignores warmup outliers.
def benchmark_generation(model, tokenizer_ref, messages, num_runs=NUM_RUNS):
    """Generate tokens and measure speed. Returns median tokens/sec."""
    prompt = tokenizer_ref.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_ref(prompt, return_tensors="pt")

    # Move inputs to model device for PyTorch model
    if hasattr(model, "device") and str(model.device) != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy for reproducible benchmarks
        )
        elapsed = time.perf_counter() - start

        # Count only newly generated tokens (exclude input)
        new_token_count = output_ids.shape[1] - inputs["input_ids"].shape[1]
        tokens_per_sec = new_token_count / elapsed
        timings.append(tokens_per_sec)

    return statistics.median(timings)


# --- Benchmark PyTorch ---
# Standard eager-mode execution with Python dispatch overhead.
print("=" * 70)
print("BENCHMARKING: PyTorch (eager mode)")
print("=" * 70)

pytorch_results = []
for prompt_info in benchmark_prompts:
    tps = benchmark_generation(pytorch_model, tokenizer, prompt_info["messages"])
    pytorch_results.append(tps)
    print(f"  {prompt_info['name']:<15} | {tps:.1f} tokens/sec (median of {NUM_RUNS} runs)")

print()

# --- Benchmark ONNX Runtime ---
# Optimized static graph execution in C++. No Python overhead per operation.
# Graph optimizations (fusion, constant folding) applied at load time.
print("=" * 70)
print("BENCHMARKING: ONNX Runtime (optimized graph)")
print("=" * 70)

onnx_results = []
for prompt_info in benchmark_prompts:
    tps = benchmark_generation(onnx_model, tokenizer, prompt_info["messages"])
    onnx_results.append(tps)
    print(f"  {prompt_info['name']:<15} | {tps:.1f} tokens/sec (median of {NUM_RUNS} runs)")

print()

# --- Calculate and display speedup ---
# Speedup = ONNX tokens/sec ÷ PyTorch tokens/sec.
# Values > 1.0 mean ONNX is faster. The improvement comes from graph
# optimizations that eliminate Python overhead and fuse operations.
print("=" * 70)
print("SUMMARY: ONNX Runtime Speedup vs PyTorch")
print("=" * 70)
print(f"{'Prompt':<15} {'PyTorch (tok/s)':<18} {'ONNX (tok/s)':<16} {'Speedup':<10}")
print("-" * 59)

speedups = []
for i, prompt_info in enumerate(benchmark_prompts):
    pytorch_tps = pytorch_results[i]
    onnx_tps = onnx_results[i]
    speedup = onnx_tps / pytorch_tps if pytorch_tps > 0 else 0
    speedups.append(speedup)
    print(f"{prompt_info['name']:<15} {pytorch_tps:<18.1f} {onnx_tps:<16.1f} {speedup:<10.2f}×")

avg_speedup = statistics.mean(speedups)
print("-" * 59)
print(f"{'Average':<15} {'':18} {'':16} {avg_speedup:.2f}×")

print()
print(f"Average Speedup: {avg_speedup:.2f}×")
print()
print("Key insight: ONNX Runtime eliminates Python dispatch overhead and applies")
print("graph-level optimizations (operator fusion, constant folding, memory planning)")
print("that PyTorch's eager mode cannot perform. The speedup is most noticeable on")
print("CPU — on GPU, the gap narrows because compute already dominates over dispatch.")
print()
print("Next step: For even more speed on NVIDIA GPUs, see Day 24 (TensorRT-LLM)")
print("which applies hardware-specific kernel optimization beyond what ONNX provides.")
