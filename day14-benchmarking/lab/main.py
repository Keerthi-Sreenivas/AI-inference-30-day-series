"""
Day 14: Benchmarking Inference — Tokens/sec, Latency, VRAM

A comprehensive benchmarking script that measures the five key inference metrics:
- Time to First Token (TTFT): how fast the model starts responding
- Tokens per Second (TPS): steady-state decode speed
- Total Latency: end-to-end wall-clock time
- Peak Memory: RAM/VRAM consumed during generation
- Per-phase breakdown: prefill vs decode timing

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache 2.0, no login required)
"""

import time
import psutil
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# --- Configuration ---
# Fixed parameters ensure reproducible benchmarks. Never compare runs with
# different output lengths — it makes TPS comparisons meaningless.
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MAX_NEW_TOKENS = 100
NUM_TRIALS = 3
WARMUP_RUNS = 2

# --- Load model and tokenizer ---
# from_pretrained downloads on first run (~3.5 GB). Subsequent runs use cache.
print(f"Loading model: {MODEL_NAME}")
print("(First run downloads ~3.5 GB — subsequent runs use cache)\n")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

# --- Detect hardware ---
# Knowing the device is essential context for any benchmark. Numbers without
# hardware info are meaningless to anyone trying to reproduce your results.
device = model.device
is_cuda = device.type == "cuda"
param_count = sum(p.numel() for p in model.parameters())

print(f"Model loaded: {param_count / 1e9:.2f}B parameters")
print(f"Device: {device}")
if is_cuda:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
print()

# --- Prepare the benchmark prompt ---
# Using the chat template ensures the model receives the prompt in the format
# it was fine-tuned to expect. A moderate-length prompt exercises the prefill
# phase meaningfully without dominating total time.
messages = [
    {"role": "user", "content": "Explain the difference between latency and throughput in computing systems. Include examples."}
]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
input_token_count = input_ids.shape[1]

print(f"Input prompt: {input_token_count} tokens")
print(f"Output target: {MAX_NEW_TOKENS} tokens")
print(f"Trials: {NUM_TRIALS} (averaged)")
print()

# --- Warm-up phase ---
# The first generation on GPU includes CUDA kernel compilation and memory
# allocation overhead that inflates measurements. We run a few throwaway
# generations so subsequent timing reflects actual inference performance.
print("=" * 70)
print("WARM-UP PHASE")
print("=" * 70)
print(f"Running {WARMUP_RUNS} warm-up generation(s) to stabilize timings...")

for i in range(WARMUP_RUNS):
    _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
    print(f"  Warm-up {i + 1}/{WARMUP_RUNS} complete")

# Reset peak memory tracking after warm-up so we only measure the benchmark runs.
if is_cuda:
    torch.cuda.reset_peak_memory_stats()

# Get baseline memory after model load + warm-up
process = psutil.Process(os.getpid())
baseline_memory_mb = process.memory_info().rss / 1e6

print(f"\nBaseline memory after warm-up: {baseline_memory_mb:.0f} MB")
print()

# --- Benchmark function ---
# This function measures a single generation run and returns all key metrics.
# We use a streamer to capture TTFT — the time until the first token is produced.
def benchmark_single_run():
    """Run one generation and return metrics dict."""

    # TextIteratorStreamer lets us detect when the first token is produced.
    # This separates prefill time (TTFT) from decode time (TPS).
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Reset GPU peak memory before each trial
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,  # Greedy for reproducibility — no sampling variance
        "streamer": streamer,
    }

    # Generate in a separate thread so we can iterate the streamer in main thread
    # to detect when the first token arrives.
    start_time = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Wait for the first chunk from the streamer — this marks TTFT.
    ttft = None
    token_chunks = []
    for chunk in streamer:
        if ttft is None:
            ttft = time.perf_counter() - start_time
        token_chunks.append(chunk)

    thread.join()
    end_time = time.perf_counter()

    # Calculate metrics
    total_latency = end_time - start_time
    decode_time = total_latency - ttft  # Time spent on decode phase only
    generated_text = "".join(token_chunks)

    # Count actual tokens generated (may be fewer than MAX_NEW_TOKENS if EOS hit)
    output_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))

    # TPS measures decode speed: tokens produced after the first one, divided by decode time.
    # We subtract 1 because the first token is produced during prefill (measured by TTFT).
    tps = (output_tokens - 1) / decode_time if decode_time > 0 else 0

    # Memory measurement
    if is_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_memory_mb = process.memory_info().rss / 1e6

    return {
        "ttft_ms": ttft * 1000,           # Convert to milliseconds
        "tps": tps,                        # Tokens per second
        "total_latency_s": total_latency,  # Seconds
        "peak_memory_mb": peak_memory_mb,  # Megabytes
        "output_tokens": output_tokens,    # Actual tokens generated
    }


# --- Run benchmark trials ---
# Multiple trials account for variance from OS scheduling, thermal throttling,
# and other background processes. We report the mean of all trials.
print("=" * 70)
print("BENCHMARK RUNS")
print("=" * 70)

results = []

for trial in range(1, NUM_TRIALS + 1):
    print(f"\n--- Trial {trial}/{NUM_TRIALS} ---")
    metrics = benchmark_single_run()
    results.append(metrics)

    print(f"  TTFT:          {metrics['ttft_ms']:.1f} ms")
    print(f"  TPS:           {metrics['tps']:.1f} tokens/sec")
    print(f"  Total Latency: {metrics['total_latency_s']:.2f} s")
    print(f"  Peak Memory:   {metrics['peak_memory_mb']:.0f} MB")
    print(f"  Tokens Out:    {metrics['output_tokens']}")

# --- Compute averages ---
# Mean across trials smooths out single-run noise and gives a more
# reliable estimate of steady-state performance.
avg_ttft = sum(r["ttft_ms"] for r in results) / NUM_TRIALS
avg_tps = sum(r["tps"] for r in results) / NUM_TRIALS
avg_latency = sum(r["total_latency_s"] for r in results) / NUM_TRIALS
avg_memory = sum(r["peak_memory_mb"] for r in results) / NUM_TRIALS
avg_tokens = sum(r["output_tokens"] for r in results) / NUM_TRIALS

# --- Print comprehensive benchmark table ---
# This is the primary output — a single table you can screenshot, paste into
# a report, or use as a baseline for comparison in later days.
print("\n")
print("=" * 70)
print("BENCHMARK RESULTS — Comprehensive Metrics Table")
print("=" * 70)
print()
print("Hardware Context:")
print(f"  Model:            {MODEL_NAME}")
print(f"  Parameters:       {param_count / 1e9:.2f}B")
print(f"  Device:           {device}")
if is_cuda:
    print(f"  GPU:              {torch.cuda.get_device_name(0)}")
print(f"  Input tokens:     {input_token_count}")
print(f"  Output tokens:    {MAX_NEW_TOKENS} (target)")
print(f"  Trials:           {NUM_TRIALS}")
print()
print(f"{'Metric':<30} {'Mean Value':<15} {'Unit':<15}")
print("-" * 60)
print(f"{'Time to First Token (TTFT)':<30} {avg_ttft:<15.1f} {'ms':<15}")
print(f"{'Tokens per Second (TPS)':<30} {avg_tps:<15.1f} {'tokens/sec':<15}")
print(f"{'Total Latency':<30} {avg_latency:<15.2f} {'seconds':<15}")
print(f"{'Peak Memory':<30} {avg_memory:<15.0f} {'MB':<15}")
print(f"{'Output Tokens (actual)':<30} {avg_tokens:<15.0f} {'tokens':<15}")
print("-" * 60)
print()

# --- Phase breakdown ---
# Showing prefill vs decode as percentages helps identify the bottleneck.
# If TTFT dominates, optimize prefill (Flash Attention, shorter prompts).
# If decode dominates, optimize decode (quantization, speculative decoding).
prefill_pct = (avg_ttft / 1000) / avg_latency * 100
decode_pct = 100 - prefill_pct

print("Phase Breakdown:")
print(f"  Prefill (TTFT):   {avg_ttft:.0f} ms ({prefill_pct:.1f}% of total)")
print(f"  Decode:           {(avg_latency - avg_ttft/1000)*1000:.0f} ms ({decode_pct:.1f}% of total)")
print()
print("Interpretation:")
if prefill_pct > 30:
    print("  → Prefill-heavy: Consider Flash Attention (Day 13) or shorter prompts.")
else:
    print("  → Decode-heavy: Consider quantization (Days 8-11) or speculative decoding (Day 22).")
print()
print("=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("TTFT tells you how responsive the model FEELS to a user.")
print("TPS tells you how fast the response STREAMS once it starts.")
print("Peak Memory tells you what FITS on your hardware.")
print("All three matter. Optimizing one often trades off against another.")
print("This metrics table is your measurement toolkit for the rest of the series.")
