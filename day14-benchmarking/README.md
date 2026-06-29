# Day 14: Benchmarking Inference — Tokens/sec, Latency, VRAM

## The Concept

When you cook a meal, you don't just say "it took a while." You measure *prep time* (chopping vegetables), *cook time* (pan on heat), and *total time* (door-to-plate). Each measurement tells you something different about where the bottleneck is. Inference benchmarking works exactly the same way — except your "prep" is loading the model into memory, your "cook" is generating tokens one by one, and your "plate" is the final response delivered to the user.

Benchmarking isn't just about measuring speed. It's about asking the *right* questions: How long does the user wait before seeing the first token? How fast do subsequent tokens stream in? How much memory does the model consume? On Day 12 you explored how the KV cache eats memory as sequence length grows. On Day 13 you saw Flash Attention speed up the attention mechanism itself. On Day 4 you measured tokens/sec at different temperatures. Today you'll bring all these measurements together into a single, repeatable benchmarking methodology — a measurement toolkit you'll use for the rest of this series.

This day wraps up Week 2. Every optimization you've studied (quantization, caching, attention tricks) only matters if you can *measure* the improvement. Without numbers, you're guessing. After today, you'll never have to guess again.

## 1. The Five Key Metrics

Inference performance isn't captured by a single number. You need a family of metrics, each illuminating a different aspect of the generation pipeline:

**Time to First Token (TTFT)** — The delay between submitting a prompt and receiving the *first* generated token. This is what the user perceives as "responsiveness." A system generating 100 tokens/sec but taking 3 seconds to start feels sluggish compared to one that starts streaming in 200ms.

**Tokens Per Second (TPS)** — The steady-state generation speed once the first token appears. This determines how fast a response "streams" to the user. It's the rate at which subsequent tokens arrive after the first one.

**Total Latency** — End-to-end wall-clock time from prompt submission to final token. This is TTFT + (number of tokens × per-token time). It's the "door-to-plate" time in our cooking analogy.

**Memory Usage (VRAM/RAM)** — Peak memory consumed during generation. On GPUs this is VRAM; on CPU it's system RAM. This determines which models fit on your hardware and how many concurrent requests you can serve.

**Throughput** — Total tokens generated per second across all concurrent requests. Single-request TPS and multi-request throughput can differ dramatically (as you'll see in Week 3 with vLLM batching).

## 2. Metrics Collection Pipeline

Here's how a benchmarking harness captures these metrics during a single inference call:

```
┌─────────────────────────────────────────────────────────────────┐
│            Metrics Collection Pipeline (Data Flow)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                                │
│  │  Input      │  Prompt text + generation config               │
│  │  Prompt     │  (max_tokens, temperature, etc.)               │
│  └─────┬───────┘                                                │
│        │                                                        │
│        ▼           ⏱ Start timer                                │
│  ┌─────────────┐                                                │
│  │  Tokenize   │  Convert text → token IDs                      │
│  │  + Prefill  │  Process all input tokens at once              │
│  └─────┬───────┘                                                │
│        │                                                        │
│        ▼           ⏱ Record TTFT (first token produced)         │
│  ┌─────────────┐                                                │
│  │  Decode     │  Generate tokens one-by-one (autoregressive)   │
│  │  Loop       │  Each iteration: attention → FFN → sample      │
│  └─────┬───────┘                                                │
│        │                                                        │
│        ▼           ⏱ Stop timer (last token produced)           │
│  ┌─────────────┐                                                │
│  │  Collect    │  Total time, token count, peak memory          │
│  │  Metrics    │  Compute TPS, TTFT, latency                    │
│  └─────┬───────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │  Report     │  Print table / export JSON / log to file       │
│  │  Results    │  Compare across runs, configs, models          │
│  └─────────────┘                                                │
│                                                                 │
│  Memory probe (psutil/torch.cuda) runs throughout ────────────▶ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The two-phase nature of generation is key: **prefill** processes all input tokens in parallel (a single forward pass), then **decode** generates output tokens one at a time (sequential forward passes). TTFT measures prefill time; TPS measures decode speed.

## 3. Prefill vs Decode — Two Distinct Bottlenecks

Understanding why we separate TTFT from TPS requires knowing the two phases of autoregressive generation:

**Prefill phase (compute-bound):**
- Processes all input tokens simultaneously
- One large matrix multiplication
- Bottlenecked by GPU compute (FLOPS)
- Longer prompts = longer prefill
- This is what TTFT measures

**Decode phase (memory-bound):**
- Generates one token per forward pass
- Reads entire model weights for each token
- Bottlenecked by memory bandwidth (GB/s)
- KV cache grows with each step (Day 12)
- This is what TPS measures

```python
# Conceptual breakdown of model.generate() timing:
#
# |--- TTFT ---|---------- Decode (N tokens) ----------|
# |  prefill   |  tok1  |  tok2  |  tok3  | ... | tokN |
# |<--------------------- Total Latency --------------->|
```

Why this matters: different optimizations target different phases. Flash Attention (Day 13) speeds up prefill. Speculative decoding (Day 22) speeds up decode. KV cache (Day 12) trades memory for decode speed. Benchmarking both phases separately tells you which optimization to apply.

## 4. Comparison Table

| Metric | What It Measures | How to Improve | Tool/Method |
|--------|-----------------|----------------|-------------|
| TTFT (Time to First Token) | Responsiveness — how fast streaming starts | Shorter prompts, Flash Attention, tensor parallelism | `time.perf_counter()` around first token |
| Tokens/sec (TPS) | Decode throughput — streaming speed | Quantization, KV cache, speculative decoding | `tokens_generated / decode_time` |
| Total Latency | End-to-end response time | All of the above + fewer output tokens | `time.perf_counter()` start to finish |
| Peak Memory (VRAM) | Hardware utilization — what fits | Quantization (Day 8-11), smaller models, offloading | `torch.cuda.max_memory_allocated()` or `psutil` |
| Throughput (batch) | Concurrent capacity — users served | Continuous batching (Day 16), PagedAttention | Requests completed / wall-clock time |

### When Each Metric Matters Most

| Scenario | Primary Metric | Why |
|----------|---------------|-----|
| Chat application | TTFT | Users expect instant response start |
| Long-form generation | TPS | Total wait time depends on decode speed |
| Batch processing | Throughput | Processing thousands of requests overnight |
| Edge deployment | Peak Memory | Must fit within device RAM/VRAM limits |
| API SLA | Total Latency | p99 latency determines user experience |

## 5. Measurement Best Practices

Accurate benchmarking requires controlling variables. Here are the practices that separate useful measurements from misleading ones:

**Warm up the model** — The first generation after loading is always slower (JIT compilation, CUDA kernel caching, memory allocation). Run 1-2 throwaway generations before measuring.

**Fix the output length** — Use `max_new_tokens` instead of relying on EOS tokens. Variable-length outputs make TPS comparisons meaningless.

**Repeat and average** — Run each configuration 3-5 times and report the mean. Single measurements are noisy due to OS scheduling, thermal throttling, and background processes.

**Report hardware context** — Always include: model name, device (CPU/GPU model), batch size, input length, output length. Without these, numbers are not reproducible.

**Isolate the variable** — Change one parameter at a time. Don't compare temperature=0.1 on GPU to temperature=1.5 on CPU and conclude that temperature affects speed.

```python
import time
import psutil
import torch

# Warm-up run (discard the result)
_ = model.generate(input_ids, max_new_tokens=10)

# Measurement run
torch.cuda.reset_peak_memory_stats()  # GPU only
start = time.perf_counter()
output = model.generate(input_ids, max_new_tokens=100)
elapsed = time.perf_counter() - start

tokens_out = output.shape[1] - input_ids.shape[1]
tps = tokens_out / elapsed
peak_mb = torch.cuda.max_memory_allocated() / 1e6  # GPU
# or: peak_mb = psutil.Process().memory_info().rss / 1e6  # CPU
```

## 6. Common Pitfalls

Benchmarking looks simple but has subtle traps:

- **Measuring tokenization as part of generation** — Tokenizing the prompt is cheap but adds noise. Tokenize *before* starting the timer.
- **Including model download in timing** — The first `from_pretrained()` call downloads the model. Always separate loading from benchmarking.
- **Ignoring warm-up effects** — First run on GPU includes CUDA kernel compilation. Always discard the first measurement.
- **Comparing different output lengths** — 50 tokens at 30 tok/s vs 200 tokens at 28 tok/s are not meaningfully different. Fix `max_new_tokens`.
- **Confusing single-request TPS with throughput** — Your model does 30 tok/s for one request. With batching (Day 16), it might do 200 tok/s total across 8 requests.

## 7. Connecting to Prior Days

Today's benchmarking toolkit ties together several concepts from Week 2:

- **Day 12 (KV Cache)** — You measured memory growth per token. Today you formalize that as "Peak Memory" and measure it alongside speed metrics.
- **Day 13 (Flash Attention)** — You observed a speedup factor. Today's framework gives you the methodology to measure that speedup reproducibly.
- **Day 4 (Sampling Strategies)** — You first measured tokens/sec when varying temperature. Today you learn that TPS is just *one* of five metrics, and that temperature barely affects speed (it's the decode phase that dominates).

In Week 3, you'll use these benchmarks to compare inference engines: vLLM (Day 15), SGLang (Day 17), and production APIs (Day 20-21). The numbers you learn to collect today become the evidence you'll use to choose the right tool for each deployment scenario.

## Setup

### Prerequisites

- Python 3.10+
- Completed Day 4 (familiar with `model.generate()` and tokens/sec)
- Completed Day 12 (understand why memory grows during generation)
- ~4 GB disk space (for SmolLM2-1.7B-Instruct download)
- No GPU required — script measures CPU or GPU automatically

### Install dependencies

```bash
pip install -r requirements.txt
```

The script uses `psutil` for cross-platform memory measurement. On GPU systems, it also uses `torch.cuda` for precise VRAM tracking.

### Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — the same model from Day 4 and Day 12. If you've already cached it from those days, no download is needed.

## What You'll Build

A comprehensive benchmarking script that measures all five key inference metrics (TTFT, TPS, total latency, peak memory, and per-phase breakdown) for SmolLM2-1.7B-Instruct. The script runs multiple trials, handles warm-up automatically, and prints a clean metrics table that you can use as a baseline for comparison in later days.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Inference performance requires at least five metrics: TTFT, TPS, total latency, peak memory, and throughput — no single number tells the full story
- TTFT measures the prefill phase (compute-bound); TPS measures the decode phase (memory-bound) — they have different bottlenecks and different optimization strategies
- Always warm up the model before benchmarking — the first run includes one-time overhead (kernel compilation, memory allocation) that inflates measurements
- Fix `max_new_tokens` and repeat measurements 3-5 times for reproducible results
- The benchmarking methodology from today becomes your measurement toolkit for evaluating every optimization in Weeks 3 and 4
- Report hardware context with every benchmark — numbers without device info are meaningless to others
