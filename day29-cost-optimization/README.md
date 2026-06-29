# Day 29: Cost Optimization — Cloud vs Local, Engine Comparison

## The Concept

**Cost optimization** for inference means maximizing the number of useful tokens you generate per dollar spent. The tradeoff isn't just speed — it's a three-way balance between cost, latency, and quality. A faster engine costs more per hour but processes more requests, potentially making each token cheaper. A quantized model uses less memory (cheaper hardware) but might produce slightly lower quality. Understanding these tradeoffs lets you pick the right setup for your budget and workload.

Think of choosing transportation. A city bus is cheap per passenger-mile but slow and inflexible. A taxi is fast and convenient but expensive. A private car is in between — high upfront cost but low per-trip cost if you drive enough. Inference engines work the same way: a cloud API is the taxi (pay per token, zero setup), running your own vLLM server is the car (hardware cost upfront, but amortized over millions of tokens), and a CPU-only setup is the bus (free hardware, but slow throughput).

On Day 14 you learned to benchmark inference speed with proper methodology — today you'll extend that into cost-per-token analysis. On Days 15 and 17 you compared vLLM and SGLang on throughput and latency — today you'll add the critical dimension: how much each token actually costs in different deployment scenarios.

## 1. The Cost Decision Tree

Choosing an inference strategy depends on your volume, latency requirements, and budget:

```
┌─────────────────────────────────────────────────────────────────┐
│               Inference Cost Decision Tree                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  How many tokens/day do you need?                               │
│       │                                                         │
│       ├── < 100K tokens/day                                     │
│       │       │                                                 │
│       │       ▼                                                 │
│       │   ┌──────────────┐                                      │
│       │   │ Cloud API    │  Pay-per-token is cheapest at low    │
│       │   │ (OpenAI,     │  volume. No hardware, no ops.       │
│       │   │  Anthropic)  │                                      │
│       │   └──────────────┘                                      │
│       │                                                         │
│       ├── 100K – 10M tokens/day                                 │
│       │       │                                                 │
│       │       ├── Latency matters? (< 500ms p99)                │
│       │       │       │                                         │
│       │       │       ▼ YES                                     │
│       │       │   ┌──────────────┐                              │
│       │       │   │ Self-hosted  │  vLLM/SGLang on rented       │
│       │       │   │ GPU (A100)   │  GPU. Full control.          │
│       │       │   └──────────────┘                              │
│       │       │                                                 │
│       │       ▼ NO                                              │
│       │   ┌──────────────┐                                      │
│       │   │ Serverless   │  Pay-per-request cloud               │
│       │   │ (Replicate,  │  endpoints. Auto-scaling.           │
│       │   │  Together)   │                                      │
│       │   └──────────────┘                                      │
│       │                                                         │
│       └── > 10M tokens/day                                      │
│               │                                                 │
│               ▼                                                  │
│           ┌──────────────┐                                      │
│           │ Dedicated    │  Reserved/owned GPUs. Bulk           │
│           │ GPU Cluster  │  pricing. Custom optimization.       │
│           └──────────────┘                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Cost Comparison: Engines and Deployment Options

| Engine / Option | Cost per 1K Tokens | Throughput (tok/s) | Latency (p99) | Best Scenario |
|----------------|:------------------:|:-----------------:|:-------------:|---------------|
| Cloud API (GPT-4o) | $0.015 | Very High | 200ms | Low volume, zero ops |
| Cloud API (Claude Sonnet) | $0.015 | Very High | 250ms | Low volume, quality focus |
| vLLM on A100 (80GB) | $0.001–0.003 | 2000–5000 | 300–800ms | High volume production |
| SGLang on A100 (80GB) | $0.001–0.003 | 2500–6000 | 250–600ms | Multi-turn, prefix caching |
| vLLM on L4 (24GB) | $0.002–0.005 | 500–1500 | 500–1500ms | Medium volume, cost-sensitive |
| CPU (llama.cpp) | $0.0005 | 10–30 | 5–30s | Batch processing, no GPU |
| Quantized (INT4) on T4 | $0.001–0.004 | 300–800 | 800–2000ms | Budget GPU, acceptable quality |

**Key insight:** Cost per token drops dramatically with scale. At 1M tokens/day on an A100, self-hosted is 5–15x cheaper than cloud APIs. Below 100K tokens/day, the operational overhead of self-hosting outweighs the savings.

## 3. Calculating Cost Per Token

The formula for self-hosted cost per 1K tokens:

```
Cost per 1K tokens = (GPU cost per hour) / (tokens per hour) × 1000

Example:
- A100 80GB rental: $2.50/hour
- vLLM throughput: 3000 tokens/sec = 10.8M tokens/hour
- Cost per 1K tokens: $2.50 / 10,800 × 1 = $0.00023

Compare to GPT-4o at $0.015 per 1K output tokens:
  Savings: $0.015 / $0.00023 = 65x cheaper (at full utilization)
```

**The catch:** you pay for the GPU whether it's idle or not. If your workload only uses 10% of capacity, your effective cost per token is 10x higher. Utilization is the key variable.

## 4. Optimization Strategies

| Strategy | Savings | Complexity | Trade-off |
|----------|:-------:|:----------:|-----------|
| Quantization (INT4) | 2–4x less VRAM | Low | ~1-3% quality loss |
| Batching (continuous) | 3–8x throughput | Medium | Slightly higher per-request latency |
| Request queuing | 2x+ utilization | Medium | Adds queue wait time |
| Spot/preemptible GPUs | 60–70% cost | High | Can be interrupted |
| Model distillation | Use smaller model | High | Quality depends on task |
| Prefix caching (SGLang) | 2–3x for shared prefixes | Low | Only helps repeated prefixes |

## 5. Break-Even Analysis

```python
# When does self-hosting become cheaper than cloud APIs?
cloud_cost_per_1k = 0.015  # $/1K tokens (e.g., GPT-4o output)
gpu_hourly_cost = 2.50      # $/hour (A100 80GB spot)
throughput_tok_per_sec = 3000

tokens_per_hour = throughput_tok_per_sec * 3600
self_hosted_per_1k = gpu_hourly_cost / tokens_per_hour * 1000

# Break-even: at what daily volume is self-hosting cheaper?
# Self-hosted daily cost = 24 * gpu_hourly_cost = $60/day (always on)
# Cloud daily cost at N tokens = N / 1000 * cloud_cost_per_1k
# Break-even: 60 / (cloud_cost_per_1k / 1000) = 4,000,000 tokens/day

print(f"Break-even point: ~4M tokens/day")
print(f"Below 4M/day: cloud API is cheaper (no idle GPU cost)")
print(f"Above 4M/day: self-hosted saves money")
```

## Setup

### Prerequisites

- Python 3.10+
- vLLM running on port 8000 (for live benchmarking)
- Optionally: SGLang running on port 8001 (for comparison)
- GPU recommended for meaningful throughput measurements

### Start the inference engines

```bash
# Start vLLM (primary engine)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Optionally start SGLang for comparison
# python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --port 8001
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## What You'll Build

A cost analysis script that benchmarks inference throughput on available engines, calculates the cost per 1,000 tokens for different deployment scenarios (cloud API, self-hosted GPU, CPU-only), and prints a comparison table showing which option is cheapest at different usage levels.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Cost per token depends on three variables: hardware cost per hour, throughput (tokens/sec), and utilization rate
- Cloud APIs are cheapest below ~4M tokens/day (no idle GPU cost); self-hosted wins above that threshold
- Quantization (INT4) cuts GPU memory needs in half, letting you use cheaper hardware or serve more concurrent requests
- SGLang's prefix caching can reduce effective cost 2–3x for workloads with repeated system prompts or few-shot examples
- Utilization is the hidden cost killer — a GPU running at 10% capacity costs 10x more per token than one at 100%
- Always benchmark YOUR workload at YOUR scale — generic cost tables are starting points, not answers
