# Day 18: Multi-GPU Inference — Tensor Parallelism and Ray

## The Concept

When a model is too large for a single GPU — or you need more throughput than one card can deliver — you split the work across multiple GPUs. Think of it like an **assembly line**: instead of one worker building an entire car, you assign each worker a section. Each station (GPU) handles its part and passes the result downstream. The model finishes faster because the bottleneck is shared.

On Day 15 you launched vLLM on a single GPU and measured throughput. On Day 5 you learned that a 7B model in FP16 needs ~14 GB of VRAM. What happens when you want to serve a 70B model that needs 140 GB? No single consumer GPU has that much memory. The answer: **tensor parallelism** — slice the model's weight matrices across multiple GPUs so each card holds a fraction of the parameters.

Multi-GPU inference isn't just about fitting big models. Even with a model that fits on one GPU, distributing it can reduce per-request latency because matrix multiplications happen in parallel across cards.

## How Tensor Parallelism Works

Tensor parallelism splits individual weight matrices along one dimension, distributes the partial computations, and then synchronizes with an all-reduce operation.

```
┌──────────────────────────────────────────────────────────────────┐
│         Tensor Parallelism: 8B Model on 2 GPUs                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Tokens                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────┐             │
│  │           Embedding Layer (replicated)           │             │
│  └─────────────────────────────────────────────────┘             │
│       │                                                          │
│       ├──────────────────────┬───────────────────────            │
│       ▼                      ▼                                   │
│  ┌──────────────┐      ┌──────────────┐                         │
│  │   GPU 0      │      │   GPU 1      │                         │
│  │              │      │              │                         │
│  │  Attention   │      │  Attention   │                         │
│  │  (heads 0-15)│      │  (heads 16-31)│                        │
│  │              │      │              │                         │
│  │  FFN (cols   │      │  FFN (cols   │                         │
│  │   0 - 2047)  │      │   2048-4095) │                         │
│  └──────┬───────┘      └──────┬───────┘                         │
│         │                      │                                 │
│         └──────────┬───────────┘                                 │
│                    ▼                                              │
│           All-Reduce (sum)                                       │
│                    │                                              │
│                    ▼                                              │
│            Next Layer ...                                         │
│                                                                  │
│  Repeat for each transformer layer.                              │
│  Each GPU holds ½ the attention heads and ½ the FFN columns.     │
└──────────────────────────────────────────────────────────────────┘
```

The key insight: attention heads are embarrassingly parallel (each head is independent), so splitting them across GPUs is natural. For the feed-forward network, you split columns of the first linear layer and rows of the second, then all-reduce to combine.

## Parallelism Strategies Compared

| Strategy | How It Splits | Speedup | VRAM per GPU | Complexity | Best For |
|----------|---------------|:-------:|:------------:|:----------:|----------|
| Tensor Parallel (TP) | Weight matrices sliced across GPUs | Near-linear for large models | Model_size / N_GPUs | Medium | Single-node, NVLink |
| Pipeline Parallel (PP) | Layers assigned to different GPUs | Moderate (bubble overhead) | Layers / N_GPUs | Low | Multi-node, slow interconnect |
| Data Parallel (DP) | Same model replicated, batch split | Linear with batch size | Full model per GPU | Low | Training (less common for inference) |
| TP + PP (hybrid) | Layers split then tensors split | Best at scale | Minimal per GPU | High | 100B+ models across nodes |

**When to use which:**
- 2–8 GPUs on one machine with NVLink → Tensor Parallelism
- Multiple machines → Pipeline Parallelism or hybrid
- Throughput focus with small model → Data Parallelism (replicas)

## How vLLM and Ray Enable Multi-GPU

vLLM uses **Ray** under the hood as its distributed execution framework. When you set `tensor_parallel_size=N`, vLLM:

1. Spawns N Ray workers (one per GPU)
2. Partitions model weights into N shards
3. Each worker loads its shard into its GPU
4. During inference, workers execute their slice and communicate via NCCL all-reduce

You don't need to manage Ray directly — vLLM handles the orchestration. But understanding Ray helps you debug and tune.

```bash
# Start vLLM with tensor parallelism across 2 GPUs
vllm serve meta-llama/Llama-3.1-8B \
    --tensor-parallel-size 2 \
    --port 8000
```

## Measuring Scaling Efficiency

Scaling efficiency tells you how much of the theoretical speedup you actually get:

```
Efficiency = (Throughput_N_GPUs / (Throughput_1_GPU × N)) × 100%
```

Perfect scaling is 100%. In practice:
- **90-95%** — excellent (NVLink, large batches)
- **70-85%** — typical (PCIe, smaller models)
- **<60%** — communication bottleneck (slow interconnect, tiny batch)

The bottleneck is usually the all-reduce step. NVLink (600 GB/s) keeps this fast; PCIe (64 GB/s) adds significant overhead for small matrix sizes.

## Setup

### Prerequisites

- Completed Day 15 (vLLM basics)
- Familiar with model sizes from Day 5
- **Multi-GPU requirement:** 2+ NVIDIA GPUs (ideally with NVLink)
- CUDA 12.0+ installed

### Hardware Note

This lab requires multiple GPUs. If you only have one GPU:
- You can still read through the code and understand the concepts
- The lab script includes a **single-GPU fallback** that runs vLLM with `tensor_parallel_size=1` and reports baseline throughput
- Cloud options: Lambda Labs, RunPod, or vast.ai offer multi-GPU instances by the hour

### Installation

```bash
pip install -r requirements.txt
```

### Start the vLLM Server

```bash
# For 2 GPUs:
vllm serve meta-llama/Llama-3.1-8B \
    --tensor-parallel-size 2 \
    --port 8000

# For single GPU fallback:
vllm serve meta-llama/Llama-3.1-8B \
    --tensor-parallel-size 1 \
    --port 8000
```

### License Notice

> **Llama 3.1 Community License:** `meta-llama/Llama-3.1-8B` requires you to accept Meta's license agreement on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B) and set `HF_TOKEN` in your environment. If you prefer an ungated model, substitute `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0) with the same commands.

## What You'll Build

A benchmarking script that sends concurrent requests to a vLLM server, measures throughput in tokens per second, and calculates scaling efficiency by comparing multi-GPU performance against a single-GPU baseline. The script reports how close you get to theoretical linear scaling.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Tensor parallelism splits weight matrices across GPUs so each card holds a fraction of the model
- vLLM uses Ray under the hood — set `tensor_parallel_size=N` and it handles the rest
- Scaling efficiency depends on interconnect speed: NVLink >> PCIe
- Multi-GPU is essential for models that don't fit on a single card (70B+ at FP16)
- Even models that fit on one GPU benefit from tensor parallelism via reduced latency
- Always measure actual scaling efficiency — theoretical linear speedup is rarely achieved
