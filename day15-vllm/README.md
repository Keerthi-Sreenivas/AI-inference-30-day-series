# Day 15: vLLM — Production Inference Serving

## The Concept

Imagine a restaurant kitchen. A home cook makes one dish at a time — take an order, cook it, serve it, then take the next order. That's fine for a family dinner, but it collapses under the lunch rush. A professional kitchen works differently: multiple orders fire simultaneously, the chef orchestrates shared resources (ovens, burners, prep stations), and dishes leave the pass in a steady stream. vLLM is that professional kitchen for language model inference.

On Day 1 you ran Ollama — a single-user inference server perfect for local experiments. On Day 4 you explored sampling parameters that control *how* a model generates text. Today you'll step into production territory. vLLM is a high-throughput serving engine that handles many concurrent requests, schedules GPU memory intelligently, and delivers tokens faster than naive `model.generate()` loops ever could.

vLLM matters because real-world applications don't serve one user at a time. A chatbot with 100 concurrent users, a batch-processing pipeline indexing thousands of documents, or a coding assistant serving a team — all of these need an engine that maximizes GPU utilization while keeping latency low. vLLM does this through PagedAttention (which we'll cover in depth tomorrow on Day 16) and continuous batching, two techniques that fundamentally change how GPU memory is shared across requests.

## How vLLM Works

vLLM replaces the standard Hugging Face `model.generate()` loop with a dedicated inference engine. Instead of processing one prompt at a time, it:

1. Accepts requests through an OpenAI-compatible HTTP API
2. Schedules requests into batches dynamically
3. Allocates GPU memory using PagedAttention (non-contiguous KV cache)
4. Streams tokens back to each client independently

```
┌──────────────────────────────────────────────────────────────────┐
│                    vLLM System Architecture                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐     ┌──────────────────────────────────────────┐  │
│  │ Client 1 │────▶│                                          │  │
│  └──────────┘     │         OpenAI-Compatible API            │  │
│  ┌──────────┐     │         (port 8000)                      │  │
│  │ Client 2 │────▶│                                          │  │
│  └──────────┘     └─────────────────┬────────────────────────┘  │
│  ┌──────────┐                       │                            │
│  │ Client N │────▶                  ▼                            │
│  └──────────┘     ┌──────────────────────────────────────────┐  │
│                    │           vLLM Engine                     │  │
│                    │  ┌────────────────────────────────────┐  │  │
│                    │  │  Scheduler (continuous batching)    │  │  │
│                    │  └────────────────┬───────────────────┘  │  │
│                    │                   │                       │  │
│                    │  ┌────────────────▼───────────────────┐  │  │
│                    │  │  PagedAttention KV Cache Manager    │  │  │
│                    │  └────────────────┬───────────────────┘  │  │
│                    │                   │                       │  │
│                    │  ┌────────────────▼───────────────────┐  │  │
│                    │  │  Model Execution (GPU)              │  │  │
│                    │  │  Qwen/Qwen2.5-7B-Instruct          │  │  │
│                    │  └────────────────────────────────────┘  │  │
│                    └──────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

The key insight: vLLM never lets the GPU sit idle. While one request is waiting for the user to read a streaming token, the GPU is already computing tokens for other requests.

## Serving Engine Comparison

How does vLLM compare to the alternatives you've already seen?

| Engine | Throughput | Concurrent Users | Ease of Setup | Best For |
|--------|-----------|-----------------|---------------|----------|
| Ollama (Day 1) | Low (~30 tok/s) | 1–2 | Very Easy | Local dev, prototyping |
| Transformers generate | Low (~25 tok/s) | 1 | Easy | Research, custom loops |
| vLLM | High (~1000+ tok/s) | 100+ | Moderate | Production GPU serving |
| TGI (Text Generation Inference) | High (~800 tok/s) | 100+ | Moderate | HuggingFace ecosystem |
| SGLang (Day 17) | High (~1100+ tok/s) | 100+ | Moderate | Multi-turn, structured |

*Throughput numbers are approximate for a 7B model on a single A100-80GB GPU.*

## The OpenAI-Compatible API

One of vLLM's best features: it exposes the same API format as OpenAI. If your code already works with the OpenAI Python client, switching to vLLM requires changing only the base URL:

```python
from openai import OpenAI

# Point to your local vLLM server instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't require an API key
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "What is inference?"}],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

This means you can develop locally against vLLM and deploy behind an OpenAI-compatible load balancer in production without changing application code.

## Why Not Just Use Ollama in Production?

Ollama is fantastic for local development (as you saw on Day 1), but it makes trade-offs that limit production use:

- **Single-request focus** — Ollama processes one request at a time by default
- **No continuous batching** — Requests queue rather than interleave
- **Limited GPU scheduling** — No PagedAttention for memory efficiency
- **No distributed serving** — Can't shard across multiple GPUs

vLLM solves all of these. It's designed from the ground up for throughput, not simplicity.

## Setup

### Prerequisites

- **GPU**: NVIDIA GPU with at least 16GB VRAM (A10, A100, L4, or RTX 4090)
- **CUDA**: CUDA 12.1+ installed
- **Python**: 3.10 or later
- **Prior days**: Familiarity with Day 4 (sampling) and Day 1 (Ollama basics)

### Install vLLM

```bash
# Install vLLM (includes torch with CUDA support)
pip install "vllm>=0.8.0"
```

### Start the vLLM Server

```bash
# Start serving Qwen2.5-7B-Instruct (Apache 2.0 license, no gating)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096
```

The first startup downloads the model (~14GB) and compiles CUDA kernels. Subsequent starts are faster.

### Verify the Server

```bash
# Check the health endpoint
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models
```

### CPU Fallback Note

vLLM requires an NVIDIA GPU. If you don't have one, you can still follow along by:
1. Using a cloud GPU instance (Lambda, RunPod, or Google Colab Pro)
2. Running the conceptual equivalent on Ollama locally (lower throughput, but same API pattern)

## What You'll Build

A Python script that connects to a running vLLM server, sends multiple concurrent requests, and measures throughput in requests per second. You'll see firsthand how vLLM handles parallelism — something that would choke a naive generate loop.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- vLLM is a production inference engine that serves LLMs at high throughput via an OpenAI-compatible API
- It uses PagedAttention and continuous batching to handle many concurrent requests efficiently
- The OpenAI-compatible API means zero code changes when migrating from OpenAI to self-hosted
- Throughput (requests/second) and latency (time-to-first-token) are the key metrics for serving engines
- Ollama is great for local dev; vLLM is what you deploy when real users start hitting your endpoint
- Always start vLLM as a separate server process — your application code talks to it via HTTP
