# Day 17: SGLang — RadixAttention and Serving Comparison

## The Concept

Think about how a web browser caches pages. When you visit a website for the second time, the browser doesn't re-download everything — it reuses the cached HTML, CSS, and images that haven't changed. Only the new or modified parts get fetched. This saves bandwidth and makes the page load dramatically faster. RadixAttention — the core innovation in SGLang — does the same thing for LLM inference. It caches the computed attention (KV cache) for common prompt prefixes so that repeated or similar requests skip redundant computation.

On Day 15 you deployed vLLM and measured its throughput. On Day 16 you learned how PagedAttention manages memory efficiently with fixed-size pages. SGLang (which stands for "Structured Generation Language") takes a different approach to the same problem. While vLLM focuses on memory efficiency through paging, SGLang focuses on *computation reuse* through RadixAttention — a radix tree data structure that stores KV cache entries for prompt prefixes. When two requests share the same system prompt (which they almost always do in production), SGLang computes the KV cache for that prefix once and shares it across all requests.

This matters enormously for production workloads. Consider a chatbot where every request starts with the same 500-token system prompt. Without prefix caching, the GPU recomputes those 500 tokens for every single request. With RadixAttention, the first request computes it, and all subsequent requests skip straight to the unique user portion. For multi-turn conversations where each turn shares the entire prior history, the savings compound dramatically.

## RadixAttention Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              RadixAttention: Prefix KV Cache Sharing              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request 1: "You are a helpful assistant. What is Python?"       │
│  Request 2: "You are a helpful assistant. Explain recursion."    │
│  Request 3: "You are a helpful assistant. What is a loop?"       │
│                                                                  │
│  Radix Tree (caches computed KV for shared prefixes):            │
│                                                                  │
│              [ROOT]                                               │
│                │                                                  │
│                ▼                                                  │
│  ┌──────────────────────────┐                                    │
│  │ "You are a helpful       │  ← Computed ONCE, shared by all    │
│  │  assistant."             │    (500 tokens of KV cache saved    │
│  │  [KV cache: cached]      │     per subsequent request)         │
│  └──────────┬───────────────┘                                    │
│             │                                                     │
│    ┌────────┼────────┐                                           │
│    ▼        ▼        ▼                                           │
│  [What    [Explain  [What                                        │
│   is       recur-    is a                                        │
│   Python?] sion.]    loop?]                                      │
│   NEW      NEW       NEW    ← Only unique suffixes computed      │
│                                                                  │
│  Without RadixAttention: 3 × full computation                    │
│  With RadixAttention:    1 × prefix + 3 × suffix only            │
│  Speedup: ~2-3x for this pattern                                 │
└──────────────────────────────────────────────────────────────────┘
```

## vLLM vs SGLang Architecture Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│           Architecture Comparison: vLLM vs SGLang                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  vLLM (Day 15):                   SGLang:                        │
│  ┌─────────────────────┐          ┌─────────────────────┐       │
│  │   OpenAI-compat API │          │   OpenAI-compat API │       │
│  └──────────┬──────────┘          └──────────┬──────────┘       │
│             │                                 │                   │
│  ┌──────────▼──────────┐          ┌──────────▼──────────┐       │
│  │  Continuous Batcher │          │  RadixAttention      │       │
│  │  (schedule by slot) │          │  (prefix tree cache) │       │
│  └──────────┬──────────┘          └──────────┬──────────┘       │
│             │                                 │                   │
│  ┌──────────▼──────────┐          ┌──────────▼──────────┐       │
│  │  PagedAttention     │          │  Chunked Prefill +   │       │
│  │  (fixed-size pages) │          │  Paged KV Cache      │       │
│  └──────────┬──────────┘          └──────────┬──────────┘       │
│             │                                 │                   │
│  ┌──────────▼──────────┐          ┌──────────▼──────────┐       │
│  │  Model Execution    │          │  Model Execution     │       │
│  │  (GPU)              │          │  (GPU)               │       │
│  └─────────────────────┘          └─────────────────────┘       │
│                                                                  │
│  Key difference: SGLang adds a prefix-sharing layer              │
│  that eliminates redundant computation for common prefixes.      │
│  vLLM optimizes memory allocation; SGLang optimizes reuse.       │
└──────────────────────────────────────────────────────────────────┘
```

## Serving Engine Comparison

| Feature | vLLM | SGLang | Winner For |
|---------|------|--------|------------|
| First-request latency | Fast | Fast | Tie |
| Repeated-prefix latency | Standard | Very Low (cached) | SGLang — multi-turn, shared system prompts |
| Memory efficiency | Excellent (PagedAttention) | Excellent (Paged + Radix) | Tie (both near-optimal) |
| Structured output (JSON) | Good (via outlines) | Native (built-in grammar) | SGLang — constrained decoding |
| Multi-turn conversations | Good | Excellent (prefix reuse) | SGLang — chatbots, agents |
| Raw throughput (batch) | Very High | Very High | Tie (workload-dependent) |
| Ecosystem maturity | Very Mature | Rapidly Growing | vLLM — production stability |
| API compatibility | OpenAI-compatible | OpenAI-compatible | Tie |
| Speculative decoding | Yes | Yes | Tie |
| Multi-GPU (tensor parallel) | Yes | Yes | Tie |

## When to Choose Which Engine

The choice between vLLM and SGLang depends on your workload:

- **Choose vLLM** when you need battle-tested stability, broad model support, and your requests have diverse, non-repeating prefixes (batch processing unique documents).

- **Choose SGLang** when your workload involves shared system prompts (chatbots), multi-turn conversations (agents), structured output (JSON APIs), or any pattern where requests share common prefixes.

- **Both are excellent** for high-throughput production serving. The performance gap is narrowing as both projects adopt each other's innovations.

```python
# SGLang server startup — same simplicity as vLLM
# Both expose OpenAI-compatible APIs, so client code is identical

# Option 1: SGLang native server
# python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --port 8001

# Option 2: After starting, use the same OpenAI client
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="none")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Setup

### Prerequisites

- **GPU**: NVIDIA GPU with at least 16GB VRAM (compute capability sm80+)
- **CUDA**: CUDA 12.1+ installed
- **Python**: 3.10 or later
- **Prior days**: Familiarity with Day 15 (vLLM) and Day 16 (PagedAttention/batching)

### Install SGLang

```bash
# Install SGLang
pip install "sglang[all]>=0.4.0"
```

### Start Both Servers (for comparison)

```bash
# Terminal 1: Start vLLM on port 8000
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096

# Terminal 2: Start SGLang on port 8001
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --dtype auto
```

### CPU Fallback Note

Both vLLM and SGLang require NVIDIA GPUs. If you don't have one:
1. Use a cloud GPU instance (Lambda, RunPod, or Colab Pro with A100)
2. Review the expected output and conceptual explanations in this README
3. On Day 17's lab, the key learning is the *comparison methodology* — understanding when prefix caching matters

## What You'll Build

A Python script that sends identical requests to both vLLM and SGLang, measures latency for each, and compares performance. You'll specifically test scenarios where SGLang's RadixAttention shines: repeated system prompts and multi-turn conversations where prefix caching provides measurable speedup.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- SGLang uses RadixAttention to cache KV computations for shared prompt prefixes (like a web browser cache)
- When requests share system prompts, SGLang skips redundant computation — giving 2-3x speedup on subsequent requests
- vLLM optimizes memory allocation (PagedAttention); SGLang optimizes computation reuse (RadixAttention)
- Both engines expose OpenAI-compatible APIs — switching between them requires changing only the port number
- For multi-turn chatbots and agents, SGLang's prefix caching provides the largest advantage
- In practice, the "best" engine depends on your workload pattern: shared prefixes favor SGLang, diverse inputs are a tie
