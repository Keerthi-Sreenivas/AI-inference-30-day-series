# Day 16: PagedAttention & Continuous Batching

## The Concept

Think about how your computer manages RAM. You have 16GB of physical memory, but applications ask for memory in unpredictable chunks — a browser tab here, a spreadsheet there. The operating system doesn't give each application one contiguous block. Instead, it uses *virtual memory* with fixed-size pages, allocating them on demand and reclaiming them when freed. This lets dozens of applications share limited RAM efficiently without wasting space on pre-allocated blocks they haven't filled yet.

PagedAttention does the same thing for GPU memory during inference. On Day 12 you learned that the KV cache (Key-Value cache) grows linearly with sequence length — each new token adds another entry. On Day 15 you saw vLLM serve many requests concurrently. But here's the problem: if vLLM pre-allocated a contiguous block of GPU memory for each request's maximum possible sequence length, most of that memory would sit empty (requests rarely use their full allocation). PagedAttention solves this by breaking the KV cache into fixed-size *pages* that are allocated only when needed, just like OS virtual memory.

Continuous batching is the second breakthrough. Traditional batching waits for a full batch of requests, processes them together, and returns all results. But requests finish at different times — short answers complete long before long ones. Continuous batching doesn't wait. As soon as one request finishes, its slot immediately fills with the next waiting request. The GPU never pauses between batches. Together, PagedAttention + continuous batching are why vLLM achieves 2-4x higher throughput than naive serving.

## How PagedAttention Works

In standard attention, each request gets a contiguous memory allocation for its KV cache:

```
┌──────────────────────────────────────────────────────────────────┐
│           Standard KV Cache Allocation (Wasteful)                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request A (max 2048 tokens, using 512):                        │
│  [████████░░░░░░░░░░░░░░░░░░░░░░░░]  75% wasted                │
│   ^^^^^^^^ used        ^^^^^^^^^^^^^^ pre-allocated but empty    │
│                                                                  │
│  Request B (max 2048 tokens, using 1800):                       │
│  [████████████████████████████░░░░░]  12% wasted                │
│                                                                  │
│  Request C (max 2048 tokens, using 100):                        │
│  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  95% wasted                │
│                                                                  │
│  Total GPU memory reserved: 6144 tokens worth                    │
│  Actually used:             2412 tokens worth (39%)              │
│  WASTED:                    3732 tokens worth (61%)              │
└──────────────────────────────────────────────────────────────────┘
```

PagedAttention fixes this by splitting memory into small, fixed-size blocks (pages):

```
┌──────────────────────────────────────────────────────────────────┐
│           PagedAttention KV Cache (Memory-Efficient)              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Physical GPU Memory Pages (16 tokens each):                     │
│  [P0][P1][P2][P3][P4][P5][P6][P7][P8]...                       │
│                                                                  │
│  Request A (512 tokens) → uses pages: P0, P1, P2, ... P31       │
│  Request B (1800 tokens) → uses pages: P32, P33, ... P144       │
│  Request C (100 tokens) → uses pages: P145, P146, ... P151      │
│                                                                  │
│  ✓ No pre-allocation — pages allocated on demand                 │
│  ✓ No fragmentation — freed pages immediately reusable           │
│  ✓ Near-zero waste — only last page of each request is partial   │
│                                                                  │
│  Total pages allocated: 152 (only what's needed)                 │
│  Waste: < 1 page per request (< 2%)                             │
└──────────────────────────────────────────────────────────────────┘
```

## Continuous Batching Timeline

Traditional batching forces all requests in a batch to wait for the longest one:

```
┌──────────────────────────────────────────────────────────────────┐
│         Continuous Batching vs Static Batching                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STATIC BATCHING (wasteful):                                     │
│  Time ───────────────────────────────────────────────▶           │
│                                                                  │
│  Batch 1:                    │ Batch 2:                          │
│  ┌──────────────────────┐    │ ┌──────────────────────┐          │
│  │ Req A ████░░░░░░░░░░ │    │ │ Req D ████████░░░░░░ │          │
│  │ Req B ████████████░░ │    │ │ Req E ██░░░░░░░░░░░░ │          │
│  │ Req C ████████████████│    │ │ Req F ██████████████ │          │
│  └──────────────────────┘    │ └──────────────────────┘          │
│  ▲ A,B finish but wait for C │                                   │
│                                                                  │
│  CONTINUOUS BATCHING (efficient):                                 │
│  Time ───────────────────────────────────────────────▶           │
│                                                                  │
│  Slot 1: [Req A ████][Req D ████████████]                        │
│  Slot 2: [Req B █████████][Req E ██████████]                     │
│  Slot 3: [Req C ████████████████][Req F ████████]                │
│                                                                  │
│  ✓ When A finishes, D immediately starts in its slot             │
│  ✓ No idle time between batches                                  │
│  ✓ GPU utilization stays near 100%                               │
└──────────────────────────────────────────────────────────────────┘
```

## Batching Strategy Comparison

| Strategy | Throughput | Memory Efficiency | Latency | Use Case |
|----------|-----------|------------------|---------|----------|
| No batching (sequential) | Very Low | Low (one request at a time) | Lowest per-request | Single-user dev |
| Static batching | Medium | Medium (wastes padding) | High (wait for slowest) | Offline batch jobs |
| Dynamic batching | High | Medium | Medium | Simple production |
| Continuous batching | Very High | High (with PagedAttention) | Low | Production serving |
| Continuous + PagedAttention | Highest | Highest (near-zero waste) | Lowest | vLLM, SGLang |

## Memory Savings in Practice

On a single A100-80GB GPU serving Qwen2.5-7B-Instruct:

| Metric | Standard Allocation | PagedAttention | Improvement |
|--------|-------------------|----------------|-------------|
| Max concurrent requests | ~8 | ~24 | 3x |
| Memory utilization | 30-40% | 85-95% | ~2.5x |
| Throughput (req/sec) | ~3 | ~10 | ~3.3x |
| Memory wasted | ~60% | <5% | 12x less waste |

The improvement is most dramatic when requests have highly variable output lengths — which is the norm in production (short answers and long answers mixed together).

## vLLM's PagedAttention Configuration

You can control PagedAttention behavior when starting vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --block-size 16
```

- `--gpu-memory-utilization 0.90`: Use 90% of GPU memory for KV cache pages (default: 0.90)
- `--block-size 16`: Number of tokens per page/block (default: 16)
- `--max-model-len 4096`: Maximum sequence length (limits page table size)

## Setup

### Prerequisites

- **GPU**: NVIDIA GPU with at least 16GB VRAM
- **CUDA**: CUDA 12.1+ installed
- **Python**: 3.10 or later
- **Prior days**: Familiarity with Day 12 (KV cache) and Day 15 (vLLM basics)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the vLLM server

```bash
# Start with explicit memory utilization setting so we can observe it
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90
```

### CPU Fallback Note

PagedAttention is a GPU memory management technique — it doesn't have a CPU equivalent. If you don't have a GPU, follow the README conceptually and review the expected output at the end. The concepts transfer to any paged memory system.

## What You'll Build

A Python script that sends concurrent requests of varying lengths to a vLLM server and monitors GPU memory utilization. You'll observe how PagedAttention allows the server to handle many simultaneous requests without running out of memory — and how memory usage scales with actual token count rather than maximum allocation.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- PagedAttention manages KV cache memory like an OS manages RAM — using fixed-size pages allocated on demand
- Standard allocation wastes 60%+ of GPU memory; PagedAttention wastes less than 5%
- Continuous batching fills GPU slots immediately when requests finish — no idle time between batches
- Together, these techniques give vLLM 2-4x throughput improvement over naive serving
- The `--gpu-memory-utilization` flag controls how much GPU memory vLLM reserves for KV cache pages
- Memory efficiency directly translates to more concurrent users on the same hardware
