# Day 24: TensorRT-LLM, Triton & NVIDIA NIMs

## The Concept

Yesterday you learned how ONNX Runtime provides cross-platform optimization — one format, many devices. Today you'll meet NVIDIA's answer to the question "what if we don't need portability at all?" **TensorRT-LLM** is a purpose-built inference engine that squeezes maximum throughput out of NVIDIA GPUs by compiling models into GPU-specific execution plans. Think of it as the Formula 1 car of inference: blindingly fast on its specific track, but it won't drive on public roads.

The analogy is a custom-built race car versus a street car. Your street car (PyTorch, ONNX Runtime) drives anywhere — city streets, highways, gravel roads. A race car (TensorRT-LLM) only works on the racetrack, but on that track, it's 2–4× faster than anything else. The trade-off is portability: TensorRT-LLM only runs on NVIDIA GPUs with specific CUDA compute capabilities, requires a lengthy compilation step, and locks you into NVIDIA's ecosystem.

On Day 23 you saw ONNX Runtime as a portable optimization layer. On Day 15 you deployed vLLM for high-throughput serving. TensorRT-LLM occupies a different niche: when you control the hardware, need absolute maximum throughput, and can accept the setup complexity. NVIDIA wraps this into **NIMs** (NVIDIA Inference Microservices) — pre-compiled, containerized model packages that eliminate the compilation step for popular models.

## 1. The TensorRT-LLM Architecture

TensorRT-LLM compiles a model through several stages, each adding hardware-specific optimizations that generic runtimes can't apply:

```
┌──────────────────────────────────────────────────────────────────┐
│     Inference Engine Comparison: TensorRT-LLM vs ONNX vs vLLM    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  vLLM (Day 15):                                                  │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐    │
│  │ PyTorch │───▶│PagedAttn │───▶│  CUDA    │───▶│ Output  │    │
│  │ Weights │    │+ Batching│    │ Kernels  │    │         │    │
│  └─────────┘    └──────────┘    └──────────┘    └─────────┘    │
│  Flexible, easy setup, good throughput                           │
│                                                                  │
│  ─────────────────────────────────────────────────────────────   │
│                                                                  │
│  ONNX Runtime (Day 23):                                          │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐    │
│  │  ONNX   │───▶│  Graph   │───▶│Execution │───▶│ Output  │    │
│  │  Graph  │    │  Fusion  │    │ Provider │    │         │    │
│  └─────────┘    └──────────┘    └──────────┘    └─────────┘    │
│  Portable, moderate optimization, multi-platform                 │
│                                                                  │
│  ─────────────────────────────────────────────────────────────   │
│                                                                  │
│  TensorRT-LLM:                                                   │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐    │
│  │ Model   │───▶│ TRT-LLM  │───▶│ Compiled │───▶│ Output  │    │
│  │ Weights │    │ Compiler │    │  Engine  │    │         │    │
│  └─────────┘    └──────────┘    └──────────┘    └─────────┘    │
│       │              │               │                           │
│       │         ┌────┴────┐    ┌─────┴─────┐                    │
│       │         │• Layer  │    │• GPU-spec  │                    │
│       │         │  fusion │    │  kernels   │                    │
│       │         │• Quant  │    │• Memory    │                    │
│       │         │  aware  │    │  planning  │                    │
│       │         │• Kernel │    │• Tensor    │                    │
│       │         │  select │    │  cores     │                    │
│       │         └─────────┘    └───────────┘                    │
│  Maximum throughput, GPU-locked, requires compilation            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 2. What Makes TensorRT-LLM Fast

TensorRT-LLM applies optimizations that neither ONNX Runtime nor vLLM can:

**Kernel auto-tuning** — The compiler benchmarks hundreds of kernel implementations for each operation on your specific GPU and selects the fastest one. A kernel optimized for an A100 may be different from one optimized for an H100.

**Layer fusion** — Goes beyond ONNX's operator fusion. TensorRT-LLM fuses entire transformer layers into single custom CUDA kernels, eliminating intermediate memory reads/writes.

**Quantization-aware compilation** — Applies INT8/INT4/FP8 quantization at compile time with calibration data, producing kernels specifically optimized for quantized operations on Tensor Cores.

**In-flight batching** — Like vLLM's continuous batching, but implemented at the CUDA kernel level for lower overhead.

**Custom attention kernels** — Implements Flash Attention variants tuned for specific GPU architectures (Ampere, Hopper, Blackwell).

## 3. Comparison Table

| Engine | Throughput (relative) | Portability | Setup Complexity | GPU Vendor Lock-in | Best For |
|--------|:---:|:---:|:---:|:---:|---|
| PyTorch (baseline) | 1.0× | Any GPU + CPU | Low | None | Development, prototyping |
| ONNX Runtime | 1.3–2.0× | Any device | Medium | None | Cross-platform deployment |
| vLLM | 2.0–3.0× | NVIDIA GPU | Medium | Moderate | Multi-user LLM serving |
| TensorRT-LLM | 3.0–5.0× | NVIDIA GPU only | High | Strong | Maximum throughput, single vendor |
| NVIDIA NIM | 3.0–5.0× | NVIDIA GPU only | Low (container) | Strong | Production deployment (pre-built) |

## 4. NVIDIA NIMs — Pre-Compiled Containers

The biggest barrier to TensorRT-LLM is the compilation step — it can take 30–60 minutes and requires matching your exact GPU architecture, driver version, and CUDA toolkit. **NVIDIA NIMs** solve this by shipping pre-compiled engines in Docker containers:

```bash
# Pull and run a pre-compiled NIM container (example)
docker run --gpus all -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

NIMs provide:
- Pre-compiled TensorRT-LLM engines for common GPU architectures
- OpenAI-compatible API (drop-in replacement for vLLM or OpenAI endpoints)
- Automatic quantization and optimization
- Health monitoring and metrics endpoints

The trade-off: you're locked into NVIDIA's ecosystem and container registry, and you need a GPU that matches the NIM's target architecture.

## 5. When to Use Each Engine

**Use vLLM when:**
- You need flexibility and fast iteration
- You serve multiple models and swap frequently
- You want open-source with community support

**Use ONNX Runtime when:**
- You deploy to mixed hardware (CPU, different GPUs, edge)
- Portability matters more than absolute peak performance
- You want a single deployment format

**Use TensorRT-LLM / NIMs when:**
- You control the hardware (all NVIDIA, known architecture)
- Maximum throughput justifies setup complexity
- You're running at scale where 2× speedup = significant cost savings
- The model is stable (not changing weekly)

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.0+ and 16+ GB VRAM (for TensorRT-LLM)
- Docker (for NIMs)
- ~25 GB disk space

### Install dependencies

```bash
pip install -r requirements.txt
```

### GPU requirement

TensorRT-LLM requires an NVIDIA GPU with compute capability 8.0+ (Ampere or newer: A100, A10G, RTX 3090, H100, RTX 4090, etc.).

**No NVIDIA GPU?** The lab script demonstrates the concepts using the vLLM and requests-based API comparison approach. For actual TensorRT-LLM compilation, you need NVIDIA hardware. ONNX Runtime (Day 23) serves as the portable alternative.

### Note on TensorRT-LLM installation

TensorRT-LLM has complex installation requirements (specific CUDA versions, driver compatibility). The recommended approach is using NVIDIA's Docker containers:

```bash
# Use NVIDIA's pre-built container
docker pull nvcr.io/nvidia/tensorrt-llm:latest
```

## What You'll Build

A Python script that demonstrates TensorRT-LLM performance concepts by benchmarking inference throughput. The script compares request throughput between a standard serving endpoint and a TensorRT-optimized endpoint (or simulates the expected gains based on published benchmarks), printing the throughput improvement percentage as the key metric.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- TensorRT-LLM compiles models into GPU-specific execution plans with kernel auto-tuning, layer fusion, and quantization-aware optimization
- It achieves 3–5× throughput versus vanilla PyTorch, but only runs on NVIDIA GPUs — strong vendor lock-in
- NVIDIA NIMs package pre-compiled TensorRT-LLM engines in Docker containers, eliminating the compilation step
- The compilation process takes 30–60 minutes and is specific to your GPU architecture — compiled engines aren't portable between GPU types
- Use TensorRT-LLM when you control the hardware and need maximum throughput at scale; use ONNX Runtime (Day 23) when you need portability
- vLLM sits in the middle: excellent throughput with easier setup and more flexibility for model swapping
