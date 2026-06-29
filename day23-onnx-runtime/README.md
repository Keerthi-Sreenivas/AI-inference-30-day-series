# Day 23: ONNX Runtime — Cross-Platform Optimization

## The Concept

So far in this series, every model you've run has been tied to PyTorch — one framework, one execution path. That works fine for development, but what happens when you need to deploy the same model on an edge device, a Windows server, a mobile app, or hardware without CUDA? **ONNX** (Open Neural Network Exchange) is the answer: a universal format that decouples your model from any single framework.

Think of ONNX like a universal charger. You have one cable (the ONNX format) that works with many devices (CPUs, GPUs, NPUs, mobile chips). Instead of needing a separate charger for each phone brand, you export your model once and run it anywhere — with hardware-specific optimizations applied automatically by the ONNX Runtime engine.

On Day 5 you learned about model formats and how precision affects memory and speed. On Day 14 you built benchmarking methodology to compare inference performance. Today you'll export a model to ONNX format, run it through ONNX Runtime's optimized execution engine, and measure the concrete speedup compared to vanilla PyTorch inference — giving you a portable, optimized alternative to framework-locked deployment.

## 1. How ONNX Works

ONNX represents neural networks as a computation graph — a directed acyclic graph (DAG) of mathematical operations. When you export a PyTorch model to ONNX, you're converting imperative Python code into a static, framework-independent graph that any ONNX-compatible runtime can execute.

```
┌──────────────────────────────────────────────────────────────────┐
│           PyTorch vs ONNX Runtime Inference Pipeline              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PYTORCH PATH (framework-dependent):                             │
│  ┌────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │ Python │───▶│ PyTorch  │───▶│  CUDA /  │───▶│  Output  │    │
│  │  Code  │    │ Dispatch │    │   CPU    │    │ Tensors  │    │
│  └────────┘    └──────────┘    └──────────┘    └──────────┘    │
│       Dynamic graph, Python overhead, GIL, eager execution       │
│                                                                  │
│  ─────────────────────────────────────────────────────────────   │
│                                                                  │
│  ONNX PATH (framework-independent):                              │
│  ┌────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │  ONNX  │───▶│   ONNX   │───▶│ Optimized│───▶│  Output  │    │
│  │ Model  │    │ Runtime  │    │ Backend  │    │ Arrays   │    │
│  └────────┘    └──────────┘    └──────────┘    └──────────┘    │
│       Static graph, C++ execution, no GIL, graph optimization    │
│                                                                  │
│  ONNX Runtime Optimizations:                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • Operator fusion (combine matmul + bias + activation)  │    │
│  │ • Constant folding (pre-compute static operations)      │    │
│  │ • Memory planning (reduce allocations during inference) │    │
│  │ • Execution provider selection (CPU/CUDA/TensorRT/etc.) │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 2. Why ONNX Runtime is Faster

ONNX Runtime applies several optimizations that PyTorch's eager mode cannot:

**Graph optimizations** — Because ONNX models are static graphs, the runtime can analyze the entire computation at load time and fuse operations together. A sequence like `MatMul → Add → ReLU` becomes a single fused kernel call.

**No Python overhead** — PyTorch dispatches every operation through Python, hitting the GIL. ONNX Runtime runs the entire graph in C++, only crossing the Python boundary at input/output.

**Execution Providers** — ONNX Runtime automatically selects the best hardware backend (called an "Execution Provider") for each operation. On NVIDIA GPUs it can use the CUDA or TensorRT Execution Provider; on Intel CPUs it can use OpenVINO; on Qualcomm chips it can use QNN.

**Quantization-aware execution** — ONNX Runtime supports INT8 and INT4 quantized models natively, with hardware-specific optimized kernels for quantized operations.

## 3. Comparison: ONNX Runtime vs Other Runtimes

| Runtime | Platform Support | Typical Speedup vs PyTorch | Model Compatibility | Setup Complexity |
|---------|:---:|:---:|:---:|:---:|
| PyTorch (baseline) | GPU/CPU (CUDA, MPS) | 1.0× | All HF models | Low |
| ONNX Runtime (CPU) | All CPUs, all OS | 1.2–2.0× | Most encoder/decoder models | Medium |
| ONNX Runtime (CUDA) | NVIDIA GPU | 1.3–2.5× | Most encoder/decoder models | Medium |
| TensorRT (Day 24) | NVIDIA GPU only | 2.0–4.0× | Subset, requires conversion | High |
| vLLM (Day 15) | NVIDIA GPU | High throughput | LLM-specific | Medium |
| llama.cpp (Day 6) | All CPUs, all OS | Varies | GGUF format only | Low |

## 4. The Export Pipeline with Optimum

Hugging Face's `optimum` library handles ONNX export and optimization in one step. It takes care of:
- Tracing the model with correct input shapes
- Applying ONNX Runtime graph optimizations
- Handling dynamic axes (variable sequence lengths)
- Setting up the correct execution provider

```python
from optimum.onnxruntime import ORTModelForCausalLM

# Export and load in one step — optimum handles the conversion
model = ORTModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    export=True,  # Converts PyTorch → ONNX automatically
)
```

For production, you'd export once and save the ONNX model to disk, then load only the ONNX file at inference time — skipping the PyTorch dependency entirely.

## 5. Execution Providers

ONNX Runtime's power comes from **Execution Providers** (EPs) — pluggable hardware backends that optimize execution for specific platforms. When you load a model, the runtime assigns each operation to the best available EP:

| Execution Provider | Platform | Typical Speedup | Use Case |
|---|---|:---:|---|
| CPU (default) | Any x86/ARM CPU | 1.0× (ONNX baseline) | Universal fallback |
| CUDA | NVIDIA GPU | 1.5–3.0× vs CPU EP | GPU inference |
| TensorRT | NVIDIA GPU | 2.0–4.0× vs CPU EP | Maximum GPU throughput |
| OpenVINO | Intel CPU/GPU/NPU | 1.3–2.5× vs CPU EP | Intel hardware |
| DirectML | Windows GPU (any vendor) | 1.2–2.0× vs CPU EP | Windows deployment |
| CoreML | Apple Silicon | 1.5–3.0× vs CPU EP | macOS/iOS |
| QNN | Qualcomm NPU | 2.0–5.0× vs CPU EP | Mobile/edge |

```python
# Select execution provider when creating the inference session
import onnxruntime as ort

# Automatically use GPU if available, fall back to CPU
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession("model.onnx", providers=providers)
```

## 6. When to Use ONNX Runtime

**Good fit:**
- Deployment on diverse hardware (CPU servers, edge devices, mobile)
- Need to eliminate PyTorch dependency in production
- Latency-sensitive applications where every millisecond matters
- Cross-platform consistency (same model, same results everywhere)

**Poor fit:**
- Active model development (frequent changes make re-export tedious)
- Very large autoregressive models (vLLM/TGI are better for multi-user serving)
- Models that heavily use dynamic control flow (hard to trace to static graph)

## Setup

### Prerequisites

- Python 3.10+
- ~8 GB disk space (model + ONNX export)
- No GPU required — ONNX Runtime optimizes CPU inference significantly

### Install dependencies

```bash
pip install -r requirements.txt
```

### Model choice

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — a small, open model (Apache 2.0, no account needed) that exports cleanly to ONNX. The export process adds ~3 GB to disk for the ONNX graph files.

### CPU vs GPU

ONNX Runtime's biggest advantage on CPU is eliminating Python overhead and applying graph fusions. On GPU, it adds operator fusion and execution provider optimization. The lab runs on CPU by default, where speedups are most noticeable for smaller models.

## What You'll Build

A Python script that exports SmolLM2-1.7B-Instruct to ONNX format, benchmarks text generation using both PyTorch and ONNX Runtime, and prints the speedup factor — showing exactly how much faster the optimized ONNX graph executes compared to standard PyTorch inference.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- ONNX is a universal model format — export once, run anywhere (CPU, GPU, mobile, edge)
- ONNX Runtime applies graph optimizations (fusion, constant folding, memory planning) that PyTorch's eager mode cannot
- The `optimum` library handles export from Hugging Face models with a single `export=True` flag
- Typical speedups are 1.2–2.0× on CPU and 1.3–2.5× on GPU compared to vanilla PyTorch
- Best for deployment scenarios where you need cross-platform portability or want to eliminate the PyTorch dependency
- Not ideal for serving large LLMs at scale — vLLM and TGI are better for that use case (as we saw on Day 15)
