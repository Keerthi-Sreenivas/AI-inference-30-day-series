# Day 5: Model Formats & Precision — FP16, BF16, INT8, INT4

## The Concept

Every parameter in a neural network is a number — and the way you store that number dramatically affects how much memory the model consumes, how fast it runs, and how accurate its outputs are. Today you'll learn the five major precision formats used in inference: FP32, FP16, BF16, INT8, and INT4. By the end, you'll understand when to reach for each one and how to load the same model at different precisions using Hugging Face Transformers.

Think of it like photo quality. A RAW photograph (FP32) captures every detail at full fidelity — but the file is enormous. Export it as a high-quality JPEG (FP16/BF16) and you lose almost nothing perceptible while cutting file size in half. Compress it further to a small JPEG thumbnail (INT8/INT4) and you can barely tell the difference at normal viewing distance — yet the file is a fraction of the original. Language models work the same way: you can represent each weight with fewer bits and lose almost nothing in practice, while gaining massive memory savings.

On Day 3 you used `from_pretrained()` to load a model — today you'll learn what happens under the hood when you add `torch_dtype=torch.float16` or load with 8-bit quantization. On Day 2 you saw how text becomes token IDs (integers) — today you'll see how the model's internal weights that process those token IDs are stored as floating-point numbers at various precision levels.

## 1. What Are Precision Formats?

A **precision format** defines how many binary digits (bits) are used to represent a single number. More bits means more accuracy but more memory. In machine learning, the standard formats are:

- **FP32 (Float32)** — 32 bits per number. Full precision. This is the default training format — maximum accuracy, maximum memory.
- **FP16 (Float16)** — 16 bits per number. Half precision. Cuts memory in half with negligible quality loss for inference.
- **BF16 (BFloat16)** — 16 bits, but with a different split between exponent and mantissa than FP16. Same memory as FP16, better handling of very large or very small numbers. Preferred on modern hardware (A100, H100, Apple M-series).
- **INT8 (8-bit integer)** — 8 bits per number. Quarter of FP32 memory. Requires quantization (mapping floating-point values to integers). Small quality loss.
- **INT4 (4-bit integer)** — 4 bits per number. One-eighth of FP32 memory. More aggressive quantization. Noticeable quality loss on complex tasks, but surprisingly good for general chat and summarization.

Each parameter in a 1.7B model is one of these numbers. Multiply `1.7 billion × bits_per_param / 8` to get the model size in bytes.

## 2. Memory Layout — Same Model, Different Precisions

Here's what happens when you load the same 7B-parameter model at different precision levels:

```
┌──────────────────────────────────────────────────────┐
│     Memory Layout: Same 7B Model, Different Formats  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  FP32 (32 bits per param):                           │
│  ████████████████████████████████  28.0 GB           │
│                                                      │
│  FP16 / BF16 (16 bits per param):                    │
│  ████████████████                  14.0 GB           │
│                                                      │
│  INT8 (8 bits per param):                            │
│  ████████                           7.0 GB           │
│                                                      │
│  INT4 (4 bits per param):                            │
│  ████                               3.5 GB           │
│                                                      │
│  ─────────────────────────────────────────────       │
│  Same model. Same knowledge. 8× less memory.         │
└──────────────────────────────────────────────────────┘
```

For our 1.7B SmolLM2 model, the numbers scale down proportionally:

| Format | Bits per Param | Approximate Model Size | RAM Needed |
|--------|:-:|:-:|:-:|
| FP32 | 32 | ~6.8 GB | ~8 GB |
| FP16 / BF16 | 16 | ~3.4 GB | ~5 GB |
| INT8 | 8 | ~1.7 GB | ~3 GB |
| INT4 | 4 | ~0.85 GB | ~2 GB |

The "RAM Needed" column includes overhead for the framework, activations, and KV cache during generation — not just the raw model weights.

## 3. Format Comparison

| Format | Bits | Memory (1.7B) | Inference Speed | Quality Impact | Best For |
|--------|:----:|:-------------:|:---------------:|:--------------:|----------|
| FP32 | 32 | ~6.8 GB | Baseline | None (reference) | Training, debugging |
| FP16 | 16 | ~3.4 GB | ~2× faster | Negligible | GPU inference (default) |
| BF16 | 16 | ~3.4 GB | ~2× faster | Negligible | Modern GPUs (A100/H100) |
| INT8 | 8 | ~1.7 GB | ~2–3× faster | Minimal | Memory-constrained serving |
| INT4 | 4 | ~0.85 GB | ~3–4× faster | Small (chat OK) | Edge devices, laptops |

**Key insight:** FP16 and BF16 give you the best bang for your buck — you halve memory with essentially zero quality loss. Going to INT8/INT4 saves even more memory but requires quantization techniques (which you'll explore in depth on Day 8).

## 4. How to Load at Different Precisions

With Hugging Face Transformers, switching precision is a single argument:

```python
from transformers import AutoModelForCausalLM
import torch

# FP32 — full precision (default on CPU)
model_fp32 = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# FP16 — half precision (common for GPU inference)
model_fp16 = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    torch_dtype=torch.float16
)

# BF16 — brain float (preferred on modern hardware)
model_bf16 = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    torch_dtype=torch.bfloat16
)
```

The `torch_dtype` argument tells `from_pretrained()` what precision to load the weights in. The model file on disk (typically in **safetensors** format — a safe, fast binary format for storing tensors) gets converted to your requested dtype during loading.

## 5. Safetensors — The Storage Format

You might wonder: if precision is about how numbers are stored in memory, what about on disk? Modern models use the **safetensors** format (`.safetensors` files). This is a binary file format developed by Hugging Face that:

- Stores tensors with zero-copy memory mapping (fast loading)
- Is safe against code injection (unlike pickle-based `.bin` files)
- Supports any precision format (FP32, FP16, BF16, etc.)

When you call `from_pretrained()`, the library reads the safetensors file and loads it into memory at whatever `torch_dtype` you specify. If the file stores FP16 weights and you request FP32, it upcasts. If you request FP16 from an FP32 file, it downcasts. The conversion happens during loading.

## 6. When to Use Each Format

Here's a practical decision guide:

- **Developing/debugging?** → Use FP32 for maximum precision during testing
- **Running on a GPU with ≥16 GB VRAM?** → Use BF16 (or FP16 if BF16 unsupported)
- **Running on a laptop/CPU?** → Use FP32 (CPUs handle it natively) or FP16 if your CPU supports it
- **Need to fit a larger model in memory?** → Use INT8 quantization (Day 8 covers this)
- **Running on mobile or edge devices?** → Use INT4 quantization (extreme compression)

## Setup

### Prerequisites

- Python 3.10+
- ~4 GB disk space (for model download)
- No GPU required — runs on CPU (slower but demonstrates the concept)
- Familiarity with `from_pretrained()` from Day 3

### Install dependencies

```bash
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — a 1.7-billion parameter model that's fully open (Apache 2.0 license), no account or token needed. It downloads automatically on first run (~3.5 GB).

## What You'll Build

A Python script that loads the same model at multiple precision formats (FP32, FP16, BF16), measures the memory footprint of each, generates a short response to verify the model works at each precision, and prints a comparison table showing memory usage in MB — giving you concrete evidence of how precision affects resource consumption.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Every model parameter is a number — the precision format determines how many bits store each number (32, 16, 8, or 4 bits)
- FP16 and BF16 halve memory usage compared to FP32 with negligible quality loss — they're the default choice for GPU inference
- BF16 handles extreme values better than FP16 (wider exponent range) and is preferred on modern hardware like A100, H100, and Apple M-series
- INT8 and INT4 use quantization to compress further, trading small quality loss for dramatic memory savings
- The `torch_dtype` parameter in `from_pretrained()` controls loading precision — one argument changes everything
- Safetensors is the modern storage format for model weights: fast, safe, and precision-flexible
