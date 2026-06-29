# Day 8: Quantization Theory — Why Fewer Bits Still Work

## The Concept

Think about prices at a grocery store. A can of soup costs $3.99 — but when someone asks how much it was, you say "about four dollars." You rounded a 4-digit number down to a 1-digit number, lost almost nothing meaningful, and made the answer easier to carry in your head. **Quantization** does the same thing to the billions of numbers (called weights) stored inside a neural network: it rounds high-precision floating-point values into fewer bits, dramatically shrinking the model while keeping its answers nearly identical.

Why does this matter for inference? On Day 5 you learned that a 7-billion parameter model stored in FP32 (32 bits per weight) consumes 28 GB of memory. If you quantize those same weights to 4-bit integers, the model shrinks to about 3.5 GB — an 8× reduction. That means a model that required a high-end GPU can now fit on a laptop. Quantization is the single most impactful technique for making large models accessible on consumer hardware, and every production inference system — from llama.cpp (Day 6) to vLLM — uses it extensively.

The fundamental insight is that neural network weights are **redundant**. Most weights cluster around small values, and the model's predictions change very little when you round them. Research shows that 4-bit quantization preserves 95–99% of a model's quality for most tasks. The trade-off is a small increase in perplexity (the model's uncertainty), which you'll measure directly in today's lab.

## 1. How Quantization Works — The Math

At its core, quantization maps a continuous range of floating-point values into a discrete set of integers. Here's the simplest version (symmetric linear quantization):

```python
# Symmetric linear quantization: FP32 → INT8
import torch

# Original FP32 weight tensor
weight = torch.tensor([0.0032, -0.0891, 0.1247, -0.0015, 0.0763])

# Step 1: Find the absolute maximum value (the scale)
abs_max = weight.abs().max()  # 0.1247

# Step 2: Compute the scale factor (maps float range → integer range)
n_bits = 8
qmax = 2 ** (n_bits - 1) - 1  # 127 for INT8
scale = abs_max / qmax         # 0.1247 / 127 ≈ 0.000982

# Step 3: Quantize — divide by scale, round to nearest integer
quantized = torch.round(weight / scale).to(torch.int8)
# Result: tensor([  3, -91, 127,  -2,  78], dtype=int8)

# Step 4: Dequantize — multiply back by scale (introduces small error)
dequantized = quantized.float() * scale
# Result: tensor([ 0.0029, -0.0894,  0.1247, -0.0020,  0.0766])
# Compare: tensor([ 0.0032, -0.0891,  0.1247, -0.0015,  0.0763])
# Error is tiny — the "rounding" barely matters!
```

The key insight: you store only the **quantized integers** (1 byte each for INT8, 4 bits each for INT4) plus one **scale factor** per group of weights. During inference, you dequantize on-the-fly just before each matrix multiplication.

## 2. Memory Layout — Bit-Width Reduction

Here's what happens to a single layer's weight matrix when you quantize to different bit widths:

```
┌────────────────────────────────────────────────────────────────────┐
│          Memory Layout: Quantization Bit-Width Reduction           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Original FP32 weight: 3.14159265                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Sign │ Exponent (8 bits) │   Mantissa (23 bits)             │   │
│  │  0   │  1 0 0 0 0 0 0 0 │ 1 0 0 1 0 0 1 0 0 0 0 1 1 1 1  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  Total: 32 bits (4 bytes) per weight                               │
│                                                                    │
│  FP16 quantized: 3.14063                                           │
│  ┌───────────────────────────────────┐                             │
│  │ Sign │ Exp (5) │ Mantissa (10)    │                             │
│  │  0   │ 1 0 0 0 │ 0 1 0 0 1 0 0   │                             │
│  └───────────────────────────────────┘                             │
│  Total: 16 bits (2 bytes) — 2× smaller                             │
│                                                                    │
│  INT8 quantized: 3 (with scale=0.0247)                             │
│  ┌───────────────────┐                                             │
│  │  0 0 0 0 0 0 1 1  │                                             │
│  └───────────────────┘                                             │
│  Total: 8 bits (1 byte) — 4× smaller                              │
│                                                                    │
│  INT4 quantized: 3 (with scale=0.209)                              │
│  ┌─────────┐                                                       │
│  │ 0 0 1 1 │                                                       │
│  └─────────┘                                                       │
│  Total: 4 bits (0.5 bytes) — 8× smaller                           │
│                                                                    │
│  ─────────────────────────────────────────────────                 │
│  Same weight value (≈3.14). Each level trades precision            │
│  for memory. The rounding error grows but stays small.             │
└────────────────────────────────────────────────────────────────────┘
```

For a full model with billions of weights, the savings multiply:

```
SmolLM2-1.7B at different precisions:
  FP32:  1.7B × 4 bytes = 6.8 GB
  FP16:  1.7B × 2 bytes = 3.4 GB
  INT8:  1.7B × 1 byte  = 1.7 GB
  INT4:  1.7B × 0.5 B   = 0.85 GB  ← fits easily in 2GB VRAM!
```

## 3. Quantization Approaches Compared

| Bit Width | Memory (1.7B model) | Quality Loss | Speed Impact | Use Case |
|:---------:|:-------------------:|:------------:|:------------:|----------|
| FP32 (32-bit) | 6.8 GB | None (baseline) | Baseline | Training, reference testing |
| FP16 (16-bit) | 3.4 GB | Negligible | ~1.5–2× faster | Standard GPU inference |
| INT8 (8-bit) | 1.7 GB | Very low (~0.1 PPL) | ~2× faster | Balanced quality/speed |
| INT4 (4-bit) | 0.85 GB | Low (~0.3–1.0 PPL) | ~2–4× faster | Memory-constrained, edge |
| INT2 (2-bit) | 0.43 GB | High (~5+ PPL) | Fastest | Experimental only |

**PPL** = perplexity increase over the FP16 baseline. Lower is better — a perplexity increase of 0.3 means the model is slightly less confident in its predictions but still produces coherent, useful text.

## 4. Types of Quantization

Not all quantization is created equal. The two major categories:

**Post-Training Quantization (PTQ)** — You take a fully trained FP16 model and quantize it after the fact. No additional training needed. This is what `bitsandbytes` does when you load a model with `load_in_4bit=True`. It's fast and easy but slightly less optimal.

**Quantization-Aware Training (QAT)** — The model is trained (or fine-tuned) with quantization built into the forward pass. The model learns to be robust to rounding errors during training. Produces higher quality at the same bit width but requires a training step. Methods like GPTQ and AWQ use calibration data to achieve QAT-like quality without full retraining.

```python
# Post-Training Quantization with bitsandbytes — just one extra argument!
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load directly in 4-bit — no separate quantization step needed
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — optimal for weights
    bnb_4bit_compute_dtype="float16",  # Compute in FP16 for speed
)

model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
# Model now uses ~0.85 GB VRAM instead of 3.4 GB!
```

## 5. Measuring Quality Loss — Perplexity

How do you know if quantization hurt your model? The standard metric is **perplexity** — a measure of how surprised the model is by a text sequence. Lower perplexity means the model predicted the text more confidently.

- **Perplexity = 1.0** — The model predicted every token perfectly (impossible in practice)
- **Perplexity ≈ 5–10** — Excellent: the model is very confident and usually correct
- **Perplexity ≈ 10–20** — Good: slight uncertainty but coherent outputs
- **Perplexity > 50** — Poor: the model is often confused

The key question: "How much does perplexity increase when I quantize?" If FP16 perplexity is 8.5 and INT4 perplexity is 9.1, that's a 0.6 increase — barely noticeable in real-world outputs. If INT2 perplexity jumps to 45, the model is broken.

## 6. Connection to What You've Already Seen

On **Day 5** you explored precision formats (FP32, FP16, BF16) and measured memory usage at each level. Quantization takes that idea further — going below 16 bits into INT8 and INT4 territory where the savings become dramatic.

On **Day 6** you used llama.cpp with a GGUF file that had `Q4_K_M` in its name. That "Q4" means 4-bit quantized — you were already running a quantized model! The GGUF format bakes quantization directly into the file format so llama.cpp can load and run quantized models without any additional conversion at inference time.

Today you'll understand the theory behind what those tools do automatically, and you'll measure the quality trade-off yourself.

## Setup

### Prerequisites

- Python 3.10+
- ~4 GB disk space (for model download)
- **NVIDIA GPU with CUDA** required for `bitsandbytes` quantization
  - Minimum: GPU with Compute Capability 6.0+ (GTX 1060 or newer)
  - If no GPU available, the script prints expected results and explains why GPU is needed

### Install dependencies

```bash
pip install -r requirements.txt
```

### GPU-specific notes

`bitsandbytes` is an NVIDIA-only library. It requires:
- NVIDIA GPU with Compute Capability 6.0+
- CUDA toolkit (usually installed with PyTorch's CUDA build)
- Linux or Windows (macOS is NOT supported)

If you're on macOS or don't have an NVIDIA GPU, the script will detect this and print the expected output with an explanation. You can still read the code and understand the concepts — the measurements shown are representative of what you'd see on a GPU machine.

### CPU-only alternative

If you don't have a supported GPU, you can still explore quantization concepts:
```bash
# On Day 6 you ran quantized GGUF models on CPU via llama.cpp
# That's the CPU-friendly approach to quantized inference
# Today's lab specifically measures bitsandbytes (GPU-only) for comparison
```

## What You'll Build

A Python script that loads `SmolLM2-1.7B-Instruct` at multiple quantization levels (FP16, INT8, INT4) using `bitsandbytes`, computes perplexity on a fixed text sample at each level, and prints a comparison table showing the quality-vs-memory trade-off quantitatively.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Quantization rounds model weights from high-precision floats (32/16 bits) to low-precision integers (8/4 bits), shrinking models by 2–8× with minimal quality loss
- The "rounding prices" intuition holds: $3.99 → $4 loses almost nothing, and neural network weights behave the same way
- Perplexity is the standard metric for measuring quantization quality — small increases (0.1–1.0) are acceptable, large jumps (5+) mean the model is damaged
- `bitsandbytes` makes quantization trivial: add `load_in_4bit=True` to your existing `from_pretrained()` call
- Post-Training Quantization (PTQ) is fast and easy; Quantization-Aware methods (GPTQ, AWQ) produce slightly better quality at the same bit width
- 4-bit quantization is the current sweet spot — it achieves 8× compression with barely measurable quality degradation on most tasks
