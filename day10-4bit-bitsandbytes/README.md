# Day 10: 4-bit Inference with bitsandbytes

## The Concept

On Day 8 you learned the theory behind quantization — reducing the number of bits per parameter to shrink models. Today you'll put that theory into practice with **bitsandbytes**, a library that lets you load any Hugging Face model in 4-bit precision directly through the familiar `from_pretrained()` pattern you learned on Day 3. No conversion step, no separate files — just one extra configuration object and your 14 GB model fits in 4 GB of VRAM.

Think of it like a ZIP file. When you compress a folder, the content is still there — every file, every byte of meaning — but it takes up dramatically less space on disk. The trade-off? A tiny overhead to decompress when you need to read something. bitsandbytes works the same way: it compresses model weights from 16-bit floats down to 4-bit integers using a technique called **NF4** (Normal Float 4-bit), keeping model quality remarkably intact while slashing VRAM requirements by 75%. The "decompression" happens on-the-fly during each forward pass, adding a small latency cost for a massive memory savings.

Why does this matter? A 7-billion parameter model in FP16 requires ~14 GB of VRAM — more than a consumer RTX 3080 (10 GB) can handle. The same model in NF4 needs only ~4 GB, comfortably fitting on a 6 GB card or even running alongside other applications on a 12 GB card. This is the technique that made running large models on consumer hardware practical. On Day 8 you measured perplexity at different bit widths — today you'll see the real-world VRAM savings and confirm that output quality remains usable.

## 1. How NF4 Quantization Works

bitsandbytes implements **NF4** — a quantization scheme optimized for neural network weights. Unlike naive rounding (round each float to the nearest integer), NF4 uses a non-uniform mapping designed around the fact that neural network weights follow a roughly normal distribution.

Here's the key insight: most weights cluster near zero. A uniform 4-bit grid wastes precision on values that rarely appear. NF4 places more quantization levels near zero where the density is highest, giving you better effective precision where it matters most.

The process at load time:

1. Read each weight tensor from disk in FP16 (the original format)
2. Compute per-block quantization constants (blocks of 64 weights)
3. Map each FP16 value to the nearest NF4 level (one of 16 values)
4. Store the 4-bit indices + the FP16 scaling constants

At inference time (each forward pass):

1. Read the 4-bit indices and scaling constants
2. Dequantize back to FP16/BF16 for matrix multiplication
3. Run the computation as normal
4. Discard the dequantized values (only 4-bit stays in memory)

This "compress in memory, decompress for compute" pattern is why the ZIP analogy fits — the file stays small, you only expand what you need in the moment.

## 2. Memory Layout: FP16 vs NF4

```
┌──────────────────────────────────────────────────────────────┐
│        Memory Layout: Qwen2.5-7B-Instruct in VRAM           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  FP16 (16 bits per parameter):                              │
│  ████████████████████████████████████████  ~14.2 GB          │
│                                                              │
│  NF4 (4 bits per parameter + overhead):                     │
│  ██████████                                ~4.2 GB           │
│                                                              │
│  ──────────────────────────────────────────────────────      │
│  Savings:  ~10 GB freed  │  ~70% VRAM reduction             │
│                                                              │
│  Breakdown of NF4 memory:                                    │
│  ┌────────────────────────────────────────┐                  │
│  │ 4-bit weights:     ~3.5 GB (7B × 4÷8) │                  │
│  │ Scaling constants:  ~0.5 GB (FP16)     │                  │
│  │ Activations/KV:     ~0.2 GB (varies)   │                  │
│  └────────────────────────────────────────┘                  │
│                                                              │
│  Same model. Same knowledge. 70% less VRAM.                  │
└──────────────────────────────────────────────────────────────┘
```

The overhead from scaling constants (about 0.5 GB) is why you don't get a perfect 4× reduction. Each block of 64 weights shares one FP16 scaling factor, so the total is approximately `weights × 4/8 + weights × 16/8 / 64` — the second term is the per-block constant overhead.

## 3. Loading a Model in 4-bit

bitsandbytes integrates directly with Hugging Face Transformers through the `BitsAndBytesConfig` class. You create a configuration specifying 4-bit loading, then pass it to `from_pretrained()` — the same function you've been using since Day 3:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NF4 is the default and best option
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16 for speed
    bnb_4bit_use_double_quant=True,        # Quantize the quantization constants too
)

# Load model — same from_pretrained, just add quantization_config
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**Double quantization** (`bnb_4bit_use_double_quant=True`) applies a second round of quantization to the scaling constants themselves. This saves roughly 0.4 GB on a 7B model with negligible quality loss — essentially free savings.

## 4. Comparison Table

| Loading Mode | VRAM (7B Model) | Speed (tokens/sec) | Quality (vs FP16) | Use Case |
|:---:|:---:|:---:|:---:|---|
| FP16 | ~14.2 GB | Fastest | Baseline (100%) | Research, maximum quality |
| NF4 (single quant) | ~4.5 GB | 5-15% slower | ~99% | Inference on consumer GPUs |
| NF4 (double quant) | ~4.2 GB | 5-15% slower | ~98.5% | Memory-constrained inference |
| INT8 (load_in_8bit) | ~7.5 GB | 10-20% slower | ~99.5% | When 4-bit quality isn't enough |

**Key trade-offs:**
- FP16 gives you maximum speed and quality but needs expensive hardware
- NF4 with double quantization gives the best memory savings — you lose ~1.5% quality for 70% VRAM reduction
- INT8 is the middle ground — less savings but nearly lossless

## 5. When to Use bitsandbytes vs Other Formats

On Day 8 you learned about quantization theory, and you saw how bit-width affects model quality. Here's where bitsandbytes fits in the broader quantization landscape:

**bitsandbytes NF4** is best when:
- You want the simplest possible workflow (no pre-quantized model files needed)
- You're doing inference on a single GPU
- You need to quantize at load time without separate conversion steps
- You're experimenting with different models and don't want pre-quantized versions of each

**GGUF (Day 6)** is better when:
- You need CPU inference (bitsandbytes requires CUDA)
- You want the model as a single portable file
- You're running on laptops or machines without NVIDIA GPUs

## 6. GPU Requirement

**bitsandbytes requires an NVIDIA GPU with CUDA support.** This is non-negotiable — the 4-bit dequantization kernels run on CUDA, and there's no CPU fallback for the quantized inference path.

Minimum requirements:
- NVIDIA GPU with Compute Capability 6.0+ (GTX 1060 or newer)
- CUDA 11.8+ installed
- 6+ GB VRAM for a 7B model in NF4

If you don't have a CUDA GPU, you have two options:
1. **Use a cloud GPU** — Google Colab (free tier has T4 GPUs with 16 GB VRAM), Lambda Labs, RunPod, or vast.ai
2. **Use GGUF quantization instead** — as covered on Day 6, llama.cpp runs quantized models on CPU. You won't get bitsandbytes-specific features, but you'll still run 4-bit models

The lab script includes a GPU check that detects whether CUDA is available and prints expected output for CPU-only users so you can still understand the workflow.

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support and 6+ GB VRAM (see GPU Requirement section above)
- CUDA toolkit 11.8+ installed
- ~15 GB disk space (for model downloads — FP16 version needed for comparison)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify CUDA availability

Before running the lab, confirm PyTorch can see your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
```

If this prints "No GPU detected," revisit the GPU Requirement section above.

### Model download

The script uses `Qwen/Qwen2.5-7B-Instruct` — a 7-billion parameter model released under the Apache 2.0 license. It's ungated (no account approval needed) and downloads automatically on first run (~14 GB for FP16). The 4-bit version is created on-the-fly during loading — no separate download required.

## What You'll Build

A Python script that loads the same model twice — first in FP16, then in 4-bit NF4 — measures the VRAM usage for each, and prints the percentage of VRAM saved. You'll see exactly how much memory bitsandbytes frees up, confirming the theory from Day 8 with concrete numbers from your own GPU.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- bitsandbytes lets you load any Hugging Face model in 4-bit precision with one extra config — no model conversion, no separate files
- NF4 (Normal Float 4) uses a non-uniform quantization grid optimized for the normal distribution of neural network weights
- A 7B model drops from ~14 GB to ~4 GB VRAM — roughly 70% savings — with minimal quality loss
- Double quantization saves an additional ~0.3 GB by quantizing the scaling constants themselves
- The trade-off is a 5-15% speed overhead from on-the-fly dequantization during each forward pass
- bitsandbytes requires NVIDIA CUDA GPUs — for CPU inference, use GGUF (Day 6) instead
