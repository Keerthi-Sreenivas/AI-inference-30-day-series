# Day 9: GGUF Quantization — Convert Your Own Model

## The Concept

On Day 6, you downloaded a pre-made GGUF file and ran it with llama.cpp. Someone else had already converted and quantized that model for you — you just consumed the finished product. Today you'll become the producer. You'll take a standard HuggingFace model, run it through the llama.cpp conversion pipeline, and output GGUF files at multiple quantization levels. You'll see exactly how a 3.4 GB model shrinks to under 1 GB while keeping most of its intelligence.

Think of it like converting a recipe from metric to imperial measurements. The dish doesn't change — the same ingredients, the same cooking steps, the same flavor at the end. But the numbers are written differently, optimized for a different kitchen. A safetensors model is the "metric" version: precise, standard, designed for the PyTorch/CUDA kitchen. A GGUF file is the "imperial" version: reformatted for the llama.cpp kitchen where CPU inference is the norm. The conversion process doesn't retrain anything — it just repackages the weights into a format that runs efficiently on different hardware.

On Day 8, you learned the theory behind quantization — how reducing bit-width trades a small amount of precision for dramatic memory savings. Today you'll put that theory into practice. Instead of loading a pre-quantized model, you'll control the entire pipeline: download the original full-precision weights, convert them to GGUF format, apply various quantization levels, and compare the resulting file sizes. This is the same process that produces the GGUF files you find on HuggingFace Hub — now you can do it yourself for any model.

## 1. The Conversion Pipeline

The llama.cpp project provides Python scripts that handle the heavy lifting. The conversion process is a three-stage pipeline: download the source model, convert to base GGUF (FP16), then quantize to smaller sizes.

```
┌─────────────────────────────────────────────────────────────────────┐
│           Data Flow: GGUF Conversion Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐                                                │
│  │  HuggingFace    │  Original model files:                         │
│  │  Hub Download   │  • model.safetensors (~3.4 GB for 1.7B)       │
│  │                 │  • config.json, tokenizer files                │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  convert_hf_    │  Reads safetensors + config.json               │
│  │  to_gguf.py     │  Restructures weights into GGUF container      │
│  │                 │  Embeds tokenizer vocabulary                    │
│  │  (llama.cpp)    │  Outputs FP16 GGUF baseline                    │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐     ┌───────────────────────────────────────┐  │
│  │  llama-quantize │     │  Output Files:                        │  │
│  │                 │────▶│                                       │  │
│  │  (llama.cpp     │     │  model-F16.gguf    (~3.4 GB)         │  │
│  │   binary)       │     │  model-Q8_0.gguf   (~1.8 GB)         │  │
│  │                 │     │  model-Q5_K_M.gguf (~1.2 GB)         │  │
│  │  Applies quant  │     │  model-Q4_K_M.gguf (~1.0 GB)         │  │
│  │  at each level  │     │  model-Q2_K.gguf   (~0.7 GB)         │  │
│  └─────────────────┘     └───────────────────────────────────────┘  │
│                                                                     │
│  Same model knowledge → progressively smaller files                 │
│  Each level trades a bit of quality for size reduction              │
└─────────────────────────────────────────────────────────────────────┘
```

The key insight: conversion and quantization are separate steps. First you reformat the model structure (safetensors → GGUF), then you compress the numerical precision (FP16 → Q8, Q5, Q4, Q2). You can stop at FP16 GGUF if you want maximum quality, or quantize aggressively if size matters more.

## 2. Quantization Levels Explained

Not all quantization is the same. The llama.cpp project offers over a dozen quantization methods, but these are the ones you'll encounter most often:

| Quant Level | Bits | Output Size (1.7B) | Quality vs FP16 | Best Use Case |
|-------------|:----:|:-------------------:|:---------------:|---------------|
| F16 | 16 | ~3.4 GB | 100% (baseline) | Maximum quality, still needs less memory than FP32 |
| Q8_0 | 8 | ~1.8 GB | ~99.9% | When you have space and want near-perfect quality |
| Q5_K_M | 5 | ~1.2 GB | ~99% | Good balance of size and quality for most uses |
| Q4_K_M | 4 | ~1.0 GB | ~98% | Best all-around choice for local inference |
| Q3_K_M | 3 | ~0.85 GB | ~95% | Tight hardware constraints, noticeable quality loss |
| Q2_K | 2 | ~0.7 GB | ~90% | Extreme compression, significant quality degradation |

The naming convention tells you what you're getting: `Q4` means 4-bit, `K` means K-quant method (smarter than naive quantization), and `M` means medium quality within that bit-width. Day 6 showed you how to read these names in filenames — now you'll produce them yourself.

The rule of thumb: **Q4_K_M is the sweet spot.** Below that, quality degrades noticeably. Above that, you get diminishing returns — Q8_0 is nearly identical to FP16 in output quality but still twice the size.

## 3. Why Convert Your Own Models?

You might wonder: why not just download pre-converted GGUF files from HuggingFace? There are several reasons to learn conversion yourself:

- **Fine-tuned models** — If you fine-tune a model (Day 25 covers LoRA), you'll need to convert your custom weights to GGUF for deployment
- **New model releases** — When a new model drops, GGUF conversions may not exist yet. You can be first.
- **Custom quantization** — You might want a specific quant level that nobody has published
- **Reproducibility** — You control the exact conversion parameters rather than trusting an unknown quantizer

## 4. The Tools You Need

The conversion requires cloning the llama.cpp repository, which contains the conversion scripts and the `llama-quantize` binary:

```bash
# Clone llama.cpp (contains convert scripts)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install Python dependencies for the convert script
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# Build the quantize tool (requires cmake and a C++ compiler)
cmake -B build
cmake --build build --config Release --target llama-quantize
```

The conversion script (`convert_hf_to_gguf.py`) is pure Python — it reads safetensors files and writes GGUF format. The quantize tool is a compiled C++ binary that applies bit-width reduction. Together they form the full pipeline.

## 5. SmolLM2-1.7B — Our Conversion Target

We'll use `HuggingFaceTB/SmolLM2-1.7B` as our source model. This is the base (non-instruct) version of SmolLM2 — it's ungated, Apache 2.0 licensed, and small enough to convert quickly on any machine. No license acceptance required, no token needed.

Why the base model instead of the instruct variant? For GGUF conversion, it doesn't matter — the conversion process handles both identically. The base model demonstrates that you can convert *any* HuggingFace model, not just instruction-tuned ones.

```python
from huggingface_hub import snapshot_download

# Download the full model — safetensors, config, tokenizer, everything
model_path = snapshot_download(
    "HuggingFaceTB/SmolLM2-1.7B",
    local_dir="./SmolLM2-1.7B"
)
print(f"Model downloaded to: {model_path}")
```

## 6. Running the Conversion

Once you have the model files and llama.cpp cloned, the conversion is two commands:

```bash
# Step 1: Convert HuggingFace format → GGUF (FP16 baseline)
python llama.cpp/convert_hf_to_gguf.py ./SmolLM2-1.7B \
    --outfile SmolLM2-1.7B-F16.gguf \
    --outtype f16

# Step 2: Quantize to various levels
./llama.cpp/build/bin/llama-quantize SmolLM2-1.7B-F16.gguf SmolLM2-1.7B-Q8_0.gguf Q8_0
./llama.cpp/build/bin/llama-quantize SmolLM2-1.7B-F16.gguf SmolLM2-1.7B-Q5_K_M.gguf Q5_K_M
./llama.cpp/build/bin/llama-quantize SmolLM2-1.7B-F16.gguf SmolLM2-1.7B-Q4_K_M.gguf Q4_K_M
./llama.cpp/build/bin/llama-quantize SmolLM2-1.7B-F16.gguf SmolLM2-1.7B-Q2_K.gguf Q2_K
```

Each quantize command takes the FP16 GGUF as input and produces a smaller file. The process is fast — usually under a minute per quantization level for a 1.7B model.

## Setup

### Prerequisites

- Python 3.10+
- `git` (to clone llama.cpp)
- `cmake` and a C++ compiler (to build llama-quantize)
- ~8 GB disk space (source model + converted files)
- No GPU required — conversion runs entirely on CPU

### Install system dependencies

**macOS:**
```bash
# cmake and compiler are included with Xcode Command Line Tools
xcode-select --install
brew install cmake
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install build-essential cmake git
```

**Windows:**
```bash
# Install Visual Studio Build Tools or use WSL
# With WSL, follow the Linux instructions above
# Or install cmake from https://cmake.org/download/
```

### Install Python dependencies

```bash
pip install huggingface-hub transformers torch numpy sentencepiece
```

### Clone and build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt
cmake -B build
cmake --build build --config Release --target llama-quantize
cd ..
```

## What You'll Build

A Python script that downloads SmolLM2-1.7B from HuggingFace, converts it to GGUF format at multiple quantization levels using the llama.cpp tools, and prints the output file size in megabytes for each quantization level — giving you a concrete comparison of how much space each level saves.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- GGUF conversion is a two-step process: reformat (safetensors → GGUF) then quantize (FP16 → Q4, Q2, etc.)
- The llama.cpp repo provides both the conversion script (`convert_hf_to_gguf.py`) and the quantize binary
- Q4_K_M is the sweet spot for most use cases — ~70% size reduction with minimal quality loss
- You can convert any HuggingFace model to GGUF — not just popular ones with pre-made downloads
- Conversion runs on CPU and takes minutes, not hours — it's repackaging, not retraining
- The same model knowledge is preserved across all quantization levels — only the numerical precision changes (Day 8 theory in practice)
