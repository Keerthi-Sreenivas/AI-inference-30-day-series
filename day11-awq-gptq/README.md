# Day 11: AWQ & GPTQ — GPU Quantization Formats

## The Concept

On Day 8 you learned the theory behind quantization — shrinking model weights from 16-bit floats down to 4-bit integers to save memory and boost throughput. On Day 10 you put that theory into practice with bitsandbytes, which quantizes on-the-fly when you load the model. Today you'll meet two formats that take a fundamentally different approach: **AWQ** (Activation-aware Weight Quantization) and **GPTQ** (Generative Pre-trained Transformer Quantization). Instead of quantizing at load time, these methods produce pre-quantized model files that are ready to run at full speed the moment you load them — no conversion step, no runtime overhead.

Think of it like audio encoding. bitsandbytes is like streaming raw audio and compressing it in real time — it works, but there's CPU overhead at every step. AWQ and GPTQ are like distributing a pre-encoded FLAC or MP3 file: the heavy encoding work happened once (during a calibration phase), and playback is instant. Different encodings, same song — you get the same model knowledge at a fraction of the original size, with the compression math baked into the weights themselves.

Both AWQ and GPTQ require a GPU for inference because their quantized weight layouts are designed specifically for GPU tensor operations. If you're on a CPU-only machine, you can still follow along with the README to understand the concepts, and the lab script includes a graceful fallback message pointing you back to Day 6's GGUF format for CPU-based quantized inference.

## 1. How Pre-Quantized Models Work

Unlike bitsandbytes (Day 10), which quantizes weights every time you call `from_pretrained()`, AWQ and GPTQ perform quantization **once** using a small calibration dataset. This one-time process determines which weights are most important to preserve at higher precision and which can be aggressively rounded. The result is a model checkpoint that's already in 4-bit format on disk.

```
┌─────────────────────────────────────────────────────────────────┐
│         Pre-Quantization Pipeline (One-Time Process)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full-Precision Model (FP16)                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────────┐                                         │
│  │ Calibration Data   │  Small dataset (128–512 samples)        │
│  │ (representative    │  Used to measure activation patterns    │
│  │  input samples)    │  and identify important weights         │
│  └────────┬───────────┘                                         │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────────────────────────┐         │
│  │         Quantization Algorithm                      │         │
│  │                                                    │         │
│  │  AWQ: Protects high-activation channels            │         │
│  │       → scales important weights UP before quant   │         │
│  │       → minimal quality loss on critical paths     │         │
│  │                                                    │         │
│  │  GPTQ: Layer-by-layer optimal rounding             │         │
│  │       → minimizes reconstruction error per layer   │         │
│  │       → uses Hessian information for decisions     │         │
│  └────────────────────┬───────────────────────────────┘         │
│                       │                                         │
│                       ▼                                         │
│  ┌────────────────────────────┐                                 │
│  │  Pre-Quantized Checkpoint  │  4-bit weights on disk          │
│  │  (upload to HuggingFace)   │  ~4 GB for an 8B model          │
│  └────────────────────────────┘                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Once the pre-quantized checkpoint exists on HuggingFace, you load it directly — no calibration dataset needed, no quantization delay. The weights come off disk already in their compressed format.

## 2. AWQ vs GPTQ vs bitsandbytes — Architecture Comparison

Each approach takes a different path from model weights to inference output. Here's how the three pipelines compare:

```
┌─────────────────────────────────────────────────────────────────┐
│     Quantization Pipelines: AWQ vs GPTQ vs bitsandbytes         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  AWQ Pipeline:                                                  │
│  ┌─────────┐    ┌────────────────┐    ┌──────────────────┐     │
│  │ Calibr. │───▶│ Scale+Quantize │───▶│ Pre-Quant Model  │──┐  │
│  │ Dataset │    │ (activation-   │    │ (.safetensors)   │  │  │
│  └─────────┘    │  aware scaling)│    └──────────────────┘  │  │
│                 └────────────────┘                           │  │
│                                                             ▼  │
│  GPTQ Pipeline:                                    ┌──────────┐│
│  ┌─────────┐    ┌────────────────┐    ┌─────────┐ │  GPU     ││
│  │ Calibr. │───▶│ Layer-by-layer │───▶│ Pre-Q.  │─▶│ Inference││
│  │ Dataset │    │ (Hessian-based │    │ Model   │ │  (fast)  ││
│  └─────────┘    │  rounding)     │    └─────────┘ └──────────┘│
│                 └────────────────┘                     ▲       │
│                                                       │       │
│  bitsandbytes Pipeline:                               │       │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────┘       │
│  │ FP16 Model   │───▶│ Quantize at    │───▶│ Dequant on      │
│  │ (full-size   │    │ load time      │    │ each forward    │
│  │  on disk)    │    │ (every load)   │    │ pass (overhead) │
│  └──────────────┘    └────────────────┘    └─────────────────┘│
│                                                                 │
│  Key Difference:                                                │
│  • AWQ/GPTQ: quantize ONCE → distribute pre-quantized model    │
│  • bitsandbytes: quantize EVERY TIME you load the model         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Format Comparison Table

| Format | Bit Width | GPU Required | Speed | Quality Loss | Best For |
|--------|:---------:|:---:|:---:|:---:|---|
| GGUF Q4_K_M | 4-bit | No | Fast (CPU) | Low | Local/laptop inference (Day 6) |
| bitsandbytes NF4 | 4-bit | Yes | Medium | Very Low | Training + inference (Day 10) |
| AWQ | 4-bit | Yes | Fast | Very Low | Production GPU serving |
| GPTQ | 4-bit | Yes | Fast | Low | Production GPU serving |
| GPTQ (8-bit) | 8-bit | Yes | Medium | Minimal | Quality-sensitive applications |

**When to choose which:**

- **GGUF** — You're on a CPU or laptop. No GPU? Use GGUF (Day 6).
- **bitsandbytes** — You want to quantize any model on the fly without needing a pre-quantized checkpoint. Great for experimentation (Day 10).
- **AWQ** — You want maximum inference speed on GPU with minimal quality loss. Best for production serving with vLLM.
- **GPTQ** — Similar to AWQ, slightly different quality profile. More pre-quantized models available on HuggingFace.

## 4. The Library Landscape (2025 Update)

The tooling for AWQ and GPTQ has changed significantly. Here's what you need to know:

- **`autoawq`** — Deprecated and archived in May 2025. Do not use it for new projects.
- **`auto-gptq`** — Also deprecated, superseded by the same successor library.
- **`gptqmodel`** (v7.0+) — The active successor to both `autoawq` and `auto-gptq`. It handles both AWQ and GPTQ formats in a single package with better performance and ongoing maintenance.
- **`transformers`** — Can load pre-quantized AWQ/GPTQ models directly via `from_pretrained()` when the model's `config.json` specifies the quantization format.

For this lab, we use `gptqmodel>=7.0.0` which provides a unified interface for loading and running both AWQ and GPTQ models.

```python
# Loading a pre-quantized GPTQ model with gptqmodel
from gptqmodel import GPTQModel

model = GPTQModel.load("TheBloke/Llama-3.1-8B-GPTQ")
```

## 5. Pre-Quantized Models on HuggingFace

You don't need to quantize models yourself. The community provides thousands of pre-quantized checkpoints on HuggingFace. For this lab, we use:

- **GPTQ variant:** `TechxGenus/Qwen2.5-7B-Instruct-GPTQ-Int4` (ungated, Apache-2.0)
- **AWQ variant:** `Qwen/Qwen2.5-7B-Instruct-AWQ` (ungated, Apache-2.0)

These are pre-quantized versions of Qwen2.5-7B-Instruct — the same base model at the same quality level, just encoded with different quantization algorithms. Using ungated Qwen2.5 variants avoids the license acceptance step required for Llama models.

**Alternative (requires license):** If you prefer Llama-3.1-8B, pre-quantized variants exist but require accepting Meta's Llama Community License at `https://huggingface.co/meta-llama/Llama-3.1-8B`.

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM (RTX 3070 or better recommended)
- CUDA 11.8+ installed
- Familiarity with Day 8 (quantization theory) and Day 10 (bitsandbytes)

### GPU Requirement

AWQ and GPTQ models require a CUDA-capable GPU. Their weight layouts are optimized for GPU tensor cores and cannot run on CPU.

**No GPU? Here's your fallback:** Use GGUF format with llama.cpp from Day 6. GGUF provides 4-bit quantized inference on CPU. The concepts (pre-quantization, calibration, quality-vs-size tradeoff) are the same — only the target hardware differs.

### Install dependencies

```bash
pip install -r requirements.txt
```

For CPU-only machines (to understand the code without running it):

```bash
pip install transformers torch  # Read-only exploration
```

### License Notice (if using Llama-3.1-8B variants)

If you choose to use Llama-3.1-8B pre-quantized models instead of Qwen2.5-7B:

1. Visit `https://huggingface.co/meta-llama/Llama-3.1-8B`
2. Accept the Llama 3.1 Community License Agreement
3. Set your HuggingFace token: `export HF_TOKEN=your_token_here`

The default lab uses ungated Qwen2.5-7B variants — no license acceptance needed.

## What You'll Build

A Python script that loads two pre-quantized models (one AWQ, one GPTQ) from HuggingFace, generates text with each using the same prompt, measures inference latency in milliseconds for both formats, and prints a side-by-side comparison so you can see how the two quantization approaches perform on the same hardware.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- AWQ and GPTQ produce pre-quantized model files — the heavy calibration work happens once, not at every load
- Both formats require a GPU; for CPU inference, use GGUF from Day 6
- AWQ preserves weights that correspond to high activations; GPTQ minimizes layer-by-layer reconstruction error — different math, similar results
- `autoawq` and `auto-gptq` are deprecated (May 2025); use `gptqmodel>=7.0.0` as the unified successor
- Pre-quantized checkpoints on HuggingFace let you skip calibration entirely — just download and run
- For production GPU serving (vLLM, TGI), AWQ and GPTQ models load faster and run with less overhead than runtime-quantized alternatives
