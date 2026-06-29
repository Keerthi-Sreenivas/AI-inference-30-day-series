# Day 6: GGUF & llama.cpp — Running Models on CPU

## The Concept

Imagine you've ordered a massive piece of furniture online. It arrives as raw lumber — planks, screws, dowels all loose in an oversized crate. You'd need a full workshop (GPU, CUDA, PyTorch) just to assemble it. That's what a **safetensors** model file is: raw, unpackaged weights that need a heavy framework to load and run.

Now imagine that same furniture arrives as a flat-pack kit — pre-cut, labeled, with an Allen wrench included. You can assemble it in your living room with no power tools. That's **GGUF**: a self-contained model format designed to run efficiently on CPUs, laptops, and edge devices using **llama.cpp** as the lightweight runtime. No CUDA. No PyTorch. No GPU required.

On Day 1, you used Ollama to run a model locally. What you might not have realized is that Ollama uses GGUF files under the hood — it's llama.cpp wrapped in a friendly CLI. On Day 5, you explored precision formats (FP16, INT8, INT4). Today you'll see how GGUF combines quantization *and* packaging into a single portable file that runs anywhere, and you'll measure the actual tokens-per-second you get on bare CPU inference.

## 1. safetensors vs GGUF — Two Approaches to Model Distribution

The PyTorch/Transformers ecosystem stores models in safetensors format (or the older .bin files). These are raw weight tensors — you load them into a Python framework that handles computation. GGUF takes a different approach: it packages weights, tokenizer, and metadata into a single file optimized for llama.cpp's inference engine.

Here's how the two pipelines compare:

```
┌─────────────────────────────────────────────────────────────────┐
│        Architecture Comparison: safetensors vs GGUF             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SAFETENSORS PIPELINE (GPU-oriented)                            │
│  ┌───────────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │ HuggingFace   │───▶│ PyTorch /    │───▶│ GPU Inference  │   │
│  │ Hub (.safet.) │    │ Transformers │    │ (CUDA needed)  │   │
│  └───────────────┘    └──────────────┘    └────────────────┘   │
│        │                     │                    │              │
│   ~14 GB (7B FP16)    Heavy framework      Fast on GPU         │
│                        Needs Python          Slow on CPU         │
│                                                                 │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│                                                                 │
│  GGUF PIPELINE (CPU-friendly)                                   │
│  ┌───────────────┐    ┌──────────────┐    ┌────────────────┐   │
│  │ HuggingFace   │───▶│ llama.cpp /  │───▶│ CPU Inference  │   │
│  │ Hub (.gguf)   │    │ llama-cpp-py │    │ (No GPU needed)│   │
│  └───────────────┘    └──────────────┘    └────────────────┘   │
│        │                     │                    │              │
│   ~2 GB (3B Q4)       Lightweight C++      Runs anywhere        │
│   Single file          No Python deps      Also supports GPU    │
│                        Tokenizer included                        │
│                                                                 │
│  KEY DIFFERENCE: GGUF is self-contained.                        │
│  The tokenizer, model config, and quantized weights are all     │
│  packed into ONE file. No separate downloads, no framework.     │
└─────────────────────────────────────────────────────────────────┘
```

## 2. What Makes GGUF Special

GGUF (GPT-Generated Unified Format) was created by the llama.cpp project as a successor to the older GGML format. It solves three problems at once:

1. **Portability** — A single `.gguf` file contains everything needed to run inference: quantized weights, tokenizer vocabulary, model architecture metadata, and generation defaults. No Python environment required.

2. **Quantization built-in** — GGUF files come pre-quantized. When you see `Q4_K_M` in a filename, that means 4-bit quantization using the K-quant method (medium quality). The quantization is baked into the file format itself — no runtime conversion needed.

3. **CPU optimization** — llama.cpp uses hand-tuned SIMD instructions (AVX2, ARM NEON) to make CPU matrix multiplication fast. It's written in C/C++ with no Python overhead. The result: reasonable inference speed on consumer hardware without a GPU.

### Quantization Naming Convention

When you download a GGUF file, the filename tells you exactly what you're getting:

```
Llama-3.2-3B-Instruct-Q4_K_M.gguf
│         │          │   │ │ │
│         │          │   │ │ └── M = Medium (quality level)
│         │          │   │ └──── K = K-quant method
│         │          │   └────── 4 = 4-bit quantization
│         │          └────────── Q = Quantized
│         └───────────────────── 3B = 3 billion parameters
└─────────────────────────────── Model family
```

Common quality levels: `Q2_K` (smallest, lowest quality) → `Q4_K_M` (good balance) → `Q8_0` (largest, best quality).

## 3. Comparison Table

| Aspect | safetensors (FP16) | GGUF Q4_K_M | GGUF Q8_0 |
|--------|:------------------:|:-----------:|:---------:|
| **GPU Required** | Practically yes | No | No |
| **File Size (3B model)** | ~6 GB | ~2 GB | ~3.5 GB |
| **Speed (CPU)** | Very slow | Fast | Medium |
| **Speed (GPU)** | Fast | Fast (if supported) | Fast |
| **Portability** | Needs Python + PyTorch | Single file, any OS | Single file, any OS |
| **Quality** | Full precision | Good (~99% of FP16) | Excellent (~99.9%) |
| **Framework** | transformers/PyTorch | llama.cpp / Ollama | llama.cpp / Ollama |
| **Tokenizer** | Separate download | Included in file | Included in file |

**The trade-off:** GGUF Q4_K_M gives you a 3× smaller file that runs on CPU at usable speed, with barely perceptible quality loss. The "flat-pack" arrives ready to run — no GPU workshop needed.

## 4. llama.cpp and llama-cpp-python

**llama.cpp** is the C/C++ inference engine that runs GGUF files. It's the runtime — like how the JVM runs .jar files or Node runs .js files. You can use it directly from the command line, but for Python developers, there's a convenient wrapper:

**llama-cpp-python** gives you a Python API that calls llama.cpp under the hood. You get the performance of compiled C++ with the ergonomics of Python:

```python
from llama_cpp import Llama

# Load a GGUF file — that's it. No tokenizer download, no config.json,
# no separate model files. Everything is in this one file.
llm = Llama(
    model_path="./Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,       # Context window size
    n_threads=8,      # CPU threads to use
    verbose=False,    # Suppress llama.cpp debug output
)

# Generate text — the API handles prompt formatting internally
output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is GGUF?"}],
    max_tokens=200,
    temperature=0.7,
)

print(output["choices"][0]["message"]["content"])
```

Compare this to the Transformers pipeline from Day 3: no `AutoTokenizer`, no `AutoModelForCausalLM`, no `device_map`, no `.safetensors` download. One file, one import, one function call.

## 5. How Ollama Connects

On Day 1 you ran `ollama run llama3.2` and got a working chatbot. Here's what happened behind the scenes:

1. Ollama downloaded a GGUF file (the specific quantization depends on your hardware)
2. Ollama started its embedded llama.cpp server
3. Your chat messages were sent to that server via HTTP
4. llama.cpp performed inference on the GGUF model using your CPU (or GPU if available)

Today you're doing the same thing Ollama does — but directly, without the wrapper. This gives you control over thread count, context size, quantization choice, and other parameters that Ollama abstracts away.

## Setup

### Prerequisites

- Python 3.10+
- ~2.5 GB disk space (for the GGUF model download)
- No GPU required — the whole point is CPU inference
- A Hugging Face account with Llama license accepted (see below)

### License Notice — Llama 3.2

The model we use today (`Llama-3.2-3B-Instruct` in GGUF format) is released under the **Llama 3.2 Community License**. This requires:

1. **Accept the license** at [https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
2. **Set your Hugging Face token** as an environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```
3. The token is used only for downloading. Once the file is cached locally, you can run offline.

If you prefer to skip the license step, you can substitute any ungated GGUF model (e.g., a SmolLM2 GGUF) — the code structure is the same.

### Install dependencies

**macOS:**
```bash
pip install llama-cpp-python huggingface-hub
```

**Linux (CPU only):**
```bash
pip install llama-cpp-python huggingface-hub
```

**Linux (with NVIDIA GPU acceleration):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python huggingface-hub
```

**Windows:**
```bash
pip install llama-cpp-python huggingface-hub
```

> **Note:** On macOS with Apple Silicon, `llama-cpp-python` automatically uses Metal for GPU acceleration. On Linux/Windows without a GPU, it falls back to optimized CPU kernels (AVX2/AVX-512).

### Install from requirements.txt

```bash
pip install -r requirements.txt
```

## What You'll Build

A Python script that downloads a quantized GGUF model, loads it with `llama-cpp-python`, generates a response on your CPU, and measures the tokens-per-second throughput — giving you a concrete number for how fast quantized CPU inference actually is on your hardware.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- GGUF is a self-contained model format: weights + tokenizer + metadata in a single file, optimized for CPU inference
- llama.cpp is the C/C++ runtime engine that executes GGUF files — Ollama uses it internally (Day 1 connection)
- Quantization (like Q4_K_M) is baked into the GGUF file — no runtime conversion needed (Day 5 connection: this is where INT4 precision ends up in practice)
- A 3B-parameter model at Q4_K_M fits in ~2 GB and runs at 10–30 tokens/sec on modern CPUs — no GPU required
- The `llama-cpp-python` package gives you a clean Python API over the C++ engine with minimal overhead
- Trade-off: GGUF sacrifices some model quality (due to quantization) for dramatically better portability and lower hardware requirements
