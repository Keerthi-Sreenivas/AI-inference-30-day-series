"""
Day 6: GGUF & llama.cpp — Running Models on CPU

Download a quantized GGUF model, load it with llama-cpp-python, and measure
tokens-per-second on CPU. This demonstrates how GGUF + llama.cpp enables
fast inference on consumer hardware without a GPU.

Model: Llama-3.2-3B-Instruct-Q4_K_M.gguf (Llama 3.2 Community License)
Note: Requires license acceptance at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
"""

import os
import time
from huggingface_hub import hf_hub_download

# --- Configuration ---
# We use a community-quantized GGUF file of Llama-3.2-3B-Instruct.
# Q4_K_M means: 4-bit quantization, K-quant method, Medium quality.
# This gives a good balance between file size (~2 GB) and output quality.
REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Number of CPU threads to use for inference. Adjust based on your machine.
# A good default is (physical cores - 1) to leave one core for OS tasks.
N_THREADS = 8
N_CTX = 2048  # Context window size (max tokens the model can "see")

# --- Step 1: Download the GGUF model ---
# huggingface_hub handles caching automatically. The first run downloads ~2 GB;
# subsequent runs load from the local cache instantly.
print("=" * 70)
print("STEP 1: Download GGUF Model")
print("=" * 70)
print(f"Repository: {REPO_ID}")
print(f"File: {FILENAME}")
print(f"Quantization: Q4_K_M (4-bit, K-quant, Medium quality)")
print()

# Check for HF_TOKEN — needed for gated Llama models
if not os.environ.get("HF_TOKEN"):
    print("WARNING: HF_TOKEN environment variable not set.")
    print("The Llama-3.2 model requires license acceptance.")
    print("Set your token: export HF_TOKEN=your_token_here")
    print("Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
    print()

print("Downloading (or loading from cache)...")
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    token=os.environ.get("HF_TOKEN"),
)
print(f"Model cached at: {model_path}")

# Show the file size to illustrate how compact quantized models are
file_size_gb = os.path.getsize(model_path) / (1024**3)
print(f"File size: {file_size_gb:.2f} GB")
print(f"(Compare: the same model in FP16 safetensors would be ~6 GB)")
print()

# --- Step 2: Load the model with llama-cpp-python ---
# This is where the magic happens. llama-cpp-python wraps the C++ llama.cpp
# engine in a Python API. Loading is fast because GGUF files are designed for
# efficient memory-mapped loading — no Python tensor operations needed.
print("=" * 70)
print("STEP 2: Load Model with llama-cpp-python")
print("=" * 70)

try:
    from llama_cpp import Llama

    load_start = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,  # Suppress llama.cpp's internal debug output
    )
    load_elapsed = time.perf_counter() - load_start
    print(f"Model loaded successfully in {load_elapsed:.1f}s")
    print(f"Context window: {N_CTX} tokens")
    print(f"CPU threads: {N_THREADS}")
    print()
except Exception as e:
    print(f"ERROR: Failed to load GGUF model.")
    print(f"Details: {e}")
    print()
    print("Troubleshooting:")
    print("  1. Ensure llama-cpp-python is installed: pip install llama-cpp-python")
    print("  2. Verify the GGUF file downloaded completely (check file size)")
    print("  3. Try re-downloading: delete the cached file and run again")
    raise SystemExit(1)

# --- Step 3: Generate text (short response) ---
# We use the chat completion API which handles prompt formatting automatically.
# The GGUF file contains the tokenizer, so no separate tokenizer download is needed.
print("=" * 70)
print("STEP 3: Generate Text — Short Response (200 tokens)")
print("=" * 70)

messages_short = [
    {"role": "user", "content": "Explain what GGUF is and why it matters for running AI models locally."}
]

print(f"Prompt: {messages_short[0]['content']}")
print("Generating...\n")

gen_start = time.perf_counter()
response_short = llm.create_chat_completion(
    messages=messages_short,
    max_tokens=200,
    temperature=0.7,
)
gen_elapsed_short = time.perf_counter() - gen_start

# Extract the response text and count tokens
response_text_short = response_short["choices"][0]["message"]["content"]
tokens_short = response_short["usage"]["completion_tokens"]
tokens_per_sec_short = tokens_short / gen_elapsed_short

print(f"Response:\n{response_text_short}\n")
print(f"--- Metrics (Short Generation) ---")
print(f"Tokens generated: {tokens_short}")
print(f"Time elapsed: {gen_elapsed_short:.1f}s")
print(f"CPU Inference: {tokens_per_sec_short:.1f} tokens/sec")
print()

# --- Step 4: Generate text (longer response) ---
# Running a longer generation shows sustained throughput. The first few tokens
# are slower (prompt processing / "prefill"), then generation reaches steady state.
print("=" * 70)
print("STEP 4: Generate Text — Longer Response (400 tokens)")
print("=" * 70)

messages_long = [
    {"role": "user", "content": "Write a detailed comparison of running AI models on GPU vs CPU. Cover speed, cost, accessibility, and when each approach makes sense."}
]

print(f"Prompt: {messages_long[0]['content']}")
print("Generating...\n")

gen_start = time.perf_counter()
response_long = llm.create_chat_completion(
    messages=messages_long,
    max_tokens=400,
    temperature=0.7,
)
gen_elapsed_long = time.perf_counter() - gen_start

response_text_long = response_long["choices"][0]["message"]["content"]
tokens_long = response_long["usage"]["completion_tokens"]
tokens_per_sec_long = tokens_long / gen_elapsed_long

print(f"Response:\n{response_text_long}\n")
print(f"--- Metrics (Long Generation) ---")
print(f"Tokens generated: {tokens_long}")
print(f"Time elapsed: {gen_elapsed_long:.1f}s")
print(f"CPU Inference: {tokens_per_sec_long:.1f} tokens/sec")
print()

# --- Step 5: Summary ---
# Print a clear comparison table showing the key metrics from both runs.
# This gives the learner a quantitative sense of GGUF + CPU performance.
print("=" * 70)
print("SUMMARY: GGUF CPU Inference Performance")
print("=" * 70)
print(f"Model: {FILENAME}")
print(f"Size: {file_size_gb:.2f} GB (4-bit quantized)")
print(f"Threads: {N_THREADS} | Context: {N_CTX} tokens")
print()
print(f"{'Run':<20} {'Tokens':<10} {'Time (s)':<12} {'Tokens/sec':<12}")
print("-" * 54)
print(f"{'Short (200 max)':<20} {tokens_short:<10} {gen_elapsed_short:<12.1f} {tokens_per_sec_short:<12.1f}")
print(f"{'Long (400 max)':<20} {tokens_long:<10} {gen_elapsed_long:<12.1f} {tokens_per_sec_long:<12.1f}")
print("-" * 54)
print()
print("Key insights:")
print("  • GGUF + llama.cpp runs entirely on CPU — no GPU, no CUDA, no PyTorch")
print("  • 4-bit quantization keeps the model small (~2 GB) with minimal quality loss")
print("  • Typical CPU throughput: 10-30 tokens/sec (varies by hardware)")
print("  • This is the same engine Ollama uses under the hood (Day 1 connection)")
print("  • Longer generations reach steady-state speed after the initial prefill")
