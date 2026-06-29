"""
Day 9: GGUF Conversion — Convert Your Own Model

This script downloads SmolLM2-1.7B from HuggingFace, converts it to GGUF format
using llama.cpp's conversion tools, quantizes at multiple levels, and prints the
output file sizes. Think of it as converting a recipe from metric to imperial —
same dish, different measures optimized for a different kitchen.

Model: HuggingFaceTB/SmolLM2-1.7B (ungated, Apache 2.0)
Key metric: output file size (MB) per quantization level
"""

import os
import sys
import subprocess
import time

# ─── Configuration ───────────────────────────────────────────────────────────
# SmolLM2-1.7B is small (1.7B params), ungated, and Apache 2.0 licensed.
# No token needed — anyone can download it without license acceptance.
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
MODEL_DIR = "./SmolLM2-1.7B"
OUTPUT_DIR = "./gguf_output"

# llama.cpp paths — these must exist (clone + build llama.cpp first)
LLAMA_CPP_DIR = "./llama.cpp"
CONVERT_SCRIPT = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
QUANTIZE_BIN = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")

# Quantization levels to produce — from largest (best quality) to smallest
QUANT_LEVELS = ["Q8_0", "Q4_K_M", "Q2_K"]

start_time = time.perf_counter()

# ─── Step 1: Verify llama.cpp tools are available ────────────────────────────
# The conversion pipeline depends on two tools from the llama.cpp repo:
# 1. convert_hf_to_gguf.py — Python script that reformats HF models to GGUF
# 2. llama-quantize — compiled C++ binary that applies bit-width reduction
print("=" * 60)
print("STEP 1: Verifying llama.cpp tools")
print("=" * 60)

if not os.path.isfile(CONVERT_SCRIPT):
    print(f"\nERROR: Convert script not found at: {CONVERT_SCRIPT}")
    print("\nTo fix this, clone and set up llama.cpp:")
    print("  git clone https://github.com/ggerganov/llama.cpp.git")
    print("  cd llama.cpp")
    print("  pip install -r requirements/requirements-convert_hf_to_gguf.txt")
    print("  cmake -B build")
    print("  cmake --build build --config Release --target llama-quantize")
    print("  cd ..")
    sys.exit(1)

if not os.path.isfile(QUANTIZE_BIN):
    # On Windows the binary might have a .exe extension
    quantize_win = QUANTIZE_BIN + ".exe"
    if os.path.isfile(quantize_win):
        QUANTIZE_BIN = quantize_win
    else:
        print(f"\nERROR: llama-quantize binary not found at: {QUANTIZE_BIN}")
        print("\nTo fix this, build llama.cpp:")
        print("  cd llama.cpp")
        print("  cmake -B build")
        print("  cmake --build build --config Release --target llama-quantize")
        print("  cd ..")
        sys.exit(1)

print(f"  Convert script: {CONVERT_SCRIPT} ✓")
print(f"  Quantize binary: {QUANTIZE_BIN} ✓")

# ─── Step 2: Download the source model ───────────────────────────────────────
# snapshot_download fetches all model files: safetensors weights, config.json,
# tokenizer files, etc. This is the "raw material" we'll convert to GGUF.
print("\n" + "=" * 60)
print("STEP 2: Downloading source model")
print("=" * 60)

from huggingface_hub import snapshot_download

print(f"  Model: {MODEL_ID}")
print(f"  Destination: {MODEL_DIR}")
print("  Downloading... (this may take a few minutes on first run)")

model_path = snapshot_download(
    MODEL_ID,
    local_dir=MODEL_DIR,
)

# Calculate total size of downloaded model files
total_source_bytes = 0
for root, dirs, files in os.walk(model_path):
    for f in files:
        total_source_bytes += os.path.getsize(os.path.join(root, f))

source_size_mb = total_source_bytes / (1024 * 1024)
print(f"  Download complete: {model_path}")
print(f"  Total source model size: {source_size_mb:.1f} MB")

# ─── Step 3: Convert to FP16 GGUF ───────────────────────────────────────────
# This step reformats the model from HuggingFace safetensors into GGUF container.
# The weights stay at FP16 precision — no quantization yet. This produces our
# baseline file that all subsequent quantizations start from.
print("\n" + "=" * 60)
print("STEP 3: Converting to FP16 GGUF (baseline)")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
fp16_output = os.path.join(OUTPUT_DIR, "SmolLM2-1.7B-F16.gguf")

convert_cmd = [
    sys.executable, CONVERT_SCRIPT,
    MODEL_DIR,
    "--outfile", fp16_output,
    "--outtype", "f16",
]

print(f"  Running: {' '.join(convert_cmd)}")
print("  Converting... (reformats safetensors → GGUF container)")

result = subprocess.run(convert_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"\n  ERROR: Conversion failed!")
    print(f"  stderr: {result.stderr[:500]}")
    sys.exit(1)

fp16_size_mb = os.path.getsize(fp16_output) / (1024 * 1024)
print(f"  Output: {fp16_output}")
print(f"  FP16 GGUF size: {fp16_size_mb:.1f} MB")

# ─── Step 4: Quantize at multiple levels ─────────────────────────────────────
# Each quantization level applies different bit-width compression to the FP16
# baseline. Lower bits = smaller file = faster inference = slightly less quality.
# The llama-quantize binary handles the math — we just specify the level.
print("\n" + "=" * 60)
print("STEP 4: Quantizing at multiple levels")
print("=" * 60)

# Store results for the summary table
results = [("F16", fp16_size_mb, 1.0)]

for quant_level in QUANT_LEVELS:
    output_file = os.path.join(OUTPUT_DIR, f"SmolLM2-1.7B-{quant_level}.gguf")

    quantize_cmd = [
        QUANTIZE_BIN,
        fp16_output,
        output_file,
        quant_level,
    ]

    print(f"\n  Quantizing to {quant_level}...")
    print(f"  Running: {' '.join(quantize_cmd)}")

    result = subprocess.run(quantize_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Quantization to {quant_level} failed!")
        print(f"  stderr: {result.stderr[:300]}")
        continue

    quant_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    compression = fp16_size_mb / quant_size_mb

    results.append((quant_level, quant_size_mb, compression))
    print(f"  {quant_level} size: {quant_size_mb:.1f} MB (compression: {compression:.1f}x)")

# ─── Step 5: Print summary table ────────────────────────────────────────────
# This is the key quantitative output — a side-by-side comparison of all
# quantization levels with their file sizes and compression ratios.
print("\n" + "=" * 60)
print("SUMMARY: GGUF Quantization Results for SmolLM2-1.7B")
print("=" * 60)

print(f"\n  Source model: {MODEL_ID}")
print(f"  Source size:  {source_size_mb:.1f} MB (safetensors + tokenizer + config)")
print()
print(f"  {'Level':<10} {'Size (MB)':<12} {'Compression':<14} {'Savings'}")
print(f"  {'─' * 10} {'─' * 12} {'─' * 14} {'─' * 12}")

for level, size_mb, compression in results:
    savings_pct = (1 - size_mb / fp16_size_mb) * 100
    if level == "F16":
        print(f"  {level:<10} {size_mb:<12.1f} {'baseline':<14} {'—'}")
    else:
        print(f"  {level:<10} {size_mb:<12.1f} {compression:<14.1f}x {savings_pct:.0f}% smaller")

# ─── Final timing ───────────────────────────────────────────────────────────
elapsed = time.perf_counter() - start_time
print(f"\n  Total conversion time: {elapsed:.1f} seconds")
print(f"  (Excludes initial model download)")
print("\n  Key insight: Same model, same knowledge, dramatically different sizes.")
print("  Q4_K_M gives you ~70% size reduction with minimal quality loss.")
print("=" * 60)
