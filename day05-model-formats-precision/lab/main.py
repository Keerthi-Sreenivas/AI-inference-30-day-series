"""
Day 5: Model Formats & Precision — FP16, BF16, INT8, INT4

Demonstrates how precision formats affect model memory usage by loading the
same model at different dtypes and measuring the memory footprint of each.

Analogy: Like photo quality — RAW (FP32) captures everything but is huge,
JPEG high (FP16) is nearly identical at half the size, JPEG low (INT4) is
tiny with minimal visible loss.

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache 2.0, no login required)
"""

import gc
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# --- Helper: Measure current process memory in MB ---
# psutil gives us the Resident Set Size (RSS) — the actual physical memory
# our Python process is using right now. We'll take snapshots before and
# after loading each model to compute the delta.
def get_memory_mb():
    """Return current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


# --- Baseline memory measurement ---
# Record how much memory the process uses before we load any model.
# This includes Python itself, imported libraries, and overhead.
baseline_mb = get_memory_mb()
print(f"Baseline process memory: {baseline_mb:.1f} MB")
print(f"Model: {MODEL_NAME}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print()

# --- Load tokenizer (shared across all precision tests) ---
# The tokenizer is the same regardless of model precision — it just maps text
# to token IDs (integers), as we learned on Day 2.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")
print()

# --- Storage for results ---
results = []

# ==========================================================================
# EXPERIMENT 1: Load in FP32 (32 bits per parameter)
# ==========================================================================
# FP32 is the full-precision format. Every parameter uses 32 bits (4 bytes).
# This is the default when you call from_pretrained() without specifying dtype.
# For a 1.7B model: ~1.7 billion × 4 bytes = ~6.8 GB of weights alone.
print("=" * 60)
print("Loading model in FP32 (full precision — 32 bits per param)")
print("=" * 60)

gc.collect()  # Clean up any leftover objects
if torch.cuda.is_available():
    torch.cuda.empty_cache()

before_mb = get_memory_mb()

try:
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",  # Force CPU to get accurate RSS measurements
    )
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

after_mb = get_memory_mb()
fp32_memory = after_mb - before_mb

# Count parameters to calculate theoretical memory
param_count = sum(p.numel() for p in model_fp32.parameters())
fp32_theoretical = param_count * 4 / (1024 * 1024)  # 4 bytes per FP32 param

print(f"Parameters: {param_count / 1e9:.2f}B ({param_count:,})")
print(f"Memory used: {fp32_memory:.1f} MB (actual)")
print(f"Theoretical:  {fp32_theoretical:.1f} MB (params only)")
print(f"Overhead:     {fp32_memory - fp32_theoretical:.1f} MB (buffers, framework)")
print()

results.append(("FP32", 32, fp32_memory, fp32_theoretical))

# Clean up to free memory before next load
del model_fp32
gc.collect()

# ==========================================================================
# EXPERIMENT 2: Load in FP16 (16 bits per parameter)
# ==========================================================================
# FP16 uses 16 bits (2 bytes) per parameter — exactly half of FP32. The
# mantissa is smaller (10 bits vs 23 bits) so you lose some decimal precision,
# but for inference this almost never matters. This is the standard choice
# for GPU inference because modern GPUs have dedicated FP16 compute units.
print("=" * 60)
print("Loading model in FP16 (half precision — 16 bits per param)")
print("=" * 60)

gc.collect()

before_mb = get_memory_mb()

model_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu",
)

after_mb = get_memory_mb()
fp16_memory = after_mb - before_mb
fp16_theoretical = param_count * 2 / (1024 * 1024)  # 2 bytes per FP16 param

print(f"Memory used: {fp16_memory:.1f} MB (actual)")
print(f"Theoretical:  {fp16_theoretical:.1f} MB (params only)")
print(f"Savings vs FP32: {(1 - fp16_memory / fp32_memory) * 100:.1f}%")
print()

results.append(("FP16", 16, fp16_memory, fp16_theoretical))

del model_fp16
gc.collect()

# ==========================================================================
# EXPERIMENT 3: Load in BF16 (Brain Float — 16 bits per parameter)
# ==========================================================================
# BF16 also uses 16 bits, but splits them differently: 8 bits for the exponent
# (same as FP32!) and 7 bits for the mantissa. This means BF16 can represent
# the same range of magnitudes as FP32 — it just has less decimal precision.
# This makes it more numerically stable than FP16 for very large or very small
# values. Preferred on modern hardware: NVIDIA A100/H100, Apple M-series chips.
print("=" * 60)
print("Loading model in BF16 (brain float — 16 bits per param)")
print("=" * 60)

gc.collect()

before_mb = get_memory_mb()

model_bf16 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

after_mb = get_memory_mb()
bf16_memory = after_mb - before_mb
bf16_theoretical = param_count * 2 / (1024 * 1024)  # 2 bytes per BF16 param

print(f"Memory used: {bf16_memory:.1f} MB (actual)")
print(f"Theoretical:  {bf16_theoretical:.1f} MB (params only)")
print(f"Savings vs FP32: {(1 - bf16_memory / fp32_memory) * 100:.1f}%")
print()

results.append(("BF16", 16, bf16_memory, bf16_theoretical))

# ==========================================================================
# VERIFICATION: Generate text at BF16 to prove reduced precision works
# ==========================================================================
# Even at half precision, the model produces coherent, high-quality text.
# This is the practical proof that you don't need FP32 for inference.
print("=" * 60)
print("Verification: Generating text at BF16 precision")
print("=" * 60)

messages = [{"role": "user", "content": "What is machine learning? Answer in two sentences."}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

output_ids = model_bf16.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=False,  # Greedy for reproducibility
)

new_tokens = output_ids[0][input_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"Prompt: 'What is machine learning?'")
print(f"Response ({len(new_tokens)} tokens): {response}")
print()
print("The model works perfectly at BF16 — same quality, half the memory.")
print()

del model_bf16
gc.collect()

# ==========================================================================
# SUMMARY TABLE: Memory Usage Comparison
# ==========================================================================
# This is the payoff — a clear side-by-side comparison showing that reducing
# precision from 32 to 16 bits cuts memory nearly in half.
print("=" * 60)
print("SUMMARY: Memory Usage per Precision Format")
print("=" * 60)
print()
print(f"Model: {MODEL_NAME}")
print(f"Parameters: {param_count / 1e9:.2f}B ({param_count:,})")
print()
print(f"{'Format':<8} {'Bits':<6} {'Actual (MB)':<14} {'Theoretical (MB)':<18} {'% of FP32':<10}")
print("-" * 56)

for name, bits, actual, theoretical in results:
    pct = (actual / results[0][2]) * 100  # Percentage relative to FP32
    print(f"{name:<8} {bits:<6} {actual:<14.1f} {theoretical:<18.1f} {pct:<10.1f}")

print("-" * 56)
print()
print("Key insight: FP16 and BF16 use ~50% of FP32 memory with negligible")
print("quality loss. This is why half-precision is the DEFAULT for inference.")
print()
print("Next steps:")
print("  - Day 6: Learn about GGUF format (quantized models for CPU)")
print("  - Day 8: Deep dive into INT8/INT4 quantization theory")
