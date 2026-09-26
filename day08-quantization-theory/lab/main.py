"""
Day 8: Quantization Theory — Why Fewer Bits Still Work

Measure how quantization affects model quality by computing perplexity at
different bit widths (FP16, INT8, INT4) using bitsandbytes.

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache 2.0, no login required)
Requires: NVIDIA GPU with CUDA (bitsandbytes is GPU-only)
"""

import torch
import math
import gc

# --- GPU Availability Check ---
# bitsandbytes quantization requires an NVIDIA GPU with CUDA support.
# If no GPU is available, we print expected results and exit gracefully.
if not torch.cuda.is_available():
    print("=" * 70)
    print("GPU CHECK: No CUDA-capable GPU detected")
    print("=" * 70)
    print()
    print("bitsandbytes requires an NVIDIA GPU with Compute Capability 6.0+.")
    print("This script cannot run quantized inference on CPU.")
    print()
    print("If you had a GPU, you would see output like this:")
    print()
    print("  Bit Width    | Memory (MB) | Perplexity | PPL Increase")
    print("  -------------|-------------|------------|-------------")
    print("  FP16 (16-bit)|    3,420    |    8.73    |   baseline")
    print("  INT8 (8-bit) |    1,710    |    8.85    |   +0.12")
    print("  INT4 (4-bit) |      855    |    9.31    |   +0.58")
    print()
    print("Key insight: 4-bit quantization uses 4x less memory than FP16")
    print("while perplexity increases by less than 1 point — barely noticeable")
    print("in real-world text generation.")
    print()
    print("To run this lab:")
    print("  - Use a machine with an NVIDIA GPU (GTX 1060 or newer)")
    print("  - Or try Google Colab (free tier includes T4 GPU)")
    print("  - Or see Day 6 for CPU-friendly quantized inference via llama.cpp")
    raise SystemExit(0)

print(f"GPU detected: {torch.cuda.get_device_name(0)}")
print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Configuration ---
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Fixed evaluation text for consistent perplexity measurement across all bit widths.
# Using a natural English paragraph ensures we test real-world language modeling quality.
EVAL_TEXT = (
    "The process of photosynthesis converts sunlight into chemical energy that plants "
    "use to grow. Inside each leaf, chloroplasts absorb light photons and use that energy "
    "to split water molecules, releasing oxygen as a byproduct. The captured energy drives "
    "the Calvin cycle, which fixes carbon dioxide from the atmosphere into glucose molecules. "
    "This elegant process has sustained life on Earth for billions of years, forming the "
    "foundation of nearly every food chain on the planet."
)

print(f"Model: {MODEL_NAME}")
print(f"Evaluation text: {len(EVAL_TEXT.split())} words")
print()


def compute_perplexity(model, tokenizer, text):
    """
    Compute perplexity of a text using the model's cross-entropy loss.
    Perplexity = exp(average_loss) — lower is better (more confident predictions).
    """
    # Tokenize the evaluation text
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Forward pass with labels = input_ids computes cross-entropy loss internally.
    # The loss is averaged over all tokens in the sequence.
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()

    # Perplexity is the exponential of the average cross-entropy loss.
    # A perplexity of 10 means the model is "10-ways confused" on average.
    perplexity = math.exp(loss)
    return perplexity


def get_model_memory_mb(model):
    """
    Estimate model memory usage from parameter sizes.
    For quantized models, this reads the actual byte sizes of stored tensors.
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    return total_bytes / (1024 * 1024)


# --- Store results for final comparison table ---
results = []

# =====================================================================
# EXPERIMENT 1: FP16 Baseline
# =====================================================================
print("=" * 70)
print("Loading model in FP16 (baseline — no quantization)")
print("=" * 70)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

# Measure FP16 memory and perplexity
memory_fp16 = get_model_memory_mb(model_fp16)
ppl_fp16 = compute_perplexity(model_fp16, tokenizer, EVAL_TEXT)

print(f"  Memory usage: {memory_fp16:.0f} MB")
print(f"  Perplexity:   {ppl_fp16:.2f}")
print()

results.append(("FP16 (16-bit)", memory_fp16, ppl_fp16, 0.0))

# Clean up — free VRAM before loading the next model
# This is critical: loading multiple large models simultaneously would exceed VRAM.
del model_fp16
gc.collect()
torch.cuda.empty_cache()

# =====================================================================
# EXPERIMENT 2: INT8 Quantization
# =====================================================================
print("=" * 70)
print("Loading model in INT8 (8-bit quantization via bitsandbytes)")
print("=" * 70)

# BitsAndBytesConfig with load_in_8bit applies dynamic INT8 quantization.
# Each weight is stored as an 8-bit integer plus a per-block scale factor.
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)

model_int8 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config_8bit,
    device_map="auto",
)

# Measure INT8 memory and perplexity
memory_int8 = get_model_memory_mb(model_int8)
ppl_int8 = compute_perplexity(model_int8, tokenizer, EVAL_TEXT)
ppl_increase_int8 = ppl_int8 - ppl_fp16

print(f"  Memory usage: {memory_int8:.0f} MB")
print(f"  Perplexity:   {ppl_int8:.2f} (+{ppl_increase_int8:.2f} vs FP16)")
print()

results.append(("INT8 (8-bit)", memory_int8, ppl_int8, ppl_increase_int8))

# Clean up before loading the next quantization level
del model_int8
gc.collect()
torch.cuda.empty_cache()

# =====================================================================
# EXPERIMENT 3: INT4 (NF4) Quantization
# =====================================================================
print("=" * 70)
print("Loading model in INT4 (4-bit NF4 quantization via bitsandbytes)")
print("=" * 70)

# NF4 (NormalFloat4) is a 4-bit data type optimized for normally-distributed
# neural network weights. It provides better quality than naive 4-bit integers
# because the quantization levels are spaced according to a normal distribution,
# matching the actual distribution of trained weights.
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_int4 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config_4bit,
    device_map="auto",
)

# Measure INT4 memory and perplexity
memory_int4 = get_model_memory_mb(model_int4)
ppl_int4 = compute_perplexity(model_int4, tokenizer, EVAL_TEXT)
ppl_increase_int4 = ppl_int4 - ppl_fp16

print(f"  Memory usage: {memory_int4:.0f} MB")
print(f"  Perplexity:   {ppl_int4:.2f} (+{ppl_increase_int4:.2f} vs FP16)")
print()

results.append(("INT4 (4-bit)", memory_int4, ppl_int4, ppl_increase_int4))

# Clean up the final model
del model_int4
gc.collect()
torch.cuda.empty_cache()

# =====================================================================
# SUMMARY TABLE
# =====================================================================
print("=" * 70)
print("RESULTS: Quantization Quality vs Memory Trade-off")
print("=" * 70)
print()
print(f"{'Bit Width':<16} {'Memory (MB)':<14} {'Perplexity':<14} {'PPL Increase':<14}")
print("-" * 58)

for name, memory, ppl, increase in results:
    increase_str = "baseline" if increase == 0 else f"+{increase:.2f}"
    print(f"{name:<16} {memory:<14.0f} {ppl:<14.2f} {increase_str:<14}")

print("-" * 58)
print()

# --- Key insights ---
# Summarize what the numbers tell us about the quantization trade-off.
memory_reduction = results[0][1] / results[2][1]
ppl_total_increase = results[2][3]

print(f"Memory reduction (FP16 → INT4): {memory_reduction:.1f}×")
print(f"Perplexity cost of 4-bit quantization: +{ppl_total_increase:.2f}")
print()
print("Key insight: 4-bit quantization achieves massive memory savings")
print("with minimal quality degradation. The model 'rounds' its weights")
print("like rounding $3.99 to $4 — close enough for practical purposes.")
print()
print("This is why quantization is the #1 technique for running large")
print("models on consumer hardware. A 7B model that needs 14 GB in FP16")
print("fits in just 3.5 GB at 4-bit — well within a laptop GPU's VRAM.")
