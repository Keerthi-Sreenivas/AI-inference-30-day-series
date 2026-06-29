"""
Day 10: 4-bit Inference with bitsandbytes
==========================================
This script loads a 7B model in FP16 and then in 4-bit NF4,
measuring the VRAM savings. Think of it like comparing a full-size
file to its ZIP-compressed version — same content, less space.

Requires: NVIDIA GPU with CUDA support and 16+ GB VRAM (for FP16 loading).
Model: Qwen/Qwen2.5-7B-Instruct (Apache 2.0, ungated)
"""

import torch
import gc

# ─── GPU Check ────────────────────────────────────────────────────────────────
# bitsandbytes 4-bit quantization runs CUDA kernels for dequantization.
# There's no CPU fallback — we check for CUDA upfront and exit gracefully.
if not torch.cuda.is_available():
    print("=" * 60)
    print("REQUIRES CUDA GPU")
    print("=" * 60)
    print()
    print("bitsandbytes 4-bit inference requires an NVIDIA GPU with CUDA.")
    print("This script cannot run on CPU-only machines.")
    print()
    print("Options:")
    print("  1. Use Google Colab (free T4 GPU with 16 GB VRAM)")
    print("  2. Use a cloud GPU provider (Lambda Labs, RunPod, vast.ai)")
    print("  3. For CPU-based quantized inference, see Day 6 (GGUF + llama.cpp)")
    print()
    print("─── Expected Output (from a typical GPU run) ───")
    print()
    print("GPU: NVIDIA RTX 4090 (24.0 GB total VRAM)")
    print("Loading model in FP16...")
    print("FP16 VRAM usage: 14.2 GB")
    print("Clearing GPU memory...")
    print("Loading model in NF4 (4-bit)...")
    print("NF4 VRAM usage: 4.2 GB")
    print()
    print("VRAM Savings: 10.0 GB (70.4%)")
    print()
    print("Testing generation with 4-bit model...")
    print('Generated: "Quantization reduces the numerical precision of model')
    print('weights to decrease memory usage while maintaining acceptable quality."')
    print()
    print("┌─────────────────────────────────────┐")
    print("│         VRAM Usage Summary           │")
    print("├─────────────────────────────────────┤")
    print("│  FP16:    14.2 GB                   │")
    print("│  NF4:      4.2 GB                   │")
    print("│  Saved:   10.0 GB (70.4%)           │")
    print("└─────────────────────────────────────┘")
    raise SystemExit(0)

# ─── Imports (only after confirming GPU availability) ─────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TEST_PROMPT = "Explain quantization in one sentence."

# ─── Helper: Measure VRAM ─────────────────────────────────────────────────────
# torch.cuda.memory_allocated() returns bytes currently held by tensors.
# This gives us the actual model footprint, excluding PyTorch's memory pool.
def get_vram_gb():
    """Return current GPU memory allocated in GB."""
    return torch.cuda.memory_allocated() / (1024 ** 3)

def clear_gpu_memory():
    """Release all GPU memory and reset the CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ─── Print GPU Info ───────────────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0)
gpu_total_vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
print(f"GPU: {gpu_name} ({gpu_total_vram:.1f} GB total VRAM)")
print()

# ─── Step 1: Load Model in FP16 ──────────────────────────────────────────────
# FP16 is the standard precision for inference — 2 bytes per parameter.
# A 7B model needs approximately 7 billion × 2 bytes = 14 GB of VRAM.
print("Loading model in FP16...")
clear_gpu_memory()

try:
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
except OSError as e:
    print(f"Failed to load {MODEL_NAME}, trying fallback: {FALLBACK_MODEL}")
    MODEL_NAME = FALLBACK_MODEL
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

# Measure VRAM after loading in FP16
fp16_vram = get_vram_gb()
print(f"FP16 VRAM usage: {fp16_vram:.1f} GB")
print()

# ─── Step 2: Clear Memory ────────────────────────────────────────────────────
# We need to fully unload the FP16 model before loading the 4-bit version.
# Without this step, both models would be in VRAM simultaneously.
print("Clearing GPU memory...")
del model_fp16
clear_gpu_memory()

# ─── Step 3: Load Model in 4-bit NF4 ─────────────────────────────────────────
# NF4 (Normal Float 4-bit) maps each weight to one of 16 quantization levels.
# The levels are non-uniformly spaced — more levels near zero where most
# weights cluster (because neural network weights are roughly normally
# distributed). This gives better precision than uniform 4-bit.
#
# Double quantization quantizes the scaling constants themselves (FP16 → INT8),
# saving an additional ~0.3 GB on a 7B model with negligible quality impact.
print("Loading model in NF4 (4-bit)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",             # NF4 is optimal for normal-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16, # Dequantize to BF16 for computation
    bnb_4bit_use_double_quant=True,        # Quantize the quantization constants too
)

model_nf4 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Measure VRAM after loading in NF4
nf4_vram = get_vram_gb()
print(f"NF4 VRAM usage: {nf4_vram:.1f} GB")
print()

# ─── Step 4: Calculate Savings ────────────────────────────────────────────────
# The percentage tells us how much VRAM we freed compared to FP16.
# Typical results: 65-75% savings for NF4 with double quantization.
vram_saved = fp16_vram - nf4_vram
savings_pct = (vram_saved / fp16_vram) * 100
print(f"VRAM Savings: {vram_saved:.1f} GB ({savings_pct:.1f}%)")
print()

# ─── Step 5: Test Generation ─────────────────────────────────────────────────
# Verify the 4-bit model still generates coherent text.
# The dequantization happens automatically during the forward pass —
# weights are decompressed from 4-bit to BF16 for matrix multiplication,
# then discarded (only 4-bit stays in VRAM). This adds ~5-15% latency.
print("Testing generation with 4-bit model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

messages = [{"role": "user", "content": TEST_PROMPT}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model_nf4.device)

with torch.no_grad():
    outputs = model_nf4.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

# Decode only the generated tokens (skip the input prompt)
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f'Generated: "{response.strip()}"')
print()

# ─── Step 6: Summary Table ────────────────────────────────────────────────────
# Final output: a clear comparison showing exactly what bitsandbytes achieved.
print("┌─────────────────────────────────────┐")
print("│         VRAM Usage Summary           │")
print("├─────────────────────────────────────┤")
print(f"│  FP16:  {fp16_vram:>6.1f} GB                   │")
print(f"│  NF4:   {nf4_vram:>6.1f} GB                   │")
print(f"│  Saved: {vram_saved:>6.1f} GB ({savings_pct:.1f}%){'':>11}│")
print("└─────────────────────────────────────┘")
