"""
Day 11: AWQ & GPTQ — GPU Quantization Formats

Load pre-quantized AWQ and GPTQ models from HuggingFace, generate text with
each, and compare inference latency in milliseconds. Demonstrates how
pre-quantized GPU formats deliver fast inference without runtime quantization.

Models: Qwen2.5-7B-Instruct (GPTQ-Int4 and AWQ variants)
Library: gptqmodel >= 7.0.0 (successor to deprecated autoawq/auto-gptq)
Requires: NVIDIA GPU with 8+ GB VRAM, CUDA 11.8+
"""

import time
import torch

# --- GPU availability check ---
# AWQ and GPTQ models are designed for GPU tensor operations. Their quantized
# weight layouts cannot be executed on CPU. We check upfront to provide a
# helpful message rather than a cryptic CUDA error later.
if not torch.cuda.is_available():
    print("=" * 60)
    print("ERROR: No CUDA GPU detected.")
    print("=" * 60)
    print()
    print("AWQ and GPTQ models require an NVIDIA GPU with CUDA support.")
    print("Their quantized weight formats are optimized for GPU tensor cores")
    print("and cannot run on CPU.")
    print()
    print("Alternatives for CPU-only machines:")
    print("  • Day 6: GGUF format with llama.cpp (runs on CPU)")
    print("  • Day 10: bitsandbytes (also requires GPU)")
    print()
    print("If you have a GPU but CUDA isn't detected, check:")
    print("  1. NVIDIA drivers are installed: nvidia-smi")
    print("  2. PyTorch was installed with CUDA: torch.version.cuda")
    raise SystemExit(0)

# Print GPU info so we know what hardware we're working with
gpu_name = torch.cuda.get_device_name(0)
gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU detected: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
print(f"CUDA version: {torch.version.cuda}")
print()

# --- Model definitions ---
# We use ungated Qwen2.5-7B-Instruct variants (Apache-2.0 license) to avoid
# needing a HuggingFace token or license acceptance. Both are ~4 GB downloads.
GPTQ_MODEL = "TechxGenus/Qwen2.5-7B-Instruct-GPTQ-Int4"
AWQ_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# --- Load the GPTQ model ---
# gptqmodel is the unified successor to auto-gptq and autoawq (both deprecated
# since May 2025). It provides a single interface for loading both formats.
from gptqmodel import GPTQModel
from transformers import AutoTokenizer

print("=" * 60)
print("Loading GPTQ model...")
print(f"Model: {GPTQ_MODEL}")
print("=" * 60)

try:
    gptq_tokenizer = AutoTokenizer.from_pretrained(GPTQ_MODEL)
    gptq_model = GPTQModel.load(GPTQ_MODEL)
    print("GPTQ model loaded successfully.")
    gptq_params = sum(p.numel() for p in gptq_model.parameters())
    print(f"Parameters: {gptq_params / 1e9:.1f}B (stored in 4-bit)")
except Exception as e:
    print(f"Failed to load GPTQ model: {e}")
    print("Check your internet connection and GPU VRAM (need ~5 GB free).")
    raise SystemExit(1)

print()

# --- Load the AWQ model ---
# AWQ models can also be loaded through gptqmodel or directly via transformers.
# We use transformers here since AWQ support is well-integrated.
from transformers import AutoModelForCausalLM

print("=" * 60)
print("Loading AWQ model...")
print(f"Model: {AWQ_MODEL}")
print("=" * 60)

try:
    awq_tokenizer = AutoTokenizer.from_pretrained(AWQ_MODEL)
    awq_model = AutoModelForCausalLM.from_pretrained(
        AWQ_MODEL,
        device_map="auto",
        dtype=torch.float16,
    )
    print("AWQ model loaded successfully.")
    awq_params = sum(p.numel() for p in awq_model.parameters())
    print(f"Parameters: {awq_params / 1e9:.1f}B (stored in 4-bit AWQ)")
except Exception as e:
    print(f"Failed to load AWQ model: {e}")
    print("Check your internet connection and GPU VRAM (need ~5 GB free).")
    raise SystemExit(1)

print()

# --- Define the test prompt ---
# We use the same prompt for both models to ensure a fair comparison.
# The chat template formats the message as the model expects.
PROMPT = "Explain the difference between lossless and lossy compression in two sentences."

messages = [{"role": "user", "content": PROMPT}]
print(f"Test prompt: \"{PROMPT}\"")
print()

# --- Measure GPTQ inference latency ---
# We generate 100 tokens and measure wall-clock time. The latency includes
# both the prefill (processing the prompt) and decode (generating tokens) phases.
print("=" * 60)
print("GPTQ Inference")
print("=" * 60)

gptq_input = gptq_tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(gptq_model.device)

# Warm-up pass to avoid measuring CUDA kernel compilation overhead
with torch.no_grad():
    _ = gptq_model.generate(gptq_input, max_new_tokens=5)

# Timed generation
torch.cuda.synchronize()
gptq_start = time.perf_counter()

with torch.no_grad():
    gptq_output = gptq_model.generate(
        gptq_input,
        max_new_tokens=100,
        do_sample=False,  # Greedy decoding for reproducible comparison
    )

torch.cuda.synchronize()
gptq_elapsed = time.perf_counter() - gptq_start

# Decode and print results
gptq_new_tokens = gptq_output[0][gptq_input.shape[1]:]
gptq_text = gptq_tokenizer.decode(gptq_new_tokens, skip_special_tokens=True)
gptq_token_count = len(gptq_new_tokens)
gptq_latency_ms = gptq_elapsed * 1000
gptq_tokens_per_sec = gptq_token_count / gptq_elapsed

print(f"Output: {gptq_text}")
print(f"Tokens generated: {gptq_token_count}")
print(f"Latency: {gptq_latency_ms:.1f} ms")
print(f"Throughput: {gptq_tokens_per_sec:.1f} tokens/sec")
print()

# --- Measure AWQ inference latency ---
# Same prompt, same token count, same decoding strategy. Only the quantization
# format differs, so any speed difference comes from the format itself.
print("=" * 60)
print("AWQ Inference")
print("=" * 60)

awq_input = awq_tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
awq_input = torch.tensor(awq_input).unsqueeze(0).to(awq_model.device) if not isinstance(awq_input, torch.Tensor) else awq_input.to(awq_model.device)

# Warm-up pass
with torch.no_grad():
    _ = awq_model.generate(awq_input, max_new_tokens=5)

# Timed generation
torch.cuda.synchronize()
awq_start = time.perf_counter()

with torch.no_grad():
    awq_output = awq_model.generate(
        awq_input,
        max_new_tokens=100,
        do_sample=False,  # Greedy decoding for fair comparison
    )

torch.cuda.synchronize()
awq_elapsed = time.perf_counter() - awq_start

# Decode and print results
awq_new_tokens = awq_output[0][awq_input.shape[1]:]
awq_text = awq_tokenizer.decode(awq_new_tokens, skip_special_tokens=True)
awq_token_count = len(awq_new_tokens)
awq_latency_ms = awq_elapsed * 1000
awq_tokens_per_sec = awq_token_count / awq_elapsed

print(f"Output: {awq_text}")
print(f"Tokens generated: {awq_token_count}")
print(f"Latency: {awq_latency_ms:.1f} ms")
print(f"Throughput: {awq_tokens_per_sec:.1f} tokens/sec")
print()

# --- Comparison summary ---
# Print a side-by-side comparison so the learner can see how AWQ and GPTQ
# perform relative to each other on the same hardware with the same prompt.
print("=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Metric':<25} {'GPTQ':<20} {'AWQ':<20}")
print("-" * 65)
print(f"{'Tokens generated':<25} {gptq_token_count:<20} {awq_token_count:<20}")
print(f"{'Latency (ms)':<25} {gptq_latency_ms:<20.1f} {awq_latency_ms:<20.1f}")
print(f"{'Tokens/sec':<25} {gptq_tokens_per_sec:<20.1f} {awq_tokens_per_sec:<20.1f}")
print("-" * 65)

# Determine the winner
if gptq_latency_ms < awq_latency_ms:
    faster = "GPTQ"
    diff_pct = ((awq_latency_ms - gptq_latency_ms) / awq_latency_ms) * 100
else:
    faster = "AWQ"
    diff_pct = ((gptq_latency_ms - awq_latency_ms) / gptq_latency_ms) * 100

print(f"\nFaster format: {faster} (by {diff_pct:.1f}%)")
print()
print("Key insight: Both AWQ and GPTQ deliver similar speed because they're")
print("both 4-bit pre-quantized formats optimized for GPU inference. The")
print("difference is in HOW they choose which weights to preserve — AWQ")
print("focuses on activation-aware scaling, GPTQ uses optimal rounding.")
print("In practice, the speed gap is usually small (<10%) and the choice")
print("often comes down to model availability on HuggingFace.")
