"""
Day 13: Flash Attention — Faster, Leaner Attention

Benchmark Flash Attention 2 against standard attention to measure the
speedup factor. Flash Attention reorganizes the attention computation into
tiles that stay in fast GPU SRAM, avoiding expensive reads/writes to
high-bandwidth memory (HBM). The result is the same attention scores but
2-4x faster on supported hardware.

Requirements:
    - NVIDIA GPU with Ampere+ architecture (sm80+): A100, RTX 3090/4090, H100
    - Linux OS (flash-attn does not support Windows or macOS)
    - CUDA 11.6+
    - pip install transformers>=5.0.0 torch>=2.4.0 accelerate>=1.0.0 flash-attn>=2.6.0

Model:
    meta-llama/Llama-3.1-8B (requires license acceptance at HuggingFace)
    Alternative: Qwen/Qwen2.5-7B-Instruct (Apache-2.0, no license needed)
"""

import time
import statistics
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Configuration ───────────────────────────────────────────────────────────

# Model to benchmark — Llama-3.1-8B gives clear Flash Attention speedups
# Change to "Qwen/Qwen2.5-7B-Instruct" if you don't have Llama access
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Benchmark parameters
MAX_NEW_TOKENS = 128       # Number of tokens to generate per run
NUM_ITERATIONS = 3         # How many timed runs (median reported)
WARMUP_RUNS = 1           # Excluded from timing (JIT compilation, caching)

# ─── Step 1: Check GPU Availability ─────────────────────────────────────────

# Flash Attention requires an NVIDIA Ampere+ GPU (compute capability 8.0+)
# This includes: A100, A10G, RTX 3090, RTX 4090, H100
has_cuda = torch.cuda.is_available()
flash_supported = False

if has_cuda:
    gpu_name = torch.cuda.get_device_name(0)
    # Get compute capability to check for Ampere+ (sm80+)
    major, minor = torch.cuda.get_device_capability(0)
    compute_cap = f"sm_{major}{minor}"
    flash_supported = major >= 8  # Ampere = 8.0, Hopper = 9.0

    if flash_supported:
        print(f"GPU: {gpu_name} ({compute_cap}) — Flash Attention supported ✓")
    else:
        print(f"GPU: {gpu_name} ({compute_cap}) — Flash Attention NOT supported")
        print("  Flash Attention requires Ampere+ (sm80+): A100, RTX 3090/4090, H100")
        print("  Running standard attention only as baseline.\n")
else:
    print("GPU not available or not Ampere+ architecture.")
    print("Flash Attention requires NVIDIA Ampere (sm80+) GPU and Linux.")
    print("Running standard attention only as baseline...\n")

# ─── Step 2: Check flash-attn Package ───────────────────────────────────────

# The flash-attn package provides the optimized CUDA kernels
# Without it, we can't run Flash Attention even on supported hardware
flash_attn_installed = False
if flash_supported:
    try:
        import flash_attn
        flash_attn_installed = True
        print(f"flash-attn version: {flash_attn.__version__}")
    except ImportError:
        print("WARNING: flash-attn package not installed.")
        print("Install with: pip install flash-attn --no-build-isolation")
        print("Running standard attention only.\n")
        flash_supported = False

# ─── Step 3: Load Tokenizer ─────────────────────────────────────────────────

# Load tokenizer first (shared between both model variants)
print(f"\nLoading tokenizer for {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except OSError as e:
    if "gated" in str(e).lower() or "access" in str(e).lower():
        print(f"\nERROR: {MODEL_NAME} requires license acceptance.")
        print(f"1. Visit: https://huggingface.co/{MODEL_NAME}")
        print("2. Accept the Llama 3.1 Community License")
        print("3. Set: export HF_TOKEN=your_token_here")
        print("\nAlternative: Change MODEL_NAME to 'Qwen/Qwen2.5-7B-Instruct'")
        sys.exit(1)
    else:
        print(f"Failed to load tokenizer: {e}")
        sys.exit(1)

# ─── Step 4: Prepare Benchmark Prompt ────────────────────────────────────────

# We need a sufficiently long prompt (256+ tokens) so that the attention
# computation is measurable. Short sequences don't show much difference
# because the overhead of tiling exceeds the memory savings.
benchmark_text = (
    "The transformer architecture revolutionized natural language processing "
    "by introducing the self-attention mechanism. Unlike recurrent neural "
    "networks that process tokens sequentially, transformers can attend to all "
    "positions in the input simultaneously. This parallelism enables much "
    "faster training and inference. The attention mechanism computes a weighted "
    "sum of value vectors, where the weights are determined by the similarity "
    "between query and key vectors. Each layer of the transformer refines "
    "these representations, building increasingly abstract features. "
    "Modern large language models stack dozens of these layers, each with "
    "multiple attention heads that capture different aspects of the input. "
) * 3  # Repeat to ensure sufficient length

# Tokenize and verify length
inputs = tokenizer(benchmark_text, return_tensors="pt")
input_length = inputs["input_ids"].shape[1]
print(f"Benchmark prompt length: {input_length} tokens")

if input_length < 100:
    print("WARNING: Prompt is short. Speedup may not be visible.")

# ─── Step 5: Define Benchmark Function ───────────────────────────────────────

def benchmark_generation(model, input_ids, attention_mask, label):
    """
    Run text generation multiple times and return median wall-clock time.

    We use time.perf_counter() for high-resolution timing.
    A warmup run is excluded because the first run includes:
    - CUDA kernel compilation (JIT)
    - Memory allocation patterns being established
    - GPU frequency ramp-up
    """
    # Move inputs to model device
    device = next(model.parameters()).device
    input_ids_device = input_ids.to(device)
    attention_mask_device = attention_mask.to(device)

    # Warmup: first run is always slower due to CUDA graph compilation
    print(f"  {label}: warming up...", end="", flush=True)
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            model.generate(
                input_ids=input_ids_device,
                attention_mask=attention_mask_device,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Greedy for reproducibility
            )
    # Synchronize GPU before timing (ensure warmup is fully complete)
    if has_cuda:
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for i in range(NUM_ITERATIONS):
        if has_cuda:
            torch.cuda.synchronize()  # Ensure previous work is done

        start = time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids=input_ids_device,
                attention_mask=attention_mask_device,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        if has_cuda:
            torch.cuda.synchronize()  # Wait for GPU to finish

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = statistics.median(times)
    print(f" median {median_time:.2f}s ({NUM_ITERATIONS} runs)")
    return median_time

# ─── Step 6: Load and Benchmark Standard Attention ───────────────────────────

# Standard attention uses the default "eager" implementation which
# materializes the full N×N attention matrix in GPU memory.
# Memory usage: O(N²) where N is sequence length.
print(f"\nLoading model with standard attention...")
model_standard = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto" if has_cuda else "cpu",
    attn_implementation="eager",  # Explicit standard attention
)
model_standard.eval()

print(f"\nBenchmarking with sequence length: {input_length} tokens")
print(f"Generating {MAX_NEW_TOKENS} new tokens per run, "
      f"{NUM_ITERATIONS} iterations (+ {WARMUP_RUNS} warmup)\n")

standard_time = benchmark_generation(
    model_standard,
    inputs["input_ids"],
    inputs["attention_mask"],
    "Standard Attention",
)

# ─── Step 7: Load and Benchmark Flash Attention ──────────────────────────────

# Flash Attention 2 processes attention in tiles that fit in SRAM,
# avoiding the costly HBM reads/writes of the full attention matrix.
# Memory usage: O(N) — linear instead of quadratic.
# Speed: 2-4x faster on Ampere+ GPUs due to reduced memory traffic.

if flash_supported and flash_attn_installed:
    # Free standard model memory before loading Flash variant
    del model_standard
    if has_cuda:
        torch.cuda.empty_cache()

    print(f"\nLoading model with Flash Attention 2...")
    model_flash = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",  # The only change needed!
    )
    model_flash.eval()

    flash_time = benchmark_generation(
        model_flash,
        inputs["input_ids"],
        inputs["attention_mask"],
        "Flash Attention 2",
    )

    # ─── Step 8: Calculate and Print Speedup ─────────────────────────────────

    # The speedup factor tells you how many times faster Flash Attention is.
    # Typical values: 2-4x on Ampere GPUs, higher for longer sequences.
    speedup = standard_time / flash_time

    print("\n" + "━" * 50)
    print(f"  Speedup: {speedup:.2f}x")
    print("━" * 50)

    # Summary table
    print(f"\n{'Metric':<25} {'Standard':<15} {'Flash Attn 2':<15}")
    print(f"{'─' * 25} {'─' * 15} {'─' * 15}")
    print(f"{'Median time (s)':<25} {standard_time:<15.2f} {flash_time:<15.2f}")
    print(f"{'Tokens/sec':<25} "
          f"{MAX_NEW_TOKENS / standard_time:<15.1f} "
          f"{MAX_NEW_TOKENS / flash_time:<15.1f}")
    print(f"{'Speedup':<25} {'1.00x':<15} {f'{speedup:.2f}x':<15}")
    print(f"{'Input length':<25} {input_length:<15} {input_length:<15}")
    print(f"{'Memory complexity':<25} {'O(N²)':<15} {'O(N)':<15}")

    # Clean up Flash model
    del model_flash
    if has_cuda:
        torch.cuda.empty_cache()

else:
    # ─── Fallback: No Flash Attention Available ──────────────────────────────

    # Without Flash Attention, we still report standard attention performance
    # so the user can see their baseline and understand what they'd gain.
    print("\n" + "━" * 50)
    print("  Flash Attention not available on this system.")
    print("  Showing standard attention baseline only.")
    print("━" * 50)

    print(f"\n{'Metric':<25} {'Standard':<15} {'Flash Attn 2':<15}")
    print(f"{'─' * 25} {'─' * 15} {'─' * 15}")
    print(f"{'Median time (s)':<25} {standard_time:<15.2f} {'N/A':<15}")
    print(f"{'Tokens/sec':<25} "
          f"{MAX_NEW_TOKENS / standard_time:<15.1f} {'N/A':<15}")
    print(f"{'Speedup':<25} {'1.00x (baseline)':<15} {'—':<15}")
    print(f"\nTo see the speedup, run on NVIDIA Ampere+ GPU with:")
    print(f"  pip install flash-attn --no-build-isolation")

    # Clean up
    del model_standard
    if has_cuda:
        torch.cuda.empty_cache()

# ─── Done ────────────────────────────────────────────────────────────────────

print("\n✓ Benchmark complete.")
print("  Flash Attention gives the same outputs as standard attention —")
print("  it only changes HOW the GPU computes attention (tiled vs full matrix).")
