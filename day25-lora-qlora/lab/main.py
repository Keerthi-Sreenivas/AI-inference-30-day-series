"""
Day 25: LoRA & QLoRA — Lightweight Fine-Tuning

Creates a LoRA adapter on SmolLM2-1.7B-Instruct, measures trainable parameter
count, calculates adapter size in MB, and demonstrates the dramatic memory
savings compared to full fine-tuning.

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache 2.0, no login required)
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --- Configuration ---
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
ADAPTER_SAVE_DIR = "./lora_adapter_output"

# --- Load base model ---
# Standard FP32/FP16 loading. This is our starting point before applying LoRA.
# The entire model is loaded into memory but we'll only train tiny adapter matrices.
print("=" * 70)
print("LOADING BASE MODEL")
print("=" * 70)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

# Calculate full model size — this is what full fine-tuning would need to update
total_params = sum(p.numel() for p in model.parameters())
model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

print(f"  Model: {MODEL_NAME}")
print(f"  Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
print(f"  Model size (FP16): {model_size_mb:.1f} MB")
print(f"  Device: {next(model.parameters()).device}")
print()

# --- Create LoRA configuration ---
# LoRA freezes ALL base model weights and injects small trainable matrices
# (called adapters) into specific layers. Only these tiny matrices are updated
# during training — the post-it notes on the textbook.
print("=" * 70)
print("APPLYING LoRA ADAPTER (rank=16)")
print("=" * 70)

lora_config = LoraConfig(
    r=16,                                    # Rank: controls adapter capacity
    lora_alpha=32,                           # Scaling factor (2× rank is standard)
    target_modules=["q_proj", "v_proj"],     # Which weight matrices get adapters
    lora_dropout=0.05,                       # Light regularization
    task_type="CAUSAL_LM",                   # Causal language model task
    bias="none",                             # Don't train bias terms
)

print(f"  Rank (r): {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Target modules: {lora_config.target_modules}")
print(f"  Dropout: {lora_config.lora_dropout}")
print()

# --- Apply LoRA adapter ---
# get_peft_model wraps the base model: all original weights are frozen (requires_grad=False)
# and only the LoRA adapter matrices become trainable (requires_grad=True).
lora_model = get_peft_model(model, lora_config)

print("Trainable parameter breakdown:")
lora_model.print_trainable_parameters()
print()

# --- Calculate adapter size ---
# The adapter is just the trainable parameters — the tiny matrices A and B.
# This is what gets saved, transferred, and swapped between tasks.
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
adapter_size_bytes = sum(
    p.numel() * p.element_size() for p in lora_model.parameters() if p.requires_grad
)
adapter_size_mb = adapter_size_bytes / (1024**2)

print("=" * 70)
print("ADAPTER SIZE MEASUREMENT")
print("=" * 70)
print(f"  Full model parameters: {total_params:,}")
print(f"  Trainable (adapter) parameters: {trainable_params:,}")
print(f"  Percentage trainable: {trainable_params / total_params * 100:.4f}%")
print()
print(f"  Full model size: {model_size_mb:.1f} MB")
print(f"  Adapter size: {adapter_size_mb:.2f} MB")
print(f"  Ratio: adapter is {model_size_mb / adapter_size_mb:.0f}× smaller than full model")
print()

# --- Compare different ranks ---
# Higher rank = more adapter capacity (better quality) but more parameters.
# This shows how adapter size scales with rank — typically you want the smallest
# rank that achieves acceptable quality for your specific task.
print("=" * 70)
print("RANK COMPARISON TABLE")
print("=" * 70)
print(f"{'Rank':<8} {'Trainable Params':<20} {'Adapter Size (MB)':<20} {'% of Total':<12}")
print("-" * 60)

# We need to recreate models for each rank to get accurate counts
ranks_to_test = [4, 8, 16, 32, 64]
for rank in ranks_to_test:
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Calculate expected trainable params without actually wrapping
    # For each target module in each layer: params = 2 * hidden_dim * rank
    # SmolLM2-1.7B has hidden_dim=2048 and 24 layers with q_proj and v_proj
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_target_modules = len(config.target_modules)

    # Each adapter has: A matrix (hidden_dim × rank) + B matrix (rank × hidden_dim)
    params_per_module = 2 * hidden_dim * rank
    total_adapter_params = params_per_module * num_layers * num_target_modules
    adapter_mb = total_adapter_params * 2 / (1024**2)  # FP16 = 2 bytes per param
    pct = total_adapter_params / total_params * 100

    print(f"{rank:<8} {total_adapter_params:<20,} {adapter_mb:<20.2f} {pct:<12.4f}%")

print("-" * 60)
print()
print("Key insight: Even at rank=64, the adapter is still <1% of total parameters.")
print("Most tasks achieve good results at rank 8-16.\n")

# --- QLoRA demonstration ---
# QLoRA = 4-bit quantized base model + FP16 LoRA adapters.
# This combines Day 8 (quantization) + Day 10 (bitsandbytes) + LoRA.
print("=" * 70)
print("QLoRA: 4-bit Base + LoRA Adapters")
print("=" * 70)

if torch.cuda.is_available():
    # Load model in 4-bit quantization using NF4 (as we learned on Day 10)
    print("CUDA GPU detected — loading model in 4-bit for QLoRA demo...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    qlora_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Apply LoRA on top of 4-bit model
    qlora_model = get_peft_model(qlora_model, lora_config)

    # Measure actual VRAM usage
    allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"  Base model (4-bit): ~{model_size_mb / 4:.0f} MB")
    print(f"  LoRA adapter (FP16): {adapter_size_mb:.1f} MB")
    print(f"  Total GPU memory allocated: {allocated_mb:.0f} MB")
    print(f"  vs full FP16 fine-tuning: ~{model_size_mb * 4:.0f} MB (with optimizer states)")
    print(f"  Memory savings: {(1 - allocated_mb / (model_size_mb * 4)) * 100:.0f}%")
else:
    # Show calculated memory savings without GPU
    print("No CUDA GPU detected — showing calculated QLoRA memory savings:")
    print()
    fp16_size = model_size_mb
    int4_size = model_size_mb / 4  # 4-bit is ~4× smaller than FP16
    full_ft_size = model_size_mb * 4  # Full FT needs ~4× model size (optimizer states)
    qlora_total = int4_size + adapter_size_mb

    print(f"  Full fine-tuning VRAM estimate: ~{full_ft_size:.0f} MB")
    print(f"  QLoRA VRAM estimate:")
    print(f"    Base model (4-bit NF4): ~{int4_size:.0f} MB")
    print(f"    LoRA adapter (FP16): ~{adapter_size_mb:.1f} MB")
    print(f"    Total: ~{qlora_total:.0f} MB")
    print(f"  Memory savings vs full FT: {(1 - qlora_total / full_ft_size) * 100:.0f}%")

print()

# --- Save adapter ---
# Only the adapter weights are saved — the base model is referenced by name.
# This makes adapters portable: share a 16 MB file instead of a 3.5 GB model.
print("=" * 70)
print("SAVING ADAPTER")
print("=" * 70)

os.makedirs(ADAPTER_SAVE_DIR, exist_ok=True)
lora_model.save_pretrained(ADAPTER_SAVE_DIR)

# List saved files and their sizes
print(f"  Saved to: {ADAPTER_SAVE_DIR}/")
total_saved_size = 0
for filename in sorted(os.listdir(ADAPTER_SAVE_DIR)):
    filepath = os.path.join(ADAPTER_SAVE_DIR, filename)
    if os.path.isfile(filepath):
        size = os.path.getsize(filepath)
        total_saved_size += size
        print(f"    {filename}: {size / 1024:.1f} KB")

print(f"  Total saved size: {total_saved_size / (1024**2):.2f} MB")
print(f"  (vs full model checkpoint: {model_size_mb:.0f} MB)")
print()

# --- Final summary ---
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Base model: {MODEL_NAME}")
print(f"  Total parameters: {total_params / 1e9:.2f}B")
print(f"  LoRA trainable params: {trainable_params:,} ({trainable_params / total_params * 100:.4f}%)")
print(f"  Adapter size: {adapter_size_mb:.2f} MB")
print(f"  Full model size: {model_size_mb:.1f} MB")
print(f"  Size reduction: {model_size_mb / adapter_size_mb:.0f}×")
print()
print(f"Adapter Size: {adapter_size_mb:.2f} MB")
print()
print("Key insight: LoRA trains less than 0.1% of the model's parameters while")
print("achieving 95-99% of full fine-tuning quality. Combined with 4-bit")
print("quantization (QLoRA), you can fine-tune an 8B model on a consumer GPU")
print("with just 8 GB of VRAM — the post-it notes are tiny compared to the textbook.")
