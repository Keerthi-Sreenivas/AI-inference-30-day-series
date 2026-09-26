"""
Day 7: Understanding config.json — What Every Parameter Means

Explore a model's config.json to understand how architectural parameters
define the model structure, then validate that the parameter count computed
from config values matches the actual model parameter count.

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache 2.0, no login required)
"""

import json
from transformers import AutoConfig, AutoModelForCausalLM

# --- Load config without downloading weights ---
# AutoConfig.from_pretrained fetches only config.json (~1 KB) without the
# multi-GB weight files. This is fast and useful for inspecting any model.
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

print(f"Loading config for: {MODEL_NAME}")
print("(Only downloads config.json, not the full model weights)\n")

config = AutoConfig.from_pretrained(MODEL_NAME)

# --- Display key architecture parameters ---
# These are the numbers that define the entire model structure. Every tensor
# shape in the model is derived from these values.
print("=" * 70)
print("KEY ARCHITECTURE PARAMETERS")
print("=" * 70)
print(f"  vocab_size:              {config.vocab_size:,}")
print(f"  hidden_size:             {config.hidden_size:,}")
print(f"  num_hidden_layers:       {config.num_hidden_layers}")
print(f"  num_attention_heads:     {config.num_attention_heads}")
print(f"  num_key_value_heads:     {config.num_key_value_heads}")
print(f"  head_dim:                {config.head_dim}")
print(f"  intermediate_size:       {config.intermediate_size:,}")
print(f"  max_position_embeddings: {config.max_position_embeddings:,}")
print(f"  dtype:                   {config.dtype}")
print()

# --- Calculate theoretical parameter count from config ---
# This proves that config.json is the blueprint: we can compute the total
# parameter count using only the config values, without loading any weights.
print("=" * 70)
print("PARAMETER COUNT BREAKDOWN (computed from config)")
print("=" * 70)

vocab_size = config.vocab_size
hidden_size = config.hidden_size
num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads
head_dim = config.head_dim
intermediate_size = config.intermediate_size

# Embedding layer: maps each token ID to a hidden_size vector
embedding_params = vocab_size * hidden_size
print(f"  Embedding (vocab × hidden):          {embedding_params:>15,}")

# Attention projections per layer:
#   Q projection: hidden_size → num_heads × head_dim
#   K projection: hidden_size → num_kv_heads × head_dim
#   V projection: hidden_size → num_kv_heads × head_dim
#   O projection: num_heads × head_dim → hidden_size
q_params = hidden_size * (num_heads * head_dim)
k_params = hidden_size * (num_kv_heads * head_dim)
v_params = hidden_size * (num_kv_heads * head_dim)
o_params = (num_heads * head_dim) * hidden_size
attention_params_per_layer = q_params + k_params + v_params + o_params
print(f"  Attention per layer (Q+K+V+O):       {attention_params_per_layer:>15,}")

# Feed-forward network per layer (SwiGLU architecture):
#   Gate projection: hidden_size → intermediate_size
#   Up projection:   hidden_size → intermediate_size
#   Down projection: intermediate_size → hidden_size
gate_params = hidden_size * intermediate_size
up_params = hidden_size * intermediate_size
down_params = intermediate_size * hidden_size
ffn_params_per_layer = gate_params + up_params + down_params
print(f"  Feed-forward per layer (gate+up+down):{ffn_params_per_layer:>14,}")

# Layer norms per layer: 2 RMSNorm layers, each with hidden_size parameters
# (input layernorm + post-attention layernorm)
norm_params_per_layer = 2 * hidden_size
print(f"  Layer norms per layer (2 × hidden):  {norm_params_per_layer:>15,}")

# Total per layer
per_layer_total = attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
print(f"  Total per layer:                     {per_layer_total:>15,}")
print(f"  × {num_layers} layers:                          {per_layer_total * num_layers:>15,}")

# Final layer norm (applied after all transformer blocks)
final_norm_params = hidden_size
print(f"  Final layer norm:                    {final_norm_params:>15,}")

# LM head: projects hidden_size back to vocab_size for next-token prediction
# (often weight-tied with embedding, but counted separately here)
lm_head_params = hidden_size * vocab_size
print(f"  LM head (hidden × vocab):            {lm_head_params:>15,}")

# Grand total from config
theoretical_total = (
    embedding_params
    + (per_layer_total * num_layers)
    + final_norm_params
    + lm_head_params
)
print(f"  {'─' * 45}")
print(f"  THEORETICAL TOTAL:                   {theoretical_total:>15,}")
print()

# --- Load the full model to get actual parameter count ---
# Now we load the actual weights to compare against our calculation.
print("=" * 70)
print("VALIDATION: Theoretical vs Actual Parameter Count")
print("=" * 70)
print(f"\nLoading full model: {MODEL_NAME}")
print("(First run downloads ~3.5 GB — subsequent runs use cache)\n")

try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

# Count actual parameters from PyTorch
actual_total = sum(p.numel() for p in model.parameters())

print(f"  Theoretical (from config): {theoretical_total:>15,}")
print(f"  Actual (from PyTorch):     {actual_total:>15,}")

# Calculate percentage difference
if actual_total > 0:
    pct_diff = abs(theoretical_total - actual_total) / actual_total * 100
    print(f"  Difference:                {abs(theoretical_total - actual_total):>15,} ({pct_diff:.2f}%)")
    if pct_diff < 1.0:
        print(f"\n  ✓ VALIDATED: Config accurately describes the model (<1% difference)")
    else:
        print(f"\n  Note: >1% difference — likely due to weight tying or architecture variant")
else:
    print("  Could not compute difference (zero actual parameters)")

# --- Memory estimate from config ---
# Using the dtype from config, we can predict memory without loading weights.
# This is what you'd do to check if a model fits in your GPU before downloading.
print(f"\n{'=' * 70}")
print("MEMORY ESTIMATE (from config)")
print("=" * 70)

dtype_str = str(config.dtype)
bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
dtype_bytes = bytes_per_param.get(dtype_str, 2)  # Default to 2 bytes if unknown

estimated_memory_gb = (actual_total * dtype_bytes) / (1024 ** 3)
print(f"  Dtype: {dtype_str} ({dtype_bytes} bytes per parameter)")
print(f"  Parameters: {actual_total:,}")
print(f"  Estimated memory: {estimated_memory_gb:.2f} GB")
print(f"\n  This is the minimum memory needed to hold the model weights.")
print(f"  Actual runtime memory will be higher due to KV cache, activations,")
print(f"  and framework overhead (typically 1.2–1.5× the weight size).")

# --- Show raw config.json ---
# Pretty-print the full config as JSON so learners see what's actually on disk.
print(f"\n{'=' * 70}")
print("RAW config.json CONTENTS")
print("=" * 70)

config_dict = config.to_dict()
print(json.dumps(config_dict, indent=2))

# --- Final summary ---
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"  Model:            {MODEL_NAME}")
print(f"  Parameters:       {actual_total / 1e9:.2f}B")
print(f"  Layers:           {num_layers}")
print(f"  Hidden size:      {hidden_size}")
print(f"  Context window:   {config.max_position_embeddings:,} tokens")
print(f"  Memory ({dtype_str}): {estimated_memory_gb:.2f} GB")
print(f"\n  Key insight: config.json IS the blueprint. Every tensor shape,")
print(f"  every layer dimension, and the total parameter count are all")
print(f"  derivable from these ~20 numbers in the config file.")
