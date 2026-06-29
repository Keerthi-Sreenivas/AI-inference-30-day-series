"""
Day 12: The KV Cache — Why Inference Eats Memory

Demonstrates how the KV cache grows linearly with sequence length by
generating tokens incrementally and measuring the actual memory consumed
by the cached key-value tensors at each step.

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache 2.0, no login required)
Key metric: memory growth per token (MB)
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Load model and tokenizer ---
# SmolLM2-1.7B-Instruct is small enough to run on CPU (though GPU is faster).
# We load with use_cache=True support to examine the KV cache behavior.
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

print(f"Loading model: {MODEL_NAME}")
print("(First run downloads ~3.5 GB — subsequent runs use cache)\n")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

# --- Print model architecture details ---
# These values determine the KV cache size: layers × heads × head_dim × 2 (K+V)
config = model.config
num_layers = config.num_hidden_layers
hidden_dim = config.hidden_size
num_attention_heads = config.num_attention_heads
num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)
head_dim = hidden_dim // num_attention_heads

print("=" * 70)
print("MODEL ARCHITECTURE (relevant to KV cache)")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print(f"Layers: {num_layers}")
print(f"Hidden dimension: {hidden_dim}")
print(f"Attention heads (Q): {num_attention_heads}")
print(f"KV heads: {num_kv_heads}")
print(f"Head dimension: {head_dim}")
print(f"Dtype: float16 (2 bytes per value)")
print(f"Device: {model.device}")
print()

# --- Calculate theoretical KV cache size per token ---
# The KV cache stores K and V tensors for each layer. Per token:
#   memory = num_layers × 2 (K+V) × num_kv_heads × head_dim × bytes_per_value
# With GQA (Grouped Query Attention), num_kv_heads < num_attention_heads,
# which reduces the cache size proportionally.
bytes_per_value = 2  # float16
cache_per_token_bytes = num_layers * 2 * num_kv_heads * head_dim * bytes_per_value
cache_per_token_kb = cache_per_token_bytes / 1024
cache_per_token_mb = cache_per_token_bytes / (1024 * 1024)

print("=" * 70)
print("THEORETICAL KV CACHE SIZE")
print("=" * 70)
print(f"Formula: num_layers × 2 (K+V) × num_kv_heads × head_dim × 2 bytes")
print(f"         = {num_layers} × 2 × {num_kv_heads} × {head_dim} × 2")
print(f"         = {cache_per_token_bytes:,} bytes per token")
print(f"         = {cache_per_token_kb:.2f} KB per token")
print(f"         = {cache_per_token_mb:.4f} MB per token")
print()
print("Projected cache at various sequence lengths:")
print(f"  128 tokens:  {128 * cache_per_token_mb:.1f} MB")
print(f"  512 tokens:  {512 * cache_per_token_mb:.1f} MB")
print(f"  2048 tokens: {2048 * cache_per_token_mb:.1f} MB")
print(f"  8192 tokens: {8192 * cache_per_token_mb:.1f} MB")
print()

# --- Prepare the input prompt ---
# We use a simple prompt that will generate coherent multi-token output.
messages = [{"role": "user", "content": "Explain how computers store information in memory."}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
prompt_length = input_ids.shape[1]

print(f"Input prompt: {prompt_length} tokens")
print()


def measure_cache_size(past_key_values):
    """Calculate total memory used by KV cache tensors in bytes."""
    # past_key_values is a tuple of (key_tensor, value_tensor) per layer.
    # Each tensor shape: [batch, num_kv_heads, seq_len, head_dim]
    total_bytes = 0
    for layer_cache in past_key_values:
        # DynamicCache uses .key_cache and .value_cache lists in newer transformers
        if hasattr(layer_cache, "element_size"):
            total_bytes += layer_cache.nelement() * layer_cache.element_size()
        else:
            for tensor in layer_cache:
                total_bytes += tensor.nelement() * tensor.element_size()
    return total_bytes


# --- Generate tokens incrementally and measure KV cache growth ---
# We generate tokens one at a time to observe the cache growing at each step.
# In practice, model.generate() does this internally, but we do it manually
# to inspect the cache after each token.
NUM_TOKENS_TO_GENERATE = 60
PRINT_INTERVAL = 10  # Print cache size every N tokens

print("=" * 70)
print("EXPERIMENT 1: Measuring KV Cache Growth Per Token")
print("=" * 70)
print(f"Generating {NUM_TOKENS_TO_GENERATE} tokens, measuring cache at every {PRINT_INTERVAL} tokens.\n")

# First forward pass: process the prompt and get initial KV cache
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

# Measure initial cache size (includes all prompt tokens)
initial_cache_bytes = 0
if hasattr(past_key_values, "key_cache"):
    # DynamicCache object (newer transformers versions)
    for layer_keys, layer_values in zip(past_key_values.key_cache, past_key_values.value_cache):
        initial_cache_bytes += layer_keys.nelement() * layer_keys.element_size()
        initial_cache_bytes += layer_values.nelement() * layer_values.element_size()
else:
    # Tuple of tuples (older format)
    for layer_kv in past_key_values:
        for tensor in layer_kv:
            initial_cache_bytes += tensor.nelement() * tensor.element_size()

initial_cache_mb = initial_cache_bytes / (1024 * 1024)
print(f"Initial cache (after {prompt_length} prompt tokens): {initial_cache_mb:.2f} MB")
print()

# Track measurements for summary
cache_measurements = [(prompt_length, initial_cache_mb)]
generation_start = time.perf_counter()

# Generate tokens one at a time, measuring cache at each step
current_input_ids = input_ids
generated_tokens = []

for step in range(NUM_TOKENS_TO_GENERATE):
    with torch.no_grad():
        # Sample the next token from the logits (using simple argmax for consistency)
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token_id.item())

        # Forward pass with just the new token + cached K/V from all prior tokens.
        # This is the core of KV cache efficiency: we only process 1 token,
        # not the entire sequence, because K/V for prior tokens are cached.
        outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    # Measure cache size at regular intervals
    current_seq_len = prompt_length + step + 1
    if (step + 1) % PRINT_INTERVAL == 0:
        cache_bytes = 0
        if hasattr(past_key_values, "key_cache"):
            for layer_keys, layer_values in zip(past_key_values.key_cache, past_key_values.value_cache):
                cache_bytes += layer_keys.nelement() * layer_keys.element_size()
                cache_bytes += layer_values.nelement() * layer_values.element_size()
        else:
            for layer_kv in past_key_values:
                for tensor in layer_kv:
                    cache_bytes += tensor.nelement() * tensor.element_size()

        cache_mb = cache_bytes / (1024 * 1024)
        growth_mb = cache_mb - cache_measurements[-1][1]
        per_token_mb = growth_mb / PRINT_INTERVAL
        cache_measurements.append((current_seq_len, cache_mb))

        print(f"  Seq length {current_seq_len:>4d}: cache = {cache_mb:.2f} MB "
              f"(+{growth_mb:.3f} MB for {PRINT_INTERVAL} tokens, "
              f"~{per_token_mb:.4f} MB/token)")

generation_time = time.perf_counter() - generation_start

# Decode and show the generated text
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"\nGenerated text ({NUM_TOKENS_TO_GENERATE} tokens):")
print(f"  \"{generated_text[:200]}\"")
print(f"\nGeneration time (cached): {generation_time:.2f}s")
print(f"Tokens/sec (cached): {NUM_TOKENS_TO_GENERATE / generation_time:.1f}")
print()

# --- Compare cached vs uncached generation speed ---
# To show why the KV cache matters for speed, we generate a short sequence
# both with and without caching. Without cache, the model recomputes attention
# for all previous tokens at every step — much slower.
SPEED_TEST_TOKENS = 20

print("=" * 70)
print("EXPERIMENT 2: Cached vs Uncached Generation Speed")
print("=" * 70)
print(f"Generating {SPEED_TEST_TOKENS} tokens with and without KV cache.\n")

# Cached generation (normal fast path)
start = time.perf_counter()
with torch.no_grad():
    cached_output = model.generate(
        input_ids,
        max_new_tokens=SPEED_TEST_TOKENS,
        use_cache=True,
        do_sample=False,
    )
cached_time = time.perf_counter() - start

# Uncached generation (slow path — recomputes everything each step)
start = time.perf_counter()
with torch.no_grad():
    uncached_output = model.generate(
        input_ids,
        max_new_tokens=SPEED_TEST_TOKENS,
        use_cache=False,
        do_sample=False,
    )
uncached_time = time.perf_counter() - start

speedup = uncached_time / cached_time if cached_time > 0 else float("inf")

print(f"  With KV cache:    {cached_time:.2f}s ({SPEED_TEST_TOKENS / cached_time:.1f} tokens/sec)")
print(f"  Without KV cache: {uncached_time:.2f}s ({SPEED_TEST_TOKENS / uncached_time:.1f} tokens/sec)")
print(f"  Speedup:          {speedup:.1f}×")
print()
print("  The cache trades memory for speed. Without it, the model recomputes")
print("  attention for all prior tokens at every step — O(N²) total work.")
print()

# --- Final summary ---
# Print a concise table summarizing all findings.
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"{'Metric':<40} {'Value':<30}")
print("-" * 70)
print(f"{'Model':<40} {MODEL_NAME}")
print(f"{'Layers':<40} {num_layers}")
print(f"{'KV heads per layer':<40} {num_kv_heads}")
print(f"{'Head dimension':<40} {head_dim}")
print(f"{'Theoretical cache/token':<40} {cache_per_token_kb:.2f} KB")

# Compute measured per-token growth from our intervals
if len(cache_measurements) >= 2:
    total_growth_mb = cache_measurements[-1][1] - cache_measurements[0][1]
    total_new_tokens = cache_measurements[-1][0] - cache_measurements[0][0]
    measured_per_token_mb = total_growth_mb / total_new_tokens if total_new_tokens > 0 else 0
    measured_per_token_kb = measured_per_token_mb * 1024
    print(f"{'Measured cache/token':<40} {measured_per_token_kb:.2f} KB")

print(f"{'Final cache size ({cache_measurements[-1][0]} tokens)':<40} {cache_measurements[-1][1]:.2f} MB")
print(f"{'Cache speedup vs no cache':<40} {speedup:.1f}×")
print("-" * 70)
print()
print("Key insight: Memory grows LINEARLY with sequence length.")
print(f"Each token adds ~{cache_per_token_kb:.1f} KB to the KV cache.")
print(f"At 4096 tokens, the cache alone would use ~{4096 * cache_per_token_mb:.0f} MB.")
print("This is why long-context inference is fundamentally a memory problem,")
print("and why production servers need techniques like PagedAttention (Day 16)")
print("to manage KV cache memory efficiently across multiple users.")
