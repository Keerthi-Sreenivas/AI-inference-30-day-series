"""
Day 4: Sampling Strategies — Temperature, Top-K, Top-P

Explore how sampling parameters affect text generation by running the same
prompt through different temperature, top-K, and top-P settings and measuring
tokens per second at each configuration.

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct (Apache 2.0, no login required)
"""

import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Load model and tokenizer ---
# SmolLM2-1.7B-Instruct is small enough to run on CPU (though GPU is faster).
# from_pretrained downloads the model on first run (~3.5 GB).
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

print(f"Loading model: {MODEL_NAME}")
print("(First run downloads ~3.5 GB — subsequent runs use cache)\n")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
except OSError as e:
    print(f"Failed to load model: {e}")
    print("Check your internet connection and try again.")
    raise SystemExit(1)

# Print model info so we know what we're working with
param_count = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {param_count / 1e9:.1f}B parameters")
print(f"Device: {model.device}\n")

# --- Prepare the prompt ---
# We use the model's chat template to format a user message. This ensures the
# model receives the prompt in the format it was fine-tuned to expect.
messages = [{"role": "user", "content": "Explain why the sky is blue in one paragraph."}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

print("=" * 70)
print("EXPERIMENT 1: Varying Temperature")
print("=" * 70)
print("Same prompt, same top-K (50) and top-P (0.9), different temperatures.")
print("Low temperature → focused/repetitive. High → creative/chaotic.\n")

# --- Generate at five temperatures ---
# Temperature divides the logits before softmax. Lower values sharpen the
# probability distribution (making the top token dominant), higher values
# flatten it (giving all tokens a more equal chance).
temperatures = [0.1, 0.5, 0.7, 1.0, 1.5]
temp_results = []

for temp in temperatures:
    start_time = time.perf_counter()

    # do_sample=True activates sampling — without it, generate() uses greedy
    # decoding and ignores temperature/top_k/top_p entirely.
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=temp,
        top_k=50,
        top_p=0.9,
    )

    elapsed = time.perf_counter() - start_time

    # Decode only the new tokens (skip the input prompt tokens)
    new_tokens = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tokens_generated = len(new_tokens)
    tokens_per_sec = tokens_generated / elapsed

    temp_results.append((temp, tokens_per_sec, tokens_generated))

    print(f"--- Temperature: {temp} ---")
    print(f"Tokens/sec: {tokens_per_sec:.1f} ({tokens_generated} tokens in {elapsed:.1f}s)")
    print(f"Output: {generated_text[:200]}")
    print()

print("=" * 70)
print("EXPERIMENT 2: Varying Top-K")
print("=" * 70)
print("Fixed temperature=0.7, top-P=0.9. Changing how many top tokens to keep.")
print("Low K → fewer candidates (safer). High K → more candidates (diverse).\n")

# --- Demonstrate top-K effect ---
# Top-K filters the vocabulary down to only the K highest-scoring tokens.
# K=5 means only 5 candidates at each step. K=100 gives more variety.
top_k_values = [5, 20, 50, 100]

for k in top_k_values:
    start_time = time.perf_counter()

    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=k,
        top_p=0.9,
    )

    elapsed = time.perf_counter() - start_time
    new_tokens = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tokens_per_sec = len(new_tokens) / elapsed

    print(f"--- Top-K: {k} ---")
    print(f"Tokens/sec: {tokens_per_sec:.1f} ({len(new_tokens)} tokens in {elapsed:.1f}s)")
    print(f"Output: {generated_text[:200]}")
    print()

print("=" * 70)
print("EXPERIMENT 3: Varying Top-P (Nucleus Sampling)")
print("=" * 70)
print("Fixed temperature=0.7, top-K=50. Changing the cumulative probability cutoff.")
print("Low P → only the most likely tokens. High P → includes longshots.\n")

# --- Demonstrate top-P effect ---
# Top-P (nucleus sampling) keeps the smallest set of tokens whose cumulative
# probability exceeds P. Unlike top-K, the number of candidates adapts to the
# model's confidence at each step.
top_p_values = [0.5, 0.8, 0.9, 0.95]

for p in top_p_values:
    start_time = time.perf_counter()

    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=p,
    )

    elapsed = time.perf_counter() - start_time
    new_tokens = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tokens_per_sec = len(new_tokens) / elapsed

    print(f"--- Top-P: {p} ---")
    print(f"Tokens/sec: {tokens_per_sec:.1f} ({len(new_tokens)} tokens in {elapsed:.1f}s)")
    print(f"Output: {generated_text[:200]}")
    print()

# --- Summary table ---
# Print a concise comparison of all temperature runs for quick reference.
print("=" * 70)
print("SUMMARY: Temperature vs Tokens/sec")
print("=" * 70)
print(f"{'Temperature':<15} {'Tokens/sec':<15} {'Tokens Generated':<20}")
print("-" * 50)
for temp, tps, count in temp_results:
    print(f"{temp:<15.1f} {tps:<15.1f} {count:<20}")
print("-" * 50)
print("\nKey insight: Temperature mainly affects OUTPUT QUALITY, not speed.")
print("All temperatures produce tokens at roughly the same rate because the")
print("computational cost of the forward pass doesn't change — only the")
print("probability distribution used for sampling changes.")
