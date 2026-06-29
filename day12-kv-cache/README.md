# Day 12: The KV Cache — Why Inference Eats Memory

## The Concept

Every time a language model generates a token, it needs to "look back" at all previous tokens in the sequence. Without any optimization, the model would recompute the attention scores for the entire history at every single step — token 50 would redo all the work from tokens 1 through 49. That's catastrophically wasteful. The **KV cache** (key-value cache) solves this by saving the intermediate attention results so the model never recalculates what it already knows.

Think of it like post-it notes during a long exam. Imagine you're answering a 100-question test where each answer depends on all previous answers. Without notes, you'd re-read and re-derive every prior answer before writing each new one — by question 50, you'd spend more time re-reading than thinking. The KV cache is your stack of post-it notes: you write down each answer as you go, so when you need to reference it later, you just glance at the note instead of re-deriving the whole thing. The trade-off? Your desk fills up with post-it notes. That stack is the memory cost.

On Day 3 you saw the `model.generate()` loop that produces tokens one at a time. On Day 4 you explored how sampling selects each token from the probability distribution. Today you'll look under the hood at what happens between those steps — specifically, how the model avoids redundant computation by caching attention keys and values, and why that cache is the dominant consumer of GPU memory during long-sequence inference.

## 1. Why Recomputation Is Expensive

In a transformer, each layer performs **self-attention**: every token attends to every previous token. The computation involves three matrices per layer:

- **Q (Query)** — "What am I looking for?" (computed fresh for the new token)
- **K (Key)** — "What do I contain?" (computed once, then cached)
- **V (Value)** — "What information do I provide?" (computed once, then cached)

Without caching, generating the Nth token requires computing K and V for all N-1 previous tokens again — that's O(N²) total work across the full sequence. With the KV cache, you compute K and V for each token exactly once, store them, and only compute Q for the new token. This reduces per-step cost from O(N) to O(1) for the new token's own K/V computation.

```python
# Without KV cache (naive): recompute everything at each step
for step in range(max_tokens):
    # Recomputes K and V for ALL previous tokens — wasteful!
    output = model(all_tokens_so_far)
    next_token = sample(output.logits[-1])
    all_tokens_so_far.append(next_token)

# With KV cache: only compute K/V for the new token
for step in range(max_tokens):
    # pass_kv_cache tells the model to reuse stored K/V
    output = model(new_token_only, past_key_values=kv_cache)
    kv_cache = output.past_key_values  # updated cache
    next_token = sample(output.logits[-1])
```

## 2. How the KV Cache Grows

The cache stores K and V tensors for every layer, for every token generated so far. This means memory grows **linearly** with sequence length:

```
┌─────────────────────────────────────────────────────────────────┐
│               KV Cache Growth (Memory vs Tokens)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Token 1:   K=[■]           V=[■]            ~0.2 MB            │
│  Token 2:   K=[■■]          V=[■■]           ~0.4 MB            │
│  Token 3:   K=[■■■]         V=[■■■]          ~0.6 MB            │
│  Token 4:   K=[■■■■]        V=[■■■■]         ~0.8 MB            │
│     ...        ...             ...              ...             │
│  Token 100: K=[■■■...■]     V=[■■■...■]      ~20 MB            │
│  Token 500: K=[■■■■■...■]   V=[■■■■■...■]   ~100 MB            │
│  Token 2048:K=[████████...█] V=[████████...█] ~410 MB            │
│                                                                 │
│  Memory grows LINEARLY with sequence length!                    │
│                                                                 │
│  Formula (per layer, FP16):                                     │
│  cache_per_token = 2 × hidden_dim × 2 bytes                    │
│  total_cache = num_layers × seq_len × cache_per_token           │
│                                                                 │
│  For SmolLM2-1.7B (24 layers, 2048 hidden_dim, FP16):          │
│  Per token: 24 × 2 × 2048 × 2 bytes = ~192 KB                  │
│  At 2048 tokens: ~384 MB                                        │
│  At 8192 tokens: ~1.5 GB                                        │
│                                                                 │
│  For a 7B model (32 layers, 4096 hidden_dim, FP16):             │
│  Per token: 32 × 2 × 4096 × 2 bytes = ~512 KB                  │
│  At 4096 tokens: ~2 GB just for the cache!                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The key insight: **model weights are fixed in memory, but the KV cache grows with every token you generate.** For long conversations or documents, the cache can rival or exceed the model weights themselves in memory consumption.

## 3. The Speed vs Memory Trade-Off

The KV cache is a classic time-memory trade-off. You spend memory to save computation time:

| Sequence Length | KV Cache Size (1.7B) | KV Cache Size (7B) | Time Saved vs No Cache | Trade-Off |
|:-:|:-:|:-:|:-:|---|
| 128 tokens | ~24 MB | ~64 MB | ~127× less recomputation | Negligible memory cost |
| 512 tokens | ~96 MB | ~256 MB | ~511× less recomputation | Noticeable on small GPUs |
| 2048 tokens | ~384 MB | ~1 GB | ~2047× less recomputation | Significant memory pressure |
| 8192 tokens | ~1.5 GB | ~4 GB | ~8191× less recomputation | May exceed GPU capacity |
| 32768 tokens | ~6 GB | ~16 GB | ~32767× less recomputation | Requires specialized techniques |

**Without the KV cache**, generating a 2048-token response would require the model to process an average of 1024 tokens at each step (the running average of the growing context). That's roughly 2047× more total computation than just processing one new token per step with a cache.

**The problem appears at scale:** When you serve multiple users simultaneously, each user gets their own KV cache. Ten concurrent 4096-token conversations on a 7B model need 10 × 2 GB = 20 GB just for caches — before you even count the model weights (~14 GB in FP16). This is exactly why production serving systems like vLLM (Day 15) invented PagedAttention (Day 16) to manage KV cache memory efficiently.

## 4. What Lives Inside the Cache

In Hugging Face Transformers, the KV cache is returned as `past_key_values` — a tuple of tuples, one per layer. Each layer's entry contains two tensors: the accumulated keys and the accumulated values.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs, use_cache=True)

# past_key_values is a tuple of (key_tensor, value_tensor) per layer
cache = outputs.past_key_values
print(f"Number of layers cached: {len(cache)}")
print(f"Key shape per layer: {cache[0][0].shape}")   # [batch, heads, seq_len, head_dim]
print(f"Value shape per layer: {cache[0][1].shape}") # [batch, heads, seq_len, head_dim]
```

Each tensor has the shape `[batch_size, num_attention_heads, sequence_length, head_dimension]`. As you generate more tokens, the `sequence_length` dimension grows by one at each step — that's the cache accumulating.

## 5. Progressive Disclosure: The Math

For those who want the full picture, here's the exact calculation. The KV cache memory for a single sequence is:

```
Total KV Cache Memory = num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_param
```

Breaking it down:
- **num_layers** — how many transformer blocks (24 for SmolLM2-1.7B, 32 for 7B models)
- **2** — we store both K and V
- **num_heads** — attention heads (could be fewer with GQA — Grouped Query Attention)
- **head_dim** — dimension per head (typically 64 or 128)
- **seq_len** — current sequence length (grows with generation)
- **bytes_per_param** — 2 for FP16/BF16, 4 for FP32

Modern models like Llama 3 use **Grouped Query Attention (GQA)** where K and V use fewer heads than Q. This reduces the KV cache by the group ratio (e.g., 8× fewer KV heads = 8× smaller cache). SmolLM2 also uses GQA, so the actual cache is smaller than the naive formula suggests.

## Setup

### Prerequisites

- Python 3.10+
- ~4 GB disk space (for model download)
- No GPU required — runs on CPU (slower but works)
- Familiarity with Day 3 (HuggingFace Transformers) and Day 4 (token generation)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — the same model from Days 3–4. If you've already run those labs, the model is cached locally (~3.5 GB). No account or token needed (Apache 2.0 license).

## What You'll Build

A Python script that generates tokens one at a time using the KV cache, measuring and printing how much memory the cache consumes at each step. You'll see the linear growth pattern in real numbers and understand why long-context inference is fundamentally a memory problem.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- The KV cache stores previously computed attention keys and values so the model doesn't recompute them at every generation step — it's the reason inference is fast
- Memory grows linearly with sequence length: each new token adds a fixed amount to the cache (determined by model architecture and precision)
- For a 7B model in FP16, the KV cache costs roughly 0.5 MB per token per layer — at 4096 tokens that's ~2 GB just for the cache
- The KV cache is the primary reason multi-user inference servers run out of GPU memory — each concurrent user needs their own cache
- Grouped Query Attention (GQA) reduces KV cache size by sharing K/V heads across multiple Q heads
- Understanding KV cache growth is essential for choosing context lengths, batch sizes, and serving configurations in production
