# Day 7: Understanding config.json — What Every Parameter Means

## The Concept

Every Hugging Face model you've loaded with `from_pretrained` (Day 3) comes with a hidden blueprint: a file called `config.json`. This file tells the library the exact specifications it needs to construct the model in memory — how many layers to stack, how wide to make each one, what vocabulary size to expect, and dozens of other architectural decisions. Without it, the library would have no idea how to assemble the billions of parameters into a functioning neural network.

Think of `config.json` like an architectural blueprint for a building. The blueprint doesn't contain the bricks, concrete, or steel (those are the weight files). Instead, it tells the construction crew exactly how to arrange those materials — how many floors, how wide each hallway, where the load-bearing walls go. In the same way, `config.json` tells Transformers how to wire up the weight tensors into attention heads, feed-forward layers, and embedding tables. Change a single number and you'd get a completely different building — or a structure that collapses because the materials don't fit the plan.

On Day 3 you called `AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")` and the model just appeared, ready to generate. Behind the scenes, that function first downloaded `config.json`, read every parameter, and used those numbers to allocate the exact tensor shapes before loading the weights. On Day 5 you learned that a model's memory footprint depends on its parameter count — today you'll see exactly where that parameter count comes from: it's computed directly from the numbers inside `config.json`.

## 1. How config.json Maps to Model Architecture

Every parameter in the config file corresponds to a structural decision in the model. Here's how the key numbers translate into layers and dimensions:

```
┌──────────────────────────────────────────────────────────────────────┐
│           config.json → Model Architecture Mapping                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  config.json                      Model Architecture                 │
│  ─────────────                    ──────────────────                 │
│                                                                      │
│  "vocab_size": 49152  ──────────▶ ┌─────────────────────────┐       │
│                                   │  Embedding Layer          │       │
│                                   │  Shape: [49152 × 2048]    │       │
│  "hidden_size": 2048  ──────────▶ │  (tokens → vectors)       │       │
│                                   └───────────┬─────────────┘       │
│                                               │                      │
│                                               ▼                      │
│  "num_hidden_layers": 24 ───────▶ ┌─────────────────────────┐       │
│                                   │  Transformer Block ×24    │       │
│                                   │                           │       │
│  "num_attention_heads": 32 ────▶  │  ┌─ Attention ─────────┐ │       │
│  "num_key_value_heads": 32 ────▶  │  │  32 Q heads          │ │       │
│                                   │  │  32 KV heads          │ │       │
│  "head_dim": 64 ──────────────▶   │  │  dim per head: 64    │ │       │
│                                   │  └──────────────────────┘ │       │
│                                   │                           │       │
│  "intermediate_size": 8192 ────▶  │  ┌─ Feed-Forward ──────┐ │       │
│                                   │  │  Up:   2048 → 8192   │ │       │
│                                   │  │  Down:  8192 → 2048   │ │       │
│                                   │  └──────────────────────┘ │       │
│                                   └───────────┬─────────────┘       │
│                                               │                      │
│                                               ▼  (×24 layers)        │
│                                   ┌─────────────────────────┐       │
│  "vocab_size": 49152  ──────────▶ │  LM Head (output)         │       │
│                                   │  Shape: [2048 × 49152]    │       │
│                                   │  (vectors → token probs)  │       │
│                                   └─────────────────────────┘       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

Each arrow shows a direct relationship: change `hidden_size` from 2048 to 4096 and every layer doubles in width, roughly quadrupling the model's parameter count.

## 2. Key Parameters Explained

Here's what the most important `config.json` fields actually mean for SmolLM2-1.7B-Instruct:

```json
{
  "architectures": ["LlamaForCausalLM"],
  "vocab_size": 49152,
  "hidden_size": 2048,
  "num_hidden_layers": 24,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "head_dim": 64,
  "intermediate_size": 8192,
  "max_position_embeddings": 8192,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "torch_dtype": "bfloat16"
}
```

Each field controls a specific part of the architecture:

- **vocab_size** — How many unique tokens the model understands. Determines embedding table dimensions.
- **hidden_size** — Width of the hidden state vector flowing through every layer. The "highway width."
- **num_hidden_layers** — How many transformer blocks are stacked. More layers = deeper reasoning, more parameters.
- **num_attention_heads** — How many parallel attention heads in each layer. Each head learns different patterns.
- **num_key_value_heads** — Number of KV heads (may differ from Q heads in Grouped Query Attention).
- **head_dim** — Dimension of each attention head. Usually `hidden_size / num_attention_heads`.
- **intermediate_size** — Width of the feed-forward network's hidden layer (typically 4× hidden_size).
- **max_position_embeddings** — Maximum sequence length the model supports (context window).
- **torch_dtype** — Default precision for weights. Affects memory as you learned on Day 5.

## 3. Parameter Count: Computing It From Config

You can calculate the total parameter count directly from the config values. For a standard Llama-style architecture:

```
Embedding:       vocab_size × hidden_size
Per Layer:
  Attention:     4 × hidden_size × hidden_size  (Q, K, V, output projections)
  Feed-Forward:  3 × hidden_size × intermediate_size  (gate, up, down)
  Layer Norms:   2 × hidden_size
Total Layers:    num_hidden_layers × (attention + feed-forward + norms)
LM Head:         hidden_size × vocab_size (often tied to embedding)
```

For SmolLM2-1.7B-Instruct, this works out to approximately 1.7 billion parameters — which is exactly what you see when you call `model.num_parameters()`. The config is the source of truth.

## 4. Comparison Table

| Parameter | Meaning | SmolLM2-1.7B Value | Llama-3.1-8B Value | Effect on Inference |
|-----------|---------|:------------------:|:------------------:|---------------------|
| vocab_size | Token vocabulary | 49,152 | 128,256 | Larger → bigger embedding, slightly more memory |
| hidden_size | Vector width | 2,048 | 4,096 | Wider → more capacity, quadratic memory growth |
| num_hidden_layers | Depth | 24 | 32 | More layers → slower per-token, more capable |
| num_attention_heads | Q heads | 32 | 32 | More heads → richer attention, more compute |
| num_key_value_heads | KV heads | 32 | 8 | Fewer KV heads → less KV cache memory (GQA) |
| intermediate_size | FFN width | 8,192 | 14,336 | Wider → more parameters per layer |
| max_position_embeddings | Context window | 8,192 | 131,072 | Longer → can process more text, more KV memory |
| torch_dtype | Weight precision | bfloat16 | bfloat16 | Lower precision → less memory (Day 5) |

## 5. Why config.json Matters for Inference

Understanding config.json helps you:

1. **Predict memory requirements** — Before downloading a model, check its config to estimate memory: `parameters × bytes_per_param`. On Day 5 you learned that bfloat16 uses 2 bytes per parameter, so SmolLM2's 1.7B params need ~3.4 GB of VRAM/RAM.

2. **Understand KV cache growth** — The KV cache per token depends on `num_key_value_heads × head_dim × num_hidden_layers × 2 (K and V)`. More layers or larger heads = faster memory growth per token generated.

3. **Choose the right model** — Comparing configs tells you exactly why an 8B model needs 4× the memory of a 1.7B model: it's not just "more parameters" — you can trace exactly which dimensions doubled.

4. **Debug loading errors** — When a model fails to load, mismatched config values are often the cause. Knowing what each field means helps you diagnose shape mismatches.

```python
from transformers import AutoConfig

# Load config without downloading weights (fast!)
config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
print(f"Hidden size: {config.hidden_size}")
print(f"Layers: {config.num_hidden_layers}")
print(f"Context window: {config.max_position_embeddings} tokens")
```

## Setup

### Prerequisites

- Python 3.10+
- Completed Day 3 (familiar with `from_pretrained`)
- Completed Day 5 (understand precision and memory)
- ~4 GB disk space (for model download)
- No GPU required — runs on CPU

### Install dependencies

```bash
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — fully open (Apache 2.0), no account needed. It downloads automatically on first run (~3.5 GB).

## What You'll Build

A Python script that loads SmolLM2-1.7B-Instruct's `config.json`, inspects every key parameter, and then validates the theoretical parameter count (calculated from the config values) against the actual parameter count reported by PyTorch — proving that the config really is the blueprint from which the entire model is built.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Every Hugging Face model has a `config.json` that acts as the blueprint — it tells the library how to construct the model from its weight tensors
- You can calculate total parameter count directly from config values: embedding size + (layers × attention params + feed-forward params)
- `hidden_size` is the most impactful parameter — doubling it roughly quadruples the model's total parameters and memory footprint
- `num_key_value_heads` controls KV cache efficiency: fewer KV heads (Grouped Query Attention) means less memory per generated token
- `max_position_embeddings` determines the maximum context window — the upper limit on how many tokens the model can process at once
- You can load just the config (without weights) using `AutoConfig.from_pretrained()` to quickly inspect any model's architecture
