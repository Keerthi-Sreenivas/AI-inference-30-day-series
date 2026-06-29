# Day 4: Sampling Strategies — Temperature, Top-K, Top-P

## The Concept

When a language model generates text, it doesn't pick the next word deterministically. At each step, the model produces a probability distribution over its entire vocabulary — tens of thousands of possible next tokens, each with a score. **Sampling** is how you choose which token actually gets selected from that distribution.

Think of it like a thermostat. Set the temperature low and the room stays stable, predictable — you know what you're getting. Crank it up and things get chaotic: creative, surprising, sometimes nonsensical. Temperature is just one of three knobs you'll learn today. Together, temperature, top-k, and top-p give you precise control over the creativity-coherence tradeoff in every inference call.

On Day 1 you used Ollama's `/api/generate` endpoint, which handled sampling behind the scenes with default parameters. On Day 3 you saw `model.generate()` in Hugging Face Transformers — today you'll finally understand what those `temperature`, `top_k`, and `top_p` arguments actually do to the token selection pipeline, and you'll measure how they affect both output quality and generation speed.

## 1. The Token Selection Pipeline

Every time the model predicts the next token, the raw scores (called **logits**) pass through a pipeline of transformations before a token is chosen. Here's the full flow:

```
┌─────────────────────────────────────────────────────────────────┐
│              Token Selection Pipeline (Data Flow)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model Forward Pass                                             │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐     Raw scores for all ~32,000 tokens             │
│  │  Logits  │     e.g., "the": 4.2, "cat": 3.8, "xyz": -1.5   │
│  └────┬─────┘                                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Divide logits by temperature            │
│  │  Temperature ÷ T │   Low T → sharpen distribution            │
│  │                  │   High T → flatten distribution            │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Keep only top-K highest-scoring tokens  │
│  │  Top-K Filter    │   K=5: only 5 candidates remain           │
│  │                  │   K=50: 50 candidates remain              │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Keep tokens until cumulative prob ≥ P   │
│  │  Top-P (Nucleus) │   P=0.5: only most likely ~3-5 tokens     │
│  │                  │   P=0.95: keeps ~50-200 tokens             │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Re-normalize remaining probabilities    │
│  │  Softmax         │   Probabilities now sum to 1.0            │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Randomly sample from the distribution   │
│  │  Sample Token    │   Higher-probability tokens more likely   │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  Selected token → append to sequence → repeat                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Important:** These filters stack. The model first adjusts logits by temperature, then top-k removes low-ranking tokens, then top-p removes tokens below the cumulative probability threshold. Only the survivors participate in the final random sample.

## 2. Temperature — The Creativity Dial

**Temperature** (usually written as `T`) scales the logits before the softmax function converts them into probabilities. Mathematically, each logit is divided by T before exponentiation.

- **T < 1.0** — Sharpens the distribution. The highest-scoring token gets an even larger share of probability. Output becomes repetitive and "safe."
- **T = 1.0** — No change. The model's original confidence levels are used as-is.
- **T > 1.0** — Flattens the distribution. Low-scoring tokens get a bigger chance. Output becomes creative but potentially incoherent.
- **T → 0** — Approaches greedy decoding (always pick the highest-scoring token).

**The thermostat analogy:** Imagine the temperature dial on your wall. Set it to 18°C (low) — the room is stable, predictable, comfortable. Set it to 35°C (high) — things get chaotic, unpredictable, maybe uncomfortable. Language model temperature works the same way: low values produce boring-but-reliable text, high values produce surprising-but-risky text.

```python
# Low temperature: focused, repetitive
output = model.generate(input_ids, temperature=0.1, max_new_tokens=50)

# High temperature: creative, diverse
output = model.generate(input_ids, temperature=1.5, max_new_tokens=50)
```

## 3. Top-K — The Shortlist Filter

**Top-K sampling** limits the selection to only the K highest-scoring tokens at each step. All other tokens are set to probability zero, regardless of their original scores.

- **K = 1** — Greedy decoding. Always picks the single best token.
- **K = 5** — Only 5 candidates. Very focused output.
- **K = 50** — Standard default. Balances creativity and coherence.
- **K = 100+** — Rarely needed. Allows rare tokens to appear.

**The restaurant analogy:** Top-K is like a restaurant with a limited menu. K=5 means only 5 dishes to choose from — quick decision, predictable meal. K=100 is the full buffet — more variety, but you might end up with something weird on your plate.

The downside of top-K: it's a fixed number. If the model is very confident (one token dominates), K=50 might include irrelevant junk. If the model is uncertain (many tokens are plausible), K=5 might cut off good options. That's where top-P helps.

## 4. Top-P (Nucleus Sampling) — The Dynamic Filter

**Top-P sampling** (also called nucleus sampling) keeps the smallest set of tokens whose cumulative probability exceeds threshold P. Unlike top-K, the number of candidates changes dynamically based on the distribution shape.

- **P = 0.5** — Only tokens covering the top 50% of probability mass. Usually just 3-5 tokens.
- **P = 0.9** — Covers 90% of mass. Typically 20-100 tokens depending on context.
- **P = 0.95** — The most common default. Generous but still filters outliers.
- **P = 1.0** — No filtering. All tokens are candidates.

**The betting analogy:** Top-P is like betting only on favorites. P=0.5 means you only consider the frontrunners who together account for 50% of winning probability. P=0.95 means you also consider longshots — more exciting, more risk.

The advantage over top-K: it adapts. When the model is confident, top-P might keep only 3 tokens. When the model is uncertain, it might keep 200. This adapts naturally to the distribution shape at each step.

## 5. Comparison Table

| Parameter | Low Value | High Value | Effect on Output | Analogy |
|-----------|-----------|------------|------------------|---------|
| Temperature | 0.1 | 1.5 | Focused/repetitive → Creative/chaotic | Thermostat: low = predictable winter, high = wild weather |
| Top-K | 5 | 100 | Few safe choices → Many diverse choices | Menu: 5-item prix fixe vs full buffet |
| Top-P | 0.5 | 0.95 | Only frontrunners → Includes longshots | Betting: favorites only vs all plausible contenders |

### Common Presets

| Use Case | Temperature | Top-K | Top-P | Why |
|----------|:-----------:|:-----:|:-----:|-----|
| Code generation | 0.1–0.3 | 10 | 0.9 | Correctness matters more than creativity |
| Creative writing | 0.8–1.2 | 50 | 0.95 | Diversity and surprise are valued |
| Chat / assistant | 0.6–0.8 | 40 | 0.9 | Balanced: helpful but not boring |
| Summarization | 0.2–0.4 | 10 | 0.8 | Stay faithful to the source |

## 6. Interaction Effects

The three parameters interact — they're applied in sequence, not independently. Some combinations to be aware of:

- **Temperature 0.1 + Top-P 0.95** — Temperature already sharpens so much that top-P has little effect. The top token dominates regardless.
- **Temperature 1.5 + Top-K 5** — Temperature flattens the distribution, but top-K only keeps 5 tokens. You get diversity among a small set.
- **Temperature 1.0 + Top-K 0 + Top-P 1.0** — No modifications at all. Pure sampling from the model's raw distribution.

```python
# The Transformers generate() function accepts all three together
output = model.generate(
    input_ids,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    do_sample=True,      # Must be True for sampling (vs greedy)
    max_new_tokens=100,
)
```

**Key detail:** In Hugging Face Transformers, you must set `do_sample=True` to activate sampling. Without it, the model defaults to greedy decoding and ignores temperature/top-k/top-p entirely.

## Setup

### Prerequisites

- Python 3.10+
- ~4 GB disk space (for model download)
- No GPU required — runs on CPU (slower but works)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — a 1.7-billion parameter model that's fully open (Apache 2.0 license), no account or token needed. It downloads automatically on first run (~3.5 GB).

If you want faster generation, a GPU with 4+ GB VRAM will speed things up significantly, but it's not required.

## What You'll Build

A Python script that generates text at five different temperatures (0.1, 0.5, 0.7, 1.0, 1.5) using the same prompt, prints the output at each setting, and measures tokens per second for each run — giving you a quantitative sense of how sampling parameters affect both quality and performance.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Sampling controls how a token is chosen from the model's probability distribution — it's not random, it's controlled randomness
- Temperature scales logits before softmax: low = focused, high = creative, zero = greedy
- Top-K keeps only the K highest-scoring candidates — simple but rigid
- Top-P (nucleus) keeps candidates until cumulative probability exceeds a threshold — adapts dynamically to model confidence
- The three parameters stack in order: temperature → top-K → top-P → sample
- Always set `do_sample=True` in Transformers to activate sampling — without it, the model ignores your temperature/top-k/top-p settings
