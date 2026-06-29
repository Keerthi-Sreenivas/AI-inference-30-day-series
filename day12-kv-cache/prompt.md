# Day 12 Prompt — The KV Cache

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch accelerate
```

## Step 2: Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — it downloads automatically on first run (~3.5 GB). No Hugging Face account or token needed (Apache 2.0, fully open).

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
```

Create a Python script called `main.py` that demonstrates KV cache behavior during text generation by measuring memory growth per token. Use `AutoModelForCausalLM` and `AutoTokenizer` from the `transformers` library with the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Load model and tokenizer** — Load `HuggingFaceTB/SmolLM2-1.7B-Instruct` using `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()`. Print model parameter count, number of layers, and hidden dimension to establish the baseline for cache size calculations.

2. **Calculate theoretical cache size** — Before generating, compute and print the theoretical KV cache memory per token based on the model's architecture (num_layers, num_kv_heads, head_dim, dtype). This gives a prediction to compare against measured values.

3. **Generate tokens incrementally with cache** — Generate tokens one at a time using `model.generate()` with `use_cache=True`, capturing the `past_key_values` at each step. Generate at least 50 tokens to observe the growth pattern clearly.

4. **Measure actual KV cache memory at each step** — After each token is generated, calculate the actual memory consumed by the KV cache tensors (sum of tensor sizes in the `past_key_values` tuple). Print the cache size in MB at regular intervals (every 10 tokens).

5. **Print memory growth per token** — Calculate and print the incremental memory added per token (in MB). This is the key metric — it should be approximately constant, confirming linear growth.

6. **Compare cached vs uncached generation speed** — Time a short generation (20 tokens) with `use_cache=True` and with `use_cache=False`. Print both times and the speedup factor to demonstrate why caching matters.

7. **Print a summary** — At the end, print a formatted summary showing: theoretical vs measured cache per token, total cache size at final sequence length, and the speedup from using the cache.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — load model, compute theory, generate with measurement, compare, summarize
- Educational inline comments — at least one comment per logical block (loading, calculating, generating, measuring, comparing)
- At most one `try/except` block (for the initial model load, in case download fails)

### Verification

After running the script, you should see:

- Model architecture info (layers, hidden dim, attention heads)
- Theoretical KV cache size per token (in KB or MB)
- A series of measurements showing cache size growing at each interval
- A constant (or near-constant) memory increment per token, confirming linear growth
- A speed comparison showing that cached generation is significantly faster than uncached
- Total script runtime should be 3–10 minutes depending on hardware (CPU vs GPU)
