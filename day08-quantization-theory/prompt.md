# Day 8 Prompt — Quantization Theory

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch accelerate bitsandbytes
```

**Note:** `bitsandbytes` requires an NVIDIA GPU with CUDA. If you don't have one, the script should detect this and exit gracefully with a message explaining what you'd see.

## Step 2: Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — it downloads automatically on first run (~3.5 GB). No Hugging Face account or token needed (Apache 2.0, fully open).

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
bitsandbytes>=0.44.0
```

Create a Python script called `main.py` that measures how quantization affects model quality by computing perplexity at different bit widths. Use `AutoModelForCausalLM`, `AutoTokenizer`, and `BitsAndBytesConfig` from the `transformers` library with the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Check GPU availability** — At the top of the script, check if a CUDA-capable GPU is available using `torch.cuda.is_available()`. If no GPU is found, print a clear message explaining that bitsandbytes requires an NVIDIA GPU, show what the expected output would look like, and exit gracefully.

2. **Define a fixed evaluation text** — Choose a paragraph of natural English text (50–100 tokens) that will be used consistently across all bit widths to compute perplexity. This ensures a fair comparison.

3. **Load model at FP16** — Load `HuggingFaceTB/SmolLM2-1.7B-Instruct` in FP16 precision (the baseline). Print memory usage and compute perplexity on the evaluation text.

4. **Load model at INT8** — Reload the model with `load_in_8bit=True` via `BitsAndBytesConfig`. Print memory usage and compute perplexity on the same evaluation text.

5. **Load model at INT4 (NF4)** — Reload the model with `load_in_4bit=True` and `bnb_4bit_quant_type="nf4"` via `BitsAndBytesConfig`. Print memory usage and compute perplexity on the same evaluation text.

6. **Compute perplexity** — For each loaded model, compute perplexity as `exp(average_cross_entropy_loss)` over the evaluation text. Use the model's own loss computation (pass `labels=input_ids` to the forward pass).

7. **Print comparison table** — At the end, print a formatted table showing: bit width, memory usage (MB), perplexity value, and perplexity increase vs FP16 baseline — for all three configurations side by side.

8. **Clean up between loads** — Delete each model and call `torch.cuda.empty_cache()` before loading the next quantization level to avoid running out of VRAM.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — GPU check, then load/measure/cleanup for each bit width in sequence
- Educational inline comments — at least one comment per logical block (GPU check, loading, perplexity computation, cleanup)
- At most one `try/except` block (for the initial model load, in case download fails)

### Verification

After running the script (on a CUDA GPU), you should see:

- Memory usage decreasing at each quantization level (FP16 ~3.4 GB → INT8 ~1.7 GB → INT4 ~0.85 GB)
- Perplexity increasing slightly at each level (e.g., FP16 baseline ≈ 8–12, INT8 ≈ +0.1–0.5, INT4 ≈ +0.3–1.5)
- A clear comparison table at the end showing the trade-offs
- Total runtime: 2–5 minutes (model loads dominate)
