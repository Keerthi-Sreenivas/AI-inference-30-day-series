# Day 4 Prompt — Sampling Strategies

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

Create a Python script called `main.py` that explores how temperature, top-k, and top-p sampling parameters affect text generation. Use `AutoModelForCausalLM` and `AutoTokenizer` from the `transformers` library with the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Load model and tokenizer** — Load `HuggingFaceTB/SmolLM2-1.7B-Instruct` using `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()`. Print the model's parameter count and device.

2. **Define a fixed prompt** — Use the chat template to format this user message: `"Explain why the sky is blue in one paragraph."` This gives a consistent baseline across all sampling variations.

3. **Generate at five temperatures** — Loop over temperatures `[0.1, 0.5, 0.7, 1.0, 1.5]`. For each temperature, generate 100 new tokens with `do_sample=True`, `top_k=50`, `top_p=0.9`. Print the temperature value, the generated text, and the tokens per second.

4. **Measure tokens per second** — For each generation, time the `model.generate()` call and compute `tokens_generated / elapsed_seconds`. Print this as a clear metric line: `"Temperature {T}: {X:.1f} tokens/sec"`.

5. **Demonstrate top-K effect** — Generate with a fixed temperature of 0.7 but vary top-K across `[5, 20, 50, 100]`. Print each result with its top-K value and tokens/sec.

6. **Demonstrate top-P effect** — Generate with a fixed temperature of 0.7 and top-K of 50, but vary top-P across `[0.5, 0.8, 0.9, 0.95]`. Print each result with its top-P value and tokens/sec.

7. **Print a summary table** — At the end, print a formatted summary showing all temperature runs with their tokens/sec metric side by side, so the learner can compare performance at a glance.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — load model, then run experiments in sequence
- Educational inline comments — at least one comment per logical block (loading, tokenizing, generating, measuring)
- At most one `try/except` block (for the initial model load, in case download fails)

### Verification

After running the script, you should see:

- Five text outputs at different temperatures (0.1 should be focused/repetitive, 1.5 should be creative/chaotic)
- A tokens/sec metric printed for each generation call
- The summary table at the end comparing all temperature settings
- Total script runtime should be 2–10 minutes depending on hardware (CPU vs GPU)
