# Day 5 Prompt — Model Formats & Precision

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch accelerate psutil safetensors
```

## Step 2: Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — it downloads automatically on first run (~3.5 GB). No Hugging Face account or token needed (Apache 2.0, fully open).

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
psutil>=5.9.0
```

Create a Python script called `main.py` that demonstrates how precision formats affect model memory usage. Use `AutoModelForCausalLM` and `AutoTokenizer` from the `transformers` library with the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model. Load the same model at different precisions (FP32, FP16, BF16) and compare memory consumption.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Measure baseline memory** — Before loading any model, record the current process memory usage in MB using `psutil`. This gives you a clean baseline for comparison.

2. **Load model in FP32** — Load `HuggingFaceTB/SmolLM2-1.7B-Instruct` with default settings (FP32 precision). Measure and print the memory increase in MB. Then delete the model and free memory.

3. **Load model in FP16** — Load the same model with `torch_dtype=torch.float16`. Measure and print the memory increase in MB. Then delete the model and free memory.

4. **Load model in BF16** — Load the same model with `torch_dtype=torch.bfloat16`. Measure and print the memory increase in MB. Then delete the model and free memory.

5. **Generate a test response** — After the final load (BF16), generate a short response (50 tokens) to the prompt "What is machine learning?" to verify the model works correctly at reduced precision. Print the response.

6. **Print a memory comparison table** — At the end, print a formatted comparison table showing each precision format alongside its memory usage in MB and the percentage of FP32 memory used.

7. **Print theoretical vs actual** — Calculate the theoretical memory (parameter_count × bits / 8 / 1024²) for each format and print it alongside the actual measured value, noting that the difference comes from framework overhead (activations, buffers, Python objects).

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — measure baseline, load each format, generate, print summary
- Educational inline comments — at least one comment per logical block (memory measurement, loading, cleanup, comparison)
- At most one `try/except` block (for the initial model load, in case download fails)

### Verification

After running the script, you should see:

- A baseline memory measurement in MB
- Three separate memory measurements (FP32, FP16, BF16) showing decreasing memory usage
- FP16 and BF16 should use approximately 50% of FP32 memory
- A generated text response proving the model works at reduced precision
- A final comparison table with both actual and theoretical memory values
- Total script runtime: 3–15 minutes depending on hardware (CPU vs GPU)
