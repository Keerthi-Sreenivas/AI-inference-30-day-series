# Day 23 Prompt — ONNX Runtime

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install optimum onnxruntime transformers torch
```

## Step 2: Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — it downloads automatically on first run (~3.5 GB). The ONNX export adds additional disk usage (~3 GB). No Hugging Face account or token needed (Apache 2.0, fully open).

## Step 3: Create the project

Create a `requirements.txt` with:

```
optimum>=1.20.0
onnxruntime>=1.17.0
transformers>=5.0.0
torch>=2.4.0
```

Create a Python script called `main.py` that benchmarks ONNX Runtime inference against standard PyTorch inference for text generation. Use `HuggingFaceTB/SmolLM2-1.7B-Instruct` for both paths.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Load PyTorch model** — Load `HuggingFaceTB/SmolLM2-1.7B-Instruct` using standard `AutoModelForCausalLM.from_pretrained()`. Print the model size and device.

2. **Load ONNX model** — Load the same model using `optimum`'s `ORTModelForCausalLM.from_pretrained()` with `export=True` to convert to ONNX format. Print confirmation of successful export.

3. **Define benchmark prompts** — Create 3 test prompts of varying lengths (short, medium, long input) to test both models under different conditions.

4. **Benchmark PyTorch** — For each prompt, generate 100 tokens using the PyTorch model. Measure wall-clock time and compute tokens per second. Run each prompt 3 times and take the median.

5. **Benchmark ONNX Runtime** — For the same prompts, generate 100 tokens using the ONNX model. Measure wall-clock time and compute tokens per second. Run each prompt 3 times and take the median.

6. **Calculate speedup** — For each prompt, compute `onnx_tokens_per_sec / pytorch_tokens_per_sec` as the speedup factor. Print speedup for each prompt.

7. **Print summary table** — Display a final comparison table showing prompt length, PyTorch tokens/sec, ONNX tokens/sec, and speedup factor (×).

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — load models, benchmark, compare, summarize
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for model loading)

### Verification

After running the script, you should see:

- Successful ONNX export confirmation
- Per-prompt timing results for both PyTorch and ONNX Runtime
- Speedup factors printed for each prompt (expect 1.1–2.0× on CPU)
- A final summary table comparing both runtimes
- Total script runtime of 5–15 minutes (includes export time)
