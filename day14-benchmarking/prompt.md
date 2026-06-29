# Day 14 Prompt — Benchmarking Inference

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch accelerate psutil
```

## Step 2: Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — it downloads automatically on first run (~3.5 GB). No Hugging Face account or token needed (Apache 2.0, fully open). If you've cached it from Day 4 or Day 12, no re-download is needed.

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
psutil>=5.9.0
```

Create a Python script called `main.py` that benchmarks inference performance for a language model, measuring all key metrics: Time to First Token (TTFT), tokens per second (TPS), total latency, and peak memory usage. Use `AutoModelForCausalLM` and `AutoTokenizer` from the `transformers` library with the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Load model and tokenizer** — Load `HuggingFaceTB/SmolLM2-1.7B-Instruct` using `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()`. Detect whether GPU or CPU is being used and print the device info.

2. **Warm-up run** — Perform 1-2 short throwaway generations before actual measurement to eliminate first-run overhead (CUDA kernel caching, JIT compilation, memory allocation). Print a message indicating warm-up is happening.

3. **Measure Time to First Token (TTFT)** — Use a streamer callback or two-phase timing approach to isolate the time from prompt submission to when the first output token is produced. This measures the prefill phase.

4. **Measure Tokens Per Second (TPS)** — Time the full generation with a fixed `max_new_tokens=100`, subtract TTFT to get pure decode time, then compute `(total_tokens - 1) / decode_time`. Report as steady-state decode speed.

5. **Measure Total Latency** — Record wall-clock time from the start of `model.generate()` to completion. This is the end-to-end time a user would wait for the full response.

6. **Measure Peak Memory Usage** — Use `torch.cuda.max_memory_allocated()` on GPU or `psutil.Process().memory_info().rss` on CPU to capture peak memory during generation. Report in MB.

7. **Run multiple trials** — Repeat the benchmark 3 times and compute mean values for each metric. This accounts for variance from OS scheduling and thermal effects.

8. **Print comprehensive benchmark table** — Display a formatted table with columns: Metric, Mean Value, Unit. Include TTFT, TPS, total latency, peak memory, input tokens, output tokens, and device info.

9. **Report hardware context** — Print model name, device type, parameter count, input prompt length (in tokens), and output length at the top of results so numbers are reproducible.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — load model, warm up, benchmark, report
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for the initial model load)
- Use `time.perf_counter()` for timing (highest resolution cross-platform timer)
- Use `psutil` for CPU memory measurement (cross-platform)

### Verification

After running the script, you should see:

- A warm-up notice followed by the actual benchmark runs
- A clear metrics table showing TTFT (in milliseconds), TPS (tokens/sec), total latency (seconds), and peak memory (MB)
- Hardware context printed at the top (model name, device, parameter count)
- All metrics averaged over 3 trials for reliability
- Total script runtime: 2-8 minutes depending on hardware (CPU slower, GPU faster)
