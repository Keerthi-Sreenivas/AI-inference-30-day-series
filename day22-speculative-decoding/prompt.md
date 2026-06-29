# Day 22 Prompt — Speculative Decoding

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install vllm transformers torch requests
```

## Step 2: Start the vLLM server with speculative decoding

You need two terminal sessions. In the first, start vLLM with a draft model:

```bash
# Terminal 1: Start vLLM with speculative decoding
vllm serve meta-llama/Llama-3.1-8B \
    --speculative-model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --num-speculative-tokens 5 \
    --port 8000
```

Then start a baseline server (no speculation) on a different port:

```bash
# Terminal 2: Baseline server without speculative decoding
vllm serve meta-llama/Llama-3.1-8B --port 8001
```

## Step 3: Create the project

Create a `requirements.txt` with:

```
vllm>=0.8.0
transformers>=5.0.0
torch>=2.4.0
requests>=2.31.0
```

Create a Python script called `main.py` that measures speculative decoding performance by comparing vLLM with and without a draft model. Use the OpenAI-compatible API at `http://localhost:8000` (with speculation) and `http://localhost:8001` (baseline).

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Check server availability** — Verify both vLLM endpoints are reachable at startup. If either is down, print an informative error message with the startup commands and exit gracefully.

2. **Define test prompts** — Create a list of 3 prompts that represent different difficulty levels: one factual (high acceptance expected), one creative (lower acceptance expected), and one code-generation task (medium acceptance expected).

3. **Benchmark baseline** — For each prompt, send a completion request to the baseline server (port 8001) with `max_tokens=200` and `temperature=0.7`. Measure and record the total time and tokens generated.

4. **Benchmark speculative** — For the same prompts, send requests to the speculative server (port 8000) with identical parameters. Measure total time and tokens generated.

5. **Calculate speedup** — For each prompt, compute the speedup ratio: `baseline_time / speculative_time`. Print the speedup factor for each prompt type.

6. **Estimate acceptance rate** — Use the vLLM metrics endpoint (`/metrics`) to extract the speculative decoding acceptance rate. Print it as a percentage.

7. **Print summary** — Display a final table showing prompt type, baseline tokens/sec, speculative tokens/sec, speedup factor, and estimated acceptance rate.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — check servers, benchmark baseline, benchmark speculative, compare
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for server connectivity check)

### Verification

After running the script, you should see:

- Confirmation that both servers are reachable
- Per-prompt timing results for baseline vs speculative decoding
- Acceptance rate printed as a percentage (typically 50–80% depending on prompt type)
- A summary table showing speedup factors (expect 1.3–2.5× for well-matched prompts)
- Factual prompts should show higher acceptance rates than creative prompts
