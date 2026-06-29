# Day 29 Prompt — Cost Optimization

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install vllm requests psutil
```

## Step 2: Start inference engine(s)

The script benchmarks against a running vLLM server. Optionally start SGLang too for comparison:

```bash
# Primary: vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Optional: SGLang (for engine comparison)
# python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --port 8001
```

Wait for "Application startup complete" before running the script.

## Step 3: Create the project

Create a `requirements.txt` with:

```
vllm>=0.8.0
requests>=2.31.0
psutil>=5.9.0
```

Create a Python script called `main.py` that analyzes the cost of inference across different deployment scenarios. The script benchmarks throughput on available engines, estimates hardware costs, and calculates the cost per 1,000 tokens for each scenario — showing when self-hosted is cheaper than cloud APIs.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Check engine availability** — Test connectivity to vLLM on port 8000 (required) and SGLang on port 8001 (optional). Print which engines are available. If vLLM is not running, exit with a helpful error message.

2. **Benchmark vLLM throughput** — Send 10 sequential requests (short prompts, 50 tokens max each) to vLLM. Measure total tokens generated and total time. Calculate tokens per second. Print the throughput metric.

3. **Benchmark SGLang throughput (if available)** — If SGLang is running, perform the same benchmark. Otherwise print "SGLang not available — skipping comparison."

4. **Define cost scenarios** — Set up a cost model with at least 4 scenarios: (a) Cloud API (e.g., $0.015/1K tokens), (b) Self-hosted A100 at $2.50/hour, (c) Self-hosted L4 at $0.80/hour, (d) CPU-only at $0.05/hour. Use measured throughput for self-hosted calculations.

5. **Calculate cost per 1K tokens** — For each self-hosted scenario, compute: `(hourly_cost / tokens_per_hour) * 1000`. For cloud API, use the listed price directly. Print a comparison table of all scenarios.

6. **Calculate break-even point** — Determine the daily token volume at which self-hosted becomes cheaper than the cloud API. Print this threshold clearly.

7. **Print optimization recommendations** — Based on the measured throughput, print recommendations for different usage tiers: low (<100K tokens/day), medium (100K–5M), high (>5M).

8. **Print summary table** — Output a formatted table showing: engine, throughput (tok/s), cost per 1K tokens, monthly cost at 1M tokens/day, and a "best for" column.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — check engines, benchmark, calculate, recommend
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for engine availability check)

### Verification

After running the script, you should see:

- Confirmation of which engines are available
- Measured throughput (tokens/sec) for each available engine
- A cost comparison table with at least 4 deployment scenarios
- A clear break-even point (tokens/day threshold)
- Tier-based recommendations for different usage levels
- Total runtime should be under 5 minutes
