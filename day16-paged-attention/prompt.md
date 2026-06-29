# Day 16 Prompt — PagedAttention & Continuous Batching

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

Install the Python dependencies for this project:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should contain:

```
vllm>=0.8.0
transformers>=5.0.0
torch>=2.4.0
openai>=1.30.0
requests>=2.31.0
```

## Step 2: Start the vLLM server

Start the vLLM inference server with explicit memory configuration:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90
```

Wait until you see "Application startup complete" before running the script.

## Step 3: Create the project

Create a flat Python script called `main.py` that demonstrates PagedAttention's memory efficiency by observing how vLLM handles varying concurrent load. The script should:

1. **Check vLLM server availability** by hitting `http://localhost:8000/health`. If the server is not running, print the startup command and exit.

2. **Define request groups with different output lengths** — create 3 groups: short responses (max_tokens=32), medium (max_tokens=128), and long (max_tokens=512). Use 4 requests per group, 12 total.

3. **Query the vLLM metrics endpoint** at `/metrics` before and after sending requests to observe GPU memory utilization. Parse the Prometheus-format metrics for `vllm:gpu_cache_usage_perc` (or similar KV cache usage metric).

4. **Send all 12 requests concurrently** using ThreadPoolExecutor, timing the entire batch.

5. **Track per-request metrics** — record each request's completion time and token count. Show that short requests complete faster even though they started simultaneously (evidence of continuous batching).

6. **Print a memory utilization summary** showing:
   - GPU KV cache utilization percentage (from the metrics endpoint)
   - Number of concurrent requests handled
   - Comparison of actual memory used vs what static allocation would require

7. **Print the key insight** — calculate and display: "PagedAttention efficiency: X% memory utilization with Y concurrent requests of variable length"

8. **Add educational comments** explaining: what PagedAttention pages are, how continuous batching works, why variable-length requests benefit most, and how this connects to the KV cache concept from Day 12.

Structure: flat script, no classes, linear top-to-bottom execution.

Model: `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0, no gating required)

## Verification

When you run the script with a healthy vLLM server:

- Short requests (32 tokens) should finish noticeably faster than long requests (512 tokens)
- GPU KV cache utilization should be well below 100% even with 12 concurrent requests (showing PagedAttention doesn't pre-allocate max for each)
- Memory utilization percentage should print as: "Memory utilization: XX.X%"
- If all requests were statically allocated at max length (4096 tokens each), you'd need 12×4096 = 49,152 tokens of KV cache. Actual usage should be far less.
