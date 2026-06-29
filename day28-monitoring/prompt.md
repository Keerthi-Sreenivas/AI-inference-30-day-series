# Day 28 Prompt — Monitoring & Observability

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install prometheus-client fastapi uvicorn httpx vllm
```

## Step 2: Start the vLLM backend

The script requires a running vLLM server as the inference backend:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Wait for vLLM to finish loading (you'll see "Application startup complete"). The script will check for vLLM availability before proceeding.

## Step 3: Create the project

Create a `requirements.txt` with:

```
prometheus-client>=0.20.0
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0
vllm>=0.8.0
```

Create a Python script called `main.py` that builds a metrics-instrumented inference API with Prometheus metrics collection. The script creates a FastAPI wrapper around vLLM, instruments it with standard Prometheus metrics (request count, latency histogram, active requests gauge), sends test requests, and validates that the `/metrics` endpoint returns the expected data.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Check vLLM availability** — At startup, verify the vLLM server is running on `localhost:8000`. If not reachable, print a helpful error message with the startup command and exit gracefully.

2. **Create Prometheus metrics** — Define at least 4 Prometheus metrics: a Counter for total requests, a Histogram for request latency (with appropriate bucket boundaries), a Gauge for active/in-flight requests, and a Counter for error count.

3. **Build instrumented FastAPI app** — Create a FastAPI application with a `/generate` endpoint that forwards requests to vLLM while recording all metrics. Each request should increment the counter, observe latency in the histogram, and track active requests via the gauge.

4. **Expose Prometheus metrics endpoint** — Add a `/metrics` route that returns Prometheus-formatted metrics using `generate_latest()` from the prometheus_client library.

5. **Start the API server** — Launch the FastAPI server in a background thread on port 9000 (separate from vLLM's 8000).

6. **Send test requests** — Make at least 5 inference requests through the instrumented API to generate metric data. Print the response status and latency for each request.

7. **Validate metrics output** — Fetch the `/metrics` endpoint and verify it contains the expected metric names (request counter, latency histogram, etc.). Print a pass/fail for each expected metric.

8. **Print metrics summary** — Parse the metrics output and print a human-readable summary: total requests served, average latency, p50/p99 latency from histogram, error rate, and currently active requests.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — define metrics, create app, run requests, validate
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for vLLM availability check)

### Verification

After running the script, you should see:

- Confirmation that vLLM is reachable
- 5+ test requests sent with latency printed for each
- `/metrics` endpoint returning valid Prometheus text format
- A summary showing request count, latency percentiles, and error rate
- All metric validation checks passing (metrics endpoint contains expected names)
