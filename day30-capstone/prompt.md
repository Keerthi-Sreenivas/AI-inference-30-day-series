# Day 30 Prompt — Capstone: Production Inference Service

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install vllm fastapi uvicorn prometheus-client httpx locust pydantic
```

## Step 2: Start the vLLM backend

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Wait for "Application startup complete" before running the script.

## Step 3: Docker Compose (production deployment)

For full production deployment, create a `docker-compose.yml` with three services: `vllm` (inference engine on port 8000 with GPU), `api` (FastAPI gateway on port 9000, depends on vllm), and `prometheus` (metrics on port 9090, scrapes both). For today's lab, the script runs with just vLLM on port 8000 — it launches the FastAPI layer internally.

## Step 4: Create the project

Create a `requirements.txt` with:

```
vllm>=0.8.0
fastapi>=0.115.0
uvicorn>=0.30.0
prometheus-client>=0.20.0
httpx>=0.27.0
locust>=2.28.0
pydantic>=2.0.0
```

Create a Python script called `main.py` that implements a production inference service integrating concepts from across the series: vLLM serving (Day 15), structured output (Day 19), API design (Day 20), load testing (Day 21), and monitoring (Day 28). The service includes health checks, structured JSON output, Prometheus metrics, and a concurrent load test.

The script must be flat and linear — no classes except Pydantic models. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Check vLLM availability** — Verify the vLLM server on port 8000 is healthy. If not, print the startup command and exit gracefully.

2. **Define Pydantic response models** — Create structured output schemas using Pydantic (e.g., a `SummaryResponse` model with fields for `summary`, `key_points`, `word_count`). This enforces type-safe structured output.

3. **Create Prometheus metrics** — Define at least 4 metrics: request counter, latency histogram (with appropriate buckets), active requests gauge, and tokens-generated counter.

4. **Build the FastAPI application** — Create endpoints: `POST /generate` (free-form generation), `POST /structured` (structured JSON output), `GET /health` (health check with vLLM backend validation), `GET /metrics` (Prometheus endpoint).

5. **Implement health check** — The `/health` endpoint must ping the vLLM backend and return overall status (healthy/unhealthy) with component-level details.

6. **Start the API server** — Launch the FastAPI application on port 9000 in a background thread.

7. **Run functional tests** — Test each endpoint: verify `/health` returns healthy, verify `/generate` returns text, verify `/structured` returns valid JSON matching the Pydantic schema.

8. **Run concurrent load test** — Send 10 concurrent requests using threading or asyncio. Measure and print: total requests, successful responses, failed responses, average latency, p50 latency, p99 latency, and throughput (RPS).

9. **Validate monitoring** — Fetch `/metrics` and verify it contains request counts and latency data from the test traffic. Print pass/fail for each expected metric.

10. **Print acceptance criteria results** — Print three acceptance checks: (a) Concurrency: handled 10 concurrent requests with p99 < threshold, (b) Correctness: structured endpoint returns valid JSON, (c) Observability: metrics endpoint reports counts and latencies.

### Deliverables

- A working `main.py` that integrates 5+ concepts from the series
- A `requirements.txt` listing all dependencies
- A `README.md` describing the architecture and how to run
- A `prompt.md` (this file) with full build instructions

### Code style

- Flat script structure — Pydantic models are the only exception (needed for schema)
- Linear top-to-bottom flow — setup, test, load-test, validate
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block per external service dependency

### Verification

After running the script, you should see:

- vLLM connectivity confirmed
- API server started on port 9000
- All functional tests passing (health, generate, structured)
- Load test results showing RPS and latency percentiles for concurrent requests
- Metrics validation passing (all expected metrics present)
- Three acceptance criteria printed with pass/fail status
- Total runtime: 3–10 minutes depending on hardware
