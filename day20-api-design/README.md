# Day 20: Building an Engine-Agnostic Inference API

## The Concept

You've learned to run models with vLLM (Day 15) and SGLang (Day 17), and you've added structured output constraints (Day 19). Now it's time to put a proper API in front of all of it. The problem: different inference engines have different APIs, different response formats, and different quirks. If your application talks directly to vLLM today, switching to SGLang tomorrow means rewriting your client code.

Think of it like a **universal remote** — one interface that controls many TVs. You build a single FastAPI service that exposes a clean, consistent API to your application. Behind it, an **engine adapter** translates requests to whatever backend is running. Swap vLLM for SGLang? Change one config line. Your application code doesn't notice.

This pattern is how production inference services work at scale. The API layer handles authentication, rate limiting, request validation, and response formatting — while the engine layer handles the actual model inference.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│          Engine-Agnostic Inference API Architecture               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                                │
│  │   Client     │  Your app, frontend, other services            │
│  │  (httpx)     │                                                │
│  └──────┬───────┘                                                │
│         │  POST /v1/completions                                   │
│         │  POST /v1/structured                                    │
│         │  GET  /v1/health                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              FastAPI Application                      │        │
│  │                                                      │        │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │        │
│  │  │  Request   │  │   Router &   │  │  Response   │  │        │
│  │  │ Validation │─▶│   Logging    │─▶│ Formatting  │  │        │
│  │  │ (Pydantic) │  │              │  │             │  │        │
│  │  └────────────┘  └──────┬───────┘  └─────────────┘  │        │
│  │                         │                            │        │
│  └─────────────────────────┼────────────────────────────┘        │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────┐        │
│  │             Engine Adapter Layer                      │        │
│  │                                                      │        │
│  │  ┌─────────┐    ┌──────────┐    ┌────────────────┐  │        │
│  │  │  vLLM   │    │  SGLang  │    │  Ollama (CPU)  │  │        │
│  │  │ Adapter │    │ Adapter  │    │    Adapter     │  │        │
│  │  └────┬────┘    └─────┬────┘    └───────┬────────┘  │        │
│  │       │               │                  │           │        │
│  └───────┼───────────────┼──────────────────┼───────────┘        │
│          ▼               ▼                  ▼                    │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐          │
│  │ vLLM Server│  │SGLang Server│  │  Ollama Server   │          │
│  │ :8001      │  │   :8002     │  │     :11434       │          │
│  └────────────┘  └─────────────┘  └──────────────────┘          │
│                                                                  │
│  The API layer is engine-agnostic. Swap backends by changing     │
│  one environment variable: ENGINE=vllm | sglang | ollama         │
└──────────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Purpose | Response Format |
|----------|:------:|---------|----------------|
| `/v1/health` | GET | Liveness + backend status | `{"status": "ok", "engine": "vllm", "model": "..."}` |
| `/v1/models` | GET | List available models | `{"models": [{"id": "...", "engine": "..."}]}` |
| `/v1/completions` | POST | Text generation | `{"text": "...", "tokens": N, "latency_ms": X}` |
| `/v1/structured` | POST | Schema-constrained generation | `{"data": {...}, "valid": true, "latency_ms": X}` |
| `/v1/metrics` | GET | Prometheus-format metrics | Counters, histograms, gauges |

## Why an Adapter Pattern?

Each engine has subtly different APIs:

```python
# vLLM uses OpenAI-compatible format
{"model": "...", "messages": [...], "extra_body": {"guided_json": ...}}

# SGLang uses its own format
{"text": "...", "sampling_params": {"max_new_tokens": 256}}

# Ollama uses yet another format
{"model": "...", "prompt": "...", "stream": false}
```

The adapter translates your unified request format into whatever the backend expects. This is the **Strategy pattern** — pick the right strategy (adapter) at startup based on configuration.

## Measuring API Performance

For a production API, you care about latency percentiles:

- **p50 (median)**: What most requests experience
- **p95**: The "slow" requests — 1 in 20 hits this or worse
- **p99**: Tail latency — how bad can it get?

```
p50 = 200ms   ← Half your users see this or better
p95 = 450ms   ← 95% of requests finish in under 450ms
p99 = 1200ms  ← 1% of requests take over a second (cold cache, long prompt)
```

The gap between p50 and p99 tells you about consistency. A large gap means some requests hit cold model paths or queue behind batch processing.

## Setup

### Prerequisites

- Completed Day 15 (vLLM), Day 17 (SGLang), Day 19 (structured output)
- NVIDIA GPU with 16+ GB VRAM
- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Start a Backend Engine

You need at least one inference engine running. vLLM is the default:

```bash
# Start vLLM as the backend (on a different port than our API)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --guided-decoding-backend outlines
```

Our FastAPI app will run on port 8000 and forward requests to port 8001.

### Run the API Server

```bash
# Start the API (uses uvicorn under the hood)
python main.py --serve
```

### Run the Benchmark

```bash
# In a second terminal, run the latency benchmark
python main.py --benchmark
```

## Design Decisions

**Why FastAPI?** It's the standard for Python APIs — async, fast, auto-generates OpenAPI docs, Pydantic integration for request/response validation.

**Why not just use vLLM's built-in API directly?** You could, but then you're locked in. The adapter layer costs near-zero latency (one local function call) and gives you engine portability, custom auth, rate limiting, and unified logging.

**Why measure p50/p99 instead of average?** Averages hide tail latency. A p50 of 200ms with p99 of 5000ms means your users regularly experience 5-second hangs — but the average might look fine at 400ms.

## What You'll Build

A FastAPI inference service with an engine adapter for vLLM, plus a benchmarking client that sends concurrent requests and reports p50/p95/p99 latency percentiles. The API validates requests with Pydantic, routes to the backend, and formats consistent responses regardless of which engine is behind it.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- An engine-agnostic API layer decouples your application from a specific inference backend
- The adapter pattern translates a unified request format into engine-specific calls
- FastAPI + Pydantic gives you request validation, OpenAPI docs, and async support out of the box
- Measure p50/p95/p99 latency — averages hide tail latency problems
- The API layer adds minimal overhead (sub-millisecond) compared to model inference time
- This pattern enables zero-downtime engine swaps in production
