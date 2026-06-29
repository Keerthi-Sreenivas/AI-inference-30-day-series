# Day 30: Capstone — Production Inference Service from Scratch

## The Concept

Today you bring everything together. A **production inference service** isn't just a model running on a GPU — it's an orchestrated system with an inference engine, an API layer that validates requests and structures responses, a monitoring stack that tracks health and performance, and load testing that proves it handles real traffic. Each piece depends on the others: without monitoring, you can't detect degradation; without structured output, clients get unpredictable formats; without load testing, you don't know your actual capacity.

Think of building a restaurant. The kitchen (inference engine) cooks the food. The menu and waitstaff (API layer) take orders, validate them, and deliver plated dishes in a consistent format. The health inspector (monitoring) checks temperatures, cleanliness, and processes regularly. And the fire-code capacity test (load testing) determines how many diners you can safely serve. A restaurant missing any of these fails — and so does a production inference service.

On Day 15 you launched vLLM as an inference engine. On Day 19 you constrained output to valid JSON. On Day 20 you designed an API with proper endpoints. On Day 21 you load-tested that API to find its breaking point. On Day 28 you added Prometheus monitoring. Today you wire all five together into a single cohesive system and prove it works under load.

## 1. System Architecture

The capstone system has three components communicating over HTTP, with Prometheus pulling metrics from both the API and inference layers:

```
┌─────────────────────────────────────────────────────────────────┐
│          Production Inference Service Architecture               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Clients (Load Test / Users)                                    │
│       │                                                         │
│       │  HTTP POST /generate, /structured                       │
│       ▼                                                         │
│  ┌──────────────────────────┐                                   │
│  │  FastAPI Gateway         │ Port 9000                         │
│  │  ┌────────────────────┐  │                                   │
│  │  │ Request Validation │  │  Pydantic schemas                 │
│  │  │ Rate Limiting      │  │  Concurrent request tracking      │
│  │  │ Prometheus Metrics │  │  Counter, Histogram, Gauge        │
│  │  │ Health Check       │──┼──── Pings vLLM /health            │
│  │  └────────────────────┘  │                                   │
│  │       │                  │                                   │
│  │       │ POST /v1/completions                                 │
│  │       ▼                  │                                   │
│  └───────┼──────────────────┘                                   │
│          │                                                      │
│          │  HTTP (internal network)                              │
│          ▼                                                      │
│  ┌──────────────────────────┐                                   │
│  │  vLLM Inference Engine   │ Port 8000                         │
│  │  ┌────────────────────┐  │                                   │
│  │  │ Qwen2.5-7B-Instruct│  │  Model serving                  │
│  │  │ Continuous Batching │  │  PagedAttention                  │
│  │  │ Built-in /metrics  │  │  GPU cache, queue depth           │
│  │  └────────────────────┘  │                                   │
│  └──────────────────────────┘                                   │
│                                                                 │
│          ▲                  ▲                                    │
│          │  Scrape /metrics │  Scrape /metrics                   │
│          │  (every 15s)     │  (every 15s)                       │
│          │                  │                                    │
│  ┌──────────────────────────┐                                   │
│  │  Prometheus             │ Port 9090                           │
│  │  ┌────────────────────┐  │                                   │
│  │  │ Time-series DB     │  │  Stores all metrics               │
│  │  │ PromQL queries     │  │  Powers dashboards + alerts        │
│  │  │ Alert Rules        │  │  Fires on threshold breaches      │
│  │  └────────────────────┘  │                                   │
│  └──────────────────────────┘                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Component Roles

| Component | Role | Technology | Alternative |
|-----------|------|-----------|-------------|
| Inference Engine | Model serving, batching, GPU management | vLLM | SGLang, TensorRT-LLM, Ollama |
| API Gateway | Request routing, validation, structured output | FastAPI + Pydantic | Flask, Django, Express |
| Monitoring | Metrics collection, alerting, dashboards | Prometheus + Grafana | Datadog, New Relic, CloudWatch |
| Load Testing | Performance validation, capacity planning | Locust / concurrent scripts | k6, wrk, Apache Bench |
| Orchestration | Multi-service deployment, networking | Docker Compose | Kubernetes, Nomad, systemd |

## 3. Concepts Integrated

This capstone actively exercises at least 7 concepts from prior days:

| Day | Concept | How It's Used in Capstone |
|-----|---------|--------------------------|
| 15 | vLLM | Inference engine serving Qwen2.5-7B |
| 19 | Structured Output | Pydantic models enforce JSON response schema |
| 20 | API Design | FastAPI endpoints with proper status codes and documentation |
| 21 | Load Testing | Concurrent request simulation measuring RPS and latency |
| 28 | Monitoring | Prometheus counters, histograms, and gauges on every request |
| 4 | Sampling | Temperature and top-p parameters exposed through API |
| 14 | Benchmarking | Latency percentiles (p50, p99) calculated from load test |

## 4. Self-Verification Acceptance Criteria

Your capstone must pass these three objective tests:

### Criterion 1: Concurrency & Load
> The service handles 10 concurrent requests with p99 latency under 30 seconds. All requests return HTTP 200.

This proves your API and inference engine handle parallel load without crashing, timing out, or returning errors.

### Criterion 2: Functional Correctness
> The `/structured` endpoint returns valid JSON matching the defined Pydantic schema. The response includes all required fields with correct types.

This proves structured output works end-to-end: the model generates text, the API parses and validates it against the schema.

### Criterion 3: Monitoring Observability
> The `/metrics` endpoint returns Prometheus-formatted data including `inference_requests_total` (count > 0) and `inference_request_duration_seconds` (histogram with observations).

This proves monitoring is wired correctly: every request updates the metrics, and the metrics endpoint exposes them in standard format.

## 5. Production Deployment (Docker Compose)

The Docker Compose file defines the full production stack as three networked services:

```yaml
version: "3.8"
services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    command: ["--model", "Qwen/Qwen2.5-7B-Instruct", "--port", "8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  api:
    build: .
    ports:
      - "9000:9000"
    environment:
      - VLLM_URL=http://vllm:8000
    depends_on:
      - vllm

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

Start everything with `docker compose up -d`, then run the load test against `localhost:9000`.

## Setup

### Prerequisites

- Python 3.10+
- vLLM running on port 8000 with Qwen2.5-7B-Instruct
- ~16 GB VRAM for vLLM
- Docker + Docker Compose (for full production deployment — optional for lab)

### Start the inference backend

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### For full Docker deployment (optional)

```bash
docker compose up -d
# Wait for vllm container to finish loading
docker compose logs -f vllm
```

## What You'll Build

A complete production inference service that exposes a FastAPI gateway with structured output validation, instruments every request with Prometheus metrics, proves it handles concurrent load by running a multi-threaded stress test, and prints acceptance criteria results showing concurrency capacity, output correctness, and monitoring health.

## Learner Deliverables

By completing this capstone, you'll produce:

1. **`main.py`** — A working production inference service script integrating 5+ prior concepts
2. **`requirements.txt`** — All Python dependencies for the project
3. **`README.md`** — Architecture documentation with system diagram and acceptance criteria
4. **`prompt.md`** — The complete build instructions (this file)
5. **`docker-compose.yml`** (optional) — Multi-service orchestration for production deployment

You should be able to explain to a colleague: what each component does, why it's needed, and how the pieces communicate.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- A production inference service has at least three layers: inference engine, API gateway, and monitoring — each is necessary
- Structured output (Pydantic validation) ensures clients get predictable response formats regardless of model behavior
- Prometheus metrics add negligible overhead but give you visibility into latency, throughput, errors, and saturation
- Concurrent load testing proves your system handles real traffic patterns, not just serial requests
- Docker Compose lets you define the entire stack as code and start it with a single command
- The three acceptance criteria (concurrency, correctness, observability) cover the minimum bar for a production system
