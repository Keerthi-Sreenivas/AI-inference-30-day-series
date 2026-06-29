# Day 21: Load Testing Across Engines

## The Concept

You've built an inference API (Day 20) and know how to measure single-request latency. But production doesn't serve one request at a time — it serves hundreds or thousands concurrently. **Load testing** answers the critical question: how many requests per second can your system handle before latency degrades unacceptably?

Think of it like **stress-testing a bridge**. An engineer doesn't just check if a bridge holds one car — they simulate rush-hour traffic to find the breaking point. How much weight before the bridge buckles? For inference APIs: how many concurrent requests before p99 latency exceeds your SLA?

On Day 20 you measured p50/p99 for a fixed number of requests. On Day 14 you learned rigorous benchmarking methodology. Today you'll use **Locust** — a professional load testing tool — to gradually increase traffic and find the maximum requests per second (RPS) your system sustains at acceptable latency.

## Load Testing Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              Load Testing Architecture                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────┐                  │
│  │           Locust Load Generator            │                  │
│  │                                            │                  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐         │                  │
│  │  │Worker 1│ │Worker 2│ │Worker N│  ...     │                  │
│  │  │(user)  │ │(user)  │ │(user)  │         │                  │
│  │  └───┬────┘ └───┬────┘ └───┬────┘         │                  │
│  │      │          │          │               │                  │
│  └──────┼──────────┼──────────┼───────────────┘                  │
│         │          │          │                                   │
│         ▼          ▼          ▼                                   │
│  ┌────────────────────────────────────────────┐                  │
│  │         Inference API (FastAPI)            │                  │
│  │              Port 8000                     │                  │
│  │                                            │                  │
│  │  Request queue → Adapter → Backend call    │                  │
│  └────────────────────┬───────────────────────┘                  │
│                       │                                           │
│                       ▼                                           │
│  ┌────────────────────────────────────────────┐                  │
│  │          vLLM Engine (GPU)                 │                  │
│  │              Port 8001                     │                  │
│  │                                            │                  │
│  │  Continuous batching → Inference → Stream  │                  │
│  └────────────────────────────────────────────┘                  │
│                                                                  │
│  Locust spawns virtual users that continuously send requests.    │
│  Each user waits for a response, then immediately sends another. │
│  As users increase, RPS rises until the backend saturates.       │
│                                                                  │
│  Metrics collected:                                              │
│  ├── Requests per second (RPS)                                   │
│  ├── Response time percentiles (p50, p95, p99)                   │
│  ├── Error rate (timeouts, 5xx)                                  │
│  └── Throughput (tokens/sec aggregate)                           │
└──────────────────────────────────────────────────────────────────┘
```

## Understanding Load Test Results

A load test ramps up concurrent users over time. You're looking for the **saturation point** — where adding more users stops increasing throughput and instead just increases latency.

| Load Level | Concurrent Users | RPS | p99 Latency | Error Rate | Verdict |
|:----------:|:----------------:|:---:|:-----------:|:----------:|---------|
| Light | 5 | 4.2 | 380 ms | 0% | Comfortable — room to grow |
| Medium | 15 | 11.5 | 620 ms | 0% | Normal operating range |
| Heavy | 30 | 16.8 | 1400 ms | 0.5% | Approaching saturation |
| Overload | 50 | 17.1 | 4200 ms | 8% | Saturated — latency exploding |
| Breaking | 100 | 15.2 | timeout | 35% | System is failing |

**Key observations:**
- RPS plateaus around 17 → that's your system's capacity
- p99 at heavy load (1400ms) might exceed your SLA (e.g., 1000ms target)
- The "safe" operating point is typically 60-70% of maximum RPS

## Finding Your Target RPS

The goal isn't maximum RPS — it's **maximum RPS within your latency budget**:

```
Target RPS = max RPS where p99 < latency_target

Example:
  Latency target: 1000ms (p99)
  Max RPS at p99 < 1000ms: 12.5 RPS
  This is your capacity for SLA compliance.
```

## Locust Basics

Locust defines virtual users as Python classes. Each user runs a loop: send request → wait for response → repeat. You control how many users spawn and how quickly.

```python
from locust import HttpUser, task, between

class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Brief pause between requests

    @task
    def generate(self):
        self.client.post("/v1/completions", json={
            "prompt": "What is machine learning?",
            "max_tokens": 128
        })
```

You can run Locust in **headless mode** (CLI only, no web UI) for automated testing, which is what our lab script does.

## When to Load Test

- **Before launch:** Find your baseline capacity
- **After changes:** Did that optimization actually help under load?
- **Capacity planning:** How many GPUs for 100 RPS at p99 < 500ms?
- **Engine comparison:** vLLM vs SGLang at the same load profile
- **SLA validation:** Can you meet the promised 99th percentile?

## Setup

### Prerequisites

- Completed Day 20 (API design) — your FastAPI app should be ready
- Familiar with Day 14 (benchmarking methodology)
- vLLM server running with a loaded model

### Installation

```bash
pip install -r requirements.txt
```

### Start the Stack

```bash
# Terminal 1: Start vLLM backend
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001

# Terminal 2: Start your API (from Day 20 or use the built-in one here)
python main.py --serve-api

# Terminal 3: Run the load test
python main.py
```

Or run everything headless with the built-in API:

```bash
# Just run the load test (starts its own API if needed)
python main.py --target-latency 1000 --max-users 30
```

## Interpreting Results

After the test, you get a summary showing the maximum sustainable RPS:

```
=== Load Test Results ===
Target latency: 1000ms (p99)
Max RPS within target: 12.5 req/sec
At that RPS:
  p50: 280ms
  p99: 940ms
  Error rate: 0%

Recommendation: Set autoscaler threshold at 10 RPS (80% of max)
```

This tells you:
- You can serve ~12.5 requests/sec per GPU within your SLA
- If you expect 50 RPS peak, you need 4 GPUs
- Set scaling triggers at 80% capacity (10 RPS) to scale before latency spikes

## What You'll Build

A load testing script using Locust (programmatic mode) that gradually increases concurrent users against your inference API, finds the maximum RPS achievable within a target p99 latency, and reports the result. The script includes a minimal API server so it's self-contained.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Load testing reveals your system's capacity under realistic concurrent traffic
- The key metric is max RPS within your latency SLA — not raw maximum RPS
- Locust simulates virtual users that continuously hit your API
- Look for the saturation point: where RPS plateaus and latency starts climbing
- Operate at 60-70% of max capacity for safety margin
- Load test after every significant change (model swap, engine upgrade, hardware change)
