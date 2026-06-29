# Day 28: Monitoring & Observability for Inference

## The Concept

**Monitoring** means continuously collecting quantitative measurements from your inference service — request counts, latencies, error rates, GPU utilization. **Observability** goes further: it's the ability to understand what's happening inside your system from the outside, using metrics, logs, and traces to diagnose problems you haven't seen before.

Think of it like a car dashboard. You don't need to open the hood to know something's wrong — the speedometer, fuel gauge, and temperature indicator tell you at a glance. If the engine temperature spikes, you pull over before it fails. Inference monitoring works the same way: track latency percentiles, token throughput, and memory usage so you can catch degradation before users notice.

On Day 15 you launched vLLM and saw it expose a `/metrics` endpoint — today you'll build a full monitoring layer on top of that, instrumenting a FastAPI wrapper with Prometheus counters, histograms, and gauges. On Day 20 you designed a production API — today you'll add the observability layer that makes it production-ready.

## 1. The Monitoring Architecture

A production inference stack has three observability layers: metric collection, metric storage, and visualization/alerting:

```
┌─────────────────────────────────────────────────────────────────┐
│              Inference Monitoring Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Client Requests                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Instrumented API layer                  │
│  │  FastAPI         │   Records: latency, count, errors         │
│  │  + Prometheus    │   Exposes: /metrics endpoint              │
│  │  Middleware      │                                           │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Inference engine with built-in metrics  │
│  │  vLLM Server     │   Records: queue depth, batch size, VRAM  │
│  │  (port 8000)     │   Exposes: /metrics on same port          │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       │  ← Prometheus scrapes both /metrics endpoints           │
│       ▼                                                         │
│  ┌──────────────────┐   Pulls metrics every 15-30 seconds       │
│  │  Prometheus      │   Stores as time-series data              │
│  │  (port 9090)     │   Supports PromQL queries                 │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Dashboards + alerting rules             │
│  │  Grafana         │   Visualizes trends over time             │
│  │  (port 3000)     │   Fires alerts on threshold breaches      │
│  └──────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. The Four Golden Signals

Google's Site Reliability Engineering (SRE) book defines four golden signals for monitoring any service. Here's how they apply to inference:

| Signal | What It Measures | Inference Metric | Alert Threshold | Action When Triggered |
|--------|-----------------|------------------|:---------------:|----------------------|
| Latency | Time to serve a request | p99 response time | > 5s | Scale up or reduce batch size |
| Traffic | Requests per second | Inference RPS | > capacity × 0.8 | Add replicas or queue |
| Errors | Failed request rate | HTTP 5xx / timeouts | > 1% | Check model health, OOM |
| Saturation | Resource utilization | GPU VRAM usage | > 90% | Reduce max batch or quantize |

## 3. Prometheus Metric Types

Prometheus offers four metric types, each suited to different measurements:

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Counter: only goes up. Good for totals.
# "How many requests have we served since startup?"
request_count = Counter("requests_total", "Total requests", ["status"])

# Histogram: tracks distribution of values in buckets.
# "What does our latency distribution look like?"
request_latency = Histogram(
    "request_duration_seconds",
    "Latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Gauge: can go up and down. Good for current state.
# "How many requests are being processed right now?"
active_requests = Gauge("active_requests", "Currently processing")

# Summary: like histogram but calculates quantiles client-side.
# Less common — histograms are usually preferred for aggregation.
```

## 4. vLLM Built-In Metrics

vLLM exposes its own Prometheus metrics on the same port as its API. Key metrics include:

| Metric Name | Type | What It Shows |
|------------|:----:|---------------|
| `vllm:num_requests_running` | Gauge | Currently active requests |
| `vllm:num_requests_waiting` | Gauge | Requests in the queue |
| `vllm:gpu_cache_usage_perc` | Gauge | Percentage of GPU KV cache used |
| `vllm:avg_generation_throughput_toks_per_s` | Gauge | Average tokens generated per second |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end latency distribution |

Access these at `http://localhost:8000/metrics` when vLLM is running.

## 5. Alerting Best Practices

Good alerts are actionable and specific:

```yaml
# Example: Prometheus alerting rule for inference latency
groups:
  - name: inference_alerts
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.99, rate(request_duration_seconds_bucket[5m])) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "p99 latency exceeds 5 seconds"
          action: "Check GPU utilization, consider scaling replicas"
```

**Alert on symptoms, not causes.** Alert when p99 latency exceeds your SLA, not when GPU utilization hits 80% (that might be perfectly fine under your workload).

## Setup

### Prerequisites

- Python 3.10+
- vLLM running on port 8000 with Qwen2.5-7B-Instruct
- ~16 GB VRAM for vLLM (or use CPU mode with `--device cpu` for testing)

### Start the vLLM backend

```bash
# Start vLLM with metrics enabled (they're on by default)
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Verify it's running:
curl http://localhost:8000/health
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify vLLM metrics

Before running the lab, confirm vLLM's built-in metrics are accessible:

```bash
curl http://localhost:8000/metrics | head -20
```

You should see Prometheus-formatted metrics starting with `# HELP` and `# TYPE` headers.

## What You'll Build

A metrics-instrumented inference API that wraps vLLM with custom Prometheus counters, histograms, and gauges — then validates that the `/metrics` endpoint correctly reports request counts, latency distributions, and error rates after handling test traffic.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Monitor the four golden signals: latency, traffic, errors, and saturation — they cover 90% of inference failure modes
- Prometheus uses a pull model: your service exposes `/metrics`, Prometheus scrapes it periodically
- Use Counters for totals (requests, tokens), Histograms for distributions (latency), and Gauges for current state (active requests, VRAM)
- vLLM exposes built-in Prometheus metrics — you get GPU cache usage, queue depth, and throughput for free
- Alert on user-visible symptoms (high p99 latency) rather than internal causes (GPU util) — high GPU util during peak traffic is expected and healthy
- The monitoring layer adds negligible overhead: incrementing a counter takes microseconds compared to seconds for inference
