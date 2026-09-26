"""
Day 28: Monitoring & Observability for Inference

Build a metrics-instrumented inference API that wraps vLLM with Prometheus
metrics. Tracks request count, latency histograms, active requests, and
error rates — then validates the /metrics endpoint returns expected data.

Requires: vLLM running on localhost:8000
  Start with: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
"""

import time
import threading
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# --- Check vLLM availability ---
# The monitoring layer sits between clients and vLLM. If vLLM isn't running,
# there's nothing to monitor — so we check first.
VLLM_URL = "http://localhost:8000"

print("=" * 70)
print("MONITORING & OBSERVABILITY FOR INFERENCE")
print("=" * 70)
print(f"\nChecking vLLM availability at {VLLM_URL}...")

try:
    health_check = httpx.get(f"{VLLM_URL}/health", timeout=5.0)
    health_check.raise_for_status()
    print("✓ vLLM server is running and healthy\n")
except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
    print("✗ ERROR: vLLM server is not reachable.")
    print(f"  Start it with: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000")
    print("  Wait for 'Application startup complete' before running this script.")
    raise SystemExit(1)

# --- Define Prometheus metrics ---
# These are the four core metrics every inference service should track.
# Prometheus uses a pull model: the /metrics endpoint exposes current values,
# and a Prometheus server scrapes it periodically (typically every 15-30s).
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests received",
    ["method", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Request latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

ACTIVE_REQUESTS = Gauge(
    "inference_active_requests",
    "Number of requests currently being processed",
)

ERROR_COUNT = Counter(
    "inference_errors_total",
    "Total number of failed inference requests",
    ["error_type"],
)

TOKENS_GENERATED = Counter(
    "inference_tokens_generated_total",
    "Total number of tokens generated across all requests",
)

print("Prometheus metrics defined:")
print("  - inference_requests_total (Counter)")
print("  - inference_request_duration_seconds (Histogram)")
print("  - inference_active_requests (Gauge)")
print("  - inference_errors_total (Counter)")
print("  - inference_tokens_generated_total (Counter)\n")

# --- Build the instrumented FastAPI app ---
# This API wraps vLLM and adds observability. Every request is measured
# for latency, counted, and tracked for concurrency.
app = FastAPI(title="Instrumented Inference API")


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics in text format for scraping."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/generate")
async def generate(request: Request):
    """Forward generation request to vLLM with full instrumentation."""
    ACTIVE_REQUESTS.inc()
    start_time = time.perf_counter()

    try:
        body = await request.json()
        prompt = body.get("prompt", "Hello")
        max_tokens = body.get("max_tokens", 50)

        # Forward to vLLM's completions endpoint
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VLLM_URL}/v1/completions",
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            result = response.json()

        # Track tokens generated
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        TOKENS_GENERATED.inc(completion_tokens)

        # Record success metrics
        latency = time.perf_counter() - start_time
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(method="POST", status="200").inc()

        return {"result": result, "latency_ms": latency * 1000}

    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        REQUEST_COUNT.labels(method="POST", status="500").inc()
        return {"error": str(e)}

    finally:
        ACTIVE_REQUESTS.dec()


# --- Start the API server in background ---
# We run uvicorn in a daemon thread so the script can continue to
# send test requests and validate metrics.
API_PORT = 9000
API_URL = f"http://localhost:{API_PORT}"

print(f"Starting instrumented API on port {API_PORT}...")


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning")


server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)  # Give uvicorn time to start
print(f"✓ API server running at {API_URL}\n")

# --- Send test requests ---
# Generate some traffic through the instrumented API so the metrics
# have real data to report. We vary the prompts to create different latencies.
print("=" * 70)
print("SENDING TEST REQUESTS")
print("=" * 70)

test_prompts = [
    "What is Python?",
    "Explain machine learning in one sentence.",
    "What is a GPU?",
    "Define tokenization.",
    "What is vLLM?",
]

request_latencies = []

for i, prompt in enumerate(test_prompts):
    start = time.perf_counter()
    response = httpx.post(
        f"{API_URL}/generate",
        json={"prompt": prompt, "max_tokens": 30},
        timeout=60.0,
    )
    latency = (time.perf_counter() - start) * 1000  # ms

    status = "✓" if response.status_code == 200 else "✗"
    request_latencies.append(latency)
    print(f"  Request {i+1}: {status} | {latency:.0f}ms | \"{prompt[:40]}\"")

print(f"\nAll {len(test_prompts)} requests completed.\n")

# --- Validate metrics endpoint ---
# Fetch the /metrics endpoint and check that all expected metrics are present.
# This confirms our instrumentation is working correctly.
print("=" * 70)
print("METRICS VALIDATION")
print("=" * 70)

metrics_response = httpx.get(f"{API_URL}/metrics", timeout=10.0)
metrics_text = metrics_response.text

expected_metrics = [
    "inference_requests_total",
    "inference_request_duration_seconds",
    "inference_active_requests",
    "inference_errors_total",
    "inference_tokens_generated_total",
]

print(f"\nFetching {API_URL}/metrics...")
print(f"Response size: {len(metrics_text)} bytes\n")

all_passed = True
for metric_name in expected_metrics:
    present = metric_name in metrics_text
    status = "✓ PASS" if present else "✗ FAIL"
    if not present:
        all_passed = False
    print(f"  {status}: {metric_name}")

print()
if all_passed:
    print("All metric checks PASSED ✓")
else:
    print("Some metric checks FAILED ✗")

# --- Print metrics summary ---
# Parse key values from the Prometheus output to show human-readable stats.
print("\n" + "=" * 70)
print("METRICS SUMMARY")
print("=" * 70)

# Extract values from Prometheus text format
avg_latency = sum(request_latencies) / len(request_latencies)
sorted_latencies = sorted(request_latencies)
p50 = sorted_latencies[len(sorted_latencies) // 2]
p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

print(f"\n{'Metric':<35} {'Value':<20}")
print("-" * 55)
print(f"{'Total requests sent':<35} {len(test_prompts)}")
print(f"{'Average latency':<35} {avg_latency:.0f} ms")
print(f"{'P50 latency':<35} {p50:.0f} ms")
print(f"{'P99 latency':<35} {p99:.0f} ms")
print(f"{'Min latency':<35} {sorted_latencies[0]:.0f} ms")
print(f"{'Max latency':<35} {sorted_latencies[-1]:.0f} ms")
print(f"{'Metrics endpoint status':<35} {'HEALTHY' if all_passed else 'DEGRADED'}")
print("-" * 55)

# Show a sample of the raw metrics output
print("\nSample /metrics output (first 500 chars):")
print("-" * 40)
print(metrics_text[:500])
print("-" * 40)

print("\nKey insight: Prometheus metrics give you observability into inference")
print("workloads. Track request count, latency percentiles, active requests,")
print("and error rates to detect degradation before users complain.")
print("In production, connect Prometheus to Grafana for dashboards + alerting.")
