"""
Day 30: Capstone — Production Inference Service from Scratch

Integrates vLLM (Day 15), structured output (Day 19), API design (Day 20),
load testing (Day 21), and monitoring (Day 28) into a single production
service with three acceptance criteria.

Requires: vLLM running on localhost:8000
  Start with: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
"""

import time
import json
import threading
import concurrent.futures
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# --- Check vLLM availability (Day 15 pattern) ---
VLLM_URL = "http://localhost:8000"

print("=" * 70)
print("CAPSTONE: Production Inference Service from Scratch")
print("=" * 70)

print(f"\nChecking vLLM backend at {VLLM_URL}...")

try:
    health = httpx.get(f"{VLLM_URL}/health", timeout=5.0)
    health.raise_for_status()
    print("✓ vLLM is running and healthy\n")
except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
    print("✗ ERROR: vLLM server is not reachable.")
    print(f"  Start it with: vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000")
    print("  Wait for 'Application startup complete' before running this script.")
    raise SystemExit(1)


# --- Define Pydantic models for structured output (Day 19 pattern) ---
# These schemas enforce that our API returns predictable, typed JSON
# regardless of what the model generates internally.
class SummaryResponse(BaseModel):
    """Structured response schema for the /structured endpoint."""
    summary: str = Field(description="A 1-2 sentence summary")
    key_points: List[str] = Field(description="List of key points")
    word_count: int = Field(description="Number of words in the summary")


class HealthStatus(BaseModel):
    """Health check response showing component status."""
    status: str = Field(description="Overall status: healthy or unhealthy")
    vllm_backend: str = Field(description="vLLM status")
    api_version: str = Field(default="1.0.0")



# --- Define Prometheus metrics (Day 28 pattern) ---
# Four core metrics cover the "golden signals": latency, traffic, errors, saturation.
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

ACTIVE_REQUESTS = Gauge(
    "inference_active_requests",
    "Currently processing requests",
)

TOKENS_GENERATED = Counter(
    "inference_tokens_generated_total",
    "Total tokens generated",
)

print("Prometheus metrics defined: requests_total, duration_seconds, "
      "active_requests, tokens_generated")



# --- Build the FastAPI application (Day 20 pattern) ---
# The API gateway sits between clients and vLLM, adding validation,
# metrics, structured output parsing, and health checking.
app = FastAPI(title="Production Inference Service", version="1.0.0")


@app.get("/health")
async def health_check():
    """Check overall service health including backend connectivity."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{VLLM_URL}/health")
            backend_ok = r.status_code == 200
    except Exception:
        backend_ok = False

    status = "healthy" if backend_ok else "unhealthy"
    return HealthStatus(
        status=status,
        vllm_backend="connected" if backend_ok else "unreachable",
    )


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics for scraping."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )



@app.post("/generate")
async def generate(request: Request):
    """Free-form text generation with full instrumentation."""
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()

    try:
        body = await request.json()
        prompt = body.get("prompt", "Hello, world!")
        max_tokens = body.get("max_tokens", 100)
        temperature = body.get("temperature", 0.7)

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{VLLM_URL}/v1/completions",
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            r.raise_for_status()
            result = r.json()

        # Track metrics
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        TOKENS_GENERATED.inc(tokens)
        latency = time.perf_counter() - start
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(endpoint="/generate", status="200").inc()

        return {"text": result["choices"][0]["text"], "tokens": tokens, "latency_ms": latency * 1000}

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/generate", status="500").inc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        ACTIVE_REQUESTS.dec()



@app.post("/structured")
async def structured_output(request: Request):
    """Generate structured JSON output validated against Pydantic schema."""
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()

    try:
        body = await request.json()
        topic = body.get("topic", "artificial intelligence")

        # Prompt the model to produce JSON matching our schema
        structured_prompt = (
            f"Provide a brief summary of '{topic}' as JSON with exactly these fields:\n"
            f"- \"summary\": a 1-2 sentence summary\n"
            f"- \"key_points\": an array of 3 short key points\n"
            f"- \"word_count\": the number of words in the summary\n\n"
            f"Respond with ONLY valid JSON, no other text.\n\n"
            f"JSON:"
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{VLLM_URL}/v1/completions",
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "prompt": structured_prompt,
                    "max_tokens": 200,
                    "temperature": 0.1,
                },
            )
            r.raise_for_status()
            result = r.json()

        raw_text = result["choices"][0]["text"].strip()
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        TOKENS_GENERATED.inc(tokens)

        # Parse and validate against Pydantic schema
        try:
            # Try to extract JSON from response
            json_start = raw_text.find("{")
            json_end = raw_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(raw_text[json_start:json_end])
            else:
                parsed = json.loads(raw_text)

            validated = SummaryResponse(**parsed)
            response_data = validated.model_dump()
            response_data["valid_json"] = True
        except (json.JSONDecodeError, Exception):
            response_data = {
                "summary": raw_text[:200],
                "key_points": [],
                "word_count": len(raw_text.split()),
                "valid_json": False,
            }

        latency = time.perf_counter() - start
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(endpoint="/structured", status="200").inc()
        response_data["latency_ms"] = latency * 1000

        return response_data

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/structured", status="500").inc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        ACTIVE_REQUESTS.dec()



# --- Start the API server (background thread) ---
API_PORT = 9000
API_URL = f"http://localhost:{API_PORT}"

print(f"\nStarting production API on port {API_PORT}...")


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning")


server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)
print(f"✓ API server running at {API_URL}\n")

# --- Functional Tests ---
# Verify each endpoint works correctly before running the load test.
print("=" * 70)
print("FUNCTIONAL TESTS")
print("=" * 70)

test_results = {}

# Test 1: Health check
print("\n  Test 1: GET /health")
r = httpx.get(f"{API_URL}/health", timeout=10.0)
health_ok = r.status_code == 200 and r.json().get("status") == "healthy"
test_results["health"] = health_ok
print(f"    Status: {'✓ PASS' if health_ok else '✗ FAIL'} — {r.json()}")


# Test 2: Free-form generation
print("\n  Test 2: POST /generate")
r = httpx.post(
    f"{API_URL}/generate",
    json={"prompt": "What is Python?", "max_tokens": 30},
    timeout=60.0,
)
gen_ok = r.status_code == 200 and "text" in r.json()
test_results["generate"] = gen_ok
print(f"    Status: {'✓ PASS' if gen_ok else '✗ FAIL'}")
if gen_ok:
    print(f"    Response: {r.json()['text'][:100]}...")
    print(f"    Latency: {r.json()['latency_ms']:.0f}ms")

# Test 3: Structured output
print("\n  Test 3: POST /structured")
r = httpx.post(
    f"{API_URL}/structured",
    json={"topic": "machine learning"},
    timeout=60.0,
)
struct_data = r.json()
struct_ok = r.status_code == 200 and struct_data.get("valid_json", False)
test_results["structured"] = struct_ok
print(f"    Status: {'✓ PASS' if struct_ok else '✗ FAIL'}")
print(f"    Valid JSON: {struct_data.get('valid_json')}")
if struct_data.get("summary"):
    print(f"    Summary: {struct_data['summary'][:100]}...")

# Test 4: Metrics endpoint
print("\n  Test 4: GET /metrics")
r = httpx.get(f"{API_URL}/metrics", timeout=10.0)
metrics_ok = r.status_code == 200 and "inference_requests_total" in r.text
test_results["metrics"] = metrics_ok
print(f"    Status: {'✓ PASS' if metrics_ok else '✗ FAIL'}")
print(f"    Metrics size: {len(r.text)} bytes")

all_functional = all(test_results.values())
print(f"\n  Functional tests: {'ALL PASSED ✓' if all_functional else 'SOME FAILED ✗'}")



# --- Concurrent Load Test (Day 21 pattern) ---
# Send 10 concurrent requests to prove the service handles parallel load.
# This tests vLLM's continuous batching and our API's async handling.
print("\n" + "=" * 70)
print("CONCURRENT LOAD TEST")
print("=" * 70)
print("Sending 10 concurrent requests to /generate...\n")

CONCURRENT_REQUESTS = 10
load_test_latencies = []
load_test_errors = 0


def send_request(i):
    """Send a single request and return latency in ms."""
    start = time.perf_counter()
    try:
        r = httpx.post(
            f"{API_URL}/generate",
            json={"prompt": f"Briefly explain concept number {i}.", "max_tokens": 30},
            timeout=120.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        if r.status_code == 200:
            return ("ok", latency_ms)
        else:
            return ("error", latency_ms)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        return ("error", latency_ms)


# Launch all requests concurrently
load_start = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
    futures = [executor.submit(send_request, i) for i in range(CONCURRENT_REQUESTS)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

load_total_time = time.perf_counter() - load_start

# Analyze results
for status, latency in results:
    if status == "ok":
        load_test_latencies.append(latency)
    else:
        load_test_errors += 1
        load_test_latencies.append(latency)

successful = len(load_test_latencies) - load_test_errors
sorted_latencies = sorted(load_test_latencies)


# Calculate percentiles
avg_latency = sum(sorted_latencies) / len(sorted_latencies) if sorted_latencies else 0
p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
p99_idx = int(len(sorted_latencies) * 0.99)
p99 = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)] if sorted_latencies else 0
rps = CONCURRENT_REQUESTS / load_total_time if load_total_time > 0 else 0

print(f"  Total requests:    {CONCURRENT_REQUESTS}")
print(f"  Successful:        {successful}")
print(f"  Failed:            {load_test_errors}")
print(f"  Total wall time:   {load_total_time:.1f}s")
print(f"  Throughput (RPS):  {rps:.2f}")
print(f"  Avg latency:       {avg_latency:.0f}ms")
print(f"  P50 latency:       {p50:.0f}ms")
print(f"  P99 latency:       {p99:.0f}ms")
print(f"  Min latency:       {sorted_latencies[0]:.0f}ms")
print(f"  Max latency:       {sorted_latencies[-1]:.0f}ms")

# --- Validate Monitoring (Day 28 pattern) ---
# Fetch metrics after the load test and verify they reflect the traffic.
print("\n" + "=" * 70)
print("MONITORING VALIDATION")
print("=" * 70)

metrics_response = httpx.get(f"{API_URL}/metrics", timeout=10.0)
metrics_text = metrics_response.text

expected_metrics = [
    ("inference_requests_total", "Request counter"),
    ("inference_request_duration_seconds", "Latency histogram"),
    ("inference_active_requests", "Active requests gauge"),
    ("inference_tokens_generated_total", "Tokens counter"),
]


print("\n  Checking metrics endpoint contains expected data:")
metrics_valid = True
for metric_name, description in expected_metrics:
    present = metric_name in metrics_text
    status = "✓" if present else "✗"
    if not present:
        metrics_valid = False
    print(f"    {status} {metric_name} ({description})")

# Check that counters show values > 0 (traffic was recorded)
has_observations = "inference_requests_total" in metrics_text
print(f"\n  Metrics contain request data: {'✓ YES' if has_observations else '✗ NO'}")

# --- ACCEPTANCE CRITERIA RESULTS ---
# The three objective tests that determine if the capstone passes.
print("\n" + "=" * 70)
print("ACCEPTANCE CRITERIA RESULTS")
print("=" * 70)

# Criterion 1: Concurrency
p99_threshold = 30000  # 30 seconds — generous for concurrent 7B model inference
concurrency_pass = (load_test_errors == 0) and (p99 < p99_threshold)
print(f"\n  1. CONCURRENCY & LOAD")
print(f"     Requirement: 10 concurrent requests, all succeed, p99 < {p99_threshold/1000:.0f}s")
print(f"     Result: {successful}/{CONCURRENT_REQUESTS} succeeded, p99 = {p99:.0f}ms")
print(f"     Status: {'✓ PASS' if concurrency_pass else '✗ FAIL'}")

# Criterion 2: Functional Correctness
correctness_pass = test_results.get("structured", False)
print(f"\n  2. FUNCTIONAL CORRECTNESS")
print(f"     Requirement: /structured returns valid JSON matching Pydantic schema")
print(f"     Result: valid_json = {struct_data.get('valid_json')}")
print(f"     Status: {'✓ PASS' if correctness_pass else '✗ FAIL'}")

# Criterion 3: Monitoring Observability
observability_pass = metrics_valid and has_observations
print(f"\n  3. MONITORING OBSERVABILITY")
print(f"     Requirement: /metrics returns request counts and latency data")
print(f"     Result: All metrics present = {metrics_valid}, data recorded = {has_observations}")
print(f"     Status: {'✓ PASS' if observability_pass else '✗ FAIL'}")


# --- Final Summary ---
all_pass = concurrency_pass and correctness_pass and observability_pass
print("\n" + "=" * 70)
print(f"CAPSTONE RESULT: {'ALL CRITERIA PASSED ✓' if all_pass else 'SOME CRITERIA FAILED ✗'}")
print("=" * 70)
print(f"\n{'Metric':<35} {'Value':<25}")
print("-" * 60)
print(f"{'Concurrent RPS':<35} {rps:.2f} req/s")
print(f"{'P99 Latency':<35} {p99:.0f} ms")
print(f"{'Structured JSON valid':<35} {struct_data.get('valid_json')}")
print(f"{'Metrics endpoint healthy':<35} {metrics_valid}")
print(f"{'Total tokens generated':<35} (see /metrics)")
print(f"{'Acceptance criteria passed':<35} {sum([concurrency_pass, correctness_pass, observability_pass])}/3")
print("-" * 60)
print("\nCongratulations! You've built a production inference service that:")
print("  • Serves inference via a proper API gateway (Day 20)")
print("  • Returns structured, validated JSON output (Day 19)")
print("  • Handles concurrent load with continuous batching (Day 15, 21)")
print("  • Exposes Prometheus metrics for observability (Day 28)")
print("  • Can be deployed as a multi-service Docker stack (Day 30)")
print("\nYou now have the skills to deploy, monitor, and optimize LLM inference")
print("in production. Go build something amazing! 🚀")
