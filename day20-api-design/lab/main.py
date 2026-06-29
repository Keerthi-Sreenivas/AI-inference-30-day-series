"""
Day 20: Building an Engine-Agnostic Inference API

This script runs in two modes:
  --serve     Start the FastAPI inference proxy on port 8000
  --benchmark Send concurrent requests and measure p50/p95/p99 latency

Architecture:
  Client → FastAPI (port 8000) → Engine Adapter → vLLM (port 8001)

The adapter pattern decouples your application from a specific engine.
Swap vLLM for SGLang by changing one function — the API stays the same.

Before running:
    # Start the vLLM backend on port 8001
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001 --guided-decoding-backend outlines

    # Then start this API:
    python main.py --serve

    # In another terminal, run the benchmark:
    python main.py --benchmark
"""

import argparse
import asyncio
import statistics
import time
import sys

import httpx
import requests

# ─── Configuration ───────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8001"
API_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_BENCHMARK_REQUESTS = 20


# ─── Service availability check ─────────────────────────────────────────────

def check_backend():
    """Verify the vLLM backend is running before starting our API."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
    except (requests.ConnectionError, requests.Timeout):
        print("ERROR: vLLM backend is not running on port 8001.")
        print()
        print("Start it with:")
        print("  vllm serve Qwen/Qwen2.5-7B-Instruct \\")
        print("      --port 8001 \\")
        print("      --guided-decoding-backend outlines")
        raise SystemExit(1)


def check_api():
    """Verify our FastAPI server is running for benchmark mode."""
    try:
        resp = requests.get(f"{API_URL}/v1/health", timeout=5)
        resp.raise_for_status()
    except (requests.ConnectionError, requests.Timeout):
        print("ERROR: API server is not running on port 8000.")
        print()
        print("Start it with:")
        print("  python main.py --serve")
        raise SystemExit(1)


# ─── FastAPI Application ─────────────────────────────────────────────────────

def create_app():
    """
    Build the FastAPI app with engine-agnostic endpoints.

    Why FastAPI?
    - Async support: crucial for proxying to inference backends without blocking
    - Pydantic integration: automatic request validation and OpenAPI docs
    - Minimal overhead: sub-millisecond routing latency
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Inference API", version="1.0.0")

    # ─── Request/Response Models ─────────────────────────────────────────

    class CompletionRequest(BaseModel):
        """Unified request format — engine-agnostic."""
        model: str = MODEL_NAME
        prompt: str
        max_tokens: int = 256
        temperature: float = 0.7

    class CompletionResponse(BaseModel):
        """Unified response format — same regardless of backend engine."""
        text: str
        tokens_generated: int
        latency_ms: float

    class HealthResponse(BaseModel):
        status: str
        engine: str
        model: str

    # ─── Engine Adapter ──────────────────────────────────────────────────

    async def vllm_adapter(req: CompletionRequest) -> dict:
        """
        Translate our unified request into vLLM's OpenAI-compatible format.

        This is the Strategy pattern: swap this function to change backends.
        For SGLang, you'd write an sglang_adapter with different payload format.
        For Ollama, an ollama_adapter posting to /api/generate.

        The key insight: this function is the ONLY place that knows about
        the backend's API format. Everything else is engine-agnostic.
        """
        payload = {
            "model": req.model,
            "messages": [
                {"role": "user", "content": req.prompt}
            ],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{BACKEND_URL}/v1/chat/completions",
                json=payload,
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Backend error")

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {"text": text, "tokens": tokens}

    # ─── Endpoints ───────────────────────────────────────────────────────

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        """Check liveness and backend connectivity."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{BACKEND_URL}/health")
                resp.raise_for_status()
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Backend engine unavailable"
            )
        return HealthResponse(status="ok", engine="vllm", model=MODEL_NAME)

    @app.get("/v1/models")
    async def list_models():
        """Return available models from the backend."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BACKEND_URL}/v1/models")
        return resp.json()

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest):
        """
        Generate text via the engine adapter.

        Latency breakdown:
        - FastAPI routing: ~0.1ms (negligible)
        - Adapter + network to backend: ~1-5ms
        - Model inference: 100-2000ms (dominates)
        - Response formatting: ~0.1ms
        """
        start = time.perf_counter()

        try:
            result = await vllm_adapter(req)
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Backend request timed out"
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Backend unavailable: {str(e)}"
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return CompletionResponse(
            text=result["text"],
            tokens_generated=result["tokens"],
            latency_ms=round(elapsed_ms, 1),
        )

    return app


# ─── Benchmark Mode ──────────────────────────────────────────────────────────

async def run_benchmark(num_requests):
    """
    Send concurrent requests and compute latency percentiles.

    Why percentiles instead of averages?
    - Average of [100, 100, 100, 5000] = 1325ms — looks bad
    - p50 = 100ms — most users are fine
    - p99 = 5000ms — but 1% hit a 5-second tail

    Percentiles tell you what different users experience.
    """
    print("=" * 60)
    print("API LATENCY BENCHMARK")
    print("=" * 60)
    print(f"  Endpoint: POST /v1/completions")
    print(f"  Requests: {num_requests}")
    print(f"  Backend:  vLLM @ {BACKEND_URL}")
    print()

    latencies = []
    errors = 0

    # Send all requests concurrently using httpx async
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = []
        for i in range(num_requests):
            payload = {
                "model": MODEL_NAME,
                "prompt": f"Explain concept #{i+1}: what is distributed computing?",
                "max_tokens": 128,
                "temperature": 0.7,
            }
            tasks.append(send_timed_request(client, payload))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            errors += 1
        else:
            latencies.append(result)

    # Compute percentiles
    if latencies:
        latencies.sort()
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)

        print("Results:")
        print(f"  p50:  {p50:.0f} ms")
        print(f"  p95:  {p95:.0f} ms")
        print(f"  p99:  {p99:.0f} ms")
        print(f"  Min:  {min(latencies):.0f} ms")
        print(f"  Max:  {max(latencies):.0f} ms")
        print(f"  Successful: {len(latencies)}/{num_requests}")
        if errors:
            print(f"  Errors: {errors}")
    else:
        print("  No successful requests!")

    print()
    print(f"Request latency p50/p99: {p50:.0f}ms / {p99:.0f}ms")
    print("=" * 60)


async def send_timed_request(client, payload):
    """Send a single request and return latency in milliseconds."""
    start = time.perf_counter()
    resp = await client.post(f"{API_URL}/v1/completions", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")

    return elapsed_ms


def percentile(sorted_data, pct):
    """Calculate the p-th percentile from sorted data."""
    if not sorted_data:
        return 0
    k = (len(sorted_data) - 1) * (pct / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Engine-agnostic inference API and benchmark"
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start the FastAPI server on port 8000"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark against the API"
    )
    parser.add_argument(
        "--num-requests", type=int, default=NUM_BENCHMARK_REQUESTS,
        help="Number of benchmark requests (default: 20)"
    )
    args = parser.parse_args()

    if args.serve:
        # Server mode: start FastAPI with uvicorn
        check_backend()
        print(f"Starting API server on port 8000...")
        print(f"Backend: vLLM @ {BACKEND_URL}")
        print()

        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)

    elif args.benchmark:
        # Benchmark mode: send concurrent requests and measure latency
        check_api()
        asyncio.run(run_benchmark(args.num_requests))

    else:
        # Default: show usage
        print("Usage:")
        print("  python main.py --serve      Start the API server")
        print("  python main.py --benchmark  Run latency benchmark")
        print()
        print("Start the vLLM backend first:")
        print("  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001")


if __name__ == "__main__":
    main()
