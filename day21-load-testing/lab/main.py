"""
Day 21: Load Testing Across Engines

This script performs programmatic load testing using Locust to find the
maximum requests per second (RPS) your inference API can sustain within
a target p99 latency.

Load testing answers: "How much traffic can this system handle before
latency becomes unacceptable?" — like stress-testing a bridge to find
the weight limit before it buckles.

Architecture:
    Locust (virtual users) → API (port 8000) → vLLM (port 8001)

Before running:
    # Terminal 1: Start vLLM
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001

    # Terminal 2: Start API (from Day 20 or use --serve-api here)
    python main.py --serve-api

    # Terminal 3: Run load test
    python main.py --target-latency 1000 --max-users 30

Usage:
    python main.py                           # Run with defaults
    python main.py --target-latency 500      # Stricter SLA
    python main.py --max-users 50            # Higher load
    python main.py --duration 180            # Longer test
    python main.py --serve-api               # Start built-in API proxy
"""

import argparse
import json
import signal
import statistics
import sys
import threading
import time
from collections import defaultdict

import requests

# ─── Configuration ───────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"
BACKEND_URL = "http://localhost:8001"
HEALTH_ENDPOINT = "/v1/health"
COMPLETIONS_ENDPOINT = "/v1/completions"

# Default test parameters
DEFAULT_TARGET_LATENCY_MS = 1000
DEFAULT_MAX_USERS = 30
DEFAULT_DURATION = 120
RAMP_RATE = 2  # Users added per step
RAMP_INTERVAL = 5  # Seconds between ramp steps
SAMPLE_INTERVAL = 5  # Seconds between metric snapshots


# ─── Service availability check ─────────────────────────────────────────────

def check_api():
    """Verify the inference API is reachable."""
    try:
        resp = requests.get(f"{API_URL}{HEALTH_ENDPOINT}", timeout=5)
        resp.raise_for_status()
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False


def check_backend():
    """Verify vLLM backend is reachable."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False


# ─── Built-in minimal API (for self-contained testing) ───────────────────────

def start_builtin_api():
    """
    Start a minimal FastAPI proxy for load testing.
    This is a simplified version of Day 20's API.
    """
    from fastapi import FastAPI
    from pydantic import BaseModel
    import httpx
    import uvicorn

    app = FastAPI()

    class CompletionReq(BaseModel):
        prompt: str = "Hello"
        max_tokens: int = 128
        temperature: float = 0.7

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "engine": "vllm"}

    @app.post("/v1/completions")
    async def completions(req: CompletionReq):
        start = time.perf_counter()
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{BACKEND_URL}/v1/chat/completions", json=payload
            )
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        latency_ms = (time.perf_counter() - start) * 1000
        return {"text": text, "tokens_generated": tokens, "latency_ms": latency_ms}

    # Run in a background thread
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(2)  # Give it time to start
    print("Built-in API started on port 8000")
    return server


# ─── Load Test Engine (Locust-style but self-contained) ──────────────────────

class LoadTestRunner:
    """
    Programmatic load test runner.

    How load testing works:
    1. Spawn virtual "users" (threads) that send requests in a loop
    2. Each user: send request → record latency → brief pause → repeat
    3. Gradually increase users to simulate traffic growth
    4. Collect metrics at intervals to see how the system responds

    The saturation point is where adding more users stops increasing
    RPS and starts increasing latency — the system can't process
    requests faster than it already is.
    """

    def __init__(self, target_url, max_users, duration, target_latency_ms):
        self.target_url = target_url
        self.max_users = max_users
        self.duration = duration
        self.target_latency_ms = target_latency_ms

        # Shared state (thread-safe via GIL for simple operations)
        self.active_users = 0
        self.running = True
        self.latencies = []  # All latencies recorded
        self.errors = 0
        self.total_requests = 0
        self.snapshots = []  # Periodic metric snapshots
        self._lock = threading.Lock()

    def _worker(self, user_id):
        """
        A single virtual user sending requests in a loop.

        Each user represents one concurrent connection to your API.
        With 10 users, the API sees 10 simultaneous requests being
        processed at any given time.
        """
        session = requests.Session()

        while self.running:
            payload = {
                "prompt": "Explain what distributed systems are and why they matter.",
                "max_tokens": 128,
                "temperature": 0.7,
            }

            start = time.perf_counter()
            try:
                resp = session.post(
                    f"{self.target_url}{COMPLETIONS_ENDPOINT}",
                    json=payload,
                    timeout=30,
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    with self._lock:
                        self.latencies.append(latency_ms)
                        self.total_requests += 1
                else:
                    with self._lock:
                        self.errors += 1
                        self.total_requests += 1

            except Exception:
                with self._lock:
                    self.errors += 1
                    self.total_requests += 1

            # Brief pause between requests (simulates user think time)
            time.sleep(0.1)

    def _take_snapshot(self):
        """Record current metrics for this time window."""
        with self._lock:
            if not self.latencies:
                return

            # Get latencies from the last SAMPLE_INTERVAL seconds
            recent = self.latencies[-100:]  # Approximate recent window
            sorted_recent = sorted(recent)

            snapshot = {
                "users": self.active_users,
                "rps": len(recent) / SAMPLE_INTERVAL if recent else 0,
                "p50": self._percentile(sorted_recent, 50),
                "p95": self._percentile(sorted_recent, 95),
                "p99": self._percentile(sorted_recent, 99),
                "errors": self.errors,
                "total": self.total_requests,
            }
            self.snapshots.append(snapshot)
            return snapshot

    def _percentile(self, sorted_data, pct):
        """Calculate percentile from sorted list."""
        if not sorted_data:
            return 0
        k = (len(sorted_data) - 1) * (pct / 100)
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        d = k - f
        return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])

    def run(self):
        """
        Execute the load test with gradual ramp-up.

        Ramp-up strategy:
        - Start with 1 user
        - Add RAMP_RATE users every RAMP_INTERVAL seconds
        - Stop ramping at max_users
        - Hold at max for remaining duration
        """
        print(f"Starting load test...")
        print(f"  Target: {self.target_url}")
        print(f"  Ramp: 1 → {self.max_users} users")
        print(f"  Duration: {self.duration}s")
        print(f"  Target latency: {self.target_latency_ms}ms (p99)")
        print()
        print(f"{'Users':>6} | {'RPS':>6} | {'p50 (ms)':>9} | {'p99 (ms)':>9} | {'Errors':>7} | Status")
        print("-" * 65)

        threads = []
        start_time = time.time()
        last_snapshot_time = start_time
        next_ramp_time = start_time + RAMP_INTERVAL

        # Start with 1 user
        self._spawn_user(threads)

        try:
            while time.time() - start_time < self.duration and self.running:
                current_time = time.time()

                # Ramp up users
                if (current_time >= next_ramp_time and
                        self.active_users < self.max_users):
                    for _ in range(min(RAMP_RATE, self.max_users - self.active_users)):
                        self._spawn_user(threads)
                    next_ramp_time = current_time + RAMP_INTERVAL

                # Take periodic snapshots
                if current_time - last_snapshot_time >= SAMPLE_INTERVAL:
                    snapshot = self._take_snapshot()
                    if snapshot:
                        status = "OK" if snapshot["p99"] < self.target_latency_ms else "OVER TARGET"
                        print(f"{snapshot['users']:>6} | {snapshot['rps']:>5.1f} | "
                              f"{snapshot['p50']:>8.0f} | {snapshot['p99']:>8.0f} | "
                              f"{snapshot['errors']:>7} | {status}")
                    last_snapshot_time = current_time

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\nTest interrupted — printing results...")

        # Stop all workers
        self.running = False
        for t in threads:
            t.join(timeout=5)

        return self._compute_results()

    def _spawn_user(self, threads):
        """Start a new virtual user thread."""
        self.active_users += 1
        t = threading.Thread(target=self._worker, args=(self.active_users,), daemon=True)
        t.start()
        threads.append(t)

    def _compute_results(self):
        """Analyze all snapshots to find max RPS within target latency."""
        # Find the best RPS where p99 was within target
        valid_snapshots = [
            s for s in self.snapshots
            if s["p99"] < self.target_latency_ms and s["rps"] > 0
        ]

        if valid_snapshots:
            best = max(valid_snapshots, key=lambda s: s["rps"])
            max_rps = best["rps"]
            at_users = best["users"]
        else:
            max_rps = 0
            at_users = 0

        # Overall stats
        all_sorted = sorted(self.latencies) if self.latencies else [0]
        overall_p50 = self._percentile(all_sorted, 50)
        overall_p99 = self._percentile(all_sorted, 99)
        error_rate = (self.errors / self.total_requests * 100) if self.total_requests > 0 else 0

        return {
            "max_rps": max_rps,
            "at_users": at_users,
            "target_latency_ms": self.target_latency_ms,
            "overall_p50": overall_p50,
            "overall_p99": overall_p99,
            "total_requests": self.total_requests,
            "errors": self.errors,
            "error_rate": error_rate,
            "snapshots": self.snapshots,
        }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load test an inference API to find max RPS at target latency"
    )
    parser.add_argument(
        "--target-latency", type=int, default=DEFAULT_TARGET_LATENCY_MS,
        help=f"Target p99 latency in ms (default: {DEFAULT_TARGET_LATENCY_MS})"
    )
    parser.add_argument(
        "--max-users", type=int, default=DEFAULT_MAX_USERS,
        help=f"Maximum concurrent users (default: {DEFAULT_MAX_USERS})"
    )
    parser.add_argument(
        "--duration", type=int, default=DEFAULT_DURATION,
        help=f"Test duration in seconds (default: {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "--target-url", type=str, default=API_URL,
        help=f"API URL to test (default: {API_URL})"
    )
    parser.add_argument(
        "--serve-api", action="store_true",
        help="Start a built-in API proxy before load testing"
    )
    args = parser.parse_args()

    # Step 1: Check if backend is available
    if not check_backend():
        print("ERROR: vLLM backend is not running on port 8001.")
        print()
        print("Start it with:")
        print("  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001")
        raise SystemExit(1)

    # Step 2: Start built-in API if requested
    if args.serve_api:
        start_builtin_api()

    # Step 3: Check if API is available
    if not check_api():
        print("ERROR: Inference API is not running on port 8000.")
        print()
        print("Options:")
        print("  1. Start Day 20's API: python day20-api-design/lab/main.py --serve")
        print("  2. Use built-in proxy: python main.py --serve-api")
        raise SystemExit(1)

    # Step 4: Run the load test
    print("=" * 65)
    print("LOAD TEST: INFERENCE API")
    print("=" * 65)

    runner = LoadTestRunner(
        target_url=args.target_url,
        max_users=args.max_users,
        duration=args.duration,
        target_latency_ms=args.target_latency,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        runner.running = False
    signal.signal(signal.SIGINT, signal_handler)

    results = runner.run()

    # Step 5: Print results summary
    print()
    print("=" * 65)
    print("LOAD TEST RESULTS")
    print("=" * 65)
    print(f"  Target latency (p99):    {results['target_latency_ms']} ms")
    print(f"  Total requests:          {results['total_requests']}")
    print(f"  Errors:                  {results['errors']} ({results['error_rate']:.1f}%)")
    print(f"  Overall p50:             {results['overall_p50']:.0f} ms")
    print(f"  Overall p99:             {results['overall_p99']:.0f} ms")
    print()
    print(f"  Max RPS at p99 < {results['target_latency_ms']}ms: {results['max_rps']:.1f} req/sec")
    print(f"  Achieved at:             {results['at_users']} concurrent users")
    print()

    # Capacity planning recommendation
    safe_rps = results['max_rps'] * 0.7
    if results['max_rps'] > 0:
        print(f"  Recommendation: operate at ≤{safe_rps:.1f} RPS (70% safety margin)")
        print(f"  For 50 RPS target: need ~{max(1, int(50 / results['max_rps']))} GPU instances")
    else:
        print("  No valid measurements within target latency.")
        print("  Try increasing --target-latency or reducing --max-users.")

    print("=" * 65)

    # Print the key metric
    print()
    print(f"RPS at target latency: {results['max_rps']:.1f} req/sec (p99 < {results['target_latency_ms']}ms)")


if __name__ == "__main__":
    main()
