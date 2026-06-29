# Day 21 Prompt — Load Testing Across Engines

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install locust>=2.28.0 httpx>=0.27.0 vllm>=0.8.0
```

## Step 2: Start the inference stack

Start the vLLM backend:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --guided-decoding-backend outlines
```

Wait until you see `Uvicorn running on http://0.0.0.0:8001`.

You can either use the API from Day 20 (port 8000) or the built-in minimal API in today's script.

## Step 3: Create the project

Create a `requirements.txt` with:
```
locust>=2.28.0
httpx>=0.27.0
vllm>=0.8.0
```

Create a flat Python script called `main.py` that performs programmatic load testing against an inference API. The script should:

1. **Check API availability** at `http://localhost:8000/v1/health` (or the target URL). If unreachable, offer to start a minimal built-in FastAPI proxy that forwards to vLLM at port 8001.

2. **Define a Locust user class** (programmatically, not as a separate locustfile) that:
   - Sends POST requests to `/v1/completions` with a fixed prompt ("Explain what distributed systems are.") and `max_tokens=128`
   - Uses minimal wait time between requests (0.1-0.5 seconds)

3. **Run Locust programmatically** in headless mode using `locust.env.Environment`:
   - Start with 1 user, ramp up by 2 users every 5 seconds
   - Run until `--max-users` is reached (default: 30)
   - Hold at max users for 30 seconds to measure steady-state
   - Total test duration configurable via `--duration` (default: 120 seconds)

4. **Collect metrics during the test** at 5-second intervals:
   - Current RPS (requests per second)
   - Response time percentiles (p50, p95, p99)
   - Error count and rate
   - Current number of active users

5. **Find the target RPS**: After the test, determine the maximum RPS achieved while p99 latency stayed below `--target-latency` (default: 1000ms).

6. **Print a results summary table** showing load level progression:
   ```
   Users | RPS  | p50 (ms) | p99 (ms) | Errors | Status
   ------+------+----------+----------+--------+--------
      5  | 4.2  |   280    |   380    |   0    | OK
     10  | 8.1  |   320    |   580    | ...
   ```

7. **Print the key metric**: RPS at target latency:
   ```
   Max RPS at p99 < 1000ms: 12.5 req/sec
   ```

8. **Add educational comments** explaining:
   - Why load testing matters (find breaking point before users do)
   - How Locust's user model works (each user = one concurrent connection)
   - What saturation looks like (RPS plateaus, latency explodes)
   - How to interpret the results for capacity planning

9. **Include a `--serve-api` flag** that starts a minimal FastAPI proxy (like Day 20) for self-contained testing without needing Day 20's code.

10. **Handle graceful shutdown**: Catch Ctrl+C and still print results summary.

### Verification

After running, you should see output like:
```
=== LOAD TEST: Inference API ===
Target: http://localhost:8000
Target latency: 1000ms (p99)
Ramp: 1 → 30 users over 75 seconds

[Progress]
  5 users  | 4.2 RPS | p99: 380ms  | 0 errors
 10 users  | 8.1 RPS | p99: 580ms  | 0 errors
 15 users  | 11.5 RPS | p99: 840ms | 0 errors
 20 users  | 13.2 RPS | p99: 1200ms | 1 error  ← exceeds target
 ...

=== RESULTS ===
Max RPS at p99 < 1000ms: 11.5 req/sec
Saturation point: ~15 concurrent users
Recommendation: operate at ≤10 RPS for safety margin
```
