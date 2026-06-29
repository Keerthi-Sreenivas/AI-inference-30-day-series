# Day 20 Prompt — Building an Engine-Agnostic Inference API

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install fastapi>=0.115.0 uvicorn>=0.30.0 httpx>=0.27.0 vllm>=0.8.0
```

## Step 2: Start the vLLM backend server

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --guided-decoding-backend outlines
```

Wait until you see `Uvicorn running on http://0.0.0.0:8001`. Note: the backend runs on port 8001, our API will run on port 8000.

## Step 3: Create the project

Create a `requirements.txt` with:
```
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0
vllm>=0.8.0
```

Create a flat Python script called `main.py` that operates in two modes: API server mode and benchmark mode. The script should:

1. **Check backend availability** at `http://localhost:8001/health` when starting. If the vLLM backend is unreachable, print startup instructions and exit cleanly.

2. **Define Pydantic request/response models**:
   - `CompletionRequest`: model (str), prompt (str), max_tokens (int, default 256), temperature (float, default 0.7)
   - `CompletionResponse`: text (str), tokens_generated (int), latency_ms (float)
   - `HealthResponse`: status (str), engine (str), model (str)

3. **Implement the FastAPI app** with these endpoints:
   - `GET /v1/health` — checks backend health and returns engine info
   - `POST /v1/completions` — forwards completion request to vLLM backend, measures latency, returns formatted response
   - `GET /v1/models` — returns list of available models from backend

4. **Implement an engine adapter** function that translates our API's request format into vLLM's OpenAI-compatible format. Use `httpx.AsyncClient` for async forwarding to `http://localhost:8001/v1/chat/completions`.

5. **Server mode** (`python main.py --serve`): Start uvicorn on port 8000 with the FastAPI app.

6. **Benchmark mode** (`python main.py --benchmark`): Send 20 concurrent requests to `http://localhost:8000/v1/completions` using `httpx` with asyncio. Collect all response latencies and compute p50, p95, and p99 percentiles.

7. **Print latency percentiles** as the quantitative metric:
   ```
   Request latency:
     p50:  XXX ms
     p95:  XXX ms
     p99:  XXX ms
   ```

8. **Add educational comments** explaining:
   - Why an adapter pattern decouples the API from the engine
   - How latency percentiles differ from averages
   - Why FastAPI's async support matters for inference proxying
   - How you'd add a second adapter (e.g., SGLang) by swapping one function

9. **Handle errors gracefully**: If a backend request times out or fails, return a JSON error response with appropriate HTTP status code (503 for backend unavailable, 504 for timeout).

Use `argparse` to switch between `--serve` and `--benchmark` modes.

### Verification

After starting the server and running the benchmark, you should see:
```
=== API Latency Benchmark ===
Endpoint: POST /v1/completions
Requests: 20
Backend: vLLM @ localhost:8001

Results:
  p50: 180 ms
  p95: 420 ms
  p99: 890 ms
  Successful: 20/20
```
