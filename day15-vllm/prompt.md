# Day 15 Prompt — vLLM Production Inference Serving

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

Install the Python dependencies for this project:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should contain:

```
vllm>=0.8.0
transformers>=5.0.0
torch>=2.4.0
openai>=1.30.0
requests>=2.31.0
```

## Step 2: Start the vLLM server

Start the vLLM inference server with the Qwen2.5-7B-Instruct model:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096
```

Wait until you see "Application startup complete" in the terminal before running the script.

## Step 3: Create the project

Create a flat Python script called `main.py` that measures vLLM throughput. The script should:

1. **Check vLLM server availability** by hitting `http://localhost:8000/health`. If the server is not running, print a helpful error message showing the startup command and exit.

2. **Define a batch of test prompts** — at least 10 diverse prompts of varying lengths (short questions, medium instructions, longer reasoning tasks) to simulate realistic load.

3. **Send requests concurrently** using Python's `concurrent.futures.ThreadPoolExecutor` to the `/v1/chat/completions` endpoint. Use the OpenAI client library pointed at the local vLLM server.

4. **Measure throughput** by timing the total wall-clock duration for all requests to complete, then calculating requests per second (`total_requests / total_seconds`).

5. **Print individual request latencies** showing how long each request took (in milliseconds), demonstrating that vLLM handles them concurrently rather than sequentially.

6. **Print aggregate throughput** as the final metric: `Throughput: X.XX req/sec`.

7. **Use the sampling parameters** from Day 4: set `temperature=0.7`, `max_tokens=128` for consistent benchmarking.

8. **Add educational comments** explaining: why we use concurrent requests, what the OpenAI-compatible API looks like, how vLLM differs from sequential generation, and what throughput means in production.

Structure: flat script, no classes, linear top-to-bottom execution. Use `if __name__ == "__main__":` only if helpful for clarity.

Model: `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0, no gating required)

## Verification

When you run the script with a healthy vLLM server:

- You should see each request's latency printed (e.g., "Request 1: 342ms")
- The total throughput should be significantly higher than 1 req/sec (since requests run in parallel)
- If vLLM is not running, you should see a clear error message with startup instructions
- Compare mentally: sequential execution of 10 requests at ~2s each = 20s total. Concurrent via vLLM should complete all 10 in roughly the same time as 2-3 sequential requests.
