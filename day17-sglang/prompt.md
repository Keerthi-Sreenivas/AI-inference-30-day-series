# Day 17 Prompt — SGLang: RadixAttention and Serving Comparison

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

Install the Python dependencies for this project:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should contain:

```
sglang[all]>=0.4.0
transformers>=5.0.0
torch>=2.4.0
requests>=2.31.0
openai>=1.30.0
```

## Step 2: Start both serving engines

You need both vLLM and SGLang running to compare them:

**Terminal 1 — vLLM (port 8000):**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096
```

**Terminal 2 — SGLang (port 8001):**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --dtype auto
```

Wait until both servers report they are ready before running the script.

## Step 3: Create the project

Create a flat Python script called `main.py` that compares latency between vLLM and SGLang, specifically testing scenarios where RadixAttention (prefix caching) gives SGLang an advantage. The script should:

1. **Check both server availability** by hitting the health endpoints for vLLM (`http://localhost:8000/health`) and SGLang (`http://localhost:8001/health`). Print clear error messages with startup commands if either is unavailable.

2. **Define a shared system prompt** — use a substantial system prompt (at least 200 words) that simulates a real production chatbot. This shared prefix is where RadixAttention excels.

3. **Run Test 1: First-request latency** — send the same single request to both engines and record latency. This establishes the baseline (no caching benefit yet).

4. **Run Test 2: Repeated-prefix latency** — send 5 different user messages, all with the same system prompt, to both engines sequentially. Measure average latency. SGLang should show improvement after the first request because the system prompt KV cache is reused.

5. **Run Test 3: Multi-turn conversation** — simulate a 4-turn conversation where each turn adds to the history. The prefix grows with each turn. Measure per-turn latency for both engines.

6. **Print a comparison table** showing latency (ms) for each test, each engine, and the speedup ratio (vLLM_latency / SGLang_latency).

7. **Print the key metric**: "SGLang latency advantage: X.X ms faster on repeated prefixes (Y.Y% speedup)"

8. **Add educational comments** explaining: what RadixAttention does, why prefix caching helps multi-turn conversations, how this connects to KV cache (Day 12) and PagedAttention (Day 16), and when vLLM might be equally fast (no shared prefixes).

Structure: flat script, no classes, linear top-to-bottom execution. Use `time.perf_counter()` for accurate timing.

Model: `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0, no gating required)

## Verification

When you run the script with both servers healthy:

- First-request latency should be similar for both engines (no cache advantage yet)
- Repeated-prefix test should show SGLang faster after the first request (system prompt cached)
- Multi-turn test should show SGLang's advantage growing with each turn (more prefix to reuse)
- A summary table should print comparing both engines across all tests
- The key metric should show SGLang's ms advantage for repeated-prefix scenarios
- If either server is down, you should see a clear error with the startup command
