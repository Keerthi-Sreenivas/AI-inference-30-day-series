# Day 18 Prompt — Multi-GPU Inference

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install vllm>=0.8.0 ray>=2.10.0 transformers>=5.0.0 torch>=2.4.0
```

## Step 2: Accept the Llama license and start the vLLM server

Accept the Llama 3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B and set your token:

```bash
export HF_TOKEN=your_token_here
```

Start vLLM with tensor parallelism (adjust `--tensor-parallel-size` to your GPU count):

```bash
# Multi-GPU (2 GPUs):
vllm serve meta-llama/Llama-3.1-8B \
    --tensor-parallel-size 2 \
    --port 8000

# Single-GPU fallback:
vllm serve meta-llama/Llama-3.1-8B \
    --tensor-parallel-size 1 \
    --port 8000
```

Alternatively, use the ungated `Qwen/Qwen2.5-7B-Instruct` model (Apache 2.0):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000
```

Wait until you see `Uvicorn running on http://0.0.0.0:8000`.

## Step 3: Create the project

Create a `requirements.txt` with:
```
vllm>=0.8.0
ray>=2.10.0
transformers>=5.0.0
torch>=2.4.0
```

Create a flat Python script called `main.py` that measures multi-GPU scaling efficiency. The script should:

1. **Check vLLM server availability** at `http://localhost:8000/health`. If unreachable, print startup instructions and exit cleanly.

2. **Query the server's model info** via `GET /v1/models` to confirm which model is loaded.

3. **Run a throughput benchmark** by sending N concurrent chat completion requests (use `concurrent.futures`). Use a fixed prompt: "Explain the concept of distributed computing in detail." with `max_tokens=256`.

4. **Measure total tokens generated** and **wall-clock time** to compute tokens/sec throughput.

5. **Calculate scaling efficiency** by comparing against a baseline. If a command-line argument `--baseline-tps` is provided (single-GPU tokens/sec), compute: `efficiency = (measured_tps / (baseline_tps * num_gpus)) * 100`. Otherwise, just report raw throughput.

6. **Print a summary table** showing: number of requests, total tokens generated, elapsed time, throughput (tokens/sec), and scaling efficiency (%) if baseline provided.

7. **Include educational comments** explaining tensor parallelism concepts: how vLLM distributes work, what all-reduce does, why NVLink matters.

8. **Handle single-GPU fallback**: If only 1 GPU is detected (or server reports TP=1), note this is the baseline measurement and suggest re-running with `--baseline-tps` after multi-GPU setup.

Use the OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`. Do NOT import vllm directly — talk to it as a server.

### Verification

After running, you should see output like:
```
Model: meta-llama/Llama-3.1-8B
Requests sent: 20
Total tokens generated: ~5120
Elapsed time: X.XX s
Throughput: XXX.X tokens/sec
Scaling efficiency: XX.X% (vs baseline of XXX tok/s × 2 GPUs)
```

Scaling efficiency above 70% on PCIe or above 90% on NVLink indicates healthy parallelism.
