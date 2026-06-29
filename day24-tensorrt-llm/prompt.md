# Day 24 Prompt — TensorRT-LLM, Triton & NVIDIA NIMs

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch requests
```

For full TensorRT-LLM usage, install via NVIDIA's container:

```bash
docker pull nvcr.io/nvidia/tensorrt-llm:latest
```

## Step 2: Setup options

**Option A (NVIDIA GPU with Docker):** Run a TensorRT-LLM optimized model using NVIDIA NIM:

```bash
docker run --gpus all -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

**Option B (Any setup):** Use a vLLM server as baseline and the script will demonstrate throughput concepts with comparative benchmarking:

```bash
vllm serve meta-llama/Llama-3.1-8B --port 8001
```

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
requests>=2.31.0
```

Create a Python script called `main.py` that benchmarks inference throughput and demonstrates TensorRT-LLM optimization concepts. The script should work with either a TensorRT-LLM/NIM endpoint or a standard vLLM endpoint, measuring throughput and printing the gain percentage.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Check endpoint availability** — Check if a TensorRT-LLM/NIM endpoint is available on port 8000. If not, check for a vLLM endpoint on port 8001. Print which mode is active. If neither is available, exit with instructions.

2. **Define throughput test** — Create 5 prompts of similar length to simulate concurrent user requests. These will be sent sequentially to measure per-request throughput.

3. **Measure single-request latency** — Send each prompt individually with `max_tokens=150` and `temperature=0.7`. Record time-to-first-token and total generation time for each.

4. **Measure batch throughput** — Send all 5 prompts in rapid succession (simulating concurrent load) and measure total elapsed time. Calculate effective requests per second and total tokens per second.

5. **Calculate throughput gain** — If both endpoints are available, compare TensorRT-LLM throughput against vLLM baseline. If only one is available, compare against published PyTorch baselines for the model size (estimate ~30 tokens/sec for 8B model on single GPU with PyTorch).

6. **Print hardware info** — Detect and print GPU name, VRAM, and CUDA version if available. This contextualizes the performance numbers.

7. **Print summary** — Display a final table showing: average tokens/sec, requests/sec, estimated throughput gain vs baseline (%), and hardware context.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — detect endpoint, benchmark, calculate gain, summarize
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for endpoint detection)

### Verification

After running the script, you should see:

- Detection of which endpoint is active (NIM/TensorRT-LLM or vLLM)
- Per-prompt latency and tokens/sec measurements
- Batch throughput metrics (total tokens/sec across all prompts)
- Throughput gain percentage compared to baseline
- GPU hardware info (if available)
- Total runtime of 2–5 minutes
