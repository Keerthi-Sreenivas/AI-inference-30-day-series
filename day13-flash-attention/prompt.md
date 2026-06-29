# Day 13 Prompt — Flash Attention

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

Install the Python packages for this project. Note: `flash-attn` requires Linux and an NVIDIA Ampere+ GPU (A100, RTX 3090/4090, H100).

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should contain:

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
flash-attn>=2.6.0
```

## Step 2: Accept model license

This lab uses `meta-llama/Llama-3.1-8B` which requires license acceptance:

1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B
2. Accept the Llama 3.1 Community License
3. Set your token: `export HF_TOKEN=your_token_here`

Alternatively, substitute `Qwen/Qwen2.5-7B-Instruct` (no license required) for the same experiment.

## Step 3: Create the project

Create a flat Python script called `main.py` that benchmarks Flash Attention vs standard attention. The script should:

1. **Check GPU availability** — verify CUDA is available and that the GPU supports Flash Attention (Ampere+ / sm80+). If no compatible GPU is found, print a message explaining the requirement and run only standard attention as a baseline.

2. **Load the model with standard attention** — use `AutoModelForCausalLM.from_pretrained` with `torch_dtype=torch.float16` and `device_map="auto"`. Use the default attention implementation (eager/SDPA).

3. **Load the model with Flash Attention 2** — use the same `from_pretrained` call but add `attn_implementation="flash_attention_2"`. Handle the case where flash-attn is not installed gracefully.

4. **Prepare a benchmark prompt** — create a prompt of at least 256 tokens (e.g., a paragraph of text repeated) to make the attention computation measurable. Use the tokenizer to verify input length.

5. **Benchmark standard attention** — run `model.generate()` with `max_new_tokens=128` and measure wall-clock time. Run at least 3 iterations and report the median time. Include a warmup run that is excluded from timing.

6. **Benchmark Flash Attention** — run the same generation with the Flash Attention model. Same parameters, same number of iterations, same warmup.

7. **Calculate and print the speedup factor** — compute `standard_time / flash_time` and print it as `Speedup: {x:.2f}x`. This is the key metric.

8. **Print a summary table** — show both timings, the speedup factor, and the sequence length used.

9. **Add educational comments** explaining what Flash Attention changes (memory access pattern, tiled computation, O(N) vs O(N²) memory), why it's faster (less HBM traffic), and what the speedup depends on (sequence length, GPU architecture).

Requirements:
- Flat script structure (no classes, top-to-bottom execution)
- Use `meta-llama/Llama-3.1-8B` (or `Qwen/Qwen2.5-7B-Instruct` as ungated fallback)
- Handle missing flash-attn package gracefully with informative error
- Handle gated model access errors with license instructions
- Use `time.perf_counter()` for timing measurements
- Print at least one quantitative metric (speedup factor) to stdout

## Verification

When you run the script on a supported GPU, you should see output like:

```
GPU: NVIDIA A100-SXM4-80GB (sm_80) — Flash Attention supported ✓

Benchmarking with sequence length: 312 tokens
Generating 128 new tokens per run, 3 iterations (+ 1 warmup)

Standard Attention:  median 2.41s
Flash Attention 2:   median 0.89s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Speedup: 2.71x
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

If no compatible GPU is available, you should see:

```
GPU not available or not Ampere+ architecture.
Flash Attention requires NVIDIA Ampere (sm80+) GPU and Linux.
Running standard attention only as baseline...

Standard Attention: median 8.32s (CPU mode, no Flash Attention comparison possible)
```
