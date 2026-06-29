# Day 6 Prompt — GGUF & llama.cpp

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install llama-cpp-python huggingface-hub
```

## Step 2: Accept the Llama license and set token

The GGUF model is based on Llama-3.2-3B-Instruct, which requires license acceptance:

1. Visit [https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and accept the license
2. Create or copy your Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set it as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

## Step 3: Create the project

Create a `requirements.txt` with:

```
llama-cpp-python>=0.3.0
huggingface-hub>=0.24.0
```

Create a Python script called `main.py` that downloads a GGUF model, loads it with `llama-cpp-python`, generates text on CPU, and measures tokens per second. Use `Llama-3.2-3B-Instruct` in Q4_K_M quantization from a GGUF provider on Hugging Face Hub.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Download the GGUF model** — Use `huggingface_hub.hf_hub_download()` to fetch a `Llama-3.2-3B-Instruct-Q4_K_M.gguf` file from a community quantization repo (e.g., `bartowski/Llama-3.2-3B-Instruct-GGUF`). Print the download path so the user knows where it's cached.

2. **Load with llama-cpp-python** — Instantiate `Llama(model_path=..., n_ctx=2048, n_threads=8, verbose=False)`. Wrap the load in a try/except to handle missing files or incompatible formats gracefully. Print a confirmation message when the model loads successfully.

3. **Generate a response** — Use `llm.create_chat_completion()` with a sample user message (`"Explain what GGUF is and why it matters for running AI models locally."`) and `max_tokens=200`, `temperature=0.7`. Print the generated response.

4. **Measure tokens per second** — Time the generation call and compute `tokens_generated / elapsed_seconds`. Print a clear metric line: `"CPU Inference: {X:.1f} tokens/sec"`. This is the key quantitative output.

5. **Run a second generation for comparison** — Generate a longer response (400 tokens) with a different prompt to show sustained throughput. Print the tokens/sec for this run as well so the user can compare short vs long generation performance.

6. **Print a summary** — At the end, print a formatted summary showing both generation runs with their tokens/sec metrics, the model file size, and the quantization level used.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — download, load, generate, measure, summarize
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block for the model load (in case GGUF file is missing/corrupt)
- Use `time.perf_counter()` for timing measurements

### Verification

After running the script, you should see:

- A printed path where the GGUF file was cached (~2 GB download on first run)
- A generated text response about GGUF
- A tokens/sec metric (expect 10–30 tok/s on modern CPUs, varies by hardware)
- A second generation run with tokens/sec for the longer output
- A summary comparing both runs
- Total script runtime: 1–5 minutes depending on CPU speed (excluding download)
