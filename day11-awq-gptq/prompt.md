# Day 11 Prompt — AWQ & GPTQ

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch accelerate gptqmodel
```

Requires an NVIDIA GPU with CUDA 11.8+ and 8+ GB VRAM. If you don't have a GPU, this lab won't run — refer to Day 6 (GGUF/llama.cpp) for CPU-based quantized inference instead.

## Step 2: Verify GPU availability

Before running the script, confirm CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"
```

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
gptqmodel>=7.0.0
```

Create a Python script called `main.py` that loads pre-quantized AWQ and GPTQ models from HuggingFace, generates text with each, and compares their inference latency. Use `gptqmodel` (the successor to the deprecated `autoawq` and `auto-gptq` libraries) for loading models. Use ungated Qwen2.5-7B variants to avoid license issues.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Check GPU availability** — At the start, verify that a CUDA GPU is available. If not, print a clear message explaining that AWQ/GPTQ require a GPU and suggest Day 6 GGUF as an alternative, then exit gracefully.

2. **Load a GPTQ model** — Load `TechxGenus/Qwen2.5-7B-Instruct-GPTQ-Int4` using `gptqmodel`. Print the model size and quantization config after loading.

3. **Load an AWQ model** — Load `Qwen/Qwen2.5-7B-Instruct-AWQ` using `gptqmodel` or `transformers` (whichever the library supports for AWQ). Print the model size and quantization config after loading.

4. **Define a fixed prompt** — Use the same prompt for both models: `"Explain the difference between lossless and lossy compression in two sentences."` Apply the chat template for proper formatting.

5. **Measure GPTQ latency** — Generate 100 tokens with the GPTQ model, timing the generation call. Print the output text, total latency in milliseconds, and tokens per second.

6. **Measure AWQ latency** — Generate 100 tokens with the AWQ model using the same prompt and generation parameters. Print the output text, total latency in milliseconds, and tokens per second.

7. **Print comparison summary** — Display a formatted comparison showing both models' latency (ms), tokens/sec, and model size on disk. Highlight which format was faster for this specific run.

### License Notice

The default models (`Qwen2.5-7B-Instruct` variants) are Apache-2.0 licensed and ungated — no HuggingFace token required. If you substitute Llama-3.1-8B variants, you must accept Meta's Llama 3.1 Community License at `https://huggingface.co/meta-llama/Llama-3.1-8B` and set `HF_TOKEN`.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — GPU check, load models, run experiments, print results
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block per model load (in case download fails)

### Verification

After running the script, you should see:

- A GPU confirmation message with device name and VRAM
- Both models loaded successfully with their quantization details printed
- Two generated text outputs (one from each model) answering the same prompt
- Latency metrics in milliseconds for each model
- A comparison summary table showing which format was faster
- Total script runtime should be 2–5 minutes (dominated by model downloads on first run)
