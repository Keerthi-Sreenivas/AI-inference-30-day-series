# Day 10 Prompt — 4-bit Inference with bitsandbytes

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.)

---

## Step 1: Install dependencies

```bash
pip install transformers>=5.0.0 torch>=2.4.0 accelerate>=1.0.0 bitsandbytes>=0.44.0
```

Or use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

## Step 2: Verify GPU availability

Confirm CUDA is available before proceeding:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

This lab requires an NVIDIA GPU with CUDA. If you don't have one, the script will print expected output so you can follow along.

## Step 3: Create the project

Create a file called `main.py` with the following requirements:

### Requirements

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
bitsandbytes>=0.44.0
```

### Script structure

- Flat script (no classes, no function definitions beyond helpers)
- Linear top-to-bottom execution
- Educational comments explaining each step

### Functional requirements

1. Check if CUDA GPU is available at startup. If no GPU, print a clear message explaining that bitsandbytes requires CUDA and show expected output for a typical GPU run, then exit gracefully.
2. Print the GPU name and total VRAM at the start.
3. Load `Qwen/Qwen2.5-7B-Instruct` in FP16 precision using `from_pretrained()` with `device_map="auto"` and `torch_dtype=torch.float16`.
4. Measure and print the VRAM usage after loading in FP16 (in GB, rounded to 1 decimal).
5. Delete the FP16 model and clear GPU memory completely (`del model`, `torch.cuda.empty_cache()`).
6. Load the same model in 4-bit NF4 using `BitsAndBytesConfig` with `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, and `bnb_4bit_use_double_quant=True`.
7. Measure and print the VRAM usage after loading in NF4 (in GB, rounded to 1 decimal).
8. Calculate and print the VRAM savings percentage: `(fp16_vram - nf4_vram) / fp16_vram * 100`.
9. Run a short generation with the 4-bit model (a simple prompt like "Explain quantization in one sentence") to confirm the model works correctly after quantization.
10. Print a summary table showing FP16 VRAM, NF4 VRAM, savings in GB, and savings percentage.

### Model specification

- Model: `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0, ungated — no token needed)
- Fallback: If download fails, try `HuggingFaceTB/SmolLM2-1.7B-Instruct` as a smaller alternative

### Educational comments

Include at least one comment per logical block explaining:
- Why we measure VRAM before and after
- What NF4 quantization does to the weights
- Why double quantization helps
- What the dequantization overhead means for speed

## Verification

When the script runs successfully on a CUDA GPU, you should see output similar to:

```
GPU: NVIDIA RTX 4090 (24.0 GB total VRAM)
Loading model in FP16...
FP16 VRAM usage: 14.2 GB
Clearing GPU memory...
Loading model in NF4 (4-bit)...
NF4 VRAM usage: 4.2 GB
VRAM Savings: 10.0 GB (70.4%)
Testing generation with 4-bit model...
Generated: "Quantization reduces model precision to use less memory while preserving accuracy."
Summary:
  FP16:    14.2 GB
  NF4:      4.2 GB
  Saved:   10.0 GB (70.4%)
```

The exact numbers will vary by GPU and driver version, but savings should be approximately 65-75%.
