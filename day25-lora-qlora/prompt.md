# Day 25 Prompt — LoRA & QLoRA

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install peft bitsandbytes transformers torch accelerate datasets
```

## Step 2: Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — it downloads automatically on first run (~3.5 GB). No Hugging Face account or token needed (Apache 2.0, fully open). This model is small enough to create LoRA adapters on modest hardware.

## Step 3: Create the project

Create a `requirements.txt` with:

```
peft>=0.18.0
bitsandbytes>=0.44.0
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
datasets>=2.14.0
```

Create a Python script called `main.py` that demonstrates LoRA adapter creation, parameter efficiency measurement, and adapter size calculation. Use `HuggingFaceTB/SmolLM2-1.7B-Instruct` as the base model with the `peft` library for adapter creation.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Load base model** — Load `HuggingFaceTB/SmolLM2-1.7B-Instruct` using `AutoModelForCausalLM.from_pretrained()`. Print total parameter count and model memory footprint.

2. **Create LoRA configuration** — Define a LoRA config with `r=16`, `lora_alpha=32`, targeting `q_proj` and `v_proj` modules, with `lora_dropout=0.05`. Print the configuration details.

3. **Apply LoRA adapter** — Wrap the base model with `get_peft_model()`. Use `model.print_trainable_parameters()` to show the dramatic reduction: how many parameters are trainable vs total, and the percentage.

4. **Calculate adapter size** — Compute the total memory size of only the trainable (adapter) parameters in megabytes. Print this as the key metric: `"Adapter size: X.X MB"`.

5. **Compare memory footprints** — Calculate and print: full model size (MB), adapter-only size (MB), and the ratio (how many times smaller the adapter is vs the full model).

6. **Test adapter with different ranks** — Create adapters with ranks `[4, 8, 16, 32, 64]` and print a comparison table showing rank, trainable parameter count, adapter size (MB), and percentage of total parameters for each.

7. **Demonstrate QLoRA concept** — If a CUDA GPU is available, load the model in 4-bit using `BitsAndBytesConfig` with `load_in_4bit=True` and `bnb_4bit_quant_type="nf4"`, then apply LoRA on top. Print the combined VRAM usage showing QLoRA's efficiency. If no GPU, print the expected memory savings based on calculations.

8. **Save adapter** — Save the LoRA adapter weights to a local directory. Print the saved file sizes to show how small adapters are compared to full model checkpoints.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — load, configure, apply, measure, compare, save
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for model loading)

### Verification

After running the script, you should see:

- Total model parameters (should be ~1.7B for SmolLM2-1.7B)
- Trainable parameters with LoRA (should be ~0.05–1% of total)
- Adapter size in MB (should be 5–50 MB depending on rank and modules)
- Rank comparison table showing size scaling
- Saved adapter files (should be small — under 50 MB)
- Clear demonstration that LoRA trains <1% of parameters while the rest stay frozen
