# Day 7 Prompt — Understanding config.json

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch accelerate
```

## Step 2: Model download

The script uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — it downloads automatically on first run (~3.5 GB). No Hugging Face account or token needed (Apache 2.0, fully open).

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
accelerate>=1.0.0
```

Create a Python script called `main.py` that explores a model's `config.json` and validates that the parameter count computed from config values matches the actual model parameter count. Use `AutoModelForCausalLM`, `AutoTokenizer`, and `AutoConfig` from the `transformers` library with the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model. Also use the `json` standard library module for pretty-printing config contents.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Load config without weights** — Use `AutoConfig.from_pretrained()` to fetch the model's config.json without downloading the full model weights. Print the config object to show all parameters.

2. **Display key architecture parameters** — Extract and print the most important values: `vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, and `max_position_embeddings`. Format them in a readable summary block.

3. **Calculate theoretical parameter count** — Compute the expected parameter count from config values using the formula: embedding params + (per-layer attention params + per-layer feed-forward params + per-layer norm params) × num_layers + final layer norm + LM head. Print each component's contribution.

4. **Load the full model** — Load `HuggingFaceTB/SmolLM2-1.7B-Instruct` using `AutoModelForCausalLM.from_pretrained()`. Print the actual parameter count from `sum(p.numel() for p in model.parameters())`.

5. **Validate config vs actual** — Compare the theoretical parameter count (from config math) with the actual parameter count (from PyTorch). Print both values and the percentage difference. A difference under 1% confirms the config accurately describes the model.

6. **Print memory estimate** — Using the config's `torch_dtype` field, calculate expected memory in GB (parameters × bytes_per_dtype). Compare with the actual model size to show that config predicts memory requirements.

7. **Show config as raw JSON** — Load and pretty-print the raw `config.json` content using the `json` module, so learners see exactly what the file looks like on disk.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — load config, analyze, then load model and validate
- Educational inline comments — at least one comment per logical block (config loading, parameter extraction, calculation, validation)
- At most one `try/except` block (for the model load, in case download fails)

### Verification

After running the script, you should see:

- A formatted list of key config parameters (hidden_size, num_layers, etc.)
- A breakdown of parameter count by component (embedding, attention, FFN, etc.)
- Both theoretical and actual parameter counts printed side by side
- A percentage difference confirming they match (should be <1%)
- A memory estimate in GB matching the expected ~3.4 GB for bfloat16
- The raw JSON content of config.json displayed at the end
