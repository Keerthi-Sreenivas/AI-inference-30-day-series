# Day 25: LoRA & QLoRA — Lightweight Fine-Tuning

## The Concept

Full fine-tuning of a large language model means updating every single parameter — billions of weights adjusted to learn your specific task. For an 8B model, that requires 60+ GB of VRAM just for the optimizer states, making it impossible on consumer hardware and expensive even on cloud GPUs. **LoRA** (Low-Rank Adaptation) solves this by freezing the original model and training only tiny "adapter" matrices that sit alongside the frozen weights.

Think of it like post-it notes on a textbook. You don't rewrite the entire textbook to add your own notes — you stick small post-it notes on the relevant pages. The textbook (base model) stays unchanged, your notes (adapter weights) capture your specific knowledge, and together they produce customized output. The post-it notes are tiny compared to the textbook, so they're cheap to create, easy to swap, and don't damage the original.

On Day 8 you learned quantization theory — representing weights in fewer bits to save memory. On Day 10 you used bitsandbytes to load models in 4-bit precision. Today those concepts combine: **QLoRA** applies LoRA training on top of a 4-bit quantized base model, reducing VRAM requirements from 60+ GB to under 10 GB for an 8B model. This makes fine-tuning accessible on a single consumer GPU.

## 1. How LoRA Works

In a standard transformer, each attention layer contains large weight matrices (Q, K, V, O) of shape `[hidden_dim × hidden_dim]`. For a 4096-dimensional model, each matrix is 4096×4096 = 16 million parameters. LoRA replaces the full weight update with a low-rank decomposition:

```
┌──────────────────────────────────────────────────────────────────┐
│          Memory Layout: Base Model + LoRA Adapter Weights         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FULL FINE-TUNING (update all weights):                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  W (4096 × 4096) = 16M params per matrix × 4 = 64M/layer  │  │
│  │  ████████████████████████████████████████████████████████  │  │
│  │  ALL parameters trainable → 8B params total → 60+ GB VRAM │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ─────────────────────────────────────────────────────────────   │
│                                                                  │
│  LoRA (freeze base, train small adapters):                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  W (frozen) = 16M params (not updated)                     │  │
│  │  ████████████████████████████████████████████████████████  │  │
│  │                                                            │  │
│  │  + LoRA adapter:                                           │  │
│  │  A (4096 × 16) = 65K params  ┐                            │  │
│  │  ░░░░                        ├─ rank=16 → 131K per matrix  │  │
│  │  B (16 × 4096) = 65K params  ┘                            │  │
│  │                                                            │  │
│  │  Total trainable: ~4M params (0.05% of 8B) → 0.5 GB VRAM  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ─────────────────────────────────────────────────────────────   │
│                                                                  │
│  QLoRA (4-bit base + LoRA adapters):                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  W (frozen, 4-bit quantized) = 4M bytes (not 32M)         │  │
│  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (4× smaller than FP16)                  │  │
│  │                                                            │  │
│  │  + LoRA adapter (FP16):                                    │  │
│  │  A (4096 × 16) ┐                                          │  │
│  │  ░░░░           ├─ still 131K params, trained in FP16      │  │
│  │  B (16 × 4096)  ┘                                          │  │
│  │                                                            │  │
│  │  Base model: ~4 GB | Adapter: ~16 MB | Total: ~6 GB VRAM  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Result: Fine-tune an 8B model on a single 8 GB GPU!            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

The math: instead of learning a full weight update `ΔW` (4096×4096), LoRA learns two small matrices `A` (4096×r) and `B` (r×4096) where `r` (the rank) is typically 8–64. The update becomes `ΔW = A × B`, a low-rank approximation that captures the most important adaptation directions with far fewer parameters.

## 2. LoRA Hyperparameters

| Parameter | Typical Range | Effect | Trade-off |
|-----------|:---:|---|---|
| `r` (rank) | 8–64 | Higher rank = more capacity | More params, more VRAM |
| `lora_alpha` | 16–64 | Scaling factor for adaptation | Higher = stronger adaptation signal |
| `target_modules` | q, k, v, o | Which matrices get adapters | More modules = better quality, more params |
| `lora_dropout` | 0.0–0.1 | Regularization during training | Prevents overfitting on small datasets |

## 3. Comparison: Fine-Tuning Methods

| Method | Trainable Params | VRAM (8B model) | Quality vs Full FT | Best Use Case |
|--------|:---:|:---:|:---:|---|
| Full fine-tuning | 100% (8B) | 60+ GB | Reference | Unlimited budget, maximum quality |
| LoRA (FP16 base) | 0.05–1% | 18–20 GB | 95–99% | GPU-rich, fast iteration |
| QLoRA (4-bit base) | 0.05–1% | 6–10 GB | 93–98% | Consumer GPU, cost-sensitive |
| Prompt tuning | <0.01% | 16 GB (inference) | 80–90% | Very few trainable params needed |
| Full FP16 + DeepSpeed | 100% (8B) | 24+ GB (multi-GPU) | Reference | Multi-GPU clusters |

## 4. Using PEFT (Parameter-Efficient Fine-Tuning)

Hugging Face's `peft` library makes LoRA simple. You wrap your base model with a LoRA configuration and only the adapter weights become trainable:

```python
from peft import LoraConfig, get_peft_model

# Define LoRA configuration
config = LoraConfig(
    r=16,                          # Rank: 16 (small but effective)
    lora_alpha=32,                 # Scaling: 2× rank is common
    target_modules=["q_proj", "v_proj"],  # Apply to Q and V matrices
    lora_dropout=0.05,             # Light regularization
    task_type="CAUSAL_LM",         # Autoregressive language model
)

# Wrap the base model — only adapter params become trainable
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# Output: "trainable params: 4,194,304 || all params: 8,030,261,248 || 0.05%"
```

## 5. Adapter Merging and Deployment

After training, you have two options for deployment:

**Keep separate:** Load base model + adapter at inference time. You can hot-swap adapters for different tasks without reloading the base model.

**Merge:** Combine the adapter weights back into the base model. This produces a single model file with no inference overhead, but you lose the ability to swap adapters.

```python
# Merge adapter into base model for deployment
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

vLLM (Day 15) supports serving multiple LoRA adapters on a single base model simultaneously — different users can use different fine-tuned variants without loading separate model copies.

## Setup

### Prerequisites

- Python 3.10+
- ~8 GB disk space (model + adapter)
- GPU with 8+ GB VRAM recommended (for QLoRA training)
- CPU works for creating and inspecting adapters (training will be very slow)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Model choice

This lab uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` — an Apache 2.0, ungated model that's small enough to create LoRA adapters on modest hardware. The concepts apply identically to larger models (Llama-3.1-8B, Qwen2.5-7B) with proportionally more VRAM.

### Note on Llama models

If you want to experiment with larger models like `meta-llama/Llama-3.1-8B`, you'll need to accept Meta's Llama 3.1 Community License at [https://huggingface.co/meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) and set your `HF_TOKEN` environment variable.

## What You'll Build

A Python script that creates a LoRA adapter on SmolLM2-1.7B-Instruct, measures how few parameters are actually trainable (compared to the full model), calculates the adapter size in megabytes, and demonstrates the dramatic memory savings — printing adapter size (MB) as the key metric.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- LoRA freezes the base model and trains small adapter matrices (typically 0.05–1% of total parameters) — dramatically reducing VRAM and training cost
- QLoRA combines 4-bit quantized base weights with FP16 LoRA adapters, enabling 8B model fine-tuning on a single 8 GB GPU
- The `rank` parameter (r) controls adapter capacity: higher rank = better quality but more parameters
- Adapters can be merged back into the base model for zero-overhead inference, or kept separate for hot-swapping between tasks
- PEFT + bitsandbytes makes the implementation straightforward: wrap model, configure adapters, train only the tiny adapter weights
- vLLM supports serving multiple LoRA adapters simultaneously on one base model — different tasks without model copies
