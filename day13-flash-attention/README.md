# Day 13: Flash Attention — Faster, Leaner Attention

## The Concept

Every time a language model generates a token, it computes attention — a mechanism where each new token looks back at every previous token to decide what's relevant. On Day 12, you saw how the KV cache stores these attention values to avoid recomputation. But there's a deeper problem: the attention computation *itself* is slow because of how it accesses GPU memory. Flash Attention solves this by reorganizing the math to be memory-friendly, delivering the same results significantly faster.

Think of it like reading a book. Standard attention is like a student who, for every new sentence, flips back through the *entire* book — page 1, page 2, page 3, all the way to the current page — writing notes on a huge sheet of paper. The paper (GPU memory) gets enormous, and the constant flipping (memory transfers) wastes time. Flash Attention is like a smarter student who reads in tiles — they process a few nearby pages at a time, keep a small notepad, and update a running summary as they go. They never need the giant sheet of paper, and they avoid flipping back and forth across the entire book.

The result? Identical attention scores but 2–4× faster and with dramatically less memory. This is why Flash Attention has become the default for modern inference engines — it doesn't change *what* the model computes, just *how* it computes it. If you used the forward pass on Day 3, your model was already silently using something like this behind the scenes.

## How Standard Attention Works (The Bottleneck)

In the original attention algorithm, the GPU must:

1. Load Q (queries) and K (keys) from high-bandwidth memory (HBM)
2. Compute the full attention matrix (N × N for sequence length N)
3. Write this massive matrix back to HBM
4. Load it again to multiply with V (values)
5. Write the final output back to HBM

The problem isn't the arithmetic — GPUs are fast at math. The problem is the **memory traffic**. Moving data between HBM (slow, large) and SRAM (fast, tiny) dominates the runtime.

```
Standard Attention Memory Access Pattern:

  HBM (slow, 40GB)              SRAM (fast, 20MB)
  ┌──────────────┐              ┌──────────┐
  │  Q, K, V     │──── load ───▶│ compute  │
  │              │              │  Q × K^T │
  │              │◀── store ────│          │
  │  S (N×N)     │              └──────────┘
  │              │──── load ───▶┌──────────┐
  │              │              │ softmax  │
  │              │◀── store ────│          │
  │  P (N×N)     │              └──────────┘
  │              │──── load ───▶┌──────────┐
  │              │              │  P × V   │
  │  Output      │◀── store ────│          │
  └──────────────┘              └──────────┘

  6 memory transfers for one attention layer!
  The N×N matrix is the killer — at 4096 tokens: 64 million entries.
```

## How Flash Attention Fixes It (Tiled Computation)

Flash Attention never materializes the full N × N attention matrix. Instead, it processes attention in **tiles** — small blocks that fit entirely in SRAM:

```
Flash Attention — Tiled Memory Access:

  HBM (slow, 40GB)              SRAM (fast, 20MB)
  ┌──────────────┐              ┌──────────────────┐
  │              │              │                  │
  │  Q block 1   │──── load ───▶│  Q₁ × K₁^T      │
  │  K block 1   │──── load ───▶│  local softmax   │
  │  V block 1   │──── load ───▶│  × V₁            │
  │              │              │  accumulate ──┐  │
  │              │              │               │  │
  │  Q block 1   │──── load ───▶│  Q₁ × K₂^T   │  │
  │  K block 2   │──── load ───▶│  local softmax│  │
  │  V block 2   │──── load ───▶│  × V₂         │  │
  │              │              │  accumulate ──┘  │
  │              │              │                  │
  │  Output      │◀── store ────│  final result    │
  └──────────────┘              └──────────────────┘

  Only 1 write to HBM per tile! No N×N matrix ever created.
  Memory: O(N) instead of O(N²)
```

The key insight: you can compute softmax *incrementally* across tiles using the online softmax algorithm. Each tile updates a running maximum and denominator, so the final result is mathematically identical to computing the full matrix at once.

## Timeline Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│         Standard vs Flash Attention — Time Comparison             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Standard Attention (sequence length = 2048):                    │
│  ─────────────────────────────────────────────                   │
│  Time: ████████████████████████████████████████  100%            │
│        ↑load  ↑compute ↑store  ↑load  ↑soft  ↑store  ↑load ↑*V │
│        │←── memory bound ──────────────────────────────────→│    │
│                                                                  │
│  Flash Attention (same sequence, tiled):                         │
│  ─────────────────────────────────────────────                   │
│  Time: ████████████████                          ~40%            │
│        ↑load tile ↑compute+softmax+accumulate ↑next tile         │
│        │←── compute bound (GPU fully utilized) ──→│              │
│                                                                  │
│  Speedup: 2–4× depending on sequence length and GPU             │
│  Memory:  O(N) vs O(N²) — enables longer sequences              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Comparison Table

| Method | Peak Memory | Speed | GPU Requirement | Implementation |
|--------|-------------|-------|-----------------|----------------|
| Standard Attention | O(N²) — grows quadratically | Baseline (1×) | Any CUDA GPU | Built into PyTorch (`F.scaled_dot_product_attention`) |
| Flash Attention 2 | O(N) — linear | 2–4× faster | NVIDIA Ampere+ (A100, RTX 3090+) | `flash-attn` package or HF `attn_implementation="flash_attention_2"` |
| Flash Attention 3 | O(N) — linear | 3–5× faster (Hopper) | NVIDIA Hopper (H100) | `flash-attn>=2.6` with Hopper support |
| SDPA (PyTorch native) | O(N) — linear* | 1.5–3× faster | Any CUDA GPU | `torch.nn.functional.scaled_dot_product_attention` |
| xFormers Memory Efficient | O(N) — linear | 2–3× faster | NVIDIA Volta+ (V100+) | `xformers` package |

*PyTorch SDPA automatically dispatches to Flash Attention when the hardware supports it.

## When to Use Flash Attention

Flash Attention gives you the biggest wins when:

- **Long sequences** — The O(N²) → O(N) memory savings matter most at 2048+ tokens
- **Batch inference** — Freed memory lets you fit more sequences per batch
- **Latency-sensitive serving** — 2–4× speed improvement directly reduces response time
- **You're already on Ampere+ GPUs** — A100, A10G, RTX 3090/4090 all support it

You don't need Flash Attention when:

- Running on CPU (not applicable)
- Using very short sequences (< 512 tokens) where overhead exceeds benefit
- On older GPUs (V100, T4) — use PyTorch SDPA instead, which has its own optimizations

## Enabling Flash Attention in HuggingFace

With `transformers`, enabling Flash Attention is a single argument:

```python
from transformers import AutoModelForCausalLM

# Standard attention (baseline)
model_standard = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Flash Attention 2 (just add one argument)
model_flash = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # ← this is the only change
)
```

That's it. Same model, same weights, same outputs — just faster attention computation.

## Setup

### Prerequisites

- Completed Day 12 (KV cache concepts) and Day 3 (forward pass basics)
- Python 3.10+
- **NVIDIA GPU with Ampere architecture or newer** (Compute Capability 8.0+):
  - A100, A10G, RTX 3090, RTX 4090, H100
- Linux OS (flash-attn does not support Windows or macOS)
- CUDA 11.6+ installed

### License Notice

This lab uses `meta-llama/Llama-3.1-8B`, which requires license acceptance:
1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B
2. Accept the Llama 3.1 Community License Agreement
3. Set your Hugging Face token: `export HF_TOKEN=your_token_here`

Without license acceptance, the model cannot be downloaded. As an alternative, you can substitute `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, no license required) — Flash Attention works identically with any transformer model.

### Installation

```bash
# Linux with CUDA (required for flash-attn)
pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=5.0.0 accelerate>=1.0.0
pip install flash-attn>=2.6.0 --no-build-isolation

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### CPU/Older GPU Fallback

If you don't have an Ampere+ GPU, you can still run the lab — it will use standard attention (or PyTorch SDPA if available) and report that Flash Attention is not supported. You'll see the standard attention timing but not the speedup comparison. The README and code explain what *would* happen on supported hardware.

## What You'll Build

A benchmarking script that loads the same model twice — once with standard attention and once with Flash Attention — then generates text with each and compares the speed. You'll see the exact speedup factor printed as a quantitative metric.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Flash Attention computes the same attention scores but reorganizes memory access into tiles
- It reduces memory from O(N²) to O(N) by never materializing the full attention matrix
- Speedup is typically 2–4× on supported hardware (NVIDIA Ampere+)
- In HuggingFace, it's a single argument: `attn_implementation="flash_attention_2"`
- PyTorch's SDPA (`scaled_dot_product_attention`) provides similar benefits on broader hardware
- The improvement is largest for long sequences (2048+ tokens) where memory traffic dominates
