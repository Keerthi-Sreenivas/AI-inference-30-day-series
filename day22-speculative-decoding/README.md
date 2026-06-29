# Day 22: Speculative Decoding — Draft Models for Speed

## The Concept

Autoregressive language models generate text one token at a time. Each token requires a full forward pass through billions of parameters — and the next token can't start until the previous one finishes. This sequential bottleneck means that even with a fast GPU, you're waiting for thousands of serial operations to produce a paragraph. **Speculative decoding** breaks this bottleneck by guessing ahead.

Think of it like autocomplete on your phone. As you type, the keyboard predicts your next few words. If the predictions are right, you tap to accept them instantly — no letter-by-letter typing needed. If they're wrong, you just keep typing normally. Speculative decoding works the same way: a small, fast "draft" model guesses several tokens ahead, and the large "target" model verifies those guesses in a single batch. Correct guesses are accepted for free. Wrong guesses are rejected and regenerated. The net effect? You get the exact same output quality as the large model, but faster.

On Day 4 you learned how sampling selects tokens from probability distributions. On Day 15 you saw how vLLM serves models at scale. On Day 12 you explored the KV cache that stores previously computed attention states. Speculative decoding builds on all three — it uses the KV cache to avoid recomputing verified tokens, relies on sampling alignment to preserve output quality, and integrates directly into serving engines like vLLM for production deployment.

## 1. How Speculative Decoding Works

The core idea is simple: verification is cheaper than generation. A large model can verify K tokens in roughly the same time it takes to generate one, because verification processes all K tokens in a single forward pass (just like processing a prompt). Here's the cycle:

```
┌──────────────────────────────────────────────────────────────────┐
│           Speculative Decoding: Draft → Verify → Accept          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: DRAFT (fast model generates K tokens)                   │
│  ┌──────────┐                                                    │
│  │Draft Model│──▶ token₁ ──▶ token₂ ──▶ token₃ ──▶ token₄       │
│  │  (1.7B)  │    (fast)     (fast)     (fast)     (fast)         │
│  └──────────┘                                                    │
│       Time: ~4 × small_forward_pass ≈ 20ms                      │
│                                                                  │
│  Step 2: VERIFY (large model checks all K tokens at once)        │
│  ┌──────────┐                                                    │
│  │Target    │──▶ [token₁, token₂, token₃, token₄] ──▶ probs     │
│  │Model (8B)│    (single batched forward pass)                   │
│  └──────────┘                                                    │
│       Time: ~1 × large_forward_pass ≈ 30ms                      │
│                                                                  │
│  Step 3: ACCEPT or REJECT (compare distributions)                │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ token₁: draft_prob=0.8, target_prob=0.7  → ✓ ACCEPT    │     │
│  │ token₂: draft_prob=0.6, target_prob=0.5  → ✓ ACCEPT    │     │
│  │ token₃: draft_prob=0.3, target_prob=0.1  → ✗ REJECT    │     │
│  │ token₄: (after rejection, not evaluated)               │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Result: 2 tokens accepted + 1 resampled = 3 tokens in 50ms     │
│  Without speculation: 3 tokens × 30ms = 90ms                    │
│  Speedup: ~1.8×                                                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐        │
│  │ Repeat: append accepted tokens, draft again from     │        │
│  │ the last accepted position, verify again...          │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 2. The Acceptance Algorithm

The key mathematical insight is that speculative decoding preserves the target model's exact distribution. It uses a rejection sampling scheme:

1. Draft model generates token with probability `q(x)`
2. Target model would generate that token with probability `p(x)`
3. Accept the draft token with probability `min(1, p(x)/q(x))`
4. If rejected, resample from an adjusted distribution: `norm(max(0, p(x) - q(x)))`

This guarantees the final output is statistically identical to what the target model would produce alone — no quality degradation whatsoever. The speedup comes purely from amortizing the verification cost across multiple draft tokens.

```python
# Conceptual acceptance logic (simplified)
# In practice, this is handled internally by the serving engine
for i, draft_token in enumerate(draft_tokens):
    target_prob = target_probs[i][draft_token]
    draft_prob = draft_probs[i][draft_token]

    # Accept with probability min(1, p/q)
    acceptance_ratio = target_prob / draft_prob
    if random.random() < min(1.0, acceptance_ratio):
        accepted_tokens.append(draft_token)
    else:
        # Reject: resample from adjusted distribution
        adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
        adjusted = adjusted / adjusted.sum()
        new_token = torch.multinomial(adjusted, 1)
        accepted_tokens.append(new_token)
        break  # Stop checking remaining draft tokens
```

## 3. When Speculative Decoding Helps Most

The speedup depends on the **acceptance rate** — what fraction of draft tokens the target model agrees with. Higher acceptance rate means more free tokens per cycle.

| Approach | Speedup | Memory Overhead | Quality | Best For |
|----------|:-------:|:---------------:|:-------:|----------|
| Standard autoregressive | 1.0× (baseline) | None | Reference | Simple deployments |
| Speculative (good draft) | 1.5–2.5× | +10–20% (draft model) | Identical | Long-form generation |
| Speculative (poor draft) | 0.8–1.2× | +10–20% (draft model) | Identical | Not recommended |
| Self-speculative (Medusa) | 1.3–2.0× | +5% (extra heads) | Identical | Memory-constrained |
| Lookahead decoding | 1.2–1.8× | Minimal | Identical | No draft model available |

**High acceptance rate scenarios (best speedup):**
- Draft and target models share the same training data
- Task is predictable (factual Q&A, code, translation)
- Temperature is low (fewer "creative" tokens to disagree on)

**Low acceptance rate scenarios (minimal benefit):**
- Draft model is from a completely different family
- Task is highly creative or open-ended
- Temperature is very high (distribution is flat, more disagreements)

## 4. Draft Model Selection

Choosing the right draft model is critical. The draft model should:

- Be **much faster** than the target (typically 4–10× fewer parameters)
- Produce tokens that the target model would **likely agree with**
- Share a **compatible vocabulary** (same tokenizer or subset)

Common pairings:

| Target Model | Draft Model | Typical Acceptance Rate |
|---|---|:---:|
| Llama-3.1-70B | Llama-3.1-8B | 70–85% |
| Llama-3.1-8B | SmolLM2-1.7B | 50–70% |
| Qwen2.5-72B | Qwen2.5-7B | 70–80% |
| Mixtral-8x7B | Mistral-7B | 65–80% |

The best draft models come from the same model family — they share vocabulary, training data patterns, and architectural decisions that maximize agreement.

## 5. Speculative Decoding in vLLM

vLLM supports speculative decoding natively. You configure it through the serving arguments:

```bash
# Start vLLM with speculative decoding enabled
vllm serve meta-llama/Llama-3.1-8B \
    --speculative-model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --num-speculative-tokens 5 \
    --port 8000
```

The `--num-speculative-tokens` flag controls how many tokens the draft model guesses per cycle. More guesses = higher potential speedup, but also more wasted computation on rejections. A value of 4–6 works well for most use cases.

## Setup

### Prerequisites

- Python 3.10+
- vLLM installed (GPU recommended for meaningful benchmarks)
- ~20 GB disk space (target model + draft model)
- NVIDIA GPU with 16+ GB VRAM (for both models simultaneously)

### Install dependencies

```bash
pip install -r requirements.txt
```

### GPU requirement

Speculative decoding requires holding both the target model and draft model in memory simultaneously. For Llama-3.1-8B + SmolLM2-1.7B-Instruct, you need approximately 18–20 GB of VRAM.

**No GPU?** The lab script includes a simulation mode that demonstrates the concept using timing measurements without actually running both models. You'll see the acceptance rate calculation logic and expected speedup numbers.

### License Notice

This lab uses `meta-llama/Llama-3.1-8B`, which requires accepting Meta's Llama 3.1 Community License. Visit [https://huggingface.co/meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) to accept the license, then set your `HF_TOKEN` environment variable.

## What You'll Build

A Python script that measures speculative decoding performance using vLLM — comparing generation speed with and without a draft model, and printing the acceptance rate (what percentage of draft tokens the target model agrees with) as the key metric.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Speculative decoding uses a fast draft model to guess ahead and a large target model to verify — correct guesses are "free" tokens
- The output quality is mathematically identical to standard decoding; only speed changes
- Acceptance rate determines actual speedup — higher acceptance means more free tokens per verification cycle
- Best results come from pairing models in the same family (shared vocabulary and training patterns)
- vLLM supports speculative decoding natively via `--speculative-model` and `--num-speculative-tokens` flags
- Typical speedups range from 1.5–2.5× on predictable tasks with well-matched draft models
