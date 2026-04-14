# Day 2: Tokens & Tokenizers — How Text Becomes Numbers

## The Concept

Models don't read text — they read numbers. Before a single neuron fires, your prompt gets split into **tokens**: subword pieces mapped to integer IDs. "Tokenization" is the invisible first step of every inference call.

Understanding tokenization matters because inference is priced per token, context windows are measured in tokens, and different tokenizers split text differently. The same sentence might be 15 tokens with one model and 19 with another. On Day 1 we saw Ollama return `eval_count` — today we'll understand what those tokens actually are.

## How Tokenizers Work

There are two dominant tokenizer families in modern LLMs. Both solve the same problem — turning text into a fixed vocabulary of subword pieces — but they approach it differently.

### Byte Pair Encoding (BPE)

Used by: GPT-2, GPT-3/4, Qwen, LLaMA 3

BPE starts with individual characters (or bytes) and iteratively merges the most frequent adjacent pair into a new token. Think of it like compression:

1. Start: `['H', 'e', 'l', 'l', 'o']`
2. Most frequent pair is `('l', 'l')` → merge into `'ll'` → `['H', 'e', 'll', 'o']`
3. Next most frequent pair is `('H', 'e')` → merge into `'He'` → `['He', 'll', 'o']`
4. Repeat until you hit your target vocabulary size (typically 32k–150k tokens)

Common words like "the" end up as single tokens. Rare words get split into pieces: "tokenization" → `["token", "ization"]`. This is why it's called *subword* tokenization — it's between character-level and word-level.

GPT-2 specifically uses **byte-level BPE**, meaning it operates on raw bytes (0–255) rather than Unicode characters. This is why you see `Ġ` in GPT-2's output — it's the byte representation of a leading space (byte 0x20). `Ġquick` literally means `" quick"` with the space baked into the token. The `Ċ` character represents a newline (`\n`). This byte-level approach means GPT-2 can tokenize any text in any language without "unknown token" errors — everything is just bytes.

### SentencePiece

Used by: Mistral, LLaMA 1/2, T5, ALBERT

SentencePiece treats the input as a raw stream of characters (including spaces) and learns subword units directly — no pre-tokenization step needed. It typically uses BPE or Unigram under the hood, but the key difference is that it handles whitespace explicitly by replacing spaces with `▁` (the Unicode block character U+2581).

So when Mistral tokenizes "The quick brown fox", you see `['▁The', '▁quick', '▁brown', '▁fox']` — each `▁` marks where a space was. It also prepends a `<s>` (beginning-of-sequence) token automatically, which is why Mistral's token counts are often 1–2 higher than GPT-2's for the same text.

### Why This Matters for Inference

The tokenizer determines:
- **Cost**: APIs charge per token. The same prompt costs different amounts with different models.
- **Context window**: A 4096-token limit means different amounts of text depending on the tokenizer.
- **Multilingual fairness**: Tokenizers trained on English-heavy data split non-English text into more pieces, making other languages more expensive to process.

## Setup

```bash
pip install -r requirements.txt
```

No Ollama, GPU, or HuggingFace login needed. The script uses three fully open tokenizers that download automatically (~few MB each):

- `gpt2` — GPT-2's BPE tokenizer
- `mistralai/Mistral-7B-v0.1` — Mistral (Apache 2.0)
- `Qwen/Qwen2.5-7B` — Qwen 2.5 with strong multilingual support (Apache 2.0)

## Run the Prompt

Open your coding assistant and paste the contents of `prompt.md`, or run the script directly:

```bash
pip install -r requirements.txt
python lab/main.py
```

## What You'll Build

A short script that runs four tokenization experiments, printing the raw token pieces at each step:

1. Same text through three tokenizers — see how splits differ
2. Code vs English — code costs more tokens per character
3. Multilingual — non-English text is significantly more expensive
4. Formatting — whitespace and pretty-printing eat tokens

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Tokenizers split text into subword pieces, not words — "tokenization" → ["token", "ization"]
- Different models use different tokenizers, so the same text has different token counts
- Code and non-English text tokenize less efficiently than English prose
- Whitespace and formatting consume tokens — minification saves cost at scale
