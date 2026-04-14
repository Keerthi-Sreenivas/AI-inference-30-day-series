# Day 2 Prompt — Tokens & Tokenizers

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers
```

## Step 2: No model download needed

Today we only use tokenizers, not full models. The tokenizer files (~few MB each) download automatically when the script runs. No Ollama, GPU, or HuggingFace login required — all models used are fully open.

## Step 3: Create the project

Create a `requirements.txt` with:
```
transformers>=4.44.0
```

Create a Python script called `main.py` that explores tokenization. Use `AutoTokenizer` from the `transformers` library.

Load three tokenizers — all fully open, no gated access:
- `gpt2` — GPT-2's BPE tokenizer
- `mistralai/Mistral-7B-v0.1` — Mistral's SentencePiece tokenizer (Apache 2.0)
- `Qwen/Qwen2.5-7B` — Qwen 2.5's tokenizer, strong multilingual support (Apache 2.0)

Add a brief comment on each explaining what kind of tokenizer it is (BPE, SentencePiece) and why it's notable.

Create a helper function `show_tokens(text)` that tokenizes the given text with all three tokenizers and prints: the tokenizer name, token count, and the list of token pieces (using `convert_ids_to_tokens`). Use this function throughout the script to show raw tokens between each step.

Run four short experiments, printing a section header before each:

1. **Same text, three tokenizers** — tokenize `"Tokenization is the first step of inference."` and show how the same sentence splits differently across models.

2. **Code vs English** — tokenize `"The quick brown fox jumps over the lazy dog."` and `"def fibonacci(n: int) -> list[int]:"` to show that code uses more tokens per character than prose.

3. **Multilingual costs** — tokenize `"Hello, how are you?"` and `"こんにちは、お元気ですか？"` to show that non-English text costs significantly more tokens.

4. **Formatting costs** — tokenize compact JSON `{"name":"Alice","age":30}` and pretty-printed JSON (same data with newlines and indentation) to show that whitespace eats tokens.

Keep the script flat and linear — no classes, no main() function, no error handling wrappers. Just load tokenizers at the top, define the helper, and run the four experiments in sequence. Add educational comments explaining what's happening at each step, written for someone learning inference for the first time.
