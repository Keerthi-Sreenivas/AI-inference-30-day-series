"""
Day 2: Tokens & Tokenizers — How Text Becomes Numbers

Models don't read text. They read numbers. Before inference starts,
a tokenizer splits your text into subword pieces called tokens and
maps each one to an integer ID. This script shows you exactly how
that works using three well-known open tokenizers.
"""

from transformers import AutoTokenizer

# --- Load three popular tokenizers ---
# These are just the vocabulary + merge rules (~few MB each), not full models.
# All are fully open — no login or license acceptance needed.

# GPT-2: the tokenizer behind the GPT family. Uses Byte Pair Encoding (BPE).
# BPE starts with individual bytes and repeatedly merges the most frequent
# pair until it builds a vocabulary of ~50k subword tokens.
gpt2 = AutoTokenizer.from_pretrained("gpt2")

# Mistral 7B: uses SentencePiece with BPE. Same core idea, different
# implementation — treats input as a raw character stream and uses the
# block character ▁ to mark word boundaries. Apache 2.0 license.
mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Qwen 2.5: Alibaba's model. Trained on a large multilingual corpus,
# so it handles non-English text better than GPT-2. Apache 2.0 license.
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

tokenizers = {"GPT-2": gpt2, "Mistral": mistral, "Qwen 2.5": qwen}


def show_tokens(text: str) -> None:
    """Tokenize text with all three tokenizers and print the pieces."""
    for name, tok in tokenizers.items():
        ids = tok.encode(text)
        pieces = tok.convert_ids_to_tokens(ids)
        print(f"  {name:10s} -> {len(ids):2d} tokens -> {pieces}")
    print()


# -----------------------------------------------------------------------
# 1. Same text, three tokenizers — see how the same sentence splits
# -----------------------------------------------------------------------
print("=" * 60)
print("1. SAME TEXT, THREE TOKENIZERS")
print("=" * 60)

text1 = "Tokenization is the first step of inference."
print(f'\n  Text: "{text1}"\n')

# The same sentence becomes a different number of tokens depending
# on which model you use. This directly affects API cost and whether
# your prompt fits in the context window.
show_tokens(text1)

# -----------------------------------------------------------------------
# 2. Code vs English — code costs more tokens per character
# -----------------------------------------------------------------------
print("=" * 60)
print("2. CODE VS ENGLISH")
print("=" * 60)

english = "The quick brown fox jumps over the lazy dog."
code = "def fibonacci(n: int) -> list[int]:"

# Code has special characters (:, ->, [, ]) that don't merge as well
# in BPE vocabularies trained mostly on natural language. Same length
# of text, but code typically costs more tokens.
print(f'\n  English: "{english}"\n')
show_tokens(english)

print(f'  Code: "{code}"\n')
show_tokens(code)

# -----------------------------------------------------------------------
# 3. Multilingual costs — non-English text is more expensive
# -----------------------------------------------------------------------
print("=" * 60)
print("3. MULTILINGUAL COSTS")
print("=" * 60)

en_text = "Hello, how are you?"
ja_text = "こんにちは、お元気ですか？"  # Same meaning in Japanese

# BPE merge rules are learned from training data. English dominates,
# so English words get compact tokens. Non-Latin scripts stay as small
# pieces or individual bytes -> same meaning, way more tokens.
print(f'\n  English: "{en_text}"\n')
show_tokens(en_text)

print(f'  Japanese: "{ja_text}"\n')
show_tokens(ja_text)

# -----------------------------------------------------------------------
# 4. Formatting costs — whitespace eats tokens
# -----------------------------------------------------------------------
print("=" * 60)
print("4. FORMATTING COSTS")
print("=" * 60)

# Same data, different formatting. Every space and newline is a byte
# the tokenizer has to encode.
compact = '{"name":"Alice","age":30}'
pretty = '{\n    "name": "Alice",\n    "age": 30\n}'

print(f'\n  Compact: {compact}\n')
show_tokens(compact)

print(f'  Pretty:  {pretty}\n')
show_tokens(pretty)
