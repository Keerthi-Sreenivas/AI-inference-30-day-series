"""
Day 3: Hugging Face Transformers — Your First Direct Model Call
"""
#run pip install transformers torch accelerate before executing this code

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# --- Step 1: Load model and tokenizer ---
# AutoTokenizer detects the right tokenizer class for any model.
# AutoModelForCausalLM loads a left-to-right text generation model.
# "CausalLM" = each token can only see tokens before it (not after).
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,  # use float16 on GPU for half the memory
    device_map="auto",          # accelerate places layers on best available device
)

print(f"Loaded {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")

# --- Step 2: Tokenize a prompt using the chat template ---
# Instruct models expect a specific format with role markers.
# apply_chat_template wraps your message so the model knows it's a conversation.
prompt = "Explain what a neural network forward pass is in 3 sentences, for a software developer."
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

print(f"Prompt: {prompt}")
print(f"Token count: {inputs['input_ids'].shape[1]}")
print(f"First 10 token IDs: {inputs['input_ids'][0][:10].tolist()}\n")

# --- Step 3: Single forward pass — see raw logits ---
# A forward pass pushes tokens through every layer and produces logits:
# one score per vocabulary token, for each position in the sequence.
# The last position's logits tell us what the model thinks comes next.
with torch.no_grad():  # no gradients — we're not training
    outputs = model(**inputs)

logits = outputs.logits
print(f"Logits shape: {list(logits.shape)}  (batch, seq_len, vocab_size)")

# Show the top 5 most likely next tokens
last_logits = logits[0, -1, :]
probs = torch.softmax(last_logits, dim=-1)
top5_probs, top5_ids = torch.topk(probs, 5)

print("Top 5 next tokens:")
for i in range(5):
    token = tokenizer.decode(top5_ids[i].item())
    print(f"  {i+1}. \"{token}\" ({top5_probs[i].item():.2%})")
print()

# --- Step 4: Generate a full response ---
# model.generate() runs the forward-pass-then-sample loop for you.
# It handles KV caching (Day 12), stopping, and sampling.
input_len = inputs["input_ids"].shape[1]
start = time.perf_counter()

output_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,      # sample from the distribution, not greedy
    temperature=0.7,     # higher = more creative, lower = more deterministic (Day 4)
    top_p=0.9,           # nucleus sampling: ignore the long tail of unlikely tokens (Day 4)
)

elapsed = time.perf_counter() - start
new_tokens = output_ids.shape[1] - input_len
response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

print(f"Response:\n{response}\n")

# --- Step 5: Performance summary ---
print(f"Output tokens: {new_tokens}")
print(f"Time: {elapsed:.2f}s")
print(f"Tokens/sec: {new_tokens / elapsed:.1f}")
print(f"\nNote: Ollama (Day 1) uses llama.cpp — optimized C++ with GGUF quantization.")
print(f"Raw Transformers on CPU is slower. That gap is what vLLM (Day 15) closes.")


# --- Step 6 (Optional): Visualize attention patterns ---
# Re-run the forward pass with output_attentions=True to get attention weights.
# Each layer produces a matrix of shape (num_heads, seq_len, seq_len) showing
# how much each token "looks at" every other token when making its prediction.
import matplotlib.pyplot as plt

with torch.no_grad():
    attn_outputs = model(**inputs, output_attentions=True)

# Pick layer 0, head 0 as a starting point
# attn_outputs.attentions is a tuple: one tensor per layer
# Each tensor shape: (batch, num_heads, seq_len, seq_len)
attn_weights = attn_outputs.attentions[0][0, 0].cpu().numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Truncate to first 30 tokens so the heatmap is readable
max_tokens = min(30, len(tokens))
attn_weights = attn_weights[:max_tokens, :max_tokens]
tokens = tokens[:max_tokens]

plt.figure(figsize=(10, 10))
plt.imshow(attn_weights, cmap="viridis")
plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
plt.yticks(range(len(tokens)), tokens, fontsize=7)
plt.xlabel("Key (token being attended to)")
plt.ylabel("Query (token doing the attending)")
plt.title("Attention Pattern — Layer 0, Head 0")
plt.colorbar(label="Attention weight")
plt.tight_layout()
plt.savefig("attention_pattern.png", dpi=150)
print("Saved attention_pattern.png")
