# Day 3: Hugging Face — Your AI Model Home Base

## The Concept

Every model you'll use in this series lives on Hugging Face. It's the GitHub of machine learning — an open platform where researchers and companies publish models, datasets, and demos. Before we write more inference code, you need to know how to navigate this ecosystem: find models, read their specs, understand file formats, and set up access for gated models.

Today is a no-code day. You'll create a Hugging Face account, learn to read a model card like a spec sheet, and understand what all those files in a model repo actually are. This knowledge pays off immediately — starting tomorrow, you'll be pulling models from here daily.

## 1. Create Your Account and Access Token

Sign up at [huggingface.co](https://huggingface.co) if you haven't already.

Then create an access token — you'll need this for gated models like Llama 3:

1. Go to **Settings → Access Tokens** (https://huggingface.co/settings/tokens)
2. Create a new token with **Read** access (that's all you need for downloading models)
3. Save it somewhere safe

Set it as an environment variable so `transformers` and other tools pick it up automatically:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export HF_TOKEN="hf_your_token_here"
```

<!-- Screenshot: HF token creation page -->

**Read vs Write tokens**: Read lets you download models (including gated ones you've accepted the license for). Write lets you push models and datasets to the Hub. For this series, Read is all you need.

## 2. The Model Hub — Finding the Right Model

The Model Hub (https://huggingface.co/models) is where you'll spend most of your time. Key filters to know:

- **Task**: filter by `text-generation` for all the models we'll use in this series
- **Library**: filter by `transformers`, `gguf`, or `vllm` depending on what engine you're using
- **License**: `apache-2.0` and `mit` are fully open; `llama3` requires license acceptance
- **Sort by**: Downloads (most popular), Trending (what's hot), or Recently Updated

<!-- Screenshot: Model Hub with filters applied -->

### Models you'll see throughout this series

| Model | Size | Why we use it |
|-------|------|---------------|
| SmolLM2-1.7B-Instruct | 1.7B | Small enough for CPU, fully open (Day 3-4) |
| Llama 3.2 3B | 3B | Via Ollama, great balance of size and quality (Day 1) |
| Qwen 2.5 7B | 7B | Strong multilingual tokenizer (Day 2), good for benchmarking |
| Llama 3.1 8B | 8B | The workhorse for Weeks 2-4 (quantization, vLLM, etc.) |

## 3. Reading a Model Card

Every model repo has a README — the model card. Think of it as the spec sheet. Here's what to look for:

- **Model size** — parameter count determines VRAM/RAM requirements. Rough rule: float16 needs ~2 bytes per parameter (7B model ≈ 14 GB)
- **Base vs Instruct/Chat** — base models do text completion, instruct models follow instructions. For inference, you almost always want the instruct variant
- **Architecture** — LlamaForCausalLM, Qwen2ForCausalLM, etc. This tells `AutoModelForCausalLM` how to load it
- **License** — determines if you can use it commercially. Apache 2.0 and MIT are permissive. Llama has a custom license with usage restrictions
- **Training data and benchmarks** — how the model was evaluated, what it's good at
- **The "Use this model" button** — gives you copy-paste code for transformers, vLLM, llama.cpp, etc.

<!-- Screenshot: A model card (e.g., SmolLM2-1.7B-Instruct) highlighting key sections -->

## 4. The Files Tab — Where Inference Engineering Lives

Click "Files and versions" on any model repo. This is where the actual model lives. Key files:

| File | What it is | Relevant day |
|------|-----------|---------------|
| `config.json` | Model architecture config — layer count, hidden size, vocab size, attention heads | Day 7 |
| `*.safetensors` | The model weights in SafeTensors format (safe, fast to load) | Day 5 |
| `*.gguf` | Quantized weights for llama.cpp / Ollama | Day 6, 9 |
| `tokenizer.json` | The tokenizer vocabulary and merge rules | Day 2 |
| `tokenizer_config.json` | Chat template and special tokens config | Day 3 |
| `generation_config.json` | Default sampling parameters (temperature, top_p, etc.) | Day 4 |

### File sizes tell the quantization story

Look at a 7B model's files tab and you'll see why quantization matters:

- `model.safetensors` (float16): ~14 GB
- `model-Q8_0.gguf` (8-bit): ~7 GB
- `model-Q4_K_M.gguf` (4-bit): ~4 GB

Same model, same knowledge, dramatically different sizes. Days 8-11 explain how this works.

<!-- Screenshot: Files tab of a model showing different file formats and sizes -->

## 5. Spaces and Inference API — Try Before You Download

Two ways to test models without downloading anything:

### Spaces
Spaces (https://huggingface.co/spaces) are hosted demos. Many models have a "Try it" Space where you can chat with the model in your browser. Good for quickly checking if a model fits your use case before committing to a multi-GB download.

<!-- Screenshot: A Space running a chat model -->

### Inference API
On many model pages you'll see a widget on the right side where you can type a prompt and get a response. This runs the model on HF's servers. It's rate-limited on the free tier but useful for quick tests.

For programmatic access:

```bash
curl https://api-inference.huggingface.co/models/HuggingFaceTB/SmolLM2-1.7B-Instruct \
  -H "Authorization: Bearer $HF_TOKEN" \
  -d '{"inputs": "What is inference?"}'
```

We won't use the Inference API in this series (we're focused on local/self-hosted inference), but it's good to know it exists.

## 6. The Hugging Face Cache — Where Models Live Locally

When you call `from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")`, here's what happens:

1. Transformers checks `~/.cache/huggingface/hub/` for a cached copy
2. If not found, it downloads the files from the Hub
3. Files are stored by model ID and revision (commit hash)
4. Next time you load the same model, it skips the download entirely

Useful commands for managing your cache:

```bash
# See what's cached and how much space it uses
huggingface-cli scan-cache

# Delete specific models to free disk space
huggingface-cli delete-cache
```

You can also change the cache location:

```bash
export HF_HOME="/path/to/bigger/drive/.cache/huggingface"
```

This explains the "first run is slow, second run is fast" behavior you saw on Days 1 and 2.

<!-- Screenshot: Output of huggingface-cli scan-cache -->

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Hugging Face is the central hub for open-source models — learn to navigate it and you'll move faster through every day in this series
- Access tokens unlock gated models like Llama 3 — create one now, you'll need it soon
- Model cards are spec sheets — check size, license, and architecture before downloading
- The Files tab reveals the real artifacts: safetensors (weights), GGUF (quantized), config.json (architecture), tokenizer.json (vocabulary)
- File sizes tell the quantization story at a glance — same model, 3-4x smaller with 4-bit quantization
- Models cache locally in `~/.cache/huggingface/` — know how to inspect and clean it
