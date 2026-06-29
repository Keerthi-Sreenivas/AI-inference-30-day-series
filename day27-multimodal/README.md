# Day 27: Multimodal Inference — Vision-Language Models

## The Concept

**Multimodal inference** means feeding a language model inputs from multiple modalities — typically images and text together — and getting a coherent text response. Vision-language models (VLMs) don't just "see" an image and describe it; they understand the relationship between visual content and textual questions, enabling tasks like visual question answering, document analysis, and image-guided reasoning.

Think of a bilingual person who speaks both English and Spanish. They don't translate word-by-word — they understand the meaning in one language and express it in the other. Vision-language models work similarly: they understand visual "language" (pixels, shapes, spatial relationships) and express their understanding in natural language. The key engineering challenge is converting an image into a sequence of tokens the text model can process.

On Day 3 you loaded your first model with `from_pretrained` — today you'll use the same pattern but with a processor that handles both image and text inputs simultaneously. On Day 2 you learned that text is split into subword tokens — today you'll see how images are split into "visual tokens" (patches), and discover that a single image can consume hundreds of tokens from your context window budget.

## 1. How Vision-Language Models Process Images

The core insight is that VLMs convert images into a sequence of token-like embeddings that sit alongside text tokens in the model's attention mechanism:

```
┌─────────────────────────────────────────────────────────────────┐
│           Vision-Language Model Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image Input              Text Input                            │
│  (640x480 JPEG)           ("What is in this image?")            │
│       │                        │                                │
│       ▼                        ▼                                │
│  ┌──────────────┐    ┌──────────────────┐                      │
│  │ Vision       │    │ Text Tokenizer   │                      │
│  │ Encoder      │    │ (BPE/SentencePc) │                      │
│  │ (ViT/CLIP)   │    │                  │                      │
│  └──────┬───────┘    └────────┬─────────┘                      │
│         │                     │                                 │
│         ▼                     ▼                                 │
│  [vis_1][vis_2]...[vis_N]  [tok_1][tok_2]...[tok_M]            │
│  (N image patch tokens)    (M text tokens)                     │
│         │                     │                                 │
│         └─────────┬───────────┘                                 │
│                   ▼                                              │
│  ┌──────────────────────────────────────────┐                  │
│  │  Combined Token Sequence                  │                  │
│  │  [vis_1]...[vis_N] [tok_1]...[tok_M]     │                  │
│  │                                           │                  │
│  │  All tokens attend to each other          │                  │
│  │  via standard transformer attention       │                  │
│  └──────────────────┬───────────────────────┘                  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────┐                  │
│  │  LLM Backbone (Qwen2.5-7B)              │                  │
│  │  Autoregressive generation               │                  │
│  └──────────────────┬───────────────────────┘                  │
│                     │                                           │
│                     ▼                                           │
│  Generated text: "The image shows colored rectangles..."        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The key insight:** images become tokens. A 640×480 image might produce 256–1024 visual tokens (depending on the model's patch size and resolution handling). These tokens compete for the same context window space as text tokens.

## 2. Image Tokenization: Patches to Tokens

Most VLMs split images into fixed-size patches (typically 14×14 or 32×32 pixels). Each patch is converted to an embedding vector by a vision encoder (usually a ViT — Vision Transformer):

- **Patch size 14×14** with a 224×224 image → 256 patches → 256 visual tokens
- **Patch size 14×14** with a 448×448 image → 1024 patches → 1024 visual tokens
- **Dynamic resolution** (Qwen2.5-VL) → adapts patch count to image content

Higher resolution means more patches, more tokens, more memory, slower inference — but better visual understanding of fine details.

## 3. Model Comparison

| Model | Modalities | Size | Image Tokens (typical) | Speed | License |
|-------|:----------:|:----:|:-----:|:-----:|---------|
| Qwen2.5-VL-7B | Text + Image + Video | 7B | 256–1280 | Medium | Apache 2.0 |
| LLaVA-1.6-7B | Text + Image | 7B | 576 | Fast | Llama license |
| InternVL2-8B | Text + Image | 8B | 256–1024 | Medium | Apache 2.0 |
| Phi-3.5-Vision-4B | Text + Image | 4B | 768 | Fast | MIT |
| GPT-4o (API) | Text + Image + Audio | Unknown | ~85 tokens/tile | Fastest (cloud) | Proprietary |

**Qwen2.5-VL-7B-Instruct** is our choice today because it's fully open (Apache 2.0), ungated (no license acceptance needed), supports dynamic resolution, and handles both images and video — all in a 7B parameter package.

## 4. The Token Budget Problem

Vision-language models face a unique context window challenge. Consider a model with a 32K token context window:

- Text-only: 32,000 tokens of text (~24,000 words)
- With one image: 32,000 - 500 image tokens = 31,500 text tokens available
- With five images: 32,000 - 2,500 image tokens = 29,500 text tokens available
- With a high-res image: 32,000 - 1,200 image tokens = 30,800 text tokens available

**Practical implication:** every image you add reduces your budget for text context, conversation history, and output length. This is why resolution control matters — don't send 4K images when 640×480 provides sufficient visual detail for your task.

## 5. Common Multimodal Tasks

```python
# Visual Question Answering (VQA)
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": chart_image},
        {"type": "text", "text": "What was the revenue in Q3 2024?"}
    ]}
]

# Document Understanding (OCR + reasoning)
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": receipt_image},
        {"type": "text", "text": "Extract the total amount and list of items."}
    ]}
]

# Image Comparison
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": before_image},
        {"type": "image", "image": after_image},
        {"type": "text", "text": "What changed between these two images?"}
    ]}
]
```

## Setup

### Prerequisites

- Python 3.10+
- ~16 GB disk space (for model download)
- 16 GB RAM minimum; GPU with 16+ GB VRAM strongly recommended
- No account or token needed (Apache 2.0 model, ungated)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Model download

The script uses **Qwen2.5-VL-7B-Instruct** — a 7B-parameter vision-language model that supports text, images, and video. It downloads automatically on first run (~15 GB). Fully open under Apache 2.0 license.

### GPU note

Multimodal models are significantly more compute-intensive than text-only models due to the vision encoder. While the script runs on CPU, generation will be very slow (minutes per response). A GPU with 16+ GB VRAM is strongly recommended for interactive use.

## What You'll Build

A Python script that loads a vision-language model, creates synthetic test images, processes them alongside text queries, measures how many visual tokens each image produces at different resolutions, and compares multimodal vs text-only generation speed — giving you a quantitative understanding of the computational cost of visual understanding.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Vision-language models convert images into token sequences that sit alongside text tokens in the transformer's attention mechanism
- Images are split into patches (typically 14×14 pixels each), and each patch becomes one visual token — more resolution means more tokens
- A single image can consume hundreds of context window tokens, reducing the budget available for text
- Multimodal inference is slower than text-only because the model must process both visual and text tokens through attention
- Choose image resolution based on your task — send 640×480 for general understanding, higher only when fine detail matters
- Qwen2.5-VL-7B is a strong open choice: Apache 2.0, ungated, supports dynamic resolution, handles images and video
