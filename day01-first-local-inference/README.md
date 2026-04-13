# Day 1: Your First Local AI Inference

## The Concept

Today you'll run an AI model on your own machine for the first time. No cloud APIs, no subscriptions — just you and a model running locally.

We'll use **Ollama**, the simplest way to get started with local inference. Think of it as Docker for LLMs — one command to download, one command to run.

By the end of this day, you'll understand:
- What inference actually is (vs training)
- How to pull and run an open-source model locally
- What happens under the hood when you send a prompt
- How to measure basic performance (tokens per second)

## What is Inference?

Training teaches a model to recognize patterns from data. **Inference** is using that trained model to generate predictions — in our case, generating text token by token.

When you type a prompt and get a response, that's inference. The model:
1. Tokenizes your input (converts text to numbers)
2. Runs a forward pass through the neural network
3. Predicts the next token (word piece)
4. Repeats steps 2-3 until it generates a stop token

No weights are updated. No learning happens. It's purely prediction.

## Setup

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com/download
```

### Pull Your First Model

```bash
# We'll start with Llama 3.2 3B — small enough for any machine
ollama pull llama3.2:3b
```

This downloads a ~2GB quantized model. It runs on CPU, no GPU required.

## Run the Prompt

Open your coding assistant and paste the contents of `prompt.md`, or run the script directly:

```bash
pip install -r requirements.txt
python main.py
```

## What You'll Build

A Python script that:
1. Checks if Ollama is running and the model is available
2. Sends a prompt and streams the response
3. Measures tokens per second and total generation time
4. Explains what's happening at each step with inline comments

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Inference = using a trained model to generate output
- Ollama makes local inference trivially easy
- Even a 3B parameter model can produce useful output
- Tokens/second is the key performance metric for inference
- Quantized models (like GGUF Q4) let you run bigger models on less hardware
