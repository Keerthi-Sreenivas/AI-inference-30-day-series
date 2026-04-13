# Day 1 Prompt — Your First Local AI Inference

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

Install the Python dependencies for this project:

```bash
pip install requests
```

## Step 2: Install Ollama and pull the model

Check if Ollama is installed. If not, install it:

- macOS: `brew install ollama`
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`
- Windows: download from https://ollama.com/download

Then start the Ollama server and pull the model we'll use:

```bash
ollama serve &
ollama pull llama3.2:3b
```

## Step 3: Create the project

Create a `requirements.txt` with:
```
requests>=2.31.0
```

Create a minimal Python script called `main.py` that demonstrates local AI inference using Ollama. The script should:

1. **Send a prompt to the model** via Ollama's `/api/generate` endpoint with `stream: false`. Use this prompt:
   > "Explain what AI inference is in 3 sentences, as if explaining to a software developer who has never worked with ML."

2. **Print the response** text and basic metadata (tokens generated, total duration).

3. **Handle errors gracefully**: If Ollama isn't running or the model isn't pulled, print a helpful message instead of crashing.

4. **Add educational comments** explaining the API structure — what the request body looks like, what the response contains, and what the endpoint does.

Use only the `requests` library to talk to Ollama's local API (http://localhost:11434). Don't use the official ollama Python package — I want to see the raw HTTP calls so I understand the API.
