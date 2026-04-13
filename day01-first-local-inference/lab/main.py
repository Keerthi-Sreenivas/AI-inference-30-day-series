"""
Day 1: Your First Local AI Inference

Call a local Ollama model with a raw HTTP request and print the response.

Before running, make sure Ollama is running and the model is pulled:
    ollama serve &
    ollama pull llama3.2:3b
"""

import requests

# Ollama runs a local REST API on port 11434
OLLAMA_URL = "http://localhost:11434/api/generate"

# The request body: pick a model and give it a prompt
payload = {
    "model": "llama3.2:3b",
    "prompt": (
        "Explain what AI inference is in 3 sentences, as if explaining "
        "to a software developer who has never worked with ML."
    ),
    "stream": False,  # get the full response at once
}

# POST to the generate endpoint — this is the core inference call
# First run can be slow (Metal shader compilation + model loading), so allow 5 min.
try:
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
except requests.ConnectionError:
    print("Could not connect to Ollama. Make sure it's running: ollama serve")
    raise SystemExit(1)
except requests.ReadTimeout:
    print("Request timed out — the model is probably still loading.")
    print("This is normal on the first run. Try again in a minute.")
    raise SystemExit(1)

result = response.json()

# If something went wrong, Ollama returns an "error" key instead of "response"
if "error" in result:
    print(f"Error: {result['error']}")
    print("Try: ollama pull llama3.2:3b")
    raise SystemExit(1)

# The response JSON contains the generated text and metadata
print(result["response"])
print(f"\nTokens generated: {result.get('eval_count', 'N/A')}")
print(f"Total duration: {result.get('total_duration', 0) / 1e9:.2f}s")
