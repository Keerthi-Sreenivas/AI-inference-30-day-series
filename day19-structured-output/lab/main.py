"""
Day 19: Structured Output — JSON Mode & Constrained Decoding

This script compares constrained decoding (guaranteed valid JSON) against
unconstrained generation (prompt-only JSON request) and measures validity rates.

Constrained decoding uses a finite-state machine (FSM) compiled from your
JSON schema. At every token-generation step, the FSM masks out tokens that
would produce invalid output. The result is always parseable — no retries needed.

Before running:
    vllm serve Qwen/Qwen2.5-7B-Instruct \
        --port 8000 \
        --guided-decoding-backend outlines

Usage:
    python main.py
"""

import json
import time
import requests
from typing import Literal
from pydantic import BaseModel

# ─── Configuration ───────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{VLLM_BASE_URL}/health"
COMPLETIONS_URL = f"{VLLM_BASE_URL}/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Number of requests to send for each mode
NUM_REQUESTS = 10


# ─── Schema Definition ───────────────────────────────────────────────────────

# Pydantic models convert directly to JSON Schema.
# The constraint engine (Outlines) compiles this schema into an FSM
# that guides token generation step by step.

class MovieReview(BaseModel):
    """Schema for structured movie review extraction."""
    title: str
    year: int
    rating: float  # 1.0 to 10.0
    genre: Literal["action", "comedy", "drama", "sci-fi", "horror", "thriller"]
    summary: str
    recommended: bool


# ─── Service availability check ─────────────────────────────────────────────

def check_server():
    """Verify vLLM is running before we start."""
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        resp.raise_for_status()
    except (requests.ConnectionError, requests.Timeout):
        print("ERROR: vLLM server is not running.")
        print()
        print("Start it with:")
        print("  vllm serve Qwen/Qwen2.5-7B-Instruct \\")
        print("      --port 8000 \\")
        print("      --guided-decoding-backend outlines")
        raise SystemExit(1)


# ─── Test prompts ────────────────────────────────────────────────────────────

# Varied prompts to test schema adherence across different content
PROMPTS = [
    "Write a review for a sci-fi movie about time travel.",
    "Review an action movie set in space with a strong female lead.",
    "Give me a review of a comedy about a dog who becomes mayor.",
    "Write a review for a horror movie set in a haunted hospital.",
    "Review a drama about a pianist losing their hearing.",
    "Write a review for a thriller about a heist gone wrong.",
    "Review a sci-fi movie about first contact with aliens.",
    "Give me a review of an action movie featuring car chases.",
    "Write a review for a comedy about time-traveling chefs.",
    "Review a drama about a family reunion that goes sideways.",
]


# ─── Constrained decoding (guided_json) ─────────────────────────────────────

def run_constrained(prompts):
    """
    Send requests with guided_json — the FSM guarantees valid output.

    How it works:
    1. Pydantic model → JSON Schema → FSM (finite-state machine)
    2. At each token step, FSM provides set of valid next tokens
    3. All other tokens get logits set to -infinity
    4. Sampling happens only over valid tokens
    5. Result ALWAYS matches the schema — no parsing errors possible
    """
    schema = MovieReview.model_json_schema()
    valid_count = 0
    results = []

    for prompt in prompts:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a movie critic. Generate a review."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.7,
            # This is the key parameter — passes JSON Schema to Outlines FSM
            "extra_body": {"guided_json": schema},
        }

        # vLLM's OpenAI-compatible API accepts guided_json in extra_body
        # Some vLLM versions accept it at top level instead
        flat_payload = {**payload}
        extra = flat_payload.pop("extra_body", {})
        flat_payload.update(extra)

        resp = requests.post(COMPLETIONS_URL, json=flat_payload, timeout=60)
        content = resp.json()["choices"][0]["message"]["content"]

        # Try to parse — with constrained decoding this should always succeed
        try:
            parsed = json.loads(content)
            # Validate against Pydantic model
            MovieReview(**parsed)
            valid_count += 1
            results.append(parsed)
        except (json.JSONDecodeError, Exception):
            results.append(None)

    return valid_count, results


# ─── Unconstrained decoding (prompt-only) ────────────────────────────────────

def run_unconstrained(prompts):
    """
    Send requests WITHOUT guided_json — rely on prompt instructions only.

    The model may or may not produce valid JSON. Common failures:
    - Extra text before/after the JSON object
    - Missing quotes around keys
    - Trailing commas
    - Invalid enum values (genre not in allowed list)
    """
    valid_count = 0
    results = []

    for prompt in prompts:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a movie critic. Respond ONLY with a JSON object "
                        "matching this schema: {title: string, year: integer, "
                        "rating: float (1-10), genre: one of [action, comedy, drama, "
                        "sci-fi, horror, thriller], summary: string, recommended: boolean}. "
                        "No other text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.7,
            # No guided_json — model is free to generate anything
        }

        resp = requests.post(COMPLETIONS_URL, json=payload, timeout=60)
        content = resp.json()["choices"][0]["message"]["content"]

        # Try to parse — may fail due to malformed JSON
        try:
            # Strip potential markdown code fences
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(clean)
            MovieReview(**parsed)
            valid_count += 1
            results.append(parsed)
        except (json.JSONDecodeError, ValueError, Exception):
            results.append(None)

    return valid_count, results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Verify server is running
    check_server()

    print("=" * 60)
    print("STRUCTURED OUTPUT BENCHMARK")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Schema: MovieReview (6 fields, 1 enum constraint)")
    print(f"Requests per mode: {NUM_REQUESTS}")
    print()

    # Step 2: Run constrained decoding test
    print("Running constrained decoding (guided_json)...")
    start = time.perf_counter()
    constrained_valid, constrained_results = run_constrained(PROMPTS)
    constrained_time = time.perf_counter() - start
    print(f"  Done in {constrained_time:.1f}s")
    print()

    # Step 3: Run unconstrained comparison
    print("Running unconstrained (prompt-only)...")
    start = time.perf_counter()
    unconstrained_valid, unconstrained_results = run_unconstrained(PROMPTS)
    unconstrained_time = time.perf_counter() - start
    print(f"  Done in {unconstrained_time:.1f}s")
    print()

    # Step 4: Calculate validity rates
    constrained_rate = (constrained_valid / NUM_REQUESTS) * 100
    unconstrained_rate = (unconstrained_valid / NUM_REQUESTS) * 100

    # Step 5: Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Constrained (guided_json):  {constrained_valid}/{NUM_REQUESTS} valid ({constrained_rate:.1f}%)")
    print(f"  Unconstrained (prompt-only): {unconstrained_valid}/{NUM_REQUESTS} valid ({unconstrained_rate:.1f}%)")
    print()
    print(f"  Constrained time:   {constrained_time:.1f}s ({constrained_time/NUM_REQUESTS:.2f}s/request)")
    print(f"  Unconstrained time: {unconstrained_time:.1f}s ({unconstrained_time/NUM_REQUESTS:.2f}s/request)")

    # Speed overhead of constrained decoding
    if unconstrained_time > 0:
        overhead = ((constrained_time - unconstrained_time) / unconstrained_time) * 100
        print(f"  Constraint overhead: {overhead:+.1f}%")

    print()

    # Step 6: Print a sample valid output
    sample = next((r for r in constrained_results if r is not None), None)
    if sample:
        print("Sample output (constrained):")
        print(json.dumps(sample, indent=2))
        print()

    # Step 7: Print the key metric
    print(f"JSON validity rate: {constrained_rate:.1f}% (constrained) vs {unconstrained_rate:.1f}% (unconstrained)")
    print("=" * 60)


if __name__ == "__main__":
    main()
