# Day 19 Prompt — Structured Output with Constrained Decoding

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install vllm>=0.8.0 outlines>=0.1.0 transformers>=5.0.0 pydantic>=2.0.0
```

## Step 2: Start the vLLM server with Outlines backend

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --guided-decoding-backend outlines
```

Wait until you see `Uvicorn running on http://0.0.0.0:8000`.

## Step 3: Create the project

Create a `requirements.txt` with:
```
vllm>=0.8.0
outlines>=0.1.0
transformers>=5.0.0
pydantic>=2.0.0
```

Create a flat Python script called `main.py` that demonstrates structured output with constrained decoding. The script should:

1. **Check vLLM server availability** at `http://localhost:8000/health`. If unreachable, print startup instructions and exit cleanly.

2. **Define a Pydantic model** for structured extraction:
   ```python
   class MovieReview(BaseModel):
       title: str
       year: int
       rating: float  # 1.0 to 10.0
       genre: Literal["action", "comedy", "drama", "sci-fi", "horror", "thriller"]
       summary: str
       recommended: bool
   ```

3. **Run constrained decoding test**: Send 10 prompts like "Write a review for a sci-fi movie about time travel" to vLLM with `guided_json` set to the Pydantic model's JSON schema. Parse each response as JSON and validate against the schema.

4. **Run unconstrained comparison**: Send the same 10 prompts WITHOUT `guided_json` (just ask the model to "respond in JSON format" in the system prompt). Attempt to parse each response as JSON.

5. **Calculate and print validity rates** for both approaches:
   - Constrained: count of valid JSON responses / total
   - Unconstrained: count of valid JSON responses / total

6. **Print a sample valid output** showing the structured data extracted.

7. **Add educational comments** explaining:
   - How the FSM (finite-state machine) masks invalid tokens
   - Why constrained decoding guarantees 100% validity
   - The speed trade-off (typically 5-15% overhead)
   - How Pydantic schema converts to JSON Schema for the constraint engine

8. **Print the quantitative metric**: JSON validity rate (%) for both constrained and unconstrained modes.

Use the OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`. Pass the JSON schema via `extra_body={"guided_json": schema}`.

### Verification

After running, you should see output like:
```
=== Structured Output Benchmark ===
Constrained (guided_json):   10/10 valid (100.0%)
Unconstrained (prompt-only):  7/10 valid (70.0%)

Sample output (constrained):
{
  "title": "Temporal Paradox",
  "year": 2024,
  "rating": 8.5,
  "genre": "sci-fi",
  "summary": "A physicist discovers...",
  "recommended": true
}

JSON validity rate: 100.0% (constrained) vs 70.0% (unconstrained)
```
