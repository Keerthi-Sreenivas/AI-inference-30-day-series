# Day 19: Structured Output — JSON Mode & Constrained Decoding

## The Concept

Language models generate text token by token, and by default those tokens can be *anything*. But what if you need the output to be valid JSON? Or match a specific schema? That's where **constrained decoding** comes in — you force the model to only pick tokens that keep the output valid at every step.

Think of it like **Mad Libs**: you have a template with blanks, and each blank can only be filled with a certain type of word (noun, verb, adjective). The model is free to be creative *within* the constraint, but it can't break the rules. Instead of "noun goes here," the constraint is "valid JSON key goes here" or "integer between 1 and 100 goes here."

On Day 15 you sent open-ended prompts to vLLM and got free-form text back. On Day 4 you explored how sampling parameters (temperature, top-k) shape token selection. Structured output adds one more constraint layer: at each decoding step, tokens that would produce invalid output are *masked out* before sampling happens.

## How Constrained Decoding Works

The magic happens at the token-selection stage. Before the model samples the next token, a **logit processor** masks out any token that would make the output invalid according to the schema.

```
┌──────────────────────────────────────────────────────────────────┐
│         Constrained Decoding: Token Selection Pipeline            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                 │
│  │   Prompt    │  "Extract name and age as JSON"                 │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │    Model    │  Forward pass → logits for all tokens           │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────┐                 │
│  │        Raw Logits (50,000+ tokens)          │                 │
│  │  "the": 3.2  "{": 5.1  "Hello": 2.8  ...   │                 │
│  └──────┬──────────────────────────────────────┘                 │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────┐                 │
│  │     Schema Constraint (JSON FSM/Grammar)    │                 │
│  │                                              │                │
│  │  Current state: expecting object start       │                │
│  │  Valid tokens:  { only                       │                │
│  │  Masked out:    everything else → -inf       │                │
│  └──────┬──────────────────────────────────────┘                 │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────┐                 │
│  │      Filtered Logits                        │                 │
│  │  "{": 5.1  (only valid choice)              │                 │
│  └──────┬──────────────────────────────────────┘                 │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │  Sampling   │  Temperature, top-k, top-p applied              │
│  └──────┬──────┘                                                 │
│         ▼                                                        │
│  Token output: "{"                                               │
│                                                                  │
│  FSM advances to next state: expecting key string                │
│  Process repeats for every token until schema complete.           │
└──────────────────────────────────────────────────────────────────┘
```

The constraint engine (like Outlines) compiles your JSON schema or Pydantic model into a **finite-state machine (FSM)**. At each step, the FSM says which tokens are valid transitions — everything else gets masked to negative infinity before softmax.

## Methods Compared

| Method | Reliability | Speed Overhead | Flexibility | How It Works |
|--------|:-----------:|:--------------:|:-----------:|--------------|
| Prompt-only ("output JSON") | ~70-80% | None | Unlimited | Hope the model follows instructions |
| `response_format: json_object` | ~95% | Minimal | JSON only (no schema) | Engine-level JSON mode |
| JSON Schema (Outlines/vLLM) | 100% | 5-15% | Any JSON Schema | FSM masks invalid tokens |
| Regex constraint | 100% | 5-10% | Pattern-level | Regex compiled to FSM |
| Grammar (GBNF/CFG) | 100% | 10-20% | Context-free grammars | Full parser guides decoding |

**The key trade-off:** guaranteed validity costs a few percent of speed. For production APIs returning structured data, that trade-off is almost always worth it.

## Using Structured Output with vLLM

vLLM integrates with Outlines for constrained decoding. You pass a JSON schema via the `guided_json` parameter:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Pass schema to vLLM's OpenAI-compatible API
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me about Marie Curie"}],
    extra_body={
        "guided_json": Person.model_json_schema()
    }
)
# Output is GUARANTEED to be valid JSON matching the schema
```

Under the hood, vLLM:
1. Compiles the JSON schema into an FSM (cached after first use)
2. At each decoding step, queries the FSM for valid next tokens
3. Masks all other tokens before sampling
4. The result always parses without errors

## Common Pitfalls

1. **Schema too restrictive** — If your schema is very narrow, the model has few valid tokens at each step. Quality can drop because creative paths are blocked.
2. **First-token latency** — FSM compilation adds a one-time cost for new schemas. Cache your schemas in production.
3. **Enum fields** — Use `Literal["option1", "option2"]` in Pydantic. The FSM forces exact string matches at the token level.
4. **Nested objects** — Fully supported. The FSM tracks state across nesting levels.
5. **Optional fields** — Use `Optional[str] = None` in your Pydantic model. The FSM allows the key to be omitted entirely.
6. **Long outputs** — Schema-constrained generation is slightly slower per token. For very long structured outputs (>1000 tokens), consider chunking.

## Setup

### Prerequisites

- Completed Day 15 (vLLM server basics)
- Familiar with Day 4 (sampling parameters)
- NVIDIA GPU with 16+ GB VRAM (for Qwen2.5-7B)

### Installation

```bash
pip install -r requirements.txt
```

### Start the vLLM Server

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --guided-decoding-backend outlines
```

Wait until you see `Uvicorn running on http://0.0.0.0:8000`.

The `--guided-decoding-backend outlines` flag enables Outlines-based constrained decoding. This is the default in recent vLLM versions but explicit is better.

## What You'll Build

A script that sends multiple extraction prompts to vLLM with JSON schema constraints, validates that every response is well-formed JSON matching the schema, and reports the validity rate. You'll compare constrained decoding (100% valid) against unconstrained (prompt-only) to see the reliability difference.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- Constrained decoding masks invalid tokens at each generation step — guaranteeing valid output
- Outlines compiles JSON schemas into finite-state machines for efficient token filtering
- vLLM integrates Outlines natively via the `guided_json` parameter
- The speed overhead is 5-15% — negligible compared to the cost of retrying malformed outputs
- Pydantic models convert directly to JSON Schema for schema definition
- Always prefer schema-constrained output over prompt-based "please output JSON" in production
