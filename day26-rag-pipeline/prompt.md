# Day 26 Prompt — RAG Inference Pipeline

> Paste this prompt into your coding assistant (Kiro, Cursor, Copilot, etc.) to scaffold today's project.

---

## Step 1: Install dependencies

```bash
pip install transformers torch faiss-cpu sentence-transformers accelerate
```

## Step 2: Model downloads

The script uses two models that download automatically:

- `sentence-transformers/all-MiniLM-L6-v2` — embedding model (~90 MB)
- `Qwen/Qwen2.5-7B-Instruct` — generation model (~14 GB)

Both are Apache 2.0 licensed, no account needed.

## Step 3: Create the project

Create a `requirements.txt` with:

```
transformers>=5.0.0
torch>=2.4.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
accelerate>=1.0.0
```

Create a Python script called `main.py` that builds a simple RAG pipeline: index documents, retrieve relevant passages, and generate grounded answers. Use `sentence-transformers` for embeddings, `faiss-cpu` for vector search, and `transformers` with `Qwen/Qwen2.5-7B-Instruct` for generation.

The script must be flat and linear — no classes, no `if __name__` guard. Top-to-bottom execution. Add at least one educational comment per logical block explaining what's happening and why.

### Functional Requirements

1. **Create a knowledge base** — Define at least 10 document chunks (strings) covering distinct topics (e.g., Python history, FAISS library, transformer architecture, GPU computing, tokenization). Each chunk should be 2-4 sentences of factual content.

2. **Load embedding model** — Load `sentence-transformers/all-MiniLM-L6-v2` for encoding documents and queries into 384-dimensional vectors. Print the embedding dimension.

3. **Build FAISS index** — Encode all document chunks into vectors, normalize them, and add to a FAISS `IndexFlatIP` index (inner product = cosine similarity on normalized vectors). Print the number of vectors indexed.

4. **Define test queries** — Create at least 5 test queries, each with a known ground-truth document index (the chunk that should be retrieved first). This enables measuring retrieval accuracy.

5. **Retrieve top-K passages** — For each query, embed it, search the FAISS index for the top-3 nearest neighbors, and check if the ground-truth chunk is in the results. Print which chunks were retrieved for each query.

6. **Calculate retrieval accuracy** — Compute the percentage of queries where the correct document appeared in the top-3 results. Print this as the primary metric: `"Retrieval Accuracy: {X}%"`.

7. **Load generation model** — Load `Qwen/Qwen2.5-7B-Instruct` with `device_map="auto"` for the generation phase. Print model parameter count.

8. **Generate RAG answers** — For at least 2 queries, build a RAG prompt (context + question), generate an answer using the model, and print both the retrieved context and the generated answer.

9. **Print summary metrics** — Print a table showing: number of documents indexed, embedding dimension, number of queries tested, retrieval accuracy percentage, and average retrieval time per query.

### Code style

- Flat script structure — no classes, no `main()` function
- Linear top-to-bottom flow — index docs, then retrieve, then generate
- Educational inline comments — at least one comment per logical block
- At most one `try/except` block (for model loading)

### Verification

After running the script, you should see:

- Confirmation of documents indexed and embedding dimensions
- For each test query: which documents were retrieved (with similarity scores)
- A retrieval accuracy percentage (target: 80%+ with well-designed chunks and queries)
- At least 2 generated answers grounded in retrieved context
- A summary metrics table at the end
