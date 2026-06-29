# Day 26: RAG Inference Pipeline — Retrieval + Generation

## The Concept

A **RAG pipeline** (Retrieval-Augmented Generation) combines two systems: a retrieval engine that finds relevant documents and a language model that generates answers grounded in those documents. Instead of relying solely on what the model memorized during training, RAG gives the model access to external knowledge at inference time — dramatically reducing hallucinations and keeping answers factual.

Think of it like an open-book exam. Without RAG, the model answers from memory alone — it might be confident but wrong. With RAG, the model first looks up relevant facts in a knowledge base, then writes its answer referencing those facts. The result is more accurate, more verifiable, and easier to update (just change the documents, no retraining needed).

On Day 2 you learned how tokenizers split text into subword tokens — today those same tokenizers power the embedding models that convert documents into searchable vectors. On Day 19 you explored structured output to ensure the model returns valid JSON — today you'll combine retrieval with generation to produce grounded, factual responses that cite their sources.

## 1. The RAG Data Flow

Every RAG system follows the same four-stage pipeline: embed the query, search the index, inject context, generate an answer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Data Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query                                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Convert query to dense vector           │
│  │  Embed Query     │   using sentence-transformers             │
│  │  (384-dim vec)   │                                           │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Search FAISS index for nearest          │
│  │  Retrieve Top-K  │   neighbors by cosine similarity          │
│  │  (FAISS index)   │   Returns K most relevant chunks          │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   Prepend retrieved chunks to the         │
│  │  Build Prompt    │   user query as context                   │
│  │  (context + Q)   │                                           │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐   LLM generates answer grounded           │
│  │  Generate Answer │   in the retrieved context                │
│  │  (Qwen2.5-7B)   │                                           │
│  └────┬─────────────┘                                           │
│       │                                                         │
│       ▼                                                         │
│  Grounded answer with source attribution                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Indexing — Building the Knowledge Base

Before you can retrieve, you need to index your documents. The indexing pipeline runs once (or periodically when documents change):

1. **Chunk** — Split documents into passages of 200–500 tokens each
2. **Embed** — Convert each chunk to a dense vector using an embedding model
3. **Store** — Add vectors to a FAISS index for fast similarity search

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a small, fast embedding model (384 dimensions)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all document chunks
chunks = ["Python was created by Guido van Rossum...", "FAISS is a library for..."]
embeddings = embedder.encode(chunks)

# Build a FAISS index for fast nearest-neighbor search
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vecs)
faiss.normalize_L2(embeddings)
index.add(embeddings)
```

## 3. Retrieval Methods Compared

Different retrieval strategies offer different tradeoffs between accuracy, speed, and complexity:

| Retrieval Method | Accuracy | Speed | Index Size | Complexity | Best For |
|-----------------|:--------:|:-----:|:----------:|:----------:|----------|
| Dense (FAISS) | High | Fast | Medium | Low | General-purpose semantic search |
| Sparse (BM25) | Medium | Very Fast | Small | Very Low | Keyword-heavy queries |
| Hybrid (Dense + BM25) | Very High | Medium | Large | Medium | Production systems |
| Re-ranking (cross-encoder) | Highest | Slow | N/A | High | Precision-critical tasks |
| ColBERT (late interaction) | Very High | Fast | Very Large | High | Large-scale retrieval |

**Dense retrieval** (what we'll use today) encodes both queries and documents into the same vector space and finds nearest neighbors. It understands semantic meaning — "automobile" matches "car" even though the words differ.

**Sparse retrieval** (BM25) uses traditional keyword matching with term frequency scoring. It's fast and simple but misses semantic relationships.

## 4. The Generation Phase

Once you have retrieved chunks, you inject them into the LLM prompt as context. The key design decisions:

- **How many chunks?** — Typically 3–5. More context means more information but slower generation and risk of distraction.
- **Context window budget** — Your chunks plus the query plus the answer must fit within the model's context length.
- **Prompt template** — Instruct the model to answer ONLY from the provided context, reducing hallucination.

```python
# Template that grounds the model in retrieved context
RAG_TEMPLATE = """Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
```

## 5. Measuring RAG Quality

RAG systems need specific metrics beyond normal generation quality:

- **Retrieval accuracy** — Did the correct document appear in top-K results?
- **Answer faithfulness** — Does the answer use information from the context (not hallucinate)?
- **Answer relevance** — Does the answer actually address the question?
- **Context precision** — What fraction of retrieved chunks are actually useful?

Today you'll measure retrieval accuracy — the most fundamental metric. If retrieval fails, generation has no chance.

## Setup

### Prerequisites

- Python 3.10+
- ~10 GB disk space (for Qwen2.5-7B + embedding model)
- 16 GB RAM minimum (model runs with CPU offloading)
- GPU with 16+ GB VRAM recommended for faster generation

### Install dependencies

```bash
pip install -r requirements.txt
```

### Model downloads

The script uses two models that download automatically on first run:

1. **`all-MiniLM-L6-v2`** — A small embedding model (~90 MB) from sentence-transformers
2. **`Qwen/Qwen2.5-7B-Instruct`** — The generation model (~14 GB)

Both are freely available under permissive licenses (Apache 2.0). No account or token needed.

## What You'll Build

A complete RAG pipeline that indexes a small knowledge base of documents, embeds user queries, retrieves the most relevant passages using FAISS similarity search, and generates grounded answers using Qwen2.5-7B — printing retrieval accuracy as a percentage to measure how well the system finds the right information.

## Video

🎬 [Watch the 90-second video](#) (link added after recording)

## Key Takeaways

- RAG combines retrieval (finding relevant docs) with generation (producing answers) — it's like giving the model an open-book exam
- The pipeline has four stages: embed query → retrieve top-K → build prompt → generate answer
- Dense retrieval with FAISS finds semantically similar passages even when exact keywords don't match
- Retrieval accuracy is the foundational metric — if retrieval fails, generation can't produce correct answers
- You can update a RAG system's knowledge by re-indexing documents — no model retraining needed
- Chunk size, number of retrieved passages, and prompt template design all significantly affect answer quality
