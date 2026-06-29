"""
Day 26: RAG Inference Pipeline — Retrieval + Generation

Build a complete RAG system: index documents with FAISS, retrieve relevant
passages via dense embeddings, and generate grounded answers with Qwen2.5-7B.

Models:
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0, ~90 MB)
- Generation: Qwen/Qwen2.5-7B-Instruct (Apache 2.0, ~14 GB)
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Define the knowledge base ---
# These document chunks represent our "database" of facts. In production,
# these would come from PDFs, wikis, databases, or web scrapes. Each chunk
# is 2-4 sentences covering a distinct topic.
documents = [
    "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "FAISS (Facebook AI Similarity Search) is a library developed by Meta for efficient similarity search and clustering of dense vectors. It can handle billions of vectors and supports both CPU and GPU execution.",
    "The transformer architecture was introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al. It replaced recurrent neural networks with self-attention mechanisms, enabling parallel processing of sequences.",
    "GPU computing uses thousands of cores for parallel processing, making it ideal for matrix operations in deep learning. NVIDIA's CUDA platform is the most widely used framework for GPU-accelerated computing.",
    "Tokenization is the process of splitting text into smaller units called tokens. Subword tokenization methods like BPE (Byte-Pair Encoding) handle rare words by breaking them into known subword pieces.",
    "vLLM is a high-throughput inference engine that uses PagedAttention to efficiently manage GPU memory. It can serve multiple concurrent requests with continuous batching for maximum throughput.",
    "Quantization reduces model size by representing weights with fewer bits. A 7B parameter model at FP16 uses 14 GB of memory, but at INT4 quantization it needs only 3.5 GB with minimal quality loss.",
    "RAG (Retrieval-Augmented Generation) combines information retrieval with language model generation. It reduces hallucinations by grounding answers in retrieved documents rather than relying on parametric memory alone.",
    "LoRA (Low-Rank Adaptation) fine-tunes large language models by adding small trainable adapter layers. Instead of updating all parameters, only the adapter weights are trained, reducing memory requirements by 10-100x.",
    "Flash Attention is an IO-aware attention algorithm that reduces memory usage from O(N²) to O(N) by processing attention in tiles. It achieves 2-4x speedup over standard attention on modern GPUs.",
    "Docker containers package applications with their dependencies into isolated environments. Docker Compose orchestrates multi-container deployments, making it easy to run complex systems with a single command.",
    "Prometheus is an open-source monitoring system that collects metrics via a pull model. It stores time-series data and provides a powerful query language (PromQL) for alerting and dashboard creation.",
]

print("=" * 70)
print("RAG PIPELINE: Retrieval-Augmented Generation")
print("=" * 70)
print(f"\nKnowledge base: {len(documents)} document chunks\n")

# --- Load embedding model ---
# all-MiniLM-L6-v2 maps text to 384-dimensional dense vectors. It's small
# (~90 MB) and fast, making it ideal for educational use. In production
# you might use a larger model (768 or 1024 dims) for better accuracy.
print("Loading embedding model: all-MiniLM-L6-v2")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = embedder.get_sentence_embedding_dimension()
print(f"Embedding dimension: {embedding_dim}\n")

# --- Build FAISS index ---
# We encode all documents into vectors, normalize them (for cosine similarity),
# and add them to a flat inner-product index. IndexFlatIP with normalized
# vectors computes cosine similarity.
print("Building FAISS index...")
start_index_time = time.perf_counter()

doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
doc_embeddings = doc_embeddings.astype(np.float32)
faiss.normalize_L2(doc_embeddings)

# Create the index — IndexFlatIP does exact nearest-neighbor search
index = faiss.IndexFlatIP(embedding_dim)
index.add(doc_embeddings)

index_time = time.perf_counter() - start_index_time
print(f"Indexed {index.ntotal} vectors in {index_time:.2f}s\n")

# --- Define test queries with ground-truth answers ---
# Each query has a known "correct" document (by index) so we can measure
# retrieval accuracy objectively. This simulates a labeled evaluation set.
test_queries = [
    {"query": "Who created the Python programming language?", "ground_truth": 0},
    {"query": "What is FAISS used for?", "ground_truth": 1},
    {"query": "When was the transformer architecture introduced?", "ground_truth": 2},
    {"query": "Why are GPUs good for deep learning?", "ground_truth": 3},
    {"query": "How does subword tokenization work?", "ground_truth": 4},
    {"query": "What is PagedAttention in vLLM?", "ground_truth": 5},
    {"query": "How does quantization reduce model size?", "ground_truth": 6},
    {"query": "What is RAG and why does it reduce hallucinations?", "ground_truth": 7},
    {"query": "How does LoRA fine-tune models efficiently?", "ground_truth": 8},
    {"query": "What makes Flash Attention faster?", "ground_truth": 9},
]

# --- Retrieve top-K passages for each query ---
# For each query, we embed it, search the index, and check if the ground-truth
# document appears in the top-3 results. This is the core retrieval evaluation.
print("=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)
print(f"Running {len(test_queries)} queries, retrieving top-3 for each\n")

TOP_K = 3
correct_retrievals = 0
total_retrieval_time = 0

for i, item in enumerate(test_queries):
    query = item["query"]
    ground_truth_idx = item["ground_truth"]

    # Embed and normalize the query
    start_search = time.perf_counter()
    query_vec = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    # Search FAISS index for top-K nearest neighbors
    scores, indices = index.search(query_vec, TOP_K)
    search_time = time.perf_counter() - start_search
    total_retrieval_time += search_time

    # Check if ground truth is in retrieved results
    retrieved_indices = indices[0].tolist()
    hit = ground_truth_idx in retrieved_indices
    if hit:
        correct_retrievals += 1

    status = "✓ HIT" if hit else "✗ MISS"
    print(f"  Query {i+1}: \"{query[:50]}...\"")
    print(f"    Retrieved: {retrieved_indices} (scores: {scores[0].tolist()})")
    print(f"    Ground truth: doc[{ground_truth_idx}] → {status}")
    print()

# --- Calculate and print retrieval accuracy ---
# This is the primary metric for RAG: what percentage of queries retrieve
# the correct document in their top-K results?
retrieval_accuracy = (correct_retrievals / len(test_queries)) * 100
avg_retrieval_time = (total_retrieval_time / len(test_queries)) * 1000  # ms

print("=" * 70)
print(f"RETRIEVAL ACCURACY: {retrieval_accuracy:.0f}%")
print(f"({correct_retrievals}/{len(test_queries)} queries found correct doc in top-{TOP_K})")
print(f"Average retrieval time: {avg_retrieval_time:.1f} ms/query")
print("=" * 70)

# --- Load generation model ---
# Qwen2.5-7B-Instruct generates answers grounded in the retrieved context.
# device_map="auto" places layers on GPU if available, falls back to CPU.
GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
print(f"\nLoading generation model: {GEN_MODEL}")
print("(First run downloads ~14 GB — subsequent runs use cache)\n")

try:
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, device_map="auto")
except OSError as e:
    print(f"Failed to load generation model: {e}")
    print("Check your internet connection and disk space, then try again.")
    raise SystemExit(1)

param_count = sum(p.numel() for p in gen_model.parameters())
print(f"Generation model loaded: {param_count / 1e9:.1f}B parameters")
print(f"Device: {gen_model.device}\n")

# --- Generate RAG answers ---
# For demonstration, we'll run the full RAG pipeline (retrieve + generate)
# on two example queries to show how retrieval grounds the generation.
RAG_TEMPLATE = """Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

print("=" * 70)
print("RAG GENERATION: Retrieve → Generate")
print("=" * 70)

demo_queries = [
    "Who created Python and when was it first released?",
    "How does quantization reduce the memory needed for large language models?",
]

for query in demo_queries:
    # Retrieve
    query_vec = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, TOP_K)

    # Build context from retrieved documents
    context_chunks = [documents[idx] for idx in indices[0]]
    context = "\n\n".join(context_chunks)

    # Format the RAG prompt
    rag_prompt = RAG_TEMPLATE.format(context=context, question=query)
    messages = [{"role": "user", "content": rag_prompt}]
    prompt_text = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = gen_tokenizer(prompt_text, return_tensors="pt").input_ids.to(gen_model.device)

    # Generate answer
    start_gen = time.perf_counter()
    output_ids = gen_model.generate(
        input_ids,
        max_new_tokens=150,
        do_sample=False,  # Greedy for factual answers
    )
    gen_time = time.perf_counter() - start_gen

    new_tokens = output_ids[0][input_ids.shape[1]:]
    answer = gen_tokenizer.decode(new_tokens, skip_special_tokens=True)
    tokens_per_sec = len(new_tokens) / gen_time

    print(f"\n  Query: {query}")
    print(f"  Retrieved docs: {indices[0].tolist()} (scores: {[f'{s:.3f}' for s in scores[0]]})")
    print(f"  Answer: {answer[:300]}")
    print(f"  Generation: {len(new_tokens)} tokens in {gen_time:.1f}s ({tokens_per_sec:.1f} tok/s)")
    print()

# --- Summary metrics ---
print("=" * 70)
print("SUMMARY METRICS")
print("=" * 70)
print(f"{'Metric':<30} {'Value':<20}")
print("-" * 50)
print(f"{'Documents indexed':<30} {len(documents):<20}")
print(f"{'Embedding dimension':<30} {embedding_dim:<20}")
print(f"{'Queries tested':<30} {len(test_queries):<20}")
print(f"{'Retrieval accuracy (top-3)':<30} {retrieval_accuracy:.0f}%")
print(f"{'Avg retrieval time':<30} {avg_retrieval_time:.1f} ms/query")
print(f"{'Generation model':<30} {GEN_MODEL}")
print(f"{'Generation parameters':<30} {param_count / 1e9:.1f}B")
print("-" * 50)
print("\nKey insight: Retrieval accuracy is the foundation of RAG quality.")
print("If the correct document isn't retrieved, the model can't generate a")
print("correct answer — no matter how powerful the language model is.")
