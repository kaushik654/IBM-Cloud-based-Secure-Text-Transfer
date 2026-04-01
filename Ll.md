You are a senior AI systems engineer specializing in Retrieval-Augmented Generation (RAG) systems deployed on mobile devices.

I have an existing Android application that implements a Classic RAG pipeline. Your task is to upgrade it to an Advanced RAG system with production-quality design and code.

You must NOT give high-level explanations only. You must provide:

- Clear architecture
- Step-by-step implementation
- Kotlin code for critical components
- Integration with my existing system

---

SYSTEM CONTEXT

Existing setup:

- Platform: Android (Kotlin)
- LLM: Qwen GGUF (on-device)
- Vector DB: ObjectBox (vector similarity search enabled)
- Current pipeline:
  1. Chunk documents
  2. Generate embeddings
  3. Store in ObjectBox
  4. Query embedding
  5. Similarity search (Top-K)
  6. Pass chunks to LLM

Data:

- Separate files per language:
  - *_eng.txt
  - *_hindi.txt
  - *_korean.txt

Current issue:

- English queries work
- Hindi and Korean queries fail to retrieve correct chunks

---

OBJECTIVE

Upgrade to Advanced RAG with:

1. Multilingual retrieval
2. Hybrid search (dense + sparse)
3. Metadata-aware filtering
4. Improved context quality
5. Fully on-device execution

---

REQUIRED ARCHITECTURE

You must implement the system in the following modules:

1. Indexing Module (offline)
2. Retrieval Module (runtime)
3. Generation Module (existing, minimal changes)

---

1. INDEXING MODULE

1.1 Chunking

- Fixed-size chunking with overlap
- Recommended:
  - chunk_size: 300–500 tokens
  - overlap: 50 tokens

1.2 Metadata Enrichment (MANDATORY)

Each stored chunk must include:

- id
- text
- embedding
- language (en / hi / ko)
- source_file

You must modify ObjectBox schema accordingly.

---

1.3 Multilingual Embeddings (MANDATORY)

Replace current embedding model with:

- multilingual-e5-small OR equivalent

Requirement:

- All languages must map into the same embedding space

---

1.4 Hybrid Indexing (MANDATORY)

Implement both:

A. Dense index:

- Already handled by ObjectBox

B. Sparse index:

- Implement BM25 locally

You must:

- Provide a Kotlin implementation OR
- Provide a lightweight BM25 design compatible with Android

---

2. RETRIEVAL MODULE

---

2.1 Query Processing (MANDATORY)

Steps:

1. Detect query language
2. Normalize query

Output:

- query_text
- query_language

---

2.2 Query Embedding

- Use same multilingual model

---

2.3 Hybrid Retrieval (MANDATORY)

Steps:

1. Dense retrieval (vector similarity)
2. Sparse retrieval (BM25)
3. Merge results

Score fusion formula:
final_score = alpha * dense_score + (1 - alpha) * sparse_score

Use:
alpha = 0.7

---

2.4 Metadata Filtering (CRITICAL)

Fix multilingual issue by:

- Filtering chunks using language metadata

Logic:

- First search only same-language chunks
- If results are insufficient, fallback to all languages

---

2.5 Reranking

If device allows:

- Implement cross-encoder reranker

Else:

- Re-rank using cosine similarity again

---

2.6 Relevance Filtering

Remove:

- Low score results
- Duplicate chunks
- Overlapping redundant chunks

Keep:

- Top 5–8 high-quality chunks

---

2.7 Context Fusion (MANDATORY)

Implement:

- Deduplication
- Merge overlapping chunks
- Sort by relevance

Output:

- Clean, structured context string

---

3. GENERATION MODULE

---

3.1 Prompt Construction

Use strict template:

"You are a helpful assistant.
Answer only from the provided context.

Context:
{context}

Question:
{query}

If the answer is not present, say 'I don’t know.'"

---

3.2 LLM Integration

- Use existing Qwen GGUF setup

---

3.3 Answer Synthesis (OPTIONAL)

- Combine multiple retrieved facts
- Ensure consistency

---

ANDROID CONSTRAINTS

- Must run fully offline
- Must be memory efficient
- Must avoid heavy models
- Must be modular and reusable

---

DEBUGGING AND LOGGING (MANDATORY)

You must include logs for:

- Query language
- Retrieved chunks (with scores)
- Dense vs sparse scores
- Final selected context

---

OUTPUT FORMAT (STRICT)

You must respond in this exact structure:

1. Final Architecture Diagram (textual)
2. Data Model (ObjectBox schema)
3. Indexing Pipeline Code (Kotlin)
4. BM25 / Sparse Retrieval Implementation
5. Hybrid Retrieval Code
6. Metadata Filtering Logic
7. Context Fusion Implementation
8. Integration with existing pipeline
9. Performance optimizations

Do not skip any section.

---

FINAL REQUIREMENT

This is not a theoretical explanation task.

You must produce implementation-ready code and architecture that can directly upgrade my existing Android RAG system into an Advanced RAG system.

Focus especially on fixing multilingual retrieval and improving retrieval accuracy.
