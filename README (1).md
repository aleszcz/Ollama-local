# RAG (Retrieval-Augmented Generation) Projects

Two RAG implementations exploring different vector store backends and embedding strategies.

---

## Project 1: Local RAG with Ollama + ChromaDB

A local-first RAG pipeline for querying restaurant reviews using Ollama for both embeddings and LLM inference.

### Architecture

```
CSV Data → mxbai-embed-large (Ollama) → ChromaDB → Retriever → llama3.2 (Ollama) → Answer
```

### Files

| File | Purpose |
|------|---------|
| `vector.py` | Loads restaurant reviews from CSV, generates embeddings via Ollama, stores/retrieves from ChromaDB |
| `main.py` | Chat loop that takes user questions, retrieves relevant reviews, and generates answers with llama3.2 |
| `realistic_restaurant_reviews.csv` | Source dataset with columns: `Title`, `Review`, `Rating`, `Date` |

### Setup

```bash
# Install dependencies
pip install langchain langchain-ollama langchain-chroma pandas

# Make sure Ollama is running with required models
ollama pull mxbai-embed-large
ollama pull llama3.2

# Run
python main.py
```

### How It Works

1. **First run**: Reads the CSV, creates embeddings with `mxbai-embed-large`, and persists them to a local ChromaDB instance at `./chrome_langchain_db`.
2. **Subsequent runs**: Loads the existing ChromaDB — skips re-embedding.
3. **Query time**: User question → top 5 similar reviews retrieved → injected into a prompt template → llama3.2 generates an answer grounded in actual review data.

---

## Project 2: Cloud RAG with MongoDB Atlas + HuggingFace

A cloud-based RAG pipeline for querying PDF documents (USP-800) using MongoDB Atlas Vector Search and HuggingFace embeddings. Originally built as a Colab notebook.

### Architecture

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → all-mpnet-base-v2 (HuggingFace) → MongoDB Atlas Vector Search → Retriever → Answer
```

### Setup

```bash
# Install dependencies
pip install langchain langchain-community langchain-core langchain-mongodb langchain-huggingface pymongo pypdf sentence-transformers
```

### Key Configuration

- **Embedding model**: `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- **Chunk size**: 500 characters with 150 overlap
- **Vector search**: MongoDB Atlas with cosine similarity
- **Database**: `book_mongodb_chunks` / `chunked_data`

### MongoDB Atlas Vector Index

Before querying, create a vector search index in the Atlas UI:

```json
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

Name it `vector_index` on the `chunked_data` collection.

### Pipeline Steps

1. Upload a PDF and chunk it with `RecursiveCharacterTextSplitter`
2. Generate embeddings with HuggingFace's `all-mpnet-base-v2`
3. Store document chunks + embeddings in MongoDB Atlas
4. Create a vector search index in the Atlas UI
5. Query via similarity search (top 3 results)

---

## Comparison

| | Local (Ollama + Chroma) | Cloud (MongoDB Atlas + HuggingFace) |
|---|---|---|
| **Data source** | CSV (restaurant reviews) | PDF (USP-800 document) |
| **Embeddings** | mxbai-embed-large (local) | all-mpnet-base-v2 (local compute, HF model) |
| **Vector store** | ChromaDB (local disk) | MongoDB Atlas Vector Search (cloud) |
| **LLM** | llama3.2 via Ollama | None (retrieval only) |
| **Top-k** | 5 | 3 |
| **Chunking** | Per-row (title + review) | RecursiveCharacterTextSplitter (500/150) |
| **Infrastructure** | Fully local | Requires MongoDB Atlas account |

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) (for the local project)
- MongoDB Atlas account (for the cloud project)
