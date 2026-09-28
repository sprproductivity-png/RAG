# NotebookLM-Style Self-Hosted Hybrid RAG System

A production-ready, fully local Retrieval-Augmented Generation (RAG) system with a React dashboard, powered by FastAPI, PostgreSQL, Redis, Qdrant, and LLaMA.cpp.

This system is designed to provide a "NotebookLM-style" experience: you upload your documents, and you ask questions. The system orchestrates advanced hybrid search (Dense + BM25 Sparse + Cross-Encoder Re-ranking) and Anthropic Contextual Retrieval to feed a locally hosted LLM (or external providers) to generate precise, grounded answers.

---

## 🏗️ Architecture

The system is fully containerized using **Docker Compose** and consists of 7 tightly integrated services:

1. **`postgres-rag-db`**: PostgreSQL database storing document metadata, collection registries, and background task progress.
2. **`redis-rag-queue`**: Redis message broker managing asynchronous document ingestion jobs.
3. **`redis-rag-commander`**: Web-based Redis GUI (port 8081) for inspecting queue health.
4. **`qdrant-hybrid-rag`**: Qdrant Vector Database storing high-dimensional semantic embeddings.
5. **`fastapi-rag-worker`**: Dedicated background worker that dequeues documents, chunks them, generates dense/sparse embeddings, and persists them to Qdrant/BM25.
6. **`fastapi-rag-backend`**: Core API server (port 8000) that handles user queries, performs Reciprocal Rank Fusion (RRF), cross-encoder re-ranking, and loads a local `.gguf` LLM for answer generation.
7. **`react-rag-frontend`**: React + Vite dashboard (port 3000) providing a NotebookLM-style Generative QA interface and document management.

### Data Flow
1. **Ingestion**: A user uploads a PDF via the frontend. The backend saves it to `/data/file_cache` and enqueues a job in Redis. The Worker picks it up, runs Semantic Chunking, generates Contextual Retrieval summaries (if Anthropic is enabled), embeds the text, and saves to Qdrant.
2. **Querying**: A user asks a question. The backend embeds the query, searches Qdrant (Dense) and BM25 (Sparse) on disk, fuses the results using RRF, re-ranks the top candidates using a Cross-Encoder, and feeds the top chunks to the LLM.
3. **Caching**: All HuggingFace embedding, re-ranking, and LLM models are permanently cached to your host machine at `./data/models`, completely eliminating cold-boot download times.

---

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose (Docker Desktop recommended for Windows/Mac)
- Optional: HuggingFace Token (for higher rate limits)
- Optional: OpenAI / Anthropic API keys (if you want to use cloud LLMs instead of local models)

### 1. Configuration
Create a `.env` file in the root of the project:

```env
# Optional: Speeds up model downloads and bypasses rate limits
HF_TOKEN=your_huggingface_token

# Optional: For Anthropic Contextual Retrieval / Claude Answers
ANTHROPIC_API_KEY=your_anthropic_key

# Optional: For OpenAI Answers
OPENAI_API_KEY=your_openai_key
```

### 2. Launching the Stack
Run the following command in the root directory:

```bash
docker-compose up -d --build
```

On the first run, the backend will download the necessary models (Embedding, Cross-Encoder, and the local LLM). Because of the local volume mount (`./data:/app/data`), these models are saved directly to your machine. Subsequent restarts will be near-instant.

### 3. Accessing the Services
- **React Dashboard**: http://localhost:3000
- **FastAPI Swagger Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Redis Commander**: http://localhost:8081

---

## 🧠 Core Technologies

- **Embedding Model**: `BAAI/bge-small-en-v1.5` (Dense retrieval)
- **Sparse Retrieval**: BM25 (Okapi) saved directly to disk.
- **Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Local LLM**: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` (Running securely on CPU via `llama-cpp-python`)
- **Semantic Chunking**: Context-aware chunking based on sentence embeddings and cosine similarity thresholds.
- **Contextual Retrieval**: Concept pioneered by Anthropic to append global document context to isolated chunks, drastically improving retrieval accuracy.

---

## 🛠️ Usage Guide

### Ingestion Studio
Navigate to the "Ingestion Studio" tab in the React frontend.
1. Upload `.txt`, `.pdf`, `.md`, or `.csv` files.
2. The upload returns immediately, and you can watch the real-time progress bar as the Redis worker chunks and embeds the document.

### Ask Studio
Navigate to the "Ask Studio" tab.
1. Type a question (e.g., "What is the battery capacity of the device?").
2. Press Enter or click **Ask AI**.
3. The system executes the full Hybrid Search + Re-ranking pipeline, gathers the most relevant sources, and generates a structured answer using your selected LLM provider.
4. The exact sources used to formulate the answer are displayed below for verification.

---

## 🧹 Maintenance

All persistent state is safely stored in named Docker volumes and local bind mounts.
- Postgres Data: `pg_data` volume
- Qdrant Vectors: `qdrant_storage` volume
- Redis Queue: `redis_data` volume
- HuggingFace Models: `./data/models` bind mount
- BM25 Indexes: `./data/bm25_indexes` bind mount

To reset the entire system and wipe all data:
```bash
docker-compose down -v
# Manually delete local files if desired
rm -rf ./data/*
```
