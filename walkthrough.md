# Walkthrough: Self-Hosted Hybrid RAG & Re-ranking Engine with React Frontend & Docker Compose

We have built, verified, and packaged a complete, self-hosted **Hybrid RAG and Re-ranking Engine** with a dedicated **React + Vite + Tailwind CSS Frontend**, backed by **Docker-containerized Qdrant**, **FastAPI**, **BM25 keyword search**, **Semantic Chunking**, **Anthropic Contextual Retrieval**, and a **Cross-Encoder Re-ranker**.

---

## 1. What Was Built

### A. 3-Tier Containerized Architecture ([docker-compose.yml](file:///c:/Users/User/PycharmProjects/RAG/docker-compose.yml))
Running `docker compose up` starts all three services simultaneously:
1. **`qdrant` (Port 6333 & 6334)**: Official `qdrant/qdrant:v1.12.0` image with persistent named volume `qdrant_storage` and built-in Web Dashboard at `http://localhost:6333/dashboard`.
2. **`backend` (Port 8000)**: FastAPI service executing Dense + BM25 hybrid search, semantic chunking, Anthropic contextual retrieval, and Cross-Encoder re-ranking. Swagger docs at `http://localhost:8000/docs`.
3. **`frontend` (Port 3000)**: Dedicated container serving the compiled React application via Nginx with reverse proxy to FastAPI. Accessible at `http://localhost:3000/`.

---

### B. React Frontend Application ([frontend/](file:///c:/Users/User/PycharmProjects/RAG/frontend/))
Built with **React 18**, **Vite**, **Tailwind CSS**, and **Lucide React**:
1. **`Navbar.jsx`**: Live system indicators for Qdrant connection, hardware device (GPU/CUDA vs CPU), and loaded models (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`).
2. **`SearchStudio.jsx`**:
   - Query input with instant sample queries.
   - Interactive sliders for **Dense Weight** and **BM25 Sparse Weight**, Reciprocal Rank Fusion constant ($k_{rrf}=60$), top-$k$, and fusion method toggle (`rrf` vs `rsf`).
   - **Cross-Encoder Toggle**: Enable/disable re-ranking to visually compare raw hybrid fusion vs cross-attention scoring.
   - Timing breakdown bar (Dense, BM25, Fusion, Re-ranking, Total ms).
   - Candidate cards highlighting the **Anthropic Contextual Retrieval Summary** alongside individual scores.
3. **`IngestStudio.jsx`**:
   - Form with 1-click sample document loaders (*Acme Robotics* and *Apex Finance*).
   - Strategy selectors: Semantic Chunking (cosine distance breakpoints) vs Sliding Window.
   - Anthropic Contextual Retrieval switch (Heuristic offline, Claude API, OpenAI, Ollama).
   - Real-time sample chunk preview displaying situating summaries.
4. **`RerankPlayground.jsx`**: Standalone playground to test cross-encoder attention on arbitrary query/passage pairs.
5. **`CollectionsView.jsx`**: Real-time table of Qdrant vector collections and BM25 indices with point counts, vector dimension (384-dim), and delete actions.

---

### C. Backend Engine Services
1. **Docker Containerized Vector Database ([app/services/vector_db.py](file:///c:/Users/User/PycharmProjects/RAG/app/services/vector_db.py))**:
   - Connects to Qdrant Docker container (`http://localhost:6333` or `http://qdrant:6333`) via REST/gRPC.
   - Resilient automatic fallback to local embedded storage if Docker is temporarily offline during local development.
2. **Semantic Chunking ([app/services/chunking.py](file:///c:/Users/User/PycharmProjects/RAG/app/services/chunking.py))**:
   - Sentence embedding cosine distance analysis using `BAAI/bge-small-en-v1.5`.
   - Adaptive percentile breakpoint detection to preserve topic integrity.
3. **Anthropic Contextual Retrieval ([app/services/contextual.py](file:///c:/Users/User/PycharmProjects/RAG/app/services/contextual.py))**:
   - Situates each chunk within the overall document context using Anthropic's prompt template.
   - Prepends situational summaries to both dense embeddings and BM25 indexing (`<summary>\n\n<chunk_text>`).
   - Multi-provider: Anthropic Claude API, OpenAI/Ollama, and zero-dependency offline heuristic fallback.
4. **BM25 Sparse Retrieval ([app/services/bm25.py](file:///c:/Users/User/PycharmProjects/RAG/app/services/bm25.py))**:
   - Persistent BM25Plus sparse index with tokenization, stopword removal, and disk serialization.
5. **Hybrid Fusion ([app/services/hybrid.py](file:///c:/Users/User/PycharmProjects/RAG/app/services/hybrid.py))**:
   - Reciprocal Rank Fusion (RRF, $k=60$) and Relative Score Fusion (RSF).
6. **Cross-Encoder Re-ranking ([app/services/reranker.py](file:///c:/Users/User/PycharmProjects/RAG/app/services/reranker.py))**:
   - Joint self-attention scoring with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## 2. Verification & Test Results

### A. Automated Test Suite (`pytest tests/test_engine.py -v`)
All 8 automated tests passed with 100% success rate:
- `test_semantic_chunking`: PASSED (Successfully separated topic boundaries)
- `test_sliding_window_chunking`: PASSED
- `test_anthropic_contextual_retrieval`: PASSED (Correctly generated and prepended context)
- `test_dense_embedding_service`: PASSED (Verified 384-dimension vector embeddings)
- `test_bm25_sparse_index`: PASSED (Verified BM25Plus indexing and retrieval)
- `test_cross_encoder_reranker`: PASSED (Relevant passage scored higher than irrelevant)
- `test_reciprocal_rank_fusion`: PASSED (Verified RRF ranking formula)
- `test_fastapi_endpoints`: PASSED (Ingestion, hybrid search, rerank, collections, health, React UI)

```
tests/test_engine.py::test_semantic_chunking PASSED                      [ 12%]
tests/test_engine.py::test_sliding_window_chunking PASSED                [ 25%]
tests/test_engine.py::test_anthropic_contextual_retrieval PASSED         [ 37%]
tests/test_engine.py::test_dense_embedding_service PASSED                [ 50%]
tests/test_engine.py::test_bm25_sparse_index PASSED                      [ 62%]
tests/test_engine.py::test_cross_encoder_reranker PASSED                 [ 75%]
tests/test_engine.py::test_reciprocal_rank_fusion PASSED                 [ 87%]
tests/test_engine.py::test_fastapi_endpoints PASSED                      [100%]
======================= 8 passed in 14.51s ========================
```

### B. React Production Build (`npm run build` in `frontend/`)
```
✓ 1594 modules transformed.
dist/index.html                   1.15 kB │ gzip:  0.65 kB
dist/assets/index-rhYAT2M5.css   24.19 kB │ gzip:  5.28 kB
dist/assets/index-CZ0JTTUH.js   187.34 kB │ gzip: 56.87 kB
✓ built in 1.59s
```

---

## 3. How to Run

### Option 1: Docker Compose (Starts Everything in 1 Command)
```bash
docker compose up -d
```
Access all services:
- **React Frontend**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Backend & Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Qdrant Vector DB & Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### Option 2: Local Development Workflow

1. **Start Qdrant container**:
   ```powershell
   .\scripts\start_qdrant_docker.ps1
   ```

2. **Start FastAPI backend**:
   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Start React frontend with Hot-Module-Replacement (HMR)**:
   ```powershell
   cd frontend
   npm.cmd run dev
   ```
   Open [http://localhost:5173](http://localhost:5173) (automatically proxies `/api` calls to FastAPI on `:8000`).
