import pytest
import os
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.chunking import chunk_document, SemanticChunker, SlidingWindowChunker
from app.services.contextual import enrich_chunks_with_context, get_contextualizer
from app.services.embeddings import get_embedding_service
from app.services.vector_db import get_vector_db_service
from app.services.bm25 import BM25CollectionIndex, get_bm25_service
from app.services.reranker import get_reranker_service
from app.services.hybrid import get_hybrid_engine
from app.main import app

# Test data with 2 distinct topics to test semantic boundary detection
SAMPLE_DOC = """
Artificial Intelligence and neural networks have transformed natural language processing.
Modern transformer architectures rely on self-attention mechanisms to weigh relationships between tokens.
Large language models can perform zero-shot and few-shot reasoning across complex domains.

Baking sourdough bread requires water, flour, salt, and wild yeast culture.
Fermentation time is critical to developing lactic and acetic acids for sour flavor.
Baking at high temperature inside a Dutch oven traps steam to produce an airy crumb and blistered crust.
"""

@pytest.fixture(scope="session", autouse=True)
def test_setup_and_teardown():
    # Setup test directory paths
    test_qdrant_dir = Path("./data/test_qdrant_storage")
    test_bm25_dir = Path("./data/test_bm25")
    settings.QDRANT_LOCAL_PATH = str(test_qdrant_dir)
    settings.BM25_STORAGE_DIR = str(test_bm25_dir)
    settings.ENABLE_LOCAL_QDRANT_FALLBACK = True

    yield

    # Teardown
    if test_qdrant_dir.exists():
        shutil.rmtree(test_qdrant_dir, ignore_errors=True)
    if test_bm25_dir.exists():
        shutil.rmtree(test_bm25_dir, ignore_errors=True)

def test_semantic_chunking():
    """Verify semantic chunker detects topic shift between AI and Bread Baking."""
    chunks = chunk_document(SAMPLE_DOC, strategy="semantic")
    assert len(chunks) >= 2, "Semantic chunker should detect at least 2 distinct semantic sections."
    
    # Verify content in separate chunks
    first_chunk_text = chunks[0]["raw_text"].lower()
    last_chunk_text = chunks[-1]["raw_text"].lower()
    
    assert "artificial intelligence" in first_chunk_text or "neural" in first_chunk_text
    assert "sourdough" in last_chunk_text or "bread" in last_chunk_text

def test_sliding_window_chunking():
    """Verify sliding window fallback chunker."""
    chunks = chunk_document(SAMPLE_DOC, strategy="sliding_window", max_characters=300)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c["raw_text"]) <= 400

def test_anthropic_contextual_retrieval():
    """Verify Anthropic Contextual Retrieval situational summary prepending."""
    raw_chunks = [
        {"chunk_index": 0, "raw_text": "In Q2, revenue increased by 14% to $5.2 billion."},
        {"chunk_index": 1, "raw_text": "Operating margin expanded by 120 basis points."}
    ]
    doc_text = "Acme Corp Financial Report 2024. In Q2, revenue increased by 14% to $5.2 billion. Operating margin expanded by 120 basis points."
    
    enriched = enrich_chunks_with_context(
        document_text=doc_text,
        chunks=raw_chunks,
        document_title="Acme Corp Q2 2024 Financial Report",
        provider="heuristic",
        enabled=True
    )
    
    assert len(enriched) == 2
    for c in enriched:
        assert c["context_summary"] is not None
        assert "contextualized_text" in c
        assert c["context_summary"] in c["contextualized_text"]
        assert c["raw_text"] in c["contextualized_text"]
        # Summary should situate the chunk in the document
        assert "Acme Corp" in c["contextualized_text"]

def test_dense_embedding_service():
    """Verify embedding service vector generation and dimensions."""
    emb_service = get_embedding_service()
    dim = emb_service.get_dimension()
    assert dim == 384  # bge-small-en-v1.5 has 384 dimensions
    
    vec = emb_service.embed_query("machine learning retrieval")
    assert len(vec) == dim
    
    vecs = emb_service.embed_texts(["hello world", "information retrieval"])
    assert vecs.shape == (2, dim)

def test_bm25_sparse_index():
    """Verify BM25 sparse keyword indexing, ranking, and persistence."""
    bm25_service = get_bm25_service()
    test_collection = "test_bm25_coll"
    
    records = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 0,
            "raw_text": "Photosynthesis is the process by which green plants convert light into energy.",
            "contextualized_text": "Biology: Photosynthesis is the process by which green plants convert light into energy."
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "chunk_index": 0,
            "raw_text": "Quantum entanglement occurs when pairs of particles interact physically.",
            "contextualized_text": "Physics: Quantum entanglement occurs when pairs of particles interact physically."
        }
    ]
    
    bm25_service.add_chunks(test_collection, records)
    
    # Query for exact keyword
    results = bm25_service.search_sparse(test_collection, query="photosynthesis plants", limit=5)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["bm25_score"] > 0
    
    # Clean up
    bm25_service.delete_index(test_collection)

def test_cross_encoder_reranker():
    """Verify Cross-Encoder re-ranking places relevant passage at top."""
    reranker = get_reranker_service()
    query = "How is sourdough bread made?"
    candidates = [
        {
            "id": "cand_unrelated",
            "text": "The theory of general relativity explains gravity as the curvature of spacetime.",
        },
        {
            "id": "cand_relevant",
            "text": "Sourdough bread is made through the fermentation of dough using naturally occurring lactobacilli and yeast.",
        }
    ]
    
    reranked = reranker.rerank(query=query, candidates=candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["id"] == "cand_relevant"
    assert reranked[0]["cross_encoder_score"] > reranked[1]["cross_encoder_score"]

def test_reciprocal_rank_fusion():
    """Verify RRF mathematical combination of dense and sparse lists."""
    engine = get_hybrid_engine()
    
    dense_res = [
        {"chunk_id": "docA", "dense_score": 0.95},
        {"chunk_id": "docB", "dense_score": 0.85},
    ]
    sparse_res = [
        {"chunk_id": "docB", "bm25_score": 4.5},
        {"chunk_id": "docC", "bm25_score": 3.2},
    ]
    
    # RRF with k=60
    fused = engine.reciprocal_rank_fusion(dense_res, sparse_res, rrf_k=60, dense_weight=0.5, sparse_weight=0.5)
    
    # docB was present in BOTH lists (dense rank 2, sparse rank 1)
    # docB score = 0.5/(60+2) + 0.5/(60+1) = 0.5/62 + 0.5/61 ~= 0.00806 + 0.00819 = 0.01625
    # docA score = 0.5/(60+1) ~= 0.00819
    # docC score = 0.5/(60+2) ~= 0.00806
    # So docB MUST be rank 1 after RRF fusion!
    assert fused[0]["chunk_id"] == "docB"
    assert fused[0]["fusion_rank"] == 1

def test_fastapi_endpoints():
    """End-to-end API integration tests."""
    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "embedding_model" in data
        assert "cross_encoder_model" in data

        # 2. Ingest document
        ingest_payload = {
            "collection_name": "test_api_coll",
            "chunking_strategy": "semantic",
            "use_contextual_retrieval": True,
            "documents": [
                {
                    "id": "apple_doc_1",
                    "title": "Apple Inc. Q4 Fiscal Results",
                    "text": "Apple today announced financial results for its fiscal 2024 fourth quarter. The Company posted quarterly revenue of $94.9 billion, up 6 percent year over year. Products revenue was driven by strong iPhone sales globally.",
                    "metadata": {"quarter": "Q4", "year": 2024}
                },
                {
                    "id": "tesla_doc_1",
                    "title": "Tesla Energy Storage Deployments",
                    "text": "Tesla Megapack installations surged in the third quarter. The energy storage division achieved record gross margin as utility-scale battery projects expanded across North America and Europe.",
                    "metadata": {"division": "Energy"}
                }
            ]
        }
        ingest_res = client.post("/api/v1/ingest", json=ingest_payload)
        assert ingest_res.status_code == 200
        ingest_data = ingest_res.json()
        assert ingest_data["documents_ingested"] == 2
        assert ingest_data["chunks_created"] >= 2
        assert len(ingest_data["sample_chunks"]) >= 1

        # 3. Hybrid Search with Cross-Encoder re-ranking
        search_payload = {
            "collection_name": "test_api_coll",
            "query": "What was Apple's quarterly revenue?",
            "top_k": 3,
            "dense_top_k": 10,
            "sparse_top_k": 10,
            "apply_reranking": True
        }
        search_res = client.post("/api/v1/search", json=search_payload)
        assert search_res.status_code == 200
        search_data = search_res.json()
        assert len(search_data["results"]) >= 1
        top_hit = search_data["results"][0]
        assert "apple" in top_hit["text"].lower() or "apple" in (top_hit["context_summary"] or "").lower()
        assert top_hit["cross_encoder_score"] is not None

        # 4. Standalone Rerank
        rerank_payload = {
            "query": "electric vehicles battery storage",
            "documents": [
                {"id": "1", "text": "Recipe for chocolate chip cookies with sea salt."},
                {"id": "2", "text": "Megapack utility batteries store gigawatt-hours of solar and wind energy for power grids."}
            ]
        }
        rerank_res = client.post("/api/v1/rerank", json=rerank_payload)
        assert rerank_res.status_code == 200
        rerank_data = rerank_res.json()
        assert rerank_data["results"][0]["id"] == "2"

        # 5. List collections
        coll_res = client.get("/api/v1/collections")
        assert coll_res.status_code == 200
        colls = coll_res.json()["collections"]
        assert any(c["collection_name"] == "test_api_coll" for c in colls)

        # 6. Delete collection
        del_res = client.delete("/api/v1/collections/test_api_coll")
        assert del_res.status_code == 200

        # 7. Frontend UI verification
        ui_res = client.get("/")
        assert ui_res.status_code == 200
        assert "Hybrid RAG & Re-ranking Engine" in ui_res.text

        ui_direct = client.get("/ui")
        assert ui_direct.status_code == 200
        assert "Hybrid RAG & Re-ranking Engine" in ui_direct.text

def test_file_parser():
    """Verify parsing of text, markdown, json, csv, and pdf files."""
    from app.services.file_parser import parse_uploaded_file
    import io
    import pypdf

    # 1. Plain text / Markdown
    md_bytes = b"# Quantum Computing\nSuperconducting qubits operate at millikelvin temperatures."
    text, title, meta = parse_uploaded_file(md_bytes, "quantum_report.md")
    assert "Superconducting qubits" in text
    assert title == "quantum report"
    assert meta["file_type"] == ".md"

    # 2. JSON
    json_bytes = b'[{"topic": "Fusion Energy", "status": "Net energy gain achieved"}]'
    text, title, meta = parse_uploaded_file(json_bytes, "fusion_data.json")
    assert "Net energy gain achieved" in text
    assert meta["file_type"] == ".json"

    # 3. CSV
    csv_bytes = b"city,temperature,condition\nSeattle,62,Rain\nMiami,88,Sunny"
    text, title, meta = parse_uploaded_file(csv_bytes, "weather_readings.csv")
    assert "Seattle" in text and "Miami" in text
    assert meta["file_type"] == ".csv"
    assert meta["row_count"] == 3

    # 4. PDF (create minimal valid PDF using pypdf)
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_buffer = io.BytesIO()
    writer.write(pdf_buffer)
    # Blank PDF produces empty text, which correctly raises 400
    with pytest.raises(Exception):
        parse_uploaded_file(pdf_buffer.getvalue(), "blank.pdf")

def test_file_upload_and_document_management_api():
    """Verify multipart file upload, listing documents, and selective document deletion."""
    with TestClient(app) as client:
        test_coll = "test_doc_mgmt_coll"

        # 1. Upload Markdown file (synchronous mode)
        md_file_content = b"# Autonomous Vehicles\nLiDAR and radar systems provide 360-degree sensor coverage."
        upload_res = client.post(
            "/api/v1/upload",
            data={
                "collection_name": test_coll,
                "title": "AV Sensor Architecture",
                "chunking_strategy": "semantic",
                "use_contextual_retrieval": "true",
                "async_processing": "false",
            },
            files={"file": ("av_sensors.md", md_file_content, "text/markdown")}
        )
        assert upload_res.status_code == 200
        upload_data = upload_res.json()
        assert upload_data["documents_ingested"] == 1
        assert upload_data["chunks_created"] >= 1
        assert upload_data["collection_name"] == test_coll

        # 2. Upload CSV file (synchronous mode)
        csv_file_content = b"sensor_id,type,resolution\nS1,LiDAR,128-beam\nS2,Radar,77GHz"
        upload_res2 = client.post(
            "/api/v1/upload",
            data={
                "collection_name": test_coll,
                "title": "Sensor Inventory",
                "chunking_strategy": "sliding_window",
                "use_contextual_retrieval": "true",
                "async_processing": "false",
            },
            files={"file": ("inventory.csv", csv_file_content, "text/csv")}
        )
        assert upload_res2.status_code == 200

        # 3. List documents in collection
        list_res = client.get(f"/api/v1/collections/{test_coll}/documents")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["collection_name"] == test_coll
        assert list_data["total_documents"] == 2
        
        doc_titles = [d["title"] for d in list_data["documents"]]
        assert "AV Sensor Architecture" in doc_titles
        assert "Sensor Inventory" in doc_titles

        # Find doc_id of the AV sensor document
        av_doc = next(d for d in list_data["documents"] if d["title"] == "AV Sensor Architecture")
        av_doc_id = av_doc["doc_id"]
        av_chunks = av_doc["chunks_count"]
        assert av_chunks >= 1
        assert av_doc["file_type"] == ".md"

        # 4. Delete the AV sensor document specifically
        del_doc_res = client.delete(f"/api/v1/collections/{test_coll}/documents/{av_doc_id}")
        assert del_doc_res.status_code == 200
        del_doc_data = del_doc_res.json()
        assert del_doc_data["doc_id"] == av_doc_id
        assert del_doc_data["deleted_chunks"] == av_chunks

        # 5. Verify only the second document remains in the collection
        list_res_after = client.get(f"/api/v1/collections/{test_coll}/documents")
        assert list_res_after.status_code == 200
        list_data_after = list_res_after.json()
        assert list_data_after["total_documents"] == 1
        assert list_data_after["documents"][0]["title"] == "Sensor Inventory"

        # 6. Clean up collection
        clean_res = client.delete(f"/api/v1/collections/{test_coll}")
        assert clean_res.status_code == 200

def test_async_task_queue_ingestion():
    """Verify 2-step caching and background queue ingestion with progress tracking."""
    import time
    with TestClient(app) as client:
        test_coll = "test_async_queue_coll"

        # 1. Submit file for async ingestion (2-step: cached & queued)
        file_content = b"# Battery Engineering\nSilicon anode chemistry increases lithium-ion volumetric energy density by 25%."
        upload_res = client.post(
            "/api/v1/upload",
            data={
                "collection_name": test_coll,
                "title": "Battery Engineering Guide",
                "chunking_strategy": "semantic",
                "use_contextual_retrieval": "true",
                "async_processing": "true",
            },
            files={"file": ("battery_specs.md", file_content, "text/markdown")}
        )
        assert upload_res.status_code == 202
        task_data = upload_res.json()
        assert "task_id" in task_data
        assert task_data["status"] in ["queued", "processing"]
        assert task_data["filename"] == "battery_specs.md"
        task_id = task_data["task_id"]

        # 2. Poll task status until complete (timeout 20s)
        max_retries = 20
        final_task = None
        for _ in range(max_retries):
            status_res = client.get(f"/api/v1/tasks/{task_id}")
            assert status_res.status_code == 200
            current_task = status_res.json()
            if current_task["status"] in ["completed", "failed"]:
                final_task = current_task
                break
            time.sleep(0.5)

        assert final_task is not None, "Task did not complete within timeout."
        assert final_task["status"] == "completed", f"Task failed: {final_task.get('error')}"
        assert final_task["progress"] == 100
        assert final_task["chunks_created"] >= 1
        assert len(final_task["sample_chunks"]) >= 1

        # 3. Verify task appears in task history list
        list_tasks_res = client.get("/api/v1/tasks")
        assert list_tasks_res.status_code == 200
        all_tasks = list_tasks_res.json()["tasks"]
        assert any(t["task_id"] == task_id for t in all_tasks)

        # 4. Clean up task record
        del_task_res = client.delete(f"/api/v1/tasks/{task_id}")
        assert del_task_res.status_code == 200

        # 5. Clean up collection
        clean_coll_res = client.delete(f"/api/v1/collections/{test_coll}")
        assert clean_coll_res.status_code == 200



