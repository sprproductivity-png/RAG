import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    ChunkDetail,
    SearchRequest,
    SearchResponse,
    ScoredChunk,
    RerankRequest,
    RerankResponse,
    RerankItem,
    CollectionsListResponse,
    CollectionStats,
    HealthResponse,
)
from app.services.embeddings import get_embedding_service
from app.services.vector_db import get_vector_db_service
from app.services.bm25 import get_bm25_service
from app.services.chunking import chunk_document
from app.services.contextual import enrich_chunks_with_context
from app.services.reranker import get_reranker_service
from app.services.hybrid import get_hybrid_engine

router = APIRouter()

def make_qdrant_uuid(doc_id: str, chunk_idx: int) -> str:
    """Generates deterministic UUID string for Qdrant point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{chunk_idx}"))

# -------------------------------------------------------------
# Ingestion Endpoints
# -------------------------------------------------------------
@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents with Semantic Chunking and Anthropic Contextual Retrieval"
)
async def ingest_documents(request: IngestRequest):
    t_start = time.perf_counter()
    collection_name = request.collection_name or settings.DEFAULT_COLLECTION

    embedding_service = get_embedding_service()
    vector_db_service = get_vector_db_service()
    bm25_service = get_bm25_service()

    # Ensure collection exists
    vector_db_service.ensure_collection(
        collection_name=collection_name,
        vector_size=embedding_service.get_dimension()
    )

    all_chunk_ids: List[str] = []
    all_contextualized_texts: List[str] = []
    all_payloads: List[Dict[str, Any]] = []
    sample_chunks_preview: List[ChunkDetail] = []

    total_chunks = 0

    for doc_idx, doc in enumerate(request.documents):
        doc_id = doc.id or f"doc_{uuid.uuid4().hex[:8]}"
        doc_title = doc.title or doc.metadata.get("title")

        # 1. Semantic or sliding-window chunking
        raw_chunks = chunk_document(
            text=doc.text,
            strategy=request.chunking_strategy
        )

        if not raw_chunks:
            continue

        # 2. Advanced Add: Anthropic Contextual Retrieval enrichment
        enriched_chunks = enrich_chunks_with_context(
            document_text=doc.text,
            chunks=raw_chunks,
            document_title=doc_title,
            provider=request.context_provider,
            enabled=request.use_contextual_retrieval
        )

        # 3. Prepare payload and IDs
        for chunk in enriched_chunks:
            chunk_idx = chunk["chunk_index"]
            point_id = make_qdrant_uuid(doc_id, chunk_idx)
            raw_text = chunk["raw_text"]
            context_summary = chunk.get("context_summary")
            contextualized_text = chunk.get("contextualized_text", raw_text)

            payload = {
                "doc_id": doc_id,
                "chunk_id": point_id,
                "chunk_index": chunk_idx,
                "raw_text": raw_text,
                "context_summary": context_summary,
                "contextualized_text": contextualized_text,
                "metadata": {**doc.metadata, **(chunk.get("metadata") or {})},
            }

            all_chunk_ids.append(point_id)
            all_contextualized_texts.append(contextualized_text)
            all_payloads.append(payload)

            if len(sample_chunks_preview) < 3:
                sample_chunks_preview.append(
                    ChunkDetail(
                        chunk_id=point_id,
                        doc_id=doc_id,
                        chunk_index=chunk_idx,
                        raw_text=raw_text,
                        context_summary=context_summary,
                        contextualized_text=contextualized_text,
                        metadata=payload["metadata"],
                    )
                )

        total_chunks += len(enriched_chunks)

    if not all_chunk_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid chunks were generated from the provided documents."
        )

    # 4. Dense embedding on contextualized chunks
    dense_embeddings = embedding_service.embed_texts(all_contextualized_texts)
    dense_vectors = [v.tolist() for v in dense_embeddings]

    # 5. Upsert into Qdrant
    vector_db_service.upsert_chunks(
        collection_name=collection_name,
        ids=all_chunk_ids,
        vectors=dense_vectors,
        payloads=all_payloads,
    )

    # 6. Index into BM25
    bm25_service.add_chunks(
        collection_name=collection_name,
        records=all_payloads,
    )

    duration_ms = (time.perf_counter() - t_start) * 1000.0

    return IngestResponse(
        collection_name=collection_name,
        documents_ingested=len(request.documents),
        chunks_created=total_chunks,
        duration_ms=round(duration_ms, 2),
        message="Documents successfully chunked, contextualized, and indexed.",
        sample_chunks=sample_chunks_preview,
    )

# -------------------------------------------------------------
# Search & Retrieval Endpoints
# -------------------------------------------------------------
@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid BM25/Dense search with Reciprocal Rank Fusion and Cross-Encoder re-ranking"
)
async def search(request: SearchRequest):
    hybrid_engine = get_hybrid_engine()
    results = hybrid_engine.search(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k,
        dense_top_k=request.dense_top_k,
        sparse_top_k=request.sparse_top_k,
        rrf_k=request.rrf_k,
        dense_weight=request.dense_weight,
        sparse_weight=request.sparse_weight,
        fusion_method=request.fusion_method,
        apply_reranking=request.apply_reranking,
        filter_metadata=request.filter_metadata,
    )

    scored_chunks = [
        ScoredChunk(
            chunk_id=r["chunk_id"],
            doc_id=r.get("doc_id", ""),
            chunk_index=r.get("chunk_index", 0),
            text=r.get("text", ""),
            context_summary=r.get("context_summary"),
            contextualized_text=r.get("contextualized_text"),
            metadata=r.get("metadata", {}),
            dense_rank=r.get("dense_rank"),
            dense_score=r.get("dense_score"),
            bm25_rank=r.get("bm25_rank"),
            bm25_score=r.get("bm25_score"),
            fusion_rank=r.get("fusion_rank"),
            fusion_score=r.get("fusion_score"),
            cross_encoder_score=r.get("cross_encoder_score"),
            final_score=r.get("final_score", 0.0),
        )
        for r in results["results"]
    ]

    return SearchResponse(
        query=results["query"],
        collection_name=results["collection_name"],
        total_dense_candidates=results["total_dense_candidates"],
        total_sparse_candidates=results["total_sparse_candidates"],
        total_fused_candidates=results["total_fused_candidates"],
        results=scored_chunks,
        timing_ms={k: round(v, 2) for k, v in results["timing_ms"].items()},
    )

# -------------------------------------------------------------
# Standalone Re-ranker Endpoint
# -------------------------------------------------------------
@router.post(
    "/rerank",
    response_model=RerankResponse,
    summary="Standalone Cross-Encoder re-ranking of arbitrary candidate documents"
)
async def rerank_documents(request: RerankRequest):
    t_start = time.perf_counter()
    reranker = get_reranker_service()

    candidates = [
        {"id": d.id, "text": d.text, "metadata": d.metadata, "original_index": i}
        for i, d in enumerate(request.documents)
    ]

    reranked = reranker.rerank(
        query=request.query,
        candidates=candidates,
        text_key="text",
        top_k=request.top_k,
    )

    results = [
        RerankItem(
            index=item["original_index"],
            id=item.get("id"),
            text=item["text"],
            score=round(item["cross_encoder_score"], 4),
            metadata=item.get("metadata", {}),
        )
        for item in reranked
    ]

    duration_ms = (time.perf_counter() - t_start) * 1000.0

    return RerankResponse(
        query=request.query,
        results=results,
        duration_ms=round(duration_ms, 2),
    )

# -------------------------------------------------------------
# Collections Management
# -------------------------------------------------------------
@router.get(
    "/collections",
    response_model=CollectionsListResponse,
    summary="List collections and vector statistics"
)
async def list_collections():
    vector_db = get_vector_db_service()
    bm25 = get_bm25_service()

    colls = vector_db.list_collections()
    stats_list = []
    for c in colls:
        s = vector_db.get_collection_stats(c)
        if s:
            s["bm25_docs_count"] = bm25.get_docs_count(c)
            stats_list.append(CollectionStats(**s))
    return CollectionsListResponse(collections=stats_list)

@router.delete(
    "/collections/{collection_name}",
    summary="Delete a collection from Qdrant and clear its BM25 index"
)
async def delete_collection(collection_name: str):
    vector_db = get_vector_db_service()
    bm25 = get_bm25_service()

    deleted_vdb = vector_db.delete_collection(collection_name)
    bm25.delete_index(collection_name)

    if not deleted_vdb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found or could not be deleted."
        )

    return {"message": f"Collection '{collection_name}' successfully deleted."}

# -------------------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and subsystem status"
)
async def health_check():
    vector_db = get_vector_db_service()
    embedding_service = get_embedding_service()
    reranker = get_reranker_service()

    is_connected, mode, url = vector_db.health_check()

    return HealthResponse(
        status="healthy" if is_connected else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        qdrant_connected=is_connected,
        qdrant_mode=mode,
        qdrant_url=url,
        embedding_model=settings.EMBEDDING_MODEL,
        embedding_device=embedding_service.device,
        cross_encoder_model=settings.CROSS_ENCODER_MODEL,
        cross_encoder_device=reranker.device,
        contextual_provider=settings.CONTEXTUAL_RETRIEVAL_PROVIDER,
    )

