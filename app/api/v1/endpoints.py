import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Response, status

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    DocumentInput,
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
    DocumentSummary,
    DocumentListResponse,
    DeleteDocumentResponse,
    TaskResponse,
    TaskListResponse,
    QueueStatusResponse,
    HousekeepingResponse,
)
from app.services.file_parser import parse_uploaded_file
from app.services.embeddings import get_embedding_service
from app.services.vector_db import get_vector_db_service
from app.services.bm25 import get_bm25_service
from app.services.chunking import chunk_document
from app.services.contextual import enrich_chunks_with_context
from app.services.reranker import get_reranker_service
from app.services.hybrid import get_hybrid_engine
from app.services.task_manager import get_task_manager
from app.models.schemas import AskRequest, AskResponse
from app.services.generator import get_generator
import asyncio


router = APIRouter()

def make_qdrant_uuid(doc_id: str, chunk_idx: int) -> str:
    """Generates deterministic UUID string for Qdrant point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{chunk_idx}"))

# -------------------------------------------------------------
# Ingestion Endpoints
# -------------------------------------------------------------
async def _process_ingest(
    documents: List[DocumentInput],
    collection_name: Optional[str] = None,
    chunking_strategy: Literal["semantic", "sliding_window"] = "semantic",
    use_contextual_retrieval: bool = True,
    context_provider: Optional[Literal["heuristic", "anthropic", "openai", "ollama"]] = None,
) -> IngestResponse:
    t_start = time.perf_counter()
    target_collection = collection_name or settings.DEFAULT_COLLECTION

    embedding_service = get_embedding_service()
    vector_db_service = get_vector_db_service()
    bm25_service = get_bm25_service()

    # Ensure collection exists
    vector_db_service.ensure_collection(
        collection_name=target_collection,
        vector_size=embedding_service.get_dimension()
    )

    all_chunk_ids: List[str] = []
    all_contextualized_texts: List[str] = []
    all_payloads: List[Dict[str, Any]] = []
    sample_chunks_preview: List[ChunkDetail] = []

    total_chunks = 0
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for doc_idx, doc in enumerate(documents):
        doc_id = doc.id or f"doc_{uuid.uuid4().hex[:8]}"
        doc_title = doc.title or doc.metadata.get("title") or doc_id
        doc_meta = {**doc.metadata, "title": doc_title}
        if "ingested_at" not in doc_meta:
            doc_meta["ingested_at"] = now_iso

        # 1. Semantic or sliding-window chunking
        raw_chunks = chunk_document(
            text=doc.text,
            strategy=chunking_strategy
        )

        if not raw_chunks:
            continue

        # 2. Advanced Add: Anthropic Contextual Retrieval enrichment
        enriched_chunks = enrich_chunks_with_context(
            document_text=doc.text,
            chunks=raw_chunks,
            document_title=doc_title,
            provider=context_provider,
            enabled=use_contextual_retrieval
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
                "metadata": {**doc_meta, **(chunk.get("metadata") or {})},
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
        collection_name=target_collection,
        ids=all_chunk_ids,
        vectors=dense_vectors,
        payloads=all_payloads,
    )

    # 6. Index into BM25
    bm25_service.add_chunks(
        collection_name=target_collection,
        records=all_payloads,
    )

    duration_ms = (time.perf_counter() - t_start) * 1000.0

    return IngestResponse(
        collection_name=target_collection,
        documents_ingested=len(documents),
        chunks_created=total_chunks,
        duration_ms=round(duration_ms, 2),
        message="Documents successfully chunked, contextualized, and indexed.",
        sample_chunks=sample_chunks_preview,
    )

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents with Semantic Chunking and Anthropic Contextual Retrieval"
)
async def ingest_documents(request: IngestRequest):
    return await _process_ingest(
        documents=request.documents,
        collection_name=request.collection_name,
        chunking_strategy=request.chunking_strategy,
        use_contextual_retrieval=request.use_contextual_retrieval,
        context_provider=request.context_provider,
    )

@router.post(
    "/upload",
    response_model=Union[TaskResponse, IngestResponse],
    summary="Upload a file (.pdf, .txt, .md, .csv, .json). Caches raw file on disk and queues asynchronous processing."
)
async def upload_document(
    response: Response,
    file: UploadFile = File(..., description="File to upload and parse (.pdf, .txt, .md, .csv, .json)"),
    collection_name: Optional[str] = Form(None, description="Target Qdrant collection name"),
    title: Optional[str] = Form(None, description="Optional custom document title"),
    chunking_strategy: Literal["semantic", "sliding_window"] = Form("semantic"),
    use_contextual_retrieval: bool = Form(True),
    context_provider: Optional[Literal["heuristic", "anthropic", "openai", "ollama"]] = Form(None),
    async_processing: bool = Form(True, description="Whether to cache file and process via background task queue (default: True)"),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{file.filename}' is empty."
        )

    filename = file.filename or "uploaded_file"
    target_coll = collection_name or settings.DEFAULT_COLLECTION

    # 2-Step Approach: Cache raw file & queue background worker
    if async_processing:
        task_manager = get_task_manager()
        task_res = await task_manager.submit_file_task(
            file_bytes=file_bytes,
            filename=filename,
            collection_name=target_coll,
            title=title,
            chunking_strategy=chunking_strategy,
            use_contextual_retrieval=use_contextual_retrieval,
            context_provider=context_provider,
        )
        response.status_code = status.HTTP_202_ACCEPTED
        return task_res

    # Synchronous processing fallback
    extracted_text, inferred_title, metadata = parse_uploaded_file(file_bytes, filename)
    final_title = title.strip() if title and title.strip() else inferred_title
    metadata["title"] = final_title

    doc = DocumentInput(
        id=f"doc_{uuid.uuid4().hex[:8]}",
        title=final_title,
        text=extracted_text,
        metadata=metadata,
    )

    return await _process_ingest(
        documents=[doc],
        collection_name=target_coll,
        chunking_strategy=chunking_strategy,
        use_contextual_retrieval=use_contextual_retrieval,
        context_provider=context_provider,
    )

# -------------------------------------------------------------
# Background Task & Ingestion Queue Endpoints
# -------------------------------------------------------------
@router.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="Get status, current stage, and progress of an ingestion task"
)
async def get_task_status(task_id: str):
    task_manager = get_task_manager()
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found."
        )
    return task

@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="List active and recent background ingestion tasks"
)
async def list_tasks(limit: int = Query(50, ge=1, le=100)):
    task_manager = get_task_manager()
    tasks = await task_manager.list_tasks(limit=limit)
    return TaskListResponse(total_tasks=len(tasks), tasks=tasks)

@router.delete(
    "/tasks/{task_id}",
    summary="Remove a task record and its cached raw file"
)
async def delete_task(task_id: str):
    task_manager = get_task_manager()
    deleted = await task_manager.delete_task(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found."
        )
    return {"message": f"Task '{task_id}' removed."}

@router.post(
    "/tasks/{task_id}/retry",
    response_model=TaskResponse,
    summary="Retry a failed task, resuming automatically from the last checkpointed stage"
)
async def retry_task(task_id: str):
    task_manager = get_task_manager()
    task = await task_manager.retry_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found."
        )
    return task.to_response()



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
# Document Management
# -------------------------------------------------------------
@router.get(
    "/collections/{collection_name}/documents",
    response_model=DocumentListResponse,
    summary="List all distinct documents indexed in a collection"
)
async def list_documents(collection_name: str):
    bm25 = get_bm25_service()
    docs = bm25.get_documents(collection_name)
    if not docs:
        vector_db = get_vector_db_service()
        docs = vector_db.get_documents(collection_name)
    return DocumentListResponse(
        collection_name=collection_name,
        total_documents=len(docs),
        documents=[DocumentSummary(**d) for d in docs]
    )

@router.delete(
    "/collections/{collection_name}/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete an individual document and all its chunks from Qdrant and BM25"
)
async def delete_document(collection_name: str, doc_id: str):
    vector_db = get_vector_db_service()
    bm25 = get_bm25_service()

    vdb_deleted = vector_db.delete_document(collection_name, doc_id)
    bm25_deleted_count = bm25.delete_document(collection_name, doc_id)

    if not vdb_deleted and bm25_deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in collection '{collection_name}'."
        )

    return DeleteDocumentResponse(
        message=f"Document '{doc_id}' and {bm25_deleted_count} chunk(s) successfully deleted.",
        collection_name=collection_name,
        doc_id=doc_id,
        deleted_chunks=bm25_deleted_count,
    )

# -------------------------------------------------------------
# Queue Status Endpoint
# -------------------------------------------------------------
@router.get(
    "/queue",
    response_model=QueueStatusResponse,
    summary="Real-time Redis queue state and task metrics"
)
async def get_queue_status():
    task_manager = get_task_manager()
    stats = await task_manager.get_queue_stats()
    return QueueStatusResponse(**stats)

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
    task_manager = get_task_manager()
    redis_connected = task_manager.redis_available
    queue_backend = "redis" if redis_connected else "in_memory"
    queue_stats = await task_manager.get_queue_stats()

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
        redis_connected=redis_connected,
        queue_backend=queue_backend,
        queue_pending_count=queue_stats.get("pending_count", 0),
        queue_processing_count=queue_stats.get("processing_count", 0),
        queue_completed_count=queue_stats.get("completed_count", 0),
    )



@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Generative QA (RAG) using Hybrid Search"
)
async def ask_question(request: AskRequest):
    t_start = time.perf_counter()
    
    # 1. Retrieve contexts
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
    
    # 2. Generate Answer
    generator = get_generator(provider=request.provider, model=request.model)
    loop = asyncio.get_running_loop()
    
    # Run sync LLM generation in threadpool
    answer = await loop.run_in_executor(
        None, 
        generator.generate_answer, 
        request.query, 
        results["results"],
        request.temperature
    )
    
    duration_ms = (time.perf_counter() - t_start) * 1000.0
    
    return AskResponse(
        query=request.query,
        answer=answer,
        sources=scored_chunks,
        duration_ms=round(duration_ms, 2)
    )


# -------------------------------------------------------------
# System Maintenance & Housekeeping Endpoints
# -------------------------------------------------------------
@router.post(
    "/system/housekeep",
    response_model=HousekeepingResponse,
    summary="Trigger system maintenance: auto-retries failed tasks, purges stale cache, and reconciles data integrity"
)
async def run_housekeeping():
    from app.services.housekeeper import get_housekeeper_service
    housekeeper = get_housekeeper_service()
    report = await housekeeper.run_housekeeping_cycle()
    return report

@router.get(
    "/system/housekeep/status",
    response_model=HousekeepingResponse,
    summary="Get status and report of the latest housekeeping maintenance cycle"
)
async def get_housekeeping_status():
    from app.services.housekeeper import get_housekeeper_service
    housekeeper = get_housekeeper_service()
    if housekeeper.last_report is None:
        report = await housekeeper.run_housekeeping_cycle()
        return report
    return housekeeper.last_report
