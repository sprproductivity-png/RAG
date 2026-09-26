from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Ingestion Schemas
# -------------------------------------------------------------
class DocumentInput(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional unique document ID. Auto-generated if omitted.")
    title: Optional[str] = Field(default=None, description="Optional document title to provide additional context.")
    text: str = Field(..., description="Full text content of the document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata dictionary.")

class IngestRequest(BaseModel):
    documents: List[DocumentInput] = Field(..., min_length=1, description="List of documents to ingest.")
    collection_name: Optional[str] = Field(default=None, description="Target Qdrant collection name.")
    chunking_strategy: Literal["semantic", "sliding_window"] = Field(
        default="semantic",
        description="Chunking algorithm: 'semantic' (embedding distance breakpoints) or 'sliding_window'."
    )
    use_contextual_retrieval: bool = Field(
        default=True,
        description="Whether to generate Anthropic Contextual Retrieval summaries and prepend to chunks."
    )
    context_provider: Optional[Literal["heuristic", "anthropic", "openai", "ollama"]] = Field(
        default=None,
        description="Override contextual summarization provider."
    )

class ChunkDetail(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    raw_text: str
    context_summary: Optional[str] = None
    contextualized_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    collection_name: str
    documents_ingested: int
    chunks_created: int
    duration_ms: float
    message: str
    sample_chunks: Optional[List[ChunkDetail]] = Field(default=None, description="First few chunks with summaries for inspection.")

# -------------------------------------------------------------
# Search & Retrieval Schemas
# -------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string.")
    collection_name: Optional[str] = Field(default=None, description="Target collection name.")
    top_k: int = Field(default=5, ge=1, le=100, description="Final number of top results to return after re-ranking.")
    dense_top_k: int = Field(default=15, ge=1, le=200, description="Number of dense vector candidates to retrieve from Qdrant.")
    sparse_top_k: int = Field(default=15, ge=1, le=200, description="Number of sparse candidates to retrieve from BM25.")
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion smoothing constant k (default 60).")
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for dense retrieval in RRF / score fusion.")
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for sparse BM25 in RRF / score fusion.")
    fusion_method: Literal["rrf", "rsf"] = Field(
        default="rrf",
        description="Fusion strategy: 'rrf' (Reciprocal Rank Fusion) or 'rsf' (Relative Score Fusion with min-max normalization)."
    )
    apply_reranking: bool = Field(
        default=True,
        description="Whether to pass fused candidates through the Cross-Encoder re-ranker."
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key-value filters to match chunk metadata."
    )

class ScoredChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    context_summary: Optional[str] = None
    contextualized_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Detailed scoring diagnostics
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    fusion_rank: Optional[int] = None
    fusion_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    final_score: float

class SearchResponse(BaseModel):
    query: str
    collection_name: str
    total_dense_candidates: int
    total_sparse_candidates: int
    total_fused_candidates: int
    results: List[ScoredChunk]
    timing_ms: Dict[str, float]

# -------------------------------------------------------------
# Standalone Reranker Schemas
# -------------------------------------------------------------
class RerankCandidate(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., description="Document or passage text.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RerankRequest(BaseModel):
    query: str = Field(..., description="Query to compare documents against.")
    documents: List[RerankCandidate] = Field(..., min_length=1, description="List of candidate documents/passages to re-rank.")
    top_k: Optional[int] = Field(default=None, description="Number of top candidates to return.")

class RerankItem(BaseModel):
    index: int
    id: Optional[str] = None
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RerankResponse(BaseModel):
    query: str
    results: List[RerankItem]
    duration_ms: float

# -------------------------------------------------------------
# Collection & System Schemas
# -------------------------------------------------------------
class CollectionStats(BaseModel):
    collection_name: str
    points_count: int
    vectors_count: int
    vector_size: int
    distance: str
    bm25_docs_count: int

class CollectionsListResponse(BaseModel):
    collections: List[CollectionStats]

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    qdrant_connected: bool
    qdrant_mode: str
    qdrant_url: Optional[str]
    embedding_model: str
    embedding_device: str
    cross_encoder_model: str
    cross_encoder_device: str
    contextual_provider: str
    redis_connected: Optional[bool] = None
    queue_backend: Optional[str] = None

# -------------------------------------------------------------
# Document Management Schemas
# -------------------------------------------------------------
class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    filename: Optional[str] = None
    file_type: Optional[str] = None
    chunks_count: int
    ingested_at: Optional[str] = None
    sample_preview: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentListResponse(BaseModel):
    collection_name: str
    total_documents: int
    documents: List[DocumentSummary]

class DeleteDocumentResponse(BaseModel):
    message: str
    collection_name: str
    doc_id: str
    deleted_chunks: int

# -------------------------------------------------------------
# Background Task & Queue Schemas
# -------------------------------------------------------------
TaskStatus = Literal["queued", "processing", "completed", "failed"]
TaskStage = Literal["queued", "parsing", "chunking", "contextualizing", "embedding", "indexing", "completed", "failed"]

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    stage: TaskStage
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage 0-100")
    stage_message: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    collection_name: str
    title: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    chunks_created: Optional[int] = None
    error: Optional[str] = None
    sample_chunks: Optional[List[ChunkDetail]] = None

class TaskListResponse(BaseModel):
    total_tasks: int
    tasks: List[TaskResponse]


