import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from app.core.config import settings
from app.core.logging import logger
from app.services.embeddings import get_embedding_service
from app.services.vector_db import get_vector_db_service
from app.services.bm25 import get_bm25_service
from app.services.reranker import get_reranker_service

class HybridSearchEngine:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_db_service = get_vector_db_service()
        self.bm25_service = get_bm25_service()
        self.reranker_service = get_reranker_service()

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Combines ranked lists using Reciprocal Rank Fusion:
        RRF_Score(d) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))
        """
        fused_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process dense rankings
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_contrib = dense_weight / (rrf_k + rank)
            if cid not in fused_map:
                fused_map[cid] = dict(item)
                fused_map[cid]["fusion_score"] = 0.0
                fused_map[cid]["bm25_rank"] = None
                fused_map[cid]["bm25_score"] = None
            fused_map[cid]["dense_rank"] = rank
            fused_map[cid]["dense_score"] = item.get("dense_score")
            fused_map[cid]["fusion_score"] += rrf_contrib

        # 2. Process sparse BM25 rankings
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            rrf_contrib = sparse_weight / (rrf_k + rank)
            if cid not in fused_map:
                fused_map[cid] = dict(item)
                fused_map[cid]["fusion_score"] = 0.0
                fused_map[cid]["dense_rank"] = None
                fused_map[cid]["dense_score"] = None
            fused_map[cid]["bm25_rank"] = rank
            fused_map[cid]["bm25_score"] = item.get("bm25_score")
            fused_map[cid]["fusion_score"] += rrf_contrib

        # 3. Sort by RRF score descending
        fused_list = list(fused_map.values())
        fused_list.sort(key=lambda x: x["fusion_score"], reverse=True)

        # Assign fusion rank
        for rank, item in enumerate(fused_list, start=1):
            item["fusion_rank"] = rank
            item["final_score"] = item["fusion_score"]

        return fused_list

    def relative_score_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Combines scores using min-max normalization:
        Norm_Score(d) = w_dense * norm(dense_score) + w_sparse * norm(bm25_score)
        """
        fused_map: Dict[str, Dict[str, Any]] = {}

        # Normalize dense
        dense_scores = [d["dense_score"] for d in dense_results if d.get("dense_score") is not None]
        d_min = min(dense_scores) if dense_scores else 0.0
        d_max = max(dense_scores) if dense_scores else 1.0
        d_range = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0

        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            raw_s = item.get("dense_score", 0.0)
            norm_s = (raw_s - d_min) / d_range
            fused_map[cid] = dict(item)
            fused_map[cid]["dense_rank"] = rank
            fused_map[cid]["dense_score"] = raw_s
            fused_map[cid]["fusion_score"] = dense_weight * norm_s
            fused_map[cid]["bm25_rank"] = None
            fused_map[cid]["bm25_score"] = None

        # Normalize sparse
        sparse_scores = [s["bm25_score"] for s in sparse_results if s.get("bm25_score") is not None]
        s_min = min(sparse_scores) if sparse_scores else 0.0
        s_max = max(sparse_scores) if sparse_scores else 1.0
        s_range = (s_max - s_min) if (s_max - s_min) > 1e-6 else 1.0

        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            raw_s = item.get("bm25_score", 0.0)
            norm_s = (raw_s - s_min) / s_range
            if cid not in fused_map:
                fused_map[cid] = dict(item)
                fused_map[cid]["dense_rank"] = None
                fused_map[cid]["dense_score"] = None
                fused_map[cid]["fusion_score"] = 0.0
            fused_map[cid]["bm25_rank"] = rank
            fused_map[cid]["bm25_score"] = raw_s
            fused_map[cid]["fusion_score"] += sparse_weight * norm_s

        fused_list = list(fused_map.values())
        fused_list.sort(key=lambda x: x["fusion_score"], reverse=True)
        for rank, item in enumerate(fused_list, start=1):
            item["fusion_rank"] = rank
            item["final_score"] = item["fusion_score"]

        return fused_list

    def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: int = settings.DEFAULT_RERANK_TOP_K,
        dense_top_k: int = settings.DEFAULT_DENSE_TOP_K,
        sparse_top_k: int = settings.DEFAULT_SPARSE_TOP_K,
        rrf_k: int = settings.DEFAULT_RRF_K,
        dense_weight: float = settings.DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = settings.DEFAULT_SPARSE_WEIGHT,
        fusion_method: Literal["rrf", "rsf"] = "rrf",
        apply_reranking: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        coll = collection_name or settings.DEFAULT_COLLECTION
        timings: Dict[str, float] = {}

        # 1. Dense retrieval
        t0 = time.perf_counter()
        query_vector = self.embedding_service.embed_query(query)
        dense_candidates = self.vector_db_service.search_dense(
            collection_name=coll,
            query_vector=query_vector,
            limit=dense_top_k,
            filter_metadata=filter_metadata,
        )
        timings["dense_retrieval_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Sparse BM25 retrieval
        t0 = time.perf_counter()
        sparse_candidates = self.bm25_service.search_sparse(
            collection_name=coll,
            query=query,
            limit=sparse_top_k,
            filter_metadata=filter_metadata,
        )
        timings["sparse_retrieval_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Hybrid Fusion
        t0 = time.perf_counter()
        if fusion_method == "rsf":
            fused_candidates = self.relative_score_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        else:
            fused_candidates = self.reciprocal_rank_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                rrf_k=rrf_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Cross-Encoder Re-ranking
        t0 = time.perf_counter()
        if apply_reranking and fused_candidates:
            # For re-ranking, we use contextualized_text if available, or raw text
            rerank_pool = fused_candidates[:max(dense_top_k, sparse_top_k)]
            # We can use text_key="contextualized_text" or "text"
            # Contextualized text allows the cross-encoder to leverage Anthropic contextual summaries!
            texts_to_score = [
                c.get("contextualized_text") or c.get("text", "")
                for c in rerank_pool
            ]
            ce_scores = self.reranker_service.compute_scores(query, texts_to_score)
            for c, s in zip(rerank_pool, ce_scores):
                c["cross_encoder_score"] = s
                c["final_score"] = s

            # Sort descending by cross-encoder score
            rerank_pool.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            final_results = rerank_pool[:top_k]
        else:
            final_results = fused_candidates[:top_k]
        timings["reranking_ms"] = (time.perf_counter() - t0) * 1000.0

        total_ms = sum(timings.values())
        timings["total_ms"] = total_ms

        return {
            "query": query,
            "collection_name": coll,
            "total_dense_candidates": len(dense_candidates),
            "total_sparse_candidates": len(sparse_candidates),
            "total_fused_candidates": len(fused_candidates),
            "results": final_results,
            "timing_ms": timings,
        }

_hybrid_engine_instance: Optional[HybridSearchEngine] = None

def get_hybrid_engine() -> HybridSearchEngine:
    global _hybrid_engine_instance
    if _hybrid_engine_instance is None:
        _hybrid_engine_instance = HybridSearchEngine()
    return _hybrid_engine_instance

