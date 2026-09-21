"""
Interactive Demonstration Client for Self-Hosted Hybrid RAG & Re-ranking Engine
Demonstrates:
  1. Document ingestion with Semantic Chunking
  2. Anthropic Contextual Retrieval situational summaries
  3. Dense Qdrant Vector Search vs BM25 Sparse Search
  4. Reciprocal Rank Fusion (RRF)
  5. Cross-Encoder Re-ranking
"""

import time
import json
from app.services.chunking import chunk_document
from app.services.contextual import enrich_chunks_with_context
from app.services.embeddings import get_embedding_service
from app.services.vector_db import get_vector_db_service
from app.services.bm25 import get_bm25_service
from app.services.hybrid import get_hybrid_engine
from app.services.reranker import get_reranker_service

SAMPLE_DOCUMENT = """
Acme Robotics Inc. - Annual Engineering & Technology Review 2024.

Section 1: Autonomous Navigation and SLAM Algorithms
Our autonomous mobile robots (AMRs) utilize graph-based Simultaneous Localization and Mapping (SLAM).
By fusing high-frequency LiDAR scans with wheel odometry and IMU telemetry via an Extended Kalman Filter,
the robots maintain sub-centimeter positioning accuracy even in dynamic warehouse environments with moving personnel.

Section 2: Battery Management and Fast Charging Systems
The new Falcon-5 powertrain integrates lithium iron phosphate (LiFePO4) battery packs delivering 18 hours of continuous operation.
Our proprietary thermal management system monitors individual cell temperatures across 48 channels,
mitigating thermal runaway risks and enabling 80% recharge within 35 minutes via inductive floor pads.

Section 3: Cloud Telemetry and Fleet Coordination
Fleet orchestration is powered by a decentralized consensus algorithm that prevents path deadlocks at narrow aisle intersections.
All robot telemetry, including motor currents and wheel slip metrics, is streamed via MQTT over private 5G networks to our centralized analytics dashboard.
"""

def main():
    print("=" * 80)
    print("  SELF-HOSTED HYBRID RAG & RE-RANKING ENGINE DEMONSTRATION")
    print("=" * 80)

    # 1. Semantic Chunking
    print("\n[Step 1] Executing Semantic Chunking on Document...")
    raw_chunks = chunk_document(SAMPLE_DOCUMENT, strategy="semantic")
    print(f"Generated {len(raw_chunks)} semantic chunks based on topic boundary detection.")
    for idx, c in enumerate(raw_chunks):
        preview = c['raw_text'].replace('\n', ' ')[:90]
        print(f"  - Chunk #{idx} ({c['sentence_count']} sentences): \"{preview}...\"")

    # 2. Anthropic Contextual Retrieval
    print("\n[Step 2] Applying Anthropic Contextual Retrieval Summaries...")
    enriched_chunks = enrich_chunks_with_context(
        document_text=SAMPLE_DOCUMENT,
        chunks=raw_chunks,
        document_title="Acme Robotics Annual Review 2024",
        enabled=True
    )
    for idx, c in enumerate(enriched_chunks):
        print(f"\n  --- Chunk #{idx} Enriched ---")
        print(f"  Summary:        {c.get('context_summary')}")
        print(f"  Raw Text:       {c['raw_text'].replace(chr(10), ' ')[:100]}...")

    # 3. Index into Vector DB & BM25
    collection_name = "demo_robotics"
    print(f"\n[Step 3] Indexing Chunks into Vector DB & BM25 Store ('{collection_name}')...")
    
    emb_service = get_embedding_service()
    vector_db = get_vector_db_service()
    bm25 = get_bm25_service()

    vector_db.ensure_collection(collection_name, emb_service.get_dimension())

    import uuid
    chunk_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"demo_chunk_{i}")) for i in range(len(enriched_chunks))]
    contextualized_texts = [c["contextualized_text"] for c in enriched_chunks]
    payloads = [
        {
            "chunk_id": cid,
            "doc_id": "doc_acme_001",
            "chunk_index": i,
            "raw_text": c["raw_text"],
            "context_summary": c.get("context_summary"),
            "contextualized_text": c["contextualized_text"],
            "metadata": {"source": "acme_review_2024.pdf"}
        }
        for i, (cid, c) in enumerate(zip(chunk_ids, enriched_chunks))
    ]

    vectors = emb_service.embed_texts(contextualized_texts)
    vector_db.upsert_chunks(collection_name, chunk_ids, [v.tolist() for v in vectors], payloads)
    bm25.add_chunks(collection_name, payloads)
    print("Indexing complete!")

    # 4. Comparison Queries
    queries = [
        "How do the robots prevent collisions and deadlocks in narrow aisles?",
        "What type of battery chemistry is used and how fast does it charge?",
        "Explain the SLAM positioning accuracy and sensors."
    ]

    hybrid_engine = get_hybrid_engine()

    for q in queries:
        print("\n" + "=" * 80)
        print(f"Query: \"{q}\"")
        print("=" * 80)

        # Search with Hybrid + Cross-Encoder
        search_res = hybrid_engine.search(
            query=q,
            collection_name=collection_name,
            top_k=2,
            dense_top_k=5,
            sparse_top_k=5,
            apply_reranking=True
        )

        print(f"Timing: Total {search_res['timing_ms']['total_ms']:.2f}ms "
              f"(Dense: {search_res['timing_ms']['dense_retrieval_ms']:.2f}ms, "
              f"BM25: {search_res['timing_ms']['sparse_retrieval_ms']:.2f}ms, "
              f"RRF: {search_res['timing_ms']['fusion_ms']:.2f}ms, "
              f"Reranker: {search_res['timing_ms']['reranking_ms']:.2f}ms)")

        print("\nTop Ranked Results:")
        for rank, r in enumerate(search_res["results"], start=1):
            ce_s = f"{r['cross_encoder_score']:.4f}" if r.get('cross_encoder_score') is not None else "N/A"
            d_s = f"{r['dense_score']:.4f}" if r.get('dense_score') is not None else "N/A"
            b_s = f"{r['bm25_score']:.4f}" if r.get('bm25_score') is not None else "N/A"
            f_s = f"{r['fusion_score']:.6f}" if r.get('fusion_score') is not None else "N/A"

            print(f"\n  [Rank {rank}] (Cross-Encoder Score: {ce_s})")
            print(f"   Dense Rank: {r.get('dense_rank')} (Score: {d_s}) | "
                  f"BM25 Rank: {r.get('bm25_rank')} (Score: {b_s}) | "
                  f"RRF Rank: {r.get('fusion_rank')} (RRF Score: {f_s})")
            print(f"   Context: {r.get('context_summary')}")
            print(f"   Passage: {r['text'][:140].replace(chr(10), ' ')}...")

    # Cleanup demo collection
    vector_db.delete_collection(collection_name)
    bm25.delete_index(collection_name)
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()

