import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Plus
from app.core.config import settings
from app.core.logging import logger

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text into lowercase words with stopword removal."""
    if not text:
        return []
    words = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
    filtered = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    # If all tokens were stopwords (e.g. short query), fallback to all length > 1 words
    if not filtered and words:
        return [w for w in words if len(w) > 1]
    return filtered

class BM25CollectionIndex:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.doc_records: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Plus] = None
        self.index_file = settings.get_bm25_path(collection_name)
        self.load()

    def load(self):
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.doc_records = data.get("records", [])
                self.tokenized_corpus = [tokenize_text(r.get("index_text", "")) for r in self.doc_records]
                if self.tokenized_corpus:
                    self.bm25 = BM25Plus(self.tokenized_corpus)
                logger.info(f"Loaded BM25 index for '{self.collection_name}' with {len(self.doc_records)} documents.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index for '{self.collection_name}': {e}")
                self.doc_records = []
                self.tokenized_corpus = []
                self.bm25 = None

    def save(self):
        try:
            # Atomic write to prevent file corruption during sudden restarts
            tmp_file = self.index_file.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump({"records": self.doc_records}, f, ensure_ascii=False)
            tmp_file.replace(self.index_file)
            logger.info(f"Persisted BM25 index for '{self.collection_name}' to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to persist BM25 index for '{self.collection_name}': {e}")

    def add_chunks(self, records: List[Dict[str, Any]]):
        """Alias for add_records for backwards and interface compatibility."""
        self.add_records(records)

    def add_records(self, records: List[Dict[str, Any]]):
        """
        Adds new chunk records with O(1) lookup.
        Each record should contain: chunk_id, doc_id, chunk_index, raw_text,
        context_summary, contextualized_text, metadata.
        """
        record_map = {r["chunk_id"]: r for r in self.doc_records}

        for r in records:
            chunk_id = r["chunk_id"]
            index_text = r.get("contextualized_text") or r.get("raw_text", "")
            record = {
                "chunk_id": chunk_id,
                "doc_id": r.get("doc_id", ""),
                "chunk_index": r.get("chunk_index", 0),
                "raw_text": r.get("raw_text", ""),
                "context_summary": r.get("context_summary"),
                "contextualized_text": r.get("contextualized_text"),
                "metadata": r.get("metadata", {}),
                "index_text": index_text
            }
            record_map[chunk_id] = record

        self.doc_records = list(record_map.values())

        # Rebuild BM25 with BM25Plus
        self.tokenized_corpus = [tokenize_text(r["index_text"]) for r in self.doc_records]
        if self.tokenized_corpus:
            self.bm25 = BM25Plus(self.tokenized_corpus)
        self.save()

    def search(
        self,
        query: str,
        limit: int = 15,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes BM25 search on the collection and returns scored candidate chunks.
        """
        if not self.bm25 or not self.doc_records:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        query_token_set = set(tokens)
        scores = self.bm25.get_scores(tokens)

        # Pair scores with documents that contain at least one query term
        scored_pairs = []
        for i, score in enumerate(scores):
            # Ensure document actually contains query terms and positive score
            if score <= 0.0:
                continue
            doc_tokens = set(self.tokenized_corpus[i])
            if not query_token_set.intersection(doc_tokens):
                continue

            rec = self.doc_records[i]
            # Check metadata filter
            if filter_metadata:
                match = True
                rec_meta = rec.get("metadata", {})
                for k, v in filter_metadata.items():
                    if rec_meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            scored_pairs.append((float(score), rec))

        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, rec) in enumerate(scored_pairs[:limit], start=1):
            results.append({
                "chunk_id": rec["chunk_id"],
                "doc_id": rec["doc_id"],
                "chunk_index": rec["chunk_index"],
                "text": rec["raw_text"],
                "context_summary": rec.get("context_summary"),
                "contextualized_text": rec.get("contextualized_text"),
                "metadata": rec.get("metadata", {}),
                "bm25_rank": rank,
                "bm25_score": score,
            })
        return results

    def get_documents(self) -> List[Dict[str, Any]]:
        docs_map: Dict[str, Dict[str, Any]] = {}
        for rec in self.doc_records:
            doc_id = rec.get("doc_id", "unknown")
            meta = rec.get("metadata", {})
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title") or doc_id,
                    "filename": meta.get("filename"),
                    "file_type": meta.get("file_type"),
                    "ingested_at": meta.get("ingested_at"),
                    "chunks_count": 0,
                    "sample_preview": rec.get("raw_text", "")[:200].replace("\n", " ").strip(),
                    "metadata": meta,
                }
            docs_map[doc_id]["chunks_count"] += 1
        return list(docs_map.values())

    def delete_document(self, doc_id: str) -> int:
        initial_count = len(self.doc_records)
        self.doc_records = [r for r in self.doc_records if r.get("doc_id") != doc_id]
        deleted_count = initial_count - len(self.doc_records)
        if deleted_count > 0:
            self.tokenized_corpus = [tokenize_text(r["index_text"]) for r in self.doc_records]
            if self.tokenized_corpus:
                self.bm25 = BM25Plus(self.tokenized_corpus)
            else:
                self.bm25 = None
            self.save()
            logger.info(f"Deleted {deleted_count} chunks for doc_id '{doc_id}' from BM25 '{self.collection_name}'")
        return deleted_count

    def clear(self):
        self.doc_records = []
        self.tokenized_corpus = []
        self.bm25 = None
        if self.index_file.exists():
            try:
                self.index_file.unlink()
            except Exception:
                pass

class BM25Service:
    def __init__(self):
        self.indexes: Dict[str, BM25CollectionIndex] = {}

    def get_index(self, collection_name: str) -> BM25CollectionIndex:
        if collection_name not in self.indexes:
            self.indexes[collection_name] = BM25CollectionIndex(collection_name)
        return self.indexes[collection_name]

    def add_chunks(self, collection_name: str, records: List[Dict[str, Any]]):
        idx = self.get_index(collection_name)
        idx.add_records(records)

    def search_sparse(
        self,
        collection_name: str,
        query: str,
        limit: int = 15,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        idx = self.get_index(collection_name)
        return idx.search(query, limit=limit, filter_metadata=filter_metadata)

    def delete_index(self, collection_name: str):
        if collection_name in self.indexes:
            self.indexes[collection_name].clear()
            del self.indexes[collection_name]
        else:
            BM25CollectionIndex(collection_name).clear()

    def get_docs_count(self, collection_name: str) -> int:
        idx = self.get_index(collection_name)
        return len(idx.doc_records)

    def get_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        idx = self.get_index(collection_name)
        return idx.get_documents()

    def delete_document(self, collection_name: str, doc_id: str) -> int:
        idx = self.get_index(collection_name)
        return idx.delete_document(doc_id)

_bm25_service_instance: Optional[BM25Service] = None

def get_bm25_service() -> BM25Service:
    global _bm25_service_instance
    if _bm25_service_instance is None:
        _bm25_service_instance = BM25Service()
    return _bm25_service_instance
