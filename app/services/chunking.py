import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.services.embeddings import get_embedding_service

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences based on punctuation and whitespace boundaries,
    preserving abbreviations reasonably.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # Split on periods, question marks, exclamation marks followed by whitespace or quotes
    raw_sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z0-9"\'`])|\n{2,}', cleaned)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences

class SemanticChunker:
    """
    Splits text into coherent semantic segments using sentence embedding cosine distance
    and adaptive thresholding.
    """
    def __init__(
        self,
        percentile_threshold: float = settings.CHUNK_SIMILARITY_PERCENTILE_THRESHOLD,
        min_sentences: int = settings.CHUNK_MIN_SENTENCES,
        max_characters: int = settings.CHUNK_MAX_CHARACTERS,
    ):
        self.percentile_threshold = percentile_threshold
        self.min_sentences = min_sentences
        self.max_characters = max_characters
        self.embedding_service = get_embedding_service()

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        sentences = split_into_sentences(text)
        if not sentences:
            return []

        if len(sentences) <= self.min_sentences:
            return [{
                "chunk_index": 0,
                "raw_text": text.strip(),
                "sentence_count": len(sentences),
            }]

        # 1. Compute embeddings for each sentence
        sentence_embeddings = self.embedding_service.embed_texts(sentences)

        # 2. Compute cosine distances between adjacent sentences
        # Normalized vectors: cosine_sim = dot(u, v), distance = 1 - sim
        distances = []
        for i in range(len(sentence_embeddings) - 1):
            u = sentence_embeddings[i]
            v = sentence_embeddings[i + 1]
            dot_val = float(np.dot(u, v))
            sim = max(-1.0, min(1.0, dot_val))
            dist = 1.0 - sim
            distances.append(dist)

        # 3. Calculate breakpoint threshold
        if distances:
            threshold = float(np.percentile(distances, self.percentile_threshold))
        else:
            threshold = 0.5

        # 4. Group sentences into chunks based on threshold and character limits
        chunks = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0

        for i, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            current_len += len(sentence) + 1

            is_last_sentence = (i == len(sentences) - 1)
            
            # Check if this sentence boundary is a breakpoint
            has_semantic_breakpoint = False
            if i < len(distances):
                if distances[i] >= threshold:
                    has_semantic_breakpoint = True

            # Check if chunk exceeds max character constraint
            exceeds_max_chars = current_len >= self.max_characters
            has_min_sentences = len(current_sentences) >= self.min_sentences

            should_split = (
                is_last_sentence or
                exceeds_max_chars or
                (has_semantic_breakpoint and has_min_sentences)
            )

            if should_split and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append({
                    "chunk_index": chunk_idx,
                    "raw_text": chunk_text,
                    "sentence_count": len(current_sentences),
                })
                chunk_idx += 1
                current_sentences = []
                current_len = 0

        # Handle any remaining dangling sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunks and len(chunk_text) < 100:
                # Merge small tail into last chunk
                chunks[-1]["raw_text"] += " " + chunk_text
                chunks[-1]["sentence_count"] += len(current_sentences)
            else:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "raw_text": chunk_text,
                    "sentence_count": len(current_sentences),
                })

        return chunks

class SlidingWindowChunker:
    """
    Fixed-size or character-based sliding window chunker with configurable overlap.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        cleaned = text.strip()
        if not cleaned:
            return []

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))
            # Try to break on whitespace if not end of string
            if end < len(cleaned):
                last_space = cleaned.rfind(" ", start, end)
                if last_space != -1 and last_space > start + (self.chunk_size // 2):
                    end = last_space

            chunk_text = cleaned[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "raw_text": chunk_text,
                    "sentence_count": len(split_into_sentences(chunk_text)),
                })
                chunk_idx += 1

            if end >= len(cleaned):
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

def chunk_document(
    text: str,
    strategy: str = "semantic",
    max_characters: int = settings.CHUNK_MAX_CHARACTERS,
) -> List[Dict[str, Any]]:
    if strategy == "semantic":
        chunker = SemanticChunker(max_characters=max_characters)
    else:
        chunker = SlidingWindowChunker(chunk_size=max_characters, chunk_overlap=150)
    return chunker.chunk(text)

