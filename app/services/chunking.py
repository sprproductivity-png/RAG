import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.services.embeddings import get_embedding_service

ABBREVIATIONS = [
    "e.g.", "i.e.", "etc.", "dr.", "mr.", "mrs.", "ms.", "prof.", "vs.",
    "inc.", "ltd.", "co.", "dept.", "fig.", "vol.", "no.", "al.", "u.s.", "u.k."
]

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into coherent sentences, preserving abbreviations, decimal numbers,
    and numbered list items without premature fragmentation.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    masked = cleaned
    replacements: Dict[str, str] = {}

    # 1. Mask common abbreviations
    for i, abbr in enumerate(ABBREVIATIONS):
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        matches = list(pattern.finditer(masked))
        for m in reversed(matches):
            token = f"__ABBR_{i}_{len(replacements)}__"
            replacements[token] = m.group(0)
            masked = masked[:m.start()] + token + masked[m.end():]

    # 2. Mask decimal numbers (e.g. 3.14, 0.05)
    dec_matches = list(re.finditer(r'\b\d+\.\d+\b', masked))
    for i, m in enumerate(reversed(dec_matches)):
        token = f"__DEC_{i}__"
        replacements[token] = m.group(0)
        masked = masked[:m.start()] + token + masked[m.end():]

    # 3. Mask numbered list items at line starts (e.g. "1. ", "2. ")
    num_list_matches = list(re.finditer(r'(?:^|\n)\s*(\d+\.)\s+', masked))
    for i, m in enumerate(reversed(num_list_matches)):
        token = f"__NUMLIST_{i}__ "
        replacements[token] = m.group(1) + " "
        start_idx = m.start(1)
        end_idx = m.end()
        masked = masked[:start_idx] + token + masked[end_idx:]

    # 4. Standard sentence split
    raw_splits = re.split(r'(?<=[.?!])\s+(?=[A-Z0-9"\'`])|\n{2,}', masked)

    sentences = []
    for s in raw_splits:
        st = s.strip()
        if st:
            # Restore tokens
            for token, orig in replacements.items():
                st = st.replace(token, orig)
            sentences.append(st)

    return sentences

def _subdivide_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """
    If a single sentence exceeds max_chars, breaks it down cleanly
    at semicolon, colon, comma, or word boundaries.
    """
    if len(sentence) <= max_chars:
        return [sentence]

    sub_parts = []
    chunks = re.split(r'(?<=[;:])\s+', sentence)
    for part in chunks:
        if len(part) <= max_chars:
            sub_parts.append(part.strip())
        else:
            sub_commas = re.split(r'(?<=,)\s+', part)
            for c_part in sub_commas:
                if len(c_part) <= max_chars:
                    sub_parts.append(c_part.strip())
                else:
                    words = c_part.split()
                    cur: List[str] = []
                    cur_len = 0
                    for w in words:
                        if cur_len + len(w) + 1 > max_chars and cur:
                            sub_parts.append(" ".join(cur).strip())
                            cur = [w]
                            cur_len = len(w)
                        else:
                            cur.append(w)
                            cur_len += len(w) + 1
                    if cur:
                        sub_parts.append(" ".join(cur).strip())

    return [p for p in sub_parts if p]

class SemanticChunker:
    """
    Splits text into coherent semantic segments using sentence embedding cosine distance
    and adaptive thresholding, with guaranteed strict character ceilings.
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
        raw_sentences = split_into_sentences(text)
        if not raw_sentences:
            return []

        # Subdivide any giant individual sentences to obey max_characters ceiling
        sentences = []
        for s in raw_sentences:
            if len(s) > self.max_characters:
                sentences.extend(_subdivide_long_sentence(s, self.max_characters))
            else:
                sentences.append(s)

        if len(sentences) <= self.min_sentences:
            chunk_text = " ".join(sentences).strip()
            return [{
                "chunk_index": 0,
                "raw_text": chunk_text,
                "sentence_count": len(sentences),
            }]

        # 1. Compute embeddings for each sentence
        sentence_embeddings = self.embedding_service.embed_texts(sentences)

        # 2. Compute cosine distances between adjacent sentences
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

        # 4. Group sentences into chunks based on semantic threshold and character limit
        chunks = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            would_overflow = (current_len + sentence_len + 1) > self.max_characters
            is_last_sentence = (i == len(sentences) - 1)
            
            has_semantic_breakpoint = False
            if i < len(distances):
                if distances[i] >= threshold:
                    has_semantic_breakpoint = True

            has_min_sentences = len(current_sentences) >= self.min_sentences

            if would_overflow and current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append({
                    "chunk_index": chunk_idx,
                    "raw_text": chunk_text,
                    "sentence_count": len(current_sentences),
                })
                chunk_idx += 1
                current_sentences = [sentence]
                current_len = sentence_len
            else:
                current_sentences.append(sentence)
                current_len += sentence_len + 1

                if (has_semantic_breakpoint and has_min_sentences and not is_last_sentence):
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
            if chunks and (len(chunks[-1]["raw_text"]) + len(chunk_text) + 1 <= self.max_characters) and len(chunk_text) < 120:
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
