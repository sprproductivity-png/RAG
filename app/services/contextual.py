import os
import re
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import httpx

from app.core.config import settings
from app.core.logging import logger
from app.services.bm25 import STOPWORDS

ANTHROPIC_CONTEXTUAL_PROMPT = """<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context (50-100 words) to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else."""

def _extract_top_keywords(text: str, top_n: int = 4) -> str:
    """Extracts frequent informative keywords from text without requiring external C libraries."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    counts: Dict[str, int] = {}
    for w in words:
        if w not in STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top = [w for w, _ in sorted_words[:top_n]]
    return ", ".join(top) if top else "general topic"

class BaseContextualizer(ABC):
    @abstractmethod
    def generate_context(self, document_text: str, chunk_text: str, document_title: Optional[str] = None) -> str:
        """Generates a brief situational context string for the given chunk."""
        pass

    def generate_contexts_for_chunks(
        self,
        document_text: str,
        chunks: List[Dict[str, Any]],
        document_title: Optional[str] = None
    ) -> List[str]:
        return [
            self.generate_context(document_text, chunk["raw_text"], document_title)
            for chunk in chunks
        ]

class AnthropicContextualizer(BaseContextualizer):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or settings.ANTHROPIC_MODEL
        if not self.api_key:
            raise ValueError("Anthropic API key is required for AnthropicContextualizer.")
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_context(self, document_text: str, chunk_text: str, document_title: Optional[str] = None) -> str:
        prompt = ANTHROPIC_CONTEXTUAL_PROMPT.format(
            document_text=document_text[:40000],  # Respect context limits safely
            chunk_text=chunk_text
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]  # type: ignore
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error calling Anthropic API for contextual retrieval: {e}")
            return f"Context from document: {document_title or 'Document'}"

class OpenAICompatibleContextualizer(BaseContextualizer):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or "none"
        self.base_url = base_url or settings.OPENAI_API_BASE
        self.model = model or settings.OPENAI_MODEL
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_context(self, document_text: str, chunk_text: str, document_title: Optional[str] = None) -> str:
        prompt = ANTHROPIC_CONTEXTUAL_PROMPT.format(
            document_text=document_text[:30000],
            chunk_text=chunk_text
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=150,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI-compatible API for contextual retrieval: {e}")
            return f"Context from document: {document_title or 'Document'}"

class OllamaContextualizer(BaseContextualizer):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL

    def generate_context(self, document_text: str, chunk_text: str, document_title: Optional[str] = None) -> str:
        prompt = ANTHROPIC_CONTEXTUAL_PROMPT.format(
            document_text=document_text[:15000],
            chunk_text=chunk_text
        )
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 100}
                    }
                )
                if res.status_code == 200:
                    return res.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Error calling Ollama for contextual retrieval: {e}")
        return f"Context from document: {document_title or 'Document'}"

class HeuristicContextualizer(BaseContextualizer):
    """
    Zero-dependency offline contextualizer.
    Derives document topic/title, extracts core topical keywords, and formulates
    an Anthropic-style situational anchor for each chunk.
    Works 100% locally with zero external dependencies.
    """
    def __init__(self):
        pass

    def generate_context(self, document_text: str, chunk_text: str, document_title: Optional[str] = None) -> str:
        # 1. Derive title / heading if not provided
        if not document_title:
            first_line = document_text.strip().split("\n")[0]
            if len(first_line) < 120 and not first_line.endswith("."):
                document_title = first_line.strip("# ")
            else:
                document_title = "Document"

        # 2. Extract top salient global keywords from document text
        key_phrase = _extract_top_keywords(document_text, top_n=4)

        # 3. Formulate Anthropic-style situational context sentence
        summary = (
            f"[Context: This passage belongs to '{document_title}', "
            f"which covers {key_phrase}. This section specifically focuses on: "
            f"{chunk_text[:90].replace(chr(10), ' ').strip()}...]"
        )
        return summary

def get_contextualizer(provider: Optional[str] = None) -> BaseContextualizer:
    chosen_provider = provider or settings.CONTEXTUAL_RETRIEVAL_PROVIDER

    if chosen_provider == "anthropic":
        if settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY"):
            return AnthropicContextualizer()
        logger.warning("Anthropic API key not detected. Falling back to HeuristicContextualizer.")
        return HeuristicContextualizer()

    elif chosen_provider == "openai":
        if settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"):
            return OpenAICompatibleContextualizer()
        logger.warning("OpenAI API key not detected. Falling back to HeuristicContextualizer.")
        return HeuristicContextualizer()

    elif chosen_provider == "ollama":
        return OllamaContextualizer()

    # Default heuristic
    return HeuristicContextualizer()

def enrich_chunks_with_context(
    document_text: str,
    chunks: List[Dict[str, Any]],
    document_title: Optional[str] = None,
    provider: Optional[str] = None,
    enabled: bool = True
) -> List[Dict[str, Any]]:
    """
    Applies Anthropic Contextual Retrieval:
    Generates situational context summary for each chunk and prepends it to chunk text.
    """
    if not enabled:
        for c in chunks:
            c["context_summary"] = None
            c["contextualized_text"] = c["raw_text"]
        return chunks

    contextualizer = get_contextualizer(provider)
    
    # Fast path for HeuristicContextualizer: extract document keywords ONCE instead of per-chunk
    if isinstance(contextualizer, HeuristicContextualizer):
        if not document_title:
            first_line = document_text.strip().split("\n")[0]
            if len(first_line) < 120 and not first_line.endswith("."):
                document_title = first_line.strip("# ")
            else:
                document_title = "Document"
        key_phrase = _extract_top_keywords(document_text[:50000], top_n=4)

        enriched = []
        for c in chunks:
            raw_text = c["raw_text"]
            summary = (
                f"[Context: This passage belongs to '{document_title}', "
                f"which covers {key_phrase}. This section specifically focuses on: "
                f"{raw_text[:90].replace(chr(10), ' ').strip()}...]"
            )
            contextualized = f"{summary}\n\n{raw_text}"
            item = dict(c)
            item["context_summary"] = summary
            item["contextualized_text"] = contextualized
            enriched.append(item)
        return enriched

    enriched = []
    for c in chunks:
        raw_text = c["raw_text"]
        summary = contextualizer.generate_context(document_text, raw_text, document_title)
        contextualized = f"{summary}\n\n{raw_text}"
        item = dict(c)
        item["context_summary"] = summary
        item["contextualized_text"] = contextualized
        enriched.append(item)

    return enriched

