import os
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger

NOTEBOOKLM_SYSTEM_PROMPT = """You are an expert AI research assistant designed to answer user questions based strictly on the provided context documents.

Follow these strict rules:
1. Synthesize answers directly from the provided source passages.
2. Structure your answer clearly and legibly using Markdown: bold key terms, use bullet points or numbered lists, and short readable paragraphs.
3. Include inline citations immediately following every factual statement or finding using bracketed numbers corresponding to the source (e.g. [1], [2], or [1, 3]).
4. If the provided context does not contain the answer, say clearly: "I cannot answer this question based on the provided documents." Do not invent information or extrapolate beyond the text.
5. Maintain an objective, professional, and helpful tone."""

_local_llm = None

def get_local_llm():
    global _local_llm
    if _local_llm is None:
        try:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama  # type: ignore
            
            repo_id = settings.LOCAL_LLM_REPO
            filename = settings.LOCAL_LLM_FILE
            
            # Use up to 8 threads on high-core CPUs for much faster token generation
            num_threads = min(8, max(4, (os.cpu_count() or 4) // 2))
            
            logger.info(f"Downloading/Loading local quantized model: {repo_id}/{filename} (threads={num_threads})...")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=settings.HF_TOKEN)
            
            chat_format = getattr(settings, "LOCAL_LLM_CHAT_FORMAT", "chatml")
            logger.info(f"Initializing Llama engine with {num_threads} threads (format={chat_format})...")
            _local_llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=num_threads,
                n_batch=512,
                n_gpu_layers=99,
                chat_format=chat_format,
                verbose=False
            )
            logger.info("Local LLM successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            raise e
    return _local_llm

def build_context_block(context_chunks: List[Dict[str, Any]], max_chars: int = 4500) -> str:
    """
    Builds a numbered context block with source references [1], [2], etc.
    Budgets characters carefully from top-ranked chunks for fast, focused CPU inference.
    """
    if not context_chunks:
        return ""

    context_parts = []
    current_chars = 0

    for idx, c in enumerate(context_chunks, start=1):
        meta = c.get("metadata", {})
        title = meta.get("title") or meta.get("filename") or f"Document {c.get('doc_id', 'unknown')}"
        page_info = f", Page {meta.get('page_count')}" if meta.get('page_count') else ""
        chunk_text = c.get("contextualized_text") or c.get("text") or ""
        chunk_text = chunk_text.strip()

        header = f"[Source {idx}] (Document: '{title}'{page_info}, Chunk ID: {c.get('chunk_id', idx)})"
        entry = f"{header}\n{chunk_text}"

        entry_len = len(entry) + 2
        if current_chars + entry_len > max_chars:
            if context_parts:
                break
            entry = entry[:max_chars]

        context_parts.append(entry)
        current_chars += len(entry) + 2

    return "\n\n".join(context_parts)

class RAGGenerator:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.CONTEXTUAL_RETRIEVAL_PROVIDER
        if self.provider == "heuristic":
            self.provider = "local" # Default to local offline quantized LLM
        self.model = model

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        # Fast exit if no sources were retrieved
        if not context_chunks:
            return (
                "No relevant documents were found in the selected collection to answer your question. "
                "Please make sure your files have been uploaded and indexed."
            )

        context_str = build_context_block(context_chunks, max_chars=6500)
        user_prompt = (
            f"Here are the reference sources:\n"
            f"<context>\n"
            f"{context_str}\n"
            f"</context>\n\n"
            f"Question: {query}\n\n"
            f"Provide a structured answer formatted in Markdown (bold key terms, bullet points) with inline citations like [1] or [2] immediately following each claim based on the sources above."
        )

        if self.provider == "anthropic":
            return self._call_anthropic(user_prompt, temperature)
        elif self.provider == "openai":
            return self._call_openai(user_prompt, temperature)
        elif self.provider == "ollama":
            return self._call_ollama(user_prompt, temperature)
        elif self.provider == "local":
            return self._call_local(user_prompt, temperature)
        else:
            return "No valid LLM provider configured for answer generation."

    def _call_local(self, user_prompt: str, temperature: float) -> str:
        try:
            llm = get_local_llm()
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": NOTEBOOKLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=temperature,
                stop=["User Question:", "---", "User:"],
            )
            return output["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Local generation error: {e}")
            return f"Generation failed: {e}"

    def _call_anthropic(self, user_prompt: str, temperature: float) -> str:
        import anthropic
        api_key = settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        model = self.model or settings.ANTHROPIC_MODEL
        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(  # type: ignore
                model=model,
                max_tokens=1024,
                temperature=temperature,
                system=NOTEBOOKLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return getattr(response.content[0], "text", "").strip()
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return f"Generation failed: {e}"

    def _call_openai(self, user_prompt: str, temperature: float) -> str:
        from openai import OpenAI
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        base_url = settings.OPENAI_API_BASE
        model = self.model or settings.OPENAI_MODEL
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": NOTEBOOKLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"Generation failed: {e}"

    def _call_ollama(self, user_prompt: str, temperature: float) -> str:
        base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        model = self.model or settings.OLLAMA_MODEL
        try:
            with httpx.Client(timeout=120.0) as client:
                res = client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": NOTEBOOKLM_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    return data.get("message", {}).get("content", "").strip()
                return f"Ollama Error: {res.status_code} - {res.text}"
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Generation failed: {e}"

def get_generator(provider: Optional[str] = None, model: Optional[str] = None) -> RAGGenerator:
    return RAGGenerator(provider=provider, model=model)
