import os
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import logger

RAG_PROMPT = """You are a helpful AI assistant.
Answer the user's question based strictly on the provided context.
If the context does not contain the answer, say "I cannot answer this based on the provided documents."

<context>
{context}
</context>

Question: {query}
Answer:"""

_local_llm = None

def get_local_llm():
    global _local_llm
    if _local_llm is None:
        try:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            
            repo_id = settings.LOCAL_LLM_REPO
            filename = settings.LOCAL_LLM_FILE
            
            logger.info(f"Downloading/Loading local quantized model: {repo_id}/{filename}...")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            
            logger.info("Initializing Llama engine...")
            _local_llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=4,
                n_batch=512,
                verbose=False
            )
            logger.info("Local LLM successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            raise e
    return _local_llm

class RAGGenerator:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.CONTEXTUAL_RETRIEVAL_PROVIDER
        if self.provider == "heuristic":
            self.provider = "local" # Default to local offline quantized LLM
        self.model = model

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        # Build context string
        context_str = "\n\n".join(
            f"--- Document: {c.get('metadata', {}).get('title', 'Unknown')} ---\n{c.get('contextualized_text', c.get('text', ''))}"
            for c in context_chunks
        )
        prompt = RAG_PROMPT.format(context=context_str, query=query)

        if self.provider == "anthropic":
            return self._call_anthropic(prompt, temperature)
        elif self.provider == "openai":
            return self._call_openai(prompt, temperature)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, temperature)
        elif self.provider == "local":
            return self._call_local(prompt, temperature)
        else:
            return "No valid LLM provider configured for answer generation."

    def _call_local(self, prompt: str, temperature: float) -> str:
        try:
            llm = get_local_llm()
            output = llm(
                prompt,
                max_tokens=1024,
                temperature=temperature,
                stop=["Question:", "---", "User:"],
                echo=False
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Local generation error: {e}")
            return f"Generation failed: {e}"

    def _call_anthropic(self, prompt: str, temperature: float) -> str:
        import anthropic
        api_key = settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        model = self.model or settings.ANTHROPIC_MODEL
        client = anthropic.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return f"Generation failed: {e}"

    def _call_openai(self, prompt: str, temperature: float) -> str:
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
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"Generation failed: {e}"

    def _call_ollama(self, prompt: str, temperature: float) -> str:
        base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        model = self.model or settings.OLLAMA_MODEL
        try:
            with httpx.Client(timeout=120.0) as client:
                res = client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    }
                )
                if res.status_code == 200:
                    return res.json().get("response", "").strip()
                return f"Ollama Error: {res.status_code} - {res.text}"
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Generation failed: {e}"

def get_generator(provider: Optional[str] = None, model: Optional[str] = None) -> RAGGenerator:
    return RAGGenerator(provider=provider, model=model)
