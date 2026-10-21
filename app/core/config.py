from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service Info
    APP_NAME: str = "Self-Hosted Hybrid RAG & Re-ranking Engine"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Qdrant Vector Database
    # Pointing to Docker container by default (http://localhost:6333)
    QDRANT_URL: str = Field(default="http://localhost:6333", description="URL to Qdrant Docker or remote instance")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Optional API key for Qdrant")
    QDRANT_TIMEOUT: float = 120.0
    QDRANT_LOCAL_PATH: str = "./data/qdrant_storage"
    ENABLE_LOCAL_QDRANT_FALLBACK: bool = Field(
        default=True,
        description="If true, gracefully falls back to local embedded Qdrant if the Docker container is not reachable"
    )

    # BM25 Storage
    BM25_STORAGE_DIR: str = "./data/bm25_indexes"

    # Raw File Cache & Ingestion Queue Storage
    FILE_CACHE_DIR: str = "./data/file_cache"

    # Redis Task Queue
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Optional Redis connection URL (e.g. redis://redis:6379/0 or redis://localhost:6379/0)"
    )
    ENABLE_REDIS_QUEUE: bool = True


    # Dense Embedding Model
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: Optional[str] = None  # None for auto (cuda/cpu)
    EMBEDDING_BATCH_SIZE: int = 32
    NORMALIZE_EMBEDDINGS: bool = True

    # Cross-Encoder Re-ranker Model
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_DEVICE: Optional[str] = None

    # Anthropic Contextual Retrieval Settings
    CONTEXTUAL_RETRIEVAL_PROVIDER: Literal["heuristic", "anthropic", "openai", "ollama"] = "heuristic"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"
    OPENAI_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "gemma3:1b"
    # Local LLM Config (llama.cpp)
    LOCAL_LLM_REPO: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    LOCAL_LLM_FILE: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    LOCAL_LLM_CHAT_FORMAT: str = "chatml"


    # Semantic Chunking Defaults
    CHUNK_SIMILARITY_PERCENTILE_THRESHOLD: float = 80.0
    CHUNK_MIN_SENTENCES: int = 1
    CHUNK_MAX_CHARACTERS: int = 1200
    CHUNK_OVERLAP_SENTENCES: int = 0

    # Hybrid Search Defaults
    DEFAULT_COLLECTION: str = "documents"
    DEFAULT_DENSE_TOP_K: int = 15
    DEFAULT_SPARSE_TOP_K: int = 15
    DEFAULT_RERANK_TOP_K: int = 5
    DEFAULT_RRF_K: int = 60
    DEFAULT_DENSE_WEIGHT: float = 0.5
    DEFAULT_SPARSE_WEIGHT: float = 0.5

    def get_bm25_path(self, collection_name: str) -> Path:
        dir_path = Path(self.BM25_STORAGE_DIR)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{collection_name}_bm25.json"

    def get_file_cache_dir(self) -> Path:
        dir_path = Path(self.FILE_CACHE_DIR)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

settings = Settings()

