from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging import logger
from app.api.v1.endpoints import router as api_v1_router
from app.services.embeddings import get_embedding_service
from app.services.reranker import get_reranker_service
from app.services.vector_db import get_vector_db_service

STATIC_DIR = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-warm models & Vector DB
    logger.info("Initializing Hybrid RAG Engine services...")
    emb_service = get_embedding_service()
    reranker = get_reranker_service()
    vector_db = get_vector_db_service()
    is_connected, mode, url = vector_db.health_check()
    logger.info(
        f"Hybrid RAG Engine started. Embedding: {emb_service.model_name} ({emb_service.dimension}d), "
        f"Cross-Encoder: {reranker.model_name}, Qdrant Mode: {mode} ({url})"
    )
    yield
    logger.info("Shutting down Hybrid RAG Engine...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready, Self-Hosted Hybrid RAG & Re-ranking Engine backed by Qdrant (Docker/local), "
        "featuring Semantic Chunking, Anthropic Contextual Retrieval, BM25/Dense hybrid fusion (RRF), "
        "and Cross-Encoder re-ranking."
    ),
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_v1_router, prefix="/api/v1")

# Mount static files directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
async def serve_ui():
    """Serves the interactive Frontend Web Dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )
