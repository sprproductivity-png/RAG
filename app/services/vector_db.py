from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
import urllib3
import uuid

from app.core.config import settings
from app.core.logging import logger

def _ensure_valid_qdrant_id(pid: Any) -> Any:
    if isinstance(pid, int):
        return pid
    pid_str = str(pid)
    try:
        uuid.UUID(pid_str)
        return pid_str
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, pid_str))

class VectorDBService:
    def __init__(self):
        self.client: QdrantClient
        self.mode: str = "remote_docker"
        self.url: Optional[str] = settings.QDRANT_URL
        self._init_client()

    def _init_client(self):
        """
        Initializes the Qdrant client. Defaults to connecting to the Docker container
        at QDRANT_URL. Falls back to local embedded storage if container is unreachable
        and ENABLE_LOCAL_QDRANT_FALLBACK is True.
        """
        if settings.QDRANT_URL:
            try:
                logger.info(f"Attempting connection to Qdrant container at '{settings.QDRANT_URL}'...")
                client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=settings.QDRANT_TIMEOUT,
                    prefer_grpc=False,
                )
                # Test connectivity
                client.get_collections()
                self.client = client
                self.mode = "docker_container"
                logger.info(f"Successfully connected to Qdrant Docker container at {settings.QDRANT_URL}")
                return
            except Exception as e:
                logger.warning(
                    f"Could not connect to Qdrant Docker container at {settings.QDRANT_URL}: {e}"
                )
                if not settings.ENABLE_LOCAL_QDRANT_FALLBACK:
                    raise ConnectionError(
                        f"Failed to connect to Qdrant Docker at {settings.QDRANT_URL} "
                        "and local fallback is disabled."
                    )

        if settings.ENABLE_LOCAL_QDRANT_FALLBACK:
            storage_path = Path(settings.QDRANT_LOCAL_PATH)
            storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Activating Qdrant local embedded storage at '{storage_path}' as fallback.")
            self.client = QdrantClient(path=str(storage_path))
            self.mode = "local_embedded"
            self.url = str(storage_path)
            logger.info("Qdrant local storage initialized successfully.")
        else:
            raise RuntimeError("No valid Qdrant connection configuration found.")

    def ensure_collection(self, collection_name: str, vector_size: int, distance: qmodels.Distance = qmodels.Distance.COSINE) -> bool:
        """
        Ensures a collection exists with the specified vector dimension and distance metric.
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            if not exists:
                logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.info(f"Collection '{collection_name}' created.")
            return True
        except Exception as e:
            logger.error(f"Error ensuring collection '{collection_name}': {e}")
            raise

    def upsert_chunks(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> bool:
        """
        Upserts dense vectors and associated chunk payloads into Qdrant.
        """
        points = [
            qmodels.PointStruct(
                id=_ensure_valid_qdrant_id(pid),
                vector=vec,
                payload=payload,
            )
            for pid, vec, payload in zip(ids, vectors, payloads)
        ]
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        return True

    def search_dense(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 15,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes dense vector search on Qdrant and returns scored chunk candidates.
        """
        query_filter = None
        if filter_metadata:
            must_conditions = []
            for key, val in filter_metadata.items():
                must_conditions.append(
                    qmodels.FieldCondition(
                        key=f"metadata.{key}",
                        match=qmodels.MatchValue(value=val),
                    )
                )
            query_filter = qmodels.Filter(must=must_conditions)

        # In qdrant-client v1.10+, query_points or search is available
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        candidates = []
        for rank, hit in enumerate(results.points, start=1):
            payload = hit.payload or {}
            candidates.append({
                "chunk_id": str(hit.id),
                "doc_id": payload.get("doc_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "text": payload.get("raw_text", ""),
                "context_summary": payload.get("context_summary"),
                "contextualized_text": payload.get("contextualized_text"),
                "metadata": payload.get("metadata", {}),
                "dense_rank": rank,
                "dense_score": float(hit.score),
            })
        return candidates

    def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        try:
            info = self.client.get_collection(collection_name=collection_name)
            # Handle vectors config
            vector_size = 0
            distance_str = "COSINE"
            if hasattr(info.config.params, "vectors"):
                vec_params = info.config.params.vectors
                if isinstance(vec_params, qmodels.VectorParams):
                    vector_size = vec_params.size
                    distance_str = str(vec_params.distance)

            points_count = info.points_count if info.points_count is not None else 0
            vectors_count = info.vectors_count if hasattr(info, "vectors_count") and info.vectors_count is not None else points_count

            return {
                "collection_name": collection_name,
                "points_count": points_count,
                "vectors_count": vectors_count,
                "vector_size": vector_size,
                "distance": distance_str,
            }
        except Exception as e:
            logger.warning(f"Could not retrieve stats for collection '{collection_name}': {e}")
            return None

    def list_collections(self) -> List[str]:
        try:
            result = self.client.get_collections()
            return [c.name for c in result.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    def health_check(self) -> Tuple[bool, str, Optional[str]]:
        """
        Returns (is_connected, mode, url)
        """
        try:
            self.client.get_collections()
            return True, self.mode, self.url
        except Exception as e:
            return False, f"error: {str(e)}", self.url

_vector_db_instance: Optional[VectorDBService] = None

def get_vector_db_service() -> VectorDBService:
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDBService()
    return _vector_db_instance

