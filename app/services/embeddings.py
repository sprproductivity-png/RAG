from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from app.core.config import settings
from app.core.logging import logger

def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]  # First element contains hidden state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        if device is None:
            if settings.EMBEDDING_DEVICE:
                self.device = settings.EMBEDDING_DEVICE
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading dense embedding model '{self.model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.dimension = self.model.config.hidden_size
        logger.info(f"Dense embedding model loaded successfully. Vector dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded)
                embeddings = _mean_pooling(model_output, encoded["attention_mask"])
                if settings.NORMALIZE_EMBEDDINGS:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_query(self, query: str) -> List[float]:
        encoded = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded)
            embeddings = _mean_pooling(model_output, encoded["attention_mask"])
            if settings.NORMALIZE_EMBEDDINGS:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().numpy().tolist()

    def get_dimension(self) -> int:
        return self.dimension

_embedding_instance: Optional[EmbeddingService] = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingService()
    return _embedding_instance

