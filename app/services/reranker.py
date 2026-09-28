from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.core.config import settings
from app.core.logging import logger

class CrossEncoderReranker:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL
        if device is None:
            if settings.CROSS_ENCODER_DEVICE:
                self.device = settings.CROSS_ENCODER_DEVICE
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Cross-Encoder model '{self.model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=settings.HF_TOKEN)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, token=settings.HF_TOKEN)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Cross-Encoder model loaded successfully.")

    def compute_scores(self, query: str, texts: List[str], batch_size: int = 32) -> List[float]:
        if not texts:
            return []

        all_scores = []
        pairs = [[query, t] for t in texts]

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            encoded = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits
                if logits.shape[1] == 1:
                    batch_scores = logits.squeeze(-1).cpu().tolist()
                else:
                    # In some multi-class sequence classifiers, score is logit of positive class
                    batch_scores = logits[:, 1].cpu().tolist()

            if isinstance(batch_scores, float):
                all_scores.append(batch_scores)
            else:
                all_scores.extend(batch_scores)

        return [float(s) for s in all_scores]

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        text_key: str = "text",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidate dictionaries based on joint cross-encoder self-attention.
        """
        if not candidates:
            return []

        texts = [c.get(text_key, "") for c in candidates]
        scores = self.compute_scores(query, texts)

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["cross_encoder_score"] = float(score)
            item["final_score"] = float(score)
            reranked.append(item)

        # Sort descending by cross_encoder_score
        reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        if top_k is not None and top_k > 0:
            reranked = reranked[:top_k]

        return reranked

_reranker_instance: Optional[CrossEncoderReranker] = None

def get_reranker_service() -> CrossEncoderReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance

