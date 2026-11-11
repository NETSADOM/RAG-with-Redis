"""
backend/rag/reranker.py
Cross-Encoder Reranker for deep semantic interaction scoring between
the customer question and candidate policy passages.
"""

import os
# Ensure instant offline local model loading from cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate passages retrieved by hybrid search using a deep Cross-Encoder.
    Unlike bi-encoders which encode query and passage independently, Cross-Encoders
    perform full cross-attention over all token pairs simultaneously, producing
    significantly higher ranking accuracy.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        lazy_load: bool = False,
    ):
        self.model_name = model_name
        self._model = None
        if not lazy_load:
            self._load_model()

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"[Reranker] Loading CrossEncoder model '{self.model_name}'...")
                # Attempt to load local cached weights immediately without network retry loop
                try:
                    self._model = CrossEncoder(self.model_name, local_files_only=True)
                except Exception:
                    self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"[Reranker] Could not load CrossEncoder '{self.model_name}' ({e}). Activating fallback scoring.")
                self._model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate chunk dicts returned by the HybridRetriever.

        Args:
            query: The customer's support question.
            candidates: List of candidate dictionaries with 'chunk' key.
            top_n: Number of top reranked passages to return (default: all).

        Returns:
            Re-ordered list of candidate dicts with 'rerank_score' attached.
        """
        if not candidates:
            return []

        self._load_model()

        # Build (query, passage) pairs
        pairs = []
        for item in candidates:
            chunk = item["chunk"]
            passage_text = f"{chunk.title} - {chunk.section}: {chunk.text}"
            pairs.append([query, passage_text])

        # Compute Cross-Encoder scores
        if self._model is not None:
            try:
                scores = self._model.predict(pairs, show_progress_bar=False)
            except Exception as e:
                logger.warning(f"[Reranker] Prediction failed: {e}. Keeping original order.")
                scores = [float(item.get("rrf_score", 0.0)) for item in candidates]
        else:
            # Fallback heuristic: length and lexical overlap score
            scores = [float(item.get("rrf_score", 0.0)) for item in candidates]

        # Attach scores to candidates
        reranked = []
        for item, score in zip(candidates, scores):
            entry = dict(item)
            entry["rerank_score"] = float(score)
            reranked.append(entry)

        # Sort descending by cross-encoder relevance score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_n is not None:
            return reranked[:top_n]
        return reranked
