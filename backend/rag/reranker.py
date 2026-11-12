"""Cross-encoder reranking and final context selection."""

import os
from typing import List, Dict, Any, Optional
import logging
from threading import Lock

logger = logging.getLogger(__name__)


def select_passages(candidates, ranked, top_n):
    'Keep the strongest hybrid result alongside reranked evidence.'
    selected = ranked[:top_n]
    if top_n > 1 and candidates and selected:
        anchor_id = candidates[0]["chunk"].chunk_id
        if all(item["chunk"].chunk_id != anchor_id for item in selected):
            anchor = next((item for item in ranked if item["chunk"].chunk_id == anchor_id), None)
            if anchor is not None:
                selected = selected[:-1] + [anchor]
    return selected


class CrossEncoderReranker:
    'Reranks candidate passages retrieved by hybrid search using a deep Cross-Encoder.'

    def __init__(
        self,
        model_name: Optional[str] = None,
        lazy_load: bool = False,
    ):
        self.model_name = model_name or os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._load_attempted = False
        self._model = None
        self._predict_lock = Lock()
        self._prediction_failed = False
        if not lazy_load:
            self._load_model()

    def _load_model(self):
        if self._model is None and not self._load_attempted:
            self._load_attempted = True
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

    @property
    def is_available(self):
        return self._model is not None and not self._prediction_failed

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        'Rerank a list of candidate chunk dicts returned by the HybridRetriever.'
        if not candidates:
            return []

        self._load_model()

        pairs = []
        for item in candidates:
            chunk = item["chunk"]
            passage_text = f"{chunk.title} - {chunk.section}: {chunk.text}"
            pairs.append([query, passage_text])

        if self._model is not None:
            try:
                with self._predict_lock:
                    scores = self._model.predict(pairs, show_progress_bar=False)
                self._prediction_failed = False
            except Exception as e:
                self._prediction_failed = True
                logger.warning(f"[Reranker] Prediction failed: {e}. Keeping original order.")
                return candidates[:top_n] if top_n is not None else candidates
        else:
            return candidates[:top_n] if top_n is not None else candidates

        reranked = []
        for item, score in zip(candidates, scores):
            entry = dict(item)
            entry["rerank_score"] = float(score)
            reranked.append(entry)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_n is not None:
            return reranked[:top_n]
        return reranked
