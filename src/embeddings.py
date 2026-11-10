"""
embeddings.py
Generates vector embeddings for document chunks and user queries
using the Sentence Transformers library (default: all-MiniLM-L6-v2).
Outputs normalized, standard Python float lists compatible with ChromaDB.
"""

import os
# Ensure HuggingFace uses local disk cache without making slow network timeout checks
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from typing import List, Optional
import numpy as np

from src.config import config


class EmbeddingGenerator:
    """
    Manages loading the embedding model and generating dense vector representations
    for document chunks and natural-language queries.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the SentenceTransformer model (defaults to config).
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.embedding_model_name
        print(f"[Embeddings] Initializing model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name, local_files_only=True)
        # Determine vector dimension dynamically
        dummy = self.model.encode("dim test")
        self._dimension = len(dummy)
        print(f"[Embeddings] Model '{self.model_name}' ready. Vector dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension (e.g., 384 for all-MiniLM-L6-v2)."""
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a normalized vector embedding for a single text string (e.g. user question).

        Args:
            text: Query or sentence to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Cannot embed an empty or whitespace-only text.")

        # normalize_embeddings=True ensures cosine similarity is proportional to dot product
        embedding = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate normalized vector embeddings for a list of document chunk texts.

        Args:
            texts: List of text chunks to embed.
            batch_size: Number of texts to process in parallel.

        Returns:
            A list of float lists, one vector per input text.
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 20,
        )
        return embeddings.tolist()
