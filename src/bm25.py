"""
bm25.py
Pure-Python implementation of BM25 (Best Matching 25) sparse keyword indexing.
Provides exact keyword, acronym, code, and proper noun retrieval to complement
dense vector search in a hybrid retrieval architecture.
"""

import math
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any


class BM25Index:
    """
    Lightweight, persistent Okapi BM25 sparse keyword search index.
    Uses inverse document frequency (IDF) and term frequency (TF) with document
    length normalization (k1=1.5, b=0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 parameters.

        Args:
            k1: Term frequency saturation parameter (standard: 1.2 to 2.0).
            b: Document length normalization parameter (standard: 0.75).
        """
        self.k1 = k1
        self.b = b

        # Corpus storage: chunk_id -> raw text and metadata
        self.corpus: Dict[str, Dict[str, Any]] = {}
        # Document lengths (in tokens): chunk_id -> int
        self.doc_lengths: Dict[str, int] = {}
        # Average document length across corpus
        self.avg_doc_len: float = 0.0
        # Inverted index: term -> {chunk_id: term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        # Document frequencies: term -> number of documents containing term
        self.doc_frequencies: Dict[str, int] = {}
        # Precomputed IDF: term -> idf score
        self.idf_cache: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into lowercase alphanumeric words.
        Preserves acronyms and numbers (e.g. 'all-minilm-l6-v2' -> ['all', 'minilm', 'l6', 'v2']).
        """
        return re.findall(r"\b\w+\b", text.lower())

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a batch of documents.

        Args:
            documents: List of dicts, each with keys: 'chunk_id', 'text', 'metadata'.
        """
        for doc in documents:
            chunk_id = doc["chunk_id"]
            text = doc["text"]
            metadata = doc.get("metadata", {})

            # Remove previous version if already present
            if chunk_id in self.corpus:
                self.remove_document(chunk_id)

            tokens = self.tokenize(text)
            doc_len = len(tokens)

            self.corpus[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
            }
            self.doc_lengths[chunk_id] = doc_len

            # Calculate term frequencies for this document
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            for token, count in tf.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][chunk_id] = count
                self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1

        self._recalculate_corpus_stats()

    def remove_document(self, chunk_id: str) -> None:
        """Remove a document from the BM25 index."""
        if chunk_id not in self.corpus:
            return

        del self.corpus[chunk_id]
        if chunk_id in self.doc_lengths:
            del self.doc_lengths[chunk_id]

        # Update inverted index
        terms_to_clean = []
        for term, postings in self.inverted_index.items():
            if chunk_id in postings:
                del postings[chunk_id]
                self.doc_frequencies[term] -= 1
                if self.doc_frequencies[term] <= 0:
                    terms_to_clean.append(term)

        for term in terms_to_clean:
            del self.inverted_index[term]
            del self.doc_frequencies[term]

        self._recalculate_corpus_stats()

    def _recalculate_corpus_stats(self) -> None:
        """Recalculate average document length and IDF values for all terms."""
        total_docs = len(self.corpus)
        if total_docs == 0:
            self.avg_doc_len = 0.0
            self.idf_cache = {}
            return

        self.avg_doc_len = sum(self.doc_lengths.values()) / total_docs

        # Calculate Okapi BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1)
        self.idf_cache = {}
        for term, df in self.doc_frequencies.items():
            idf = math.log(((total_docs - df + 0.5) / (df + 0.5)) + 1.0)
            self.idf_cache[term] = max(0.0, idf)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search the BM25 index for matching documents with optional document scoping.

        Args:
            query: User's question or keywords.
            top_k: Number of highest-scoring documents to return.
            filter_doc_ids: Optional list of document filenames/IDs to restrict search to.

        Returns:
            List of (chunk_id, bm25_score) sorted by score in descending order.
        """
        query_tokens = self.tokenize(query)
        if not query_tokens or not self.corpus:
            return []

        doc_filter_set = set(filter_doc_ids) if filter_doc_ids else None
        scores: Dict[str, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf_cache.get(token, 0.0)
            postings = self.inverted_index[token]

            for chunk_id, tf in postings.items():
                # Enforce document scope filter if provided
                if doc_filter_set is not None:
                    meta = self.corpus.get(chunk_id, {}).get("metadata", {})
                    c_doc_id = meta.get("doc_id") or meta.get("filename")
                    if c_doc_id not in doc_filter_set:
                        continue

                doc_len = self.doc_lengths.get(chunk_id, 1)
                # Length normalization factor
                len_norm = 1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0))
                # Okapi BM25 score component
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * len_norm
                term_score = idf * (numerator / denominator)

                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        # Sort chunk_ids by BM25 score descending
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def save(self, filepath: Path | str) -> None:
        """Save the index and corpus metadata to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "corpus": self.corpus,
            "doc_lengths": self.doc_lengths,
            "inverted_index": self.inverted_index,
            "doc_frequencies": self.doc_frequencies,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: Path | str) -> "BM25Index":
        """Load an index from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        idx.corpus = data.get("corpus", {})
        idx.doc_lengths = data.get("doc_lengths", {})
        idx.inverted_index = data.get("inverted_index", {})
        idx.doc_frequencies = data.get("doc_frequencies", {})
        idx._recalculate_corpus_stats()
        return idx

