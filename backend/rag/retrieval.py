"""Keyword and semantic policy retrieval with reciprocal rank fusion."""

import os
import math
import re
import json
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

FILTER_FIELDS = {"doc_id", "section", "doc_version", "page_number"}
SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="policy-search")


def filter_results(results, metadata_filters=None):
    """Apply explicit, exact metadata constraints without guessing from the question."""
    filters = metadata_filters or {}
    if filters.keys() - FILTER_FIELDS:
        raise ValueError("Unsupported metadata filter.")
    return [(chunk, score) for chunk, score in results
            if all(getattr(chunk, name) == value for name, value in filters.items())]


class KBChunk:
    """Represents a single knowledge base policy chunk with metadata."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        title: str,
        section: str,
        source_ref: str,
        text: str,
        page_number: int = 1,
        source_url: str = "",
        doc_version: str = "v1.0",
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.section = section
        self.source_ref = source_ref
        self.text = text
        self.page_number = page_number
        self.source_url = source_url or ""
        self.doc_version = doc_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "source_ref": self.source_ref,
            "text": self.text,
            "page_number": self.page_number,
            "source_url": self.source_url,
            "doc_version": self.doc_version,
        }


class KnowledgeBaseManager:
    'Loads, parses, and versions customer support documentation.'

    def __init__(self, kb_path: Optional[Path | str] = None):
        if kb_path is None:
            self.kb_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
        else:
            self.kb_path = Path(kb_path)
        self.chunks: List[KBChunk] = []
        self.doc_revisions: Dict[str, str] = {}
        self.collection_revision: str = ""
        self.version_hash: str = ""
        self.load()

    def get_collection_revision(self) -> str:
        """
        Return the pre-persisted collection revision in O(1).
        Zero disk read, zero collection chunk iteration on incoming queries.
        """
        return self.collection_revision

    def load(self) -> None:
        """
        Load documentation from JSON, parse chunks, and persist document revisions in memory.
        Calculates doc-level hashes once during startup or reload.
        """
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at: {self.kb_path}")

        raw_content = self.kb_path.read_text(encoding="utf-8")
        data = json.loads(raw_content)

        self.load_data(data)

    def load_data(self, data: Dict[str, Any]) -> None:
        """Parse staged data using the same rules as startup."""
        self.chunks = []
        self.doc_revisions = {}

        for doc in data.get("documents", []):
            doc_id = doc.get("doc_id", "")
            title = doc.get("title", "")
            doc_ver = doc.get("version", "v1.0")
            default_source = doc.get("source_ref", "")

            doc_raw = json.dumps(doc, sort_keys=True)
            self.doc_revisions[doc_id] = hashlib.sha256(doc_raw.encode("utf-8")).hexdigest()[:12]

            for c in doc.get("chunks", []):
                chunk = KBChunk(
                    chunk_id=c.get("chunk_id", ""),
                    doc_id=doc_id,
                    title=c.get("title", title),
                    section=c.get("section", ""),
                    source_ref=c.get("source_ref", default_source),
                    text=c.get("text", ""),
                    page_number=int(c.get("page_number", 1)),
                    source_url=c.get("source_url", ""),
                    doc_version=doc_ver,
                )
                self.chunks.append(chunk)

        # Global collection revision derived deterministically from persisted doc revisions
        rev_manifest = json.dumps(self.doc_revisions, sort_keys=True)
        self.collection_revision = hashlib.sha256(rev_manifest.encode("utf-8")).hexdigest()[:16]
        self.version_hash = self.collection_revision


class BM25Retriever:
    """Okapi BM25 sparse keyword retriever."""

    def __init__(self, chunks: List[KBChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        corpus = [self._tokenize(c.text + " " + c.title) for c in chunks]
        self.term_counts = [Counter(doc) for doc in corpus]
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lengths) / max(self.corpus_size, 1)

        # Precompute document frequencies and IDFs
        frequencies = Counter(term for counts in self.term_counts for term in counts)
        self.idf = {
            term: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in frequencies.items()
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Normalize and tokenize text with simple English stopword removal."""
        stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
            "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
            "during", "each", "few", "for", "from", "further", "had", "has", "have",
            "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
            "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
            "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
            "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
            "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
            "their", "theirs", "them", "themselves", "then", "there", "these", "they",
            "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
            "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
            "with", "would", "you", "your", "yours", "yourself", "yourselves"
        }
        tokens = re.findall(r"\b[a-zA-Z0-9$]+\b", text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KBChunk, float]]:
        """Compute BM25 score for all chunks and return top_k candidates."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(c, 0.0) for c in self.chunks[:top_k]]

        scores = []
        for idx, token_counts in enumerate(self.term_counts):
            score = 0.0
            doc_len = self.doc_lengths[idx]

            for q_term in query_tokens:
                if q_term not in token_counts:
                    continue
                tf = token_counts[q_term]
                idf = self.idf.get(q_term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-6)

            scores.append((self.chunks[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DenseRetriever:
    """Dense semantic retriever using Sentence Transformers and in-memory cosine search."""

    def __init__(self, chunks: List[KBChunk], model_name: Optional[str] = None, model=None, previous=None):
        self.chunks = chunks
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        if model is not None:
            self.model = model
        else:
            self.model = self._load_model()
        self._encode_lock = previous._encode_lock if previous is not None and previous.model is self.model else Lock()
        texts = [f"{c.title}: {c.text}" for c in self.chunks]
        reused = {}
        if previous is not None and previous.model is self.model:
            reused = {f"{c.title}: {c.text}": vector for c, vector in zip(previous.chunks, previous.embeddings)}
        missing = list(dict.fromkeys(text for text in texts if text not in reused))
        if missing:
            with self._encode_lock:
                vectors = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            reused.update(zip(missing, vectors))
        self.embeddings = np.array([reused[text] for text in texts], dtype=np.float32)

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        try:
            return SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            return SentenceTransformer(self.model_name)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KBChunk, float]]:
        """Encode query and return top_k chunks ranked by cosine similarity."""
        if not self.chunks:
            return []
        with self._encode_lock:
            query_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        # Dot product of normalized vectors equals cosine similarity in [-1.0, 1.0]
        sims = np.dot(self.embeddings, query_vec)

        ranked_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in ranked_indices]


class HybridRetriever:
    """
    Combines BM25 and Dense Retrieval using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score = 1 / (k + rank_bm25) + 1 / (k + rank_dense)
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        candidate_pool_size: int = 50,
        final_top_k: int = 4,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search both branches concurrently, filter each list, then fuse with RRF.
        """
        if (metadata_filters or {}).keys() - FILTER_FIELDS:
            raise ValueError("Unsupported metadata filter.")
        keyword = SEARCH_EXECUTOR.submit(self.bm25.search, query, top_k=candidate_pool_size)
        semantic = SEARCH_EXECUTOR.submit(self.dense.search, query, top_k=candidate_pool_size)
        bm25_results = filter_results([(chunk, score) for chunk, score in keyword.result() if score > 0], metadata_filters)
        dense_results = filter_results(semantic.result(), metadata_filters)

        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, KBChunk] = {}
        provenance: Dict[str, Dict[str, Any]] = {}
        for branch, ranking in (("keyword", bm25_results), ("semantic", dense_results)):
            for rank, (chunk, score) in enumerate(ranking, start=1):
                cid = chunk.chunk_id
                chunk_lookup[cid] = chunk
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                provenance.setdefault(cid, {}).update({f"{branch}_rank": rank, f"{branch}_score": score})

        sorted_chunks = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [
            {"chunk": chunk_lookup[cid], "rrf_score": rrf_scores[cid], **provenance[cid]}
            for cid in sorted_chunks[:final_top_k]
        ]

