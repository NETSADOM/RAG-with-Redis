"""
backend/rag/retrieval.py
Implements:
1. BM25 Sparse Keyword Retrieval
2. Sentence Transformers Dense Semantic Retrieval
3. Reciprocal Rank Fusion (RRF) Hybrid Merger
"""

import math
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np


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
        doc_version: str = "v1.0",
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.section = section
        self.source_ref = source_ref
        self.text = text
        self.doc_version = doc_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "source_ref": self.source_ref,
            "text": self.text,
            "doc_version": self.doc_version,
        }


class KnowledgeBaseManager:
    """Loads, parses, and versions the customer support documentation."""

    def __init__(self, kb_path: Optional[Path | str] = None):
        if kb_path is None:
            self.kb_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
        else:
            self.kb_path = Path(kb_path)
        self.chunks: List[KBChunk] = []
        self.version_hash: str = ""
        self.load()

    def load(self) -> None:
        """Load documentation from JSON and calculate global content hash."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at: {self.kb_path}")

        raw_content = self.kb_path.read_text(encoding="utf-8")
        data = json.loads(raw_content)

        # Calculate stable SHA-256 fingerprint of the knowledge base content
        self.version_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()[:16]

        self.chunks = []
        for doc in data.get("documents", []):
            doc_id = doc.get("doc_id", "")
            title = doc.get("title", "")
            doc_ver = doc.get("version", "v1.0")
            default_source = doc.get("source_ref", "")

            for c in doc.get("chunks", []):
                chunk = KBChunk(
                    chunk_id=c.get("chunk_id", ""),
                    doc_id=doc_id,
                    title=c.get("title", title),
                    section=c.get("section", ""),
                    source_ref=c.get("source_ref", default_source),
                    text=c.get("text", ""),
                    doc_version=doc_ver,
                )
                self.chunks.append(chunk)

    def update_document_chunk(self, chunk_id: str, new_text: str) -> str:
        """Update a chunk text to demonstrate knowledge-base version invalidation."""
        raw_content = self.kb_path.read_text(encoding="utf-8")
        data = json.loads(raw_content)

        updated = False
        for doc in data.get("documents", []):
            for c in doc.get("chunks", []):
                if c.get("chunk_id") == chunk_id:
                    c["text"] = new_text
                    updated = True
                    break

        if not updated:
            raise ValueError(f"Chunk '{chunk_id}' not found.")

        # Save back to file and reload
        self.kb_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.load()
        return self.version_hash


class BM25Retriever:
    """Okapi BM25 sparse keyword retriever."""

    def __init__(self, chunks: List[KBChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.tokenized_corpus = [self._tokenize(c.text + " " + c.title) for c in chunks]
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lengths) / max(self.corpus_size, 1)

        # Precompute document frequencies and IDFs
        self.df: Dict[str, int] = {}
        for doc in self.tokenized_corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

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
        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            token_counts = {}
            for t in doc_tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

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

    def __init__(self, chunks: List[KBChunk], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Precompute normalized embeddings for all chunks once at startup
        texts = [f"{c.title}: {c.text}" for c in self.chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = np.array(embeddings, dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KBChunk, float]]:
        """Encode query and return top_k chunks ranked by cosine similarity."""
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
        candidate_pool_size: int = 8,
        final_top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 and Dense search, fuse results using RRF, and return top candidates.
        """
        bm25_results = self.bm25.search(query, top_k=candidate_pool_size)
        dense_results = self.dense.search(query, top_k=candidate_pool_size)

        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, KBChunk] = {}
        debug_ranks: Dict[str, Dict[str, int]] = {}

        # 1. RRF from BM25 ranks
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            chunk_lookup[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            debug_ranks[cid] = {"bm25_rank": rank, "bm25_score": score}

        # 2. RRF from Dense ranks
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_lookup[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            if cid not in debug_ranks:
                debug_ranks[cid] = {}
            debug_ranks[cid]["dense_rank"] = rank
            debug_ranks[cid]["dense_score"] = score

        # 3. Sort by combined RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cid, rrf_score in sorted_chunks[:final_top_k]:
            chunk = chunk_lookup[cid]
            ranks = debug_ranks.get(cid, {})
            results.append({
                "chunk": chunk,
                "rrf_score": rrf_score,
                "bm25_rank": ranks.get("bm25_rank"),
                "dense_rank": ranks.get("dense_rank"),
                "dense_score": ranks.get("dense_score"),
            })

        return results

