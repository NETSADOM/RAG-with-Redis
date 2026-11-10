"""
retriever.py
Hybrid semantic search and relevance filtering engine.
Combines Dense Vector Search (ChromaDB + all-MiniLM-L6-v2) and
Sparse Keyword Search (BM25) using Reciprocal Rank Fusion (RRF).
Enforces strict relevance thresholds to prevent returning irrelevant context.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.config import config
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """Represents a single document chunk retrieved for a user query."""

    text: str
    filename: str
    page_number: int
    total_pages: int
    chunk_id: str
    chunk_index: int
    distance: float          # Cosine distance in [0.0, 2.0] (0.0 = identical)
    similarity_score: float  # Normalized similarity: 1.0 - distance
    doc_id: str = ""
    doc_name: str = ""
    rrf_score: Optional[float] = None
    search_method: str = "hybrid"  # "hybrid", "dense", or "bm25"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self.metadata.get("doc_id", self.filename)
        if not self.doc_name:
            self.doc_name = self.metadata.get("doc_name", self.filename)


@dataclass
class RetrievalResult:
    """Encapsulates the complete retrieval outcome for a user query."""

    query: str
    chunks: List[RetrievedChunk]
    has_sufficient_evidence: bool
    closest_distance: Optional[float]
    relevance_threshold: float
    search_method: str = "hybrid"
    document_scope: Optional[List[str]] = None
    message: str = ""

    @property
    def source_documents(self) -> List[str]:
        """Returns sorted list of unique source filenames present in the retrieved chunks."""
        return sorted(list({c.filename for c in self.chunks}))

    @property
    def citations(self) -> List[Dict[str, Any]]:
        """Returns structured citations indicating file and page numbers."""
        citations = []
        for c in self.chunks:
            citations.append({
                "filename": c.filename,
                "page_number": c.page_number,
                "distance": round(c.distance, 4),
                "similarity": round(c.similarity_score, 4),
                "method": c.search_method,
            })
        return citations


class Retriever:
    """
    Executes hybrid semantic search over persistent vector storage.
    Fuses Dense Vector Search and BM25 Sparse Search using Reciprocal Rank Fusion (RRF),
    and enforces configurable distance thresholds.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        mode: str = "hybrid",
    ):
        """
        Initialize Retriever.

        Args:
            vector_store: VectorStore instance. Defaults to configured store.
            embedding_generator: EmbeddingGenerator instance.
            top_k: Number of chunks to retrieve (defaults to config.top_k, usually 3 to 5).
            relevance_threshold: Max allowed cosine distance (defaults to config.relevance_threshold).
            mode: Search strategy: 'hybrid' (default), 'dense', or 'bm25'.
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or self.vector_store._ensure_embedding_generator()
        self.top_k = top_k if top_k is not None else config.top_k
        self.relevance_threshold = (
            relevance_threshold if relevance_threshold is not None else config.relevance_threshold
        )
        self.mode = mode.lower()

    def _dense_search(
        self,
        query: str,
        k: int,
        where_filter: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run dense vector search via ChromaDB with optional document scoping."""
        query_vector = self.embedding_generator.embed_text(query)
        raw_results = self.vector_store.query(
            query_embedding=query_vector,
            n_results=k,
            where=where_filter,
            doc_ids=document_ids,
        )

        doc_lists = raw_results.get("documents", [[]])[0]
        meta_lists = raw_results.get("metadatas", [[]])[0]
        dist_lists = raw_results.get("distances", [[]])[0]
        id_lists = raw_results.get("ids", [[]])[0]

        items = []
        for i in range(len(doc_lists)):
            dist = float(dist_lists[i])
            items.append({
                "chunk_id": id_lists[i],
                "text": doc_lists[i],
                "metadata": meta_lists[i] if i < len(meta_lists) else {},
                "distance": dist,
                "similarity": max(0.0, min(1.0, 1.0 - dist)),
            })
        return items

    def _sparse_search(
        self,
        query: str,
        k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run sparse keyword search via BM25 with optional document scoping."""
        return self.vector_store.query_bm25(query, top_k=k, doc_ids=document_ids)

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k_const: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF).
        RRF Score(d) = sum(1 / (k_const + rank(d))) across dense and sparse lists.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}

        # 1. Score dense rankings
        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_const + rank))
            chunk_data[cid] = item

        # 2. Score sparse (BM25) rankings
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            bm25_sc = item.get("bm25_score", 0.0)
            # Only award BM25 rank boost if score > 0
            if bm25_sc > 0.0:
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_const + rank))
                if cid in chunk_data:
                    chunk_data[cid]["bm25_score"] = bm25_sc
                else:
                    chunk_data[cid] = item

        # 3. Sort by fused RRF score descending
        fused = []
        for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            entry = dict(chunk_data[cid])
            entry["rrf_score"] = score
            fused.append(entry)

        return fused

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant document passages using Hybrid, Dense, or BM25 search.
        Supports document scope filtering (document_ids).

        Args:
            query: User's question or search query.
            top_k: Optional override for number of chunks to retrieve.
            relevance_threshold: Optional override for cosine distance cutoff.
            where_filter: Optional metadata filter.
            document_ids: Optional list of document filenames/IDs to restrict search scope to.
            mode: Optional override ('hybrid', 'dense', or 'bm25').

        Returns:
            RetrievalResult containing filtered chunks, distance metrics, and status.
        """
        cleaned_query = query.strip()
        k = top_k if top_k is not None else self.top_k
        threshold = relevance_threshold if relevance_threshold is not None else self.relevance_threshold
        search_mode = (mode or self.mode).lower()

        # Handle empty query
        if not cleaned_query:
            return RetrievalResult(
                query=query,
                chunks=[],
                has_sufficient_evidence=False,
                closest_distance=None,
                relevance_threshold=threshold,
                search_method=search_mode,
                document_scope=document_ids,
                message="Query was empty or whitespace only.",
            )

        # Handle empty database
        total_records = self.vector_store.count()
        if total_records == 0:
            return RetrievalResult(
                query=cleaned_query,
                chunks=[],
                has_sufficient_evidence=False,
                closest_distance=None,
                relevance_threshold=threshold,
                search_method=search_mode,
                document_scope=document_ids,
                message="Vector database is empty. No documents have been ingested yet.",
            )

        # Retrieve candidates from both sources with document scope
        candidate_pool_size = max(k * 4, 32)
        dense_results = self._dense_search(
            cleaned_query,
            k=candidate_pool_size,
            where_filter=where_filter,
            document_ids=document_ids,
        )
        sparse_results = self._sparse_search(
            cleaned_query,
            k=candidate_pool_size,
            document_ids=document_ids,
        )

        # Fuse or select results based on chosen search_mode
        if search_mode == "bm25":
            ranked_items = sparse_results[:candidate_pool_size]
            for it in ranked_items:
                it["distance"] = 0.5  # placeholder for pure BM25
                it["similarity"] = 0.5
        elif search_mode == "dense":
            ranked_items = dense_results[:candidate_pool_size]
        else:
            # Default: Hybrid Search with Reciprocal Rank Fusion
            ranked_items = self._reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
            )

        if not ranked_items:
            scope_info = f" within selected documents {document_ids}" if document_ids else ""
            return RetrievalResult(
                query=cleaned_query,
                chunks=[],
                has_sufficient_evidence=False,
                closest_distance=None,
                relevance_threshold=threshold,
                search_method=search_mode,
                document_scope=document_ids,
                message=f"No matching document chunks found{scope_info}.",
            )

        # Track closest distance across candidates for reporting
        distances = [it.get("distance", 1.0) for it in ranked_items if "distance" in it]
        closest_distance = min(distances) if distances else None

        # Determine max chunks allowed per document during multi-document search
        is_multi_doc = (document_ids is None or len(document_ids) > 1)
        max_per_doc = max(1, k // 3) if (is_multi_doc and k >= 3) else k

        # Filter by relevance threshold with document diversity
        relevant_chunks: List[RetrievedChunk] = []
        overflow_candidates: List[RetrievedChunk] = []
        doc_chunk_counts: Dict[str, int] = {}

        for item in ranked_items:
            cid = item["chunk_id"]
            metadata = item.get("metadata", {})
            text = item.get("text", "")

            # If item came strictly from BM25 and has no pre-computed dense distance,
            # calculate its exact distance to query embedding
            dist = item.get("distance")
            if dist is None:
                q_vec = self.embedding_generator.embed_text(cleaned_query)
                c_vec = self.embedding_generator.embed_text(text)
                dot = sum(a * b for a, b in zip(q_vec, c_vec))
                dist = max(0.0, 1.0 - dot)
                item["distance"] = dist

            sim = max(0.0, min(1.0, 1.0 - dist))

            # Apply relevance threshold
            # A chunk passes if:
            # 1. Its semantic distance satisfies the threshold (dist <= threshold), OR
            # 2. It has a strong keyword match from BM25 (bm25_score >= 1.0) with dist <= 0.92,
            #    which allows exact named entities, dates, or terms that dense embeddings rate higher.
            is_relevant = dist <= threshold or (item.get("bm25_score", 0.0) >= 1.0 and dist <= 0.92)
            if is_relevant:
                fname = metadata.get("filename", "unknown")
                retrieved_chunk = RetrievedChunk(
                    text=text,
                    filename=fname,
                    page_number=int(metadata.get("page_number", 1)),
                    total_pages=int(metadata.get("total_pages", 1)),
                    chunk_id=cid,
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    distance=dist,
                    similarity_score=sim,
                    doc_id=metadata.get("doc_id", fname),
                    doc_name=metadata.get("doc_name", fname),
                    rrf_score=item.get("rrf_score"),
                    search_method=search_mode,
                    metadata=metadata,
                )

                # Check document diversity quota
                current_count = doc_chunk_counts.get(fname, 0)
                if current_count < max_per_doc:
                    relevant_chunks.append(retrieved_chunk)
                    doc_chunk_counts[fname] = current_count + 1
                else:
                    overflow_candidates.append(retrieved_chunk)

            if len(relevant_chunks) >= k:
                break

        # Pass 2: Backfill from overflow candidates if quota left empty slots
        if len(relevant_chunks) < k and overflow_candidates:
            needed = k - len(relevant_chunks)
            relevant_chunks.extend(overflow_candidates[:needed])

        has_evidence = len(relevant_chunks) > 0

        if has_evidence:
            msg = (
                f"Retrieved {len(relevant_chunks)} relevant passage(s) via {search_mode.upper()} "
                f"(best distance: {closest_distance:.4f}, threshold: {threshold:.2f})."
            )
        else:
            msg = (
                f"No sufficiently relevant passages found. "
                f"Closest passage distance was {closest_distance:.4f}, which exceeds the relevance threshold of {threshold:.2f}."
            )

        return RetrievalResult(
            query=cleaned_query,
            chunks=relevant_chunks,
            has_sufficient_evidence=has_evidence,
            closest_distance=closest_distance,
            relevance_threshold=threshold,
            search_method=search_mode,
            document_scope=document_ids,
            message=msg,
        )
