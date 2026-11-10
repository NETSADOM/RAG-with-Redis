"""
rag_pipeline.py
The central orchestrator connecting all modular RAG components:
1. PDF text extraction (PDFLoader)
2. Semantic chunking (TextSplitter)
3. Embedding generation (EmbeddingGenerator)
4. Vector storage & duplicate prevention (VectorStore)
5. Hybrid retrieval with multi-document scoping (Retriever)
6. Grounded LLM answer synthesis (LLMGenerator)

Designed with decoupled stages and deterministic cache key generation
to make Redis answer caching seamless without future architecture changes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import time
import logging

from src.config import config
from src.pdf_loader import PDFLoader, DocumentPage
from src.text_splitter import TextSplitter, TextChunk
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever, RetrievalResult, RetrievedChunk
from src.generator import LLMGenerator, GenerationResult
from src.cache import CacheStore, get_cache_strategy, BaseCacheStrategy

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Standardized response returned by the RAG pipeline."""

    question: str
    answer: str
    has_evidence: bool
    sources: List[Dict[str, Any]]
    retrieved_passages: List[Dict[str, Any]]
    model_used: str
    document_scope: Optional[List[str]] = None
    collection_version: Optional[str] = None
    cache_key: Optional[str] = None
    error: Optional[str] = None
    is_cached: bool = False
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert RAGAnswer to a serializable dictionary for Redis caching."""
        return {
            "question": self.question,
            "answer": self.answer,
            "has_evidence": self.has_evidence,
            "sources": self.sources,
            "retrieved_passages": self.retrieved_passages,
            "model_used": self.model_used,
            "document_scope": self.document_scope,
            "collection_version": self.collection_version,
            "cache_key": self.cache_key,
            "error": self.error,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        is_cached: bool = True,
        latency_ms: Optional[float] = None,
    ) -> "RAGAnswer":
        """Reconstruct RAGAnswer from cached Redis JSON payload."""
        return cls(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            has_evidence=data.get("has_evidence", False),
            sources=data.get("sources", []),
            retrieved_passages=data.get("retrieved_passages", []),
            model_used=data.get("model_used", ""),
            document_scope=data.get("document_scope"),
            collection_version=data.get("collection_version"),
            cache_key=data.get("cache_key"),
            error=data.get("error"),
            is_cached=is_cached,
            latency_ms=latency_ms,
        )

    def display(self, show_passages: bool = False) -> str:
        """Format the answer and citations into a clean, human-readable terminal output."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append(f"QUESTION: {self.question}")
        if self.document_scope:
            lines.append(f"SCOPE   : {', '.join(self.document_scope)}")
        else:
            lines.append("SCOPE   : All Available Documents")
        lines.append("-" * 70)
        lines.append(f"ANSWER:\n{self.answer}\n")

        if self.has_evidence and self.sources:
            lines.append("-" * 70)
            lines.append("SOURCES & CITATIONS:")
            for s in self.sources:
                doc_lbl = s.get("doc_name") or s.get("filename")
                lines.append(
                    f"  • {doc_lbl} (Page {s['page_number']}) — "
                    f"Distance: {s['distance']:.4f} | Similarity: {s['similarity']:.4f}"
                )

        if show_passages and self.retrieved_passages:
            lines.append("-" * 70)
            lines.append("RETRIEVED PASSAGES USED:")
            for idx, p in enumerate(self.retrieved_passages, start=1):
                doc_lbl = p.get("doc_name") or p.get("filename")
                lines.append(
                    f"\n[Passage {idx}] Source: {doc_lbl}, Page {p['page_number']} "
                    f"(Dist: {p['distance']:.4f}):"
                )
                lines.append(f"  \"{p['text']}\"")

        lines.append("=" * 70 + "\n")
        return "\n".join(lines)


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.
    Provides clear, independent entrypoints for document ingestion,
    context retrieval, answer generation, and caching key derivation.
    """

    def __init__(
        self,
        loader: Optional[PDFLoader] = None,
        splitter: Optional[TextSplitter] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[LLMGenerator] = None,
        cache_store: Optional[CacheStore] = None,
        cache_strategy_name: Optional[str] = None,
    ):
        """Initialize all pipeline components."""
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.loader = loader or PDFLoader()
        self.splitter = splitter or TextSplitter()
        self.vector_store = vector_store or VectorStore(
            embedding_generator=self.embedding_generator
        )
        self.retriever = retriever or Retriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
        )
        self.generator = generator or LLMGenerator()

        # Modular Cache Layer (Phase 1 & 2 & 3)
        self.cache_store = cache_store or CacheStore()
        strategy_name = cache_strategy_name or config.redis_strategy
        self.cache_strategy = get_cache_strategy(strategy_name, self.cache_store)

    def set_cache_strategy(self, strategy_name: str) -> None:
        """Switch active caching pattern on the fly (e.g. for comparative benchmarking)."""
        self.cache_strategy = get_cache_strategy(strategy_name, self.cache_store)
        logger.info(f"[Pipeline] Active caching strategy set to '{self.cache_strategy.name}'.")

    def clear_cache(self) -> bool:
        """Purge all cached query answers."""
        return self.cache_store.clear()

    # -------------------------------------------------------------------------
    # Ingestion Methods (Support Multiple Document Uploads & Incremental Updates)
    # -------------------------------------------------------------------------
    def ingest_file(
        self,
        file_path: Path | str,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest a single PDF document incrementally without touching others.

        Args:
            file_path: Path to the target PDF.
            force_reindex: Whether to re-embed if already present.

        Returns:
            Dict summarizing extracted pages and stored chunks.
        """
        path = Path(file_path).resolve()
        try:
            pages = self.loader.load_file(path)
        except Exception as e:
            print(f"[Pipeline] Skipped '{path.name}' during ingestion: {e}")
            return {
                "filename": path.name,
                "doc_id": path.name,
                "pages": 0,
                "chunks": 0,
                "skipped": True,
                "reason": str(e),
                "total_records": self.vector_store.count(),
            }

        if not pages:
            return {"filename": path.name, "doc_id": path.name, "pages": 0, "chunks": 0, "total_records": self.vector_store.count()}

        chunks = self.splitter.split_pages(pages)
        stored_count = self.vector_store.add_chunks(chunks, force_reindex=force_reindex)

        return {
            "filename": path.name,
            "doc_id": pages[0].doc_id if pages else path.name,
            "pages": len(pages),
            "chunks": stored_count,
            "total_records": self.vector_store.count(),
        }

    def ingest_files(
        self,
        file_paths: List[Path | str],
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest a batch of specific PDF files incrementally.

        Args:
            file_paths: List of paths to PDF files.
            force_reindex: Whether to re-embed if already present.

        Returns:
            Dictionary with aggregated statistics.
        """
        total_pages = 0
        total_chunks = 0
        processed_files = []

        for p in file_paths:
            res = self.ingest_file(p, force_reindex=force_reindex)
            total_pages += res.get("pages", 0)
            total_chunks += res.get("chunks", 0)
            processed_files.append(res)

        return {
            "files_processed": len(processed_files),
            "pages_extracted": total_pages,
            "chunks_stored": total_chunks,
            "total_records": self.vector_store.count(),
            "details": processed_files,
        }

    def ingest_documents(
        self,
        documents_dir: Optional[Path | str] = None,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute document ingestion across a directory:
        PDFs -> Page Extraction -> Chunking -> Embeddings -> ChromaDB.

        Args:
            documents_dir: Optional override for documents directory.
            force_reindex: If True, re-embeds and overwrites existing files.

        Returns:
            Dictionary summarizing ingestion statistics.
        """
        print("\n--- Starting Document Ingestion ---")
        pages = self.loader.load_directory(documents_dir)
        if not pages:
            print("[Pipeline] No extractable pages found to process.")
            return {
                "pages_extracted": 0,
                "chunks_created": 0,
                "chunks_stored": 0,
                "total_records": self.vector_store.count(),
            }

        chunks = self.splitter.split_pages(pages)
        print(f"[Pipeline] Created {len(chunks)} text chunk(s) from {len(pages)} page(s).")

        stored_count = self.vector_store.add_chunks(chunks, force_reindex=force_reindex)

        summary = {
            "pages_extracted": len(pages),
            "chunks_created": len(chunks),
            "chunks_stored": stored_count,
            "total_records": self.vector_store.count(),
        }
        print(f"[Pipeline] Ingestion complete. Stored {stored_count} chunk(s). Total DB records: {summary['total_records']}")
        return summary

    # -------------------------------------------------------------------------
    # Decoupled Querying, Caching, and Answering (Redis-Ready Architecture)
    # -------------------------------------------------------------------------
    def compute_cache_key(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        collection_version: Optional[str] = None,
    ) -> str:
        """
        Derive a deterministic SHA256 cache key for the query, scope, and database version.
        Ready for future Redis caching (e.g. redis.get(cache_key)).

        Args:
            question: Cleaned question string.
            document_ids: List of scoped document IDs, or None for all.
            collection_version: Database collection fingerprint.

        Returns:
            Hex string cache key.
        """
        sorted_scope = sorted(document_ids) if document_ids else ["__ALL__"]
        version = collection_version or self.vector_store.get_collection_fingerprint(document_ids)
        model = self.generator.model_name
        
        raw_key = f"{question.strip().lower()}::scope={','.join(sorted_scope)}::ver={version}::model={model}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def retrieve_context(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Stage 1: Retrieve scoped relevant passages using hybrid semantic search.
        """
        return self.retriever.retrieve(
            query=question,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            document_ids=document_ids,
        )

    def generate_answer(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        available_documents: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Stage 2: Synthesize a factual, grounded answer via LLM.
        """
        return self.generator.generate(
            question=question,
            retrieval_result=retrieval_result,
            available_documents=available_documents,
        )

    def _compute_rag_answer(
        self,
        cleaned_query: str,
        document_ids: Optional[List[str]],
        effective_k: int,
        relevance_threshold: Optional[float],
        collection_ver: str,
        cache_key: str,
    ) -> RAGAnswer:
        """
        Execute core decoupled RAG retrieval and LLM synthesis.
        """
        # 1. Retrieve scoped context
        retrieval_result = self.retrieve_context(
            question=cleaned_query,
            document_ids=document_ids,
            top_k=effective_k,
            relevance_threshold=relevance_threshold,
        )

        # 2. Build structured citations and passage data
        citations = []
        passages = []
        for c in retrieval_result.chunks:
            citations.append({
                "filename": c.filename,
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "distance": c.distance,
                "similarity": c.similarity_score,
            })
            passages.append({
                "chunk_id": c.chunk_id,
                "filename": c.filename,
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "text": c.text,
                "distance": c.distance,
                "similarity": c.similarity_score,
            })

        # 3. Determine available documents for broader awareness
        all_doc_names = [d["filename"] for d in self.list_documents()]
        scope_docs = document_ids if document_ids else all_doc_names

        # 4. Generate answer via LLM
        gen_result = self.generate_answer(
            question=cleaned_query,
            retrieval_result=retrieval_result,
            available_documents=scope_docs,
        )

        return RAGAnswer(
            question=cleaned_query,
            answer=gen_result.answer,
            has_evidence=retrieval_result.has_sufficient_evidence,
            sources=citations,
            retrieved_passages=passages,
            model_used=gen_result.model,
            document_scope=document_ids,
            collection_version=collection_ver,
            cache_key=cache_key,
            error=gen_result.error,
        )

    def answer_question(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
    ) -> RAGAnswer:
        """
        Complete question-answering workflow with caching:
        - On HIT: Return answer and preserved sources directly from cache (no retrieval, no LLM).
        - On MISS: Compute answer via RAG pipeline, cache the result with TTL, and return.
        - Graceful fallback: If Redis is unavailable or fails, compute answer without crashing.

        Args:
            question: The natural-language question.
            document_ids: Optional list of document filenames/IDs to scope the query to.
            top_k: Optional override for max retrieved chunks.
            relevance_threshold: Optional override for distance cutoff.

        Returns:
            RAGAnswer with final answer, source citations, passages, and cache metadata.
        """
        start_time = time.perf_counter()
        cleaned_query = question.strip()
        collection_ver = self.vector_store.get_collection_fingerprint(document_ids)
        cache_key = self.compute_cache_key(cleaned_query, document_ids, collection_ver)

        if not cleaned_query:
            return RAGAnswer(
                question=question,
                answer="Please provide a non-empty question.",
                has_evidence=False,
                sources=[],
                retrieved_passages=[],
                model_used=self.generator.model_name,
                document_scope=document_ids,
                collection_version=collection_ver,
                cache_key=cache_key,
                is_cached=False,
                latency_ms=0.0,
            )

        # Scale top_k appropriately: 6 for multi-document/global queries, 4 for single-document queries
        effective_k = top_k
        if effective_k is None:
            if document_ids and len(document_ids) == 1:
                effective_k = 4
            else:
                effective_k = max(config.top_k, 6)

        def _compute() -> RAGAnswer:
            return self._compute_rag_answer(
                cleaned_query=cleaned_query,
                document_ids=document_ids,
                effective_k=effective_k,
                relevance_threshold=relevance_threshold,
                collection_ver=collection_ver,
                cache_key=cache_key,
            )

        try:
            raw_result, is_cached = self.cache_strategy.get_or_compute_answer(
                question=cleaned_query,
                document_ids=document_ids,
                compute_fn=_compute,
                cache_key=cache_key,
                ttl=config.redis_ttl,
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            if is_cached and isinstance(raw_result, dict):
                return RAGAnswer.from_dict(raw_result, is_cached=True, latency_ms=latency_ms)
            elif isinstance(raw_result, RAGAnswer):
                raw_result.is_cached = is_cached
                raw_result.latency_ms = latency_ms
                return raw_result
            elif isinstance(raw_result, dict):
                return RAGAnswer.from_dict(raw_result, is_cached=is_cached, latency_ms=latency_ms)
            else:
                ans = _compute()
                ans.is_cached = False
                ans.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return ans

        except Exception as e:
            logger.warning(f"[Pipeline] Cache execution failed, falling back to direct computation: {e}")
            ans = _compute()
            ans.is_cached = False
            ans.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ans

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all currently indexed documents and their chunk counts."""
        return self.vector_store.list_documents()

    def delete_document(self, filename: str) -> int:
        """Remove a document and its chunks from the database."""
        return self.vector_store.delete_document(filename)
