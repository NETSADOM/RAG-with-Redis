"""
backend/rag/pipeline.py
The central orchestrator implementing the complete Customer Support RAG workflow:
Step 1: Receive customer question
Step 2: Check Redis before retrieval or generation (Exact-Match lookup O(1) <0.2ms)
        - O(1) collection revision lookup without full collection disk scan
Step 2b: Semantic Cache check for equivalent/paraphrased questions (Cosine similarity >= 0.88)
Step 3: Cache Hit -> return cached answer & verified citations immediately, skip pipeline
Step 4: Cache Miss -> Hybrid Retrieval (BM25 + Dense) -> Cross-Encoder -> LLM with Citation Validation
Step 4b: Write-Through on Miss -> persist answer, validated citations, and query vector to Redis
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import logging
import numpy as np

from backend.rag.retrieval import KnowledgeBaseManager, BM25Retriever, DenseRetriever, HybridRetriever
from backend.rag.reranker import CrossEncoderReranker
from backend.rag.generator import CustomerSupportGenerator, CustomerSupportAnswer
from backend.cache.redis_cache import CustomerSupportCache

logger = logging.getLogger(__name__)


@dataclass
class SupportRAGResponse:
    """Standardized response returned to the client and API."""
    question: str
    answer: str
    cache_hit: bool
    latency_ms: float
    sources: List[Dict[str, Any]]
    knowledge_base_version: str
    model_used: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    citation_validation_passed: bool = True
    cache_type: str = "none"  # "exact", "semantic", or "none"
    similarity_score: float = 1.0
    is_supported: bool = True
    rerank_used: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_references"] = self.sources
        return d


class CustomerSupportRAGPipeline:
    """
    Complete production-ready Customer Support RAG Pipeline.
    Initializes knowledge base, BM25, and dense embeddings once at startup.
    Supports Cache-Aside, Semantic Caching, Write-Through on Misses, and LFU Eviction.
    """

    def __init__(
        self,
        kb_manager: Optional[KnowledgeBaseManager] = None,
        cache: Optional[CustomerSupportCache] = None,
        generator: Optional[CustomerSupportGenerator] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        lazy_dense: bool = False,
    ):
        logger.info("[Pipeline] Initializing Customer Support RAG Pipeline...")

        # 1. Knowledge Base Manager (with O(1) persisted revisions)
        self.kb_manager = kb_manager or KnowledgeBaseManager()
        logger.info(f"[Pipeline] Loaded {len(self.kb_manager.chunks)} policy chunks. KB Revision: {self.kb_manager.get_collection_revision()}")

        # 2. Redis Answer Cache
        self.cache = cache or CustomerSupportCache()

        # 3. LLM Generator
        self.generator = generator or CustomerSupportGenerator()

        # 4. Sparse (BM25) Retriever
        self.bm25_retriever = BM25Retriever(self.kb_manager.chunks)

        # 5. Dense (Sentence Transformer) Retriever
        if not lazy_dense:
            self.dense_retriever = DenseRetriever(self.kb_manager.chunks)
            self.hybrid_retriever = HybridRetriever(self.bm25_retriever, self.dense_retriever)
        else:
            self.dense_retriever = None
            self.hybrid_retriever = None

        # 6. Cross-Encoder Reranker
        self.reranker = reranker or CrossEncoderReranker(lazy_load=lazy_dense)

        logger.info("[Pipeline] Pipeline initialization complete and ready for queries.")

    def reload_knowledge_base(self) -> str:
        """Reload documentation, rebuild indexes, and update collection revision."""
        self.kb_manager.load()
        self.bm25_retriever = BM25Retriever(self.kb_manager.chunks)
        if self.dense_retriever:
            self.dense_retriever = DenseRetriever(self.kb_manager.chunks)
            self.hybrid_retriever = HybridRetriever(self.bm25_retriever, self.dense_retriever)
        return self.kb_manager.get_collection_revision()

    def answer_question(
        self,
        question: str,
        skip_cache: bool = False,
        use_reranker: bool = True,
        top_k_retrieve: int = 6,
        top_k_final: int = 3,
    ) -> SupportRAGResponse:
        """
        Execute the exact RAG workflow:
        1. Receive customer question
        2. Check Redis before retrieval or generation (O(1) revision lookup, zero collection scan)
        3. On HIT -> return immediately, skip all retrieval & LLM
        4. On MISS -> BM25 + Dense -> RRF -> Cross-Encoder -> LLM -> Write-Through Cache -> return
        """
        start_time = time.perf_counter()
        clean_q = question.strip()

        # O(1) Precomputed collection revision - avoids full collection scans on queries
        if hasattr(self.kb_manager, "get_collection_revision"):
            rev = self.kb_manager.get_collection_revision()
            kb_ver = rev if isinstance(rev, str) else getattr(self.kb_manager, "version_hash", "v1.0.0")
        else:
            kb_ver = getattr(self.kb_manager, "version_hash", "v1.0.0")
        model_name = self.generator.model

        if not clean_q:
            return SupportRAGResponse(
                question=question,
                answer="Please enter a valid customer support question.",
                cache_hit=False,
                latency_ms=0.0,
                sources=[],
                citations=[],
                knowledge_base_version=kb_ver,
                model_used=model_name,
                is_supported=False,
            )

        # STEP 2 & 3: Check Redis before retrieval or generation
        if not skip_cache:
            try:
                # 2.1 Fast Exact-Match check (O(1), <0.2ms)
                cached_data, is_hit = self.cache.get_answer(
                    query=clean_q,
                    kb_version=kb_ver,
                    model=model_name,
                )
                cache_type = "exact"
                sim_score = 1.0

                # 2.2 Semantic Cache lookup for equivalent/paraphrased questions
                if not is_hit and self.dense_retriever and getattr(self.cache, "semantic_caching_enabled", True):
                    query_vec = self.dense_retriever.model.encode([clean_q], normalize_embeddings=True, show_progress_bar=False)[0]
                    sem_data, sem_hit, best_sim = self.cache.get_answer_semantic(
                        query_vec=query_vec,
                        kb_version=kb_ver,
                        model=model_name,
                        query_text=clean_q,
                    )
                    if sem_hit and sem_data:
                        cached_data = sem_data
                        is_hit = True
                        cache_type = "semantic"
                        sim_score = best_sim

                if is_hit and cached_data:
                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.info(f"[Pipeline] Cache HIT ({cache_type}, score={sim_score:.2f}) in {latency_ms}ms for: '{clean_q[:35]}...'")
                    cached_sources = cached_data.get("sources", [])
                    return SupportRAGResponse(
                        question=clean_q,
                        answer=cached_data.get("answer", ""),
                        cache_hit=True,
                        cache_type=cache_type,
                        similarity_score=round(sim_score, 3),
                        latency_ms=latency_ms,
                        sources=cached_sources,
                        citations=cached_sources,
                        retrieved_context=cached_data.get("retrieved_context", cached_sources),
                        citation_validation_passed=True,
                        knowledge_base_version=kb_ver,
                        model_used=cached_data.get("model", model_name),
                        is_supported=True,
                        rerank_used=False,
                    )
            except Exception as e:
                logger.warning(f"[Pipeline] Cache lookup failed ({e}). Proceeding to normal RAG processing.")

        logger.info(f"[Pipeline] Cache MISS for query: '{clean_q[:35]}...'. Running Hybrid Retrieval & Reranker.")

        # STEP 4: Complete RAG Pipeline on Cache Miss
        # 4.1 Hybrid Retrieval (BM25 + Dense + RRF)
        if self.hybrid_retriever:
            candidates = self.hybrid_retriever.search(
                query=clean_q,
                candidate_pool_size=top_k_retrieve,
                final_top_k=top_k_retrieve,
            )
        else:
            bm25_res = self.bm25_retriever.search(clean_q, top_k=top_k_retrieve)
            candidates = [{"chunk": c, "rrf_score": s} for c, s in bm25_res]

        # 4.2 Cross-Encoder Reranking
        if use_reranker and self.reranker:
            reranked_passages = self.reranker.rerank(
                query=clean_q,
                candidates=candidates,
                top_n=top_k_final,
            )
        else:
            reranked_passages = candidates[:top_k_final]

        # 4.3 LLM Answer Generation with Active Citation Tracking & Validation
        gen_result: CustomerSupportAnswer = self.generator.generate(
            question=clean_q,
            retrieved_passages=reranked_passages,
            company_name="NovaCart Essentials",
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        try:
            self.cache.record_miss_latency(latency_ms)
        except Exception:
            pass

        effective_sources = gen_result.citations if gen_result.citations else gen_result.sources

        # 4.4 Write-Through Storage on Miss: Persist answer + validated citations + query vector
        # Rule: Cache a supported answer ONLY if it has at least one citation AND citation validation passed
        if (
            gen_result.is_supported
            and gen_result.answer
            and not gen_result.error
            and not skip_cache
            and len(effective_sources) > 0
            and gen_result.citation_validation_passed
        ):
            try:
                query_vec = None
                if self.dense_retriever:
                    query_vec = self.dense_retriever.model.encode([clean_q], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

                self.cache.set_answer(
                    query=clean_q,
                    kb_version=kb_ver,
                    model=model_name,
                    answer=gen_result.answer,
                    sources=effective_sources,
                    query_embedding=query_vec,
                )
                logger.info(f"[Pipeline] Successfully write-through cached answer in Redis with validated citations.")
            except Exception as e:
                logger.warning(f"[Pipeline] Failed to cache answer in Redis: {e}")

        return SupportRAGResponse(
            question=clean_q,
            answer=gen_result.answer,
            cache_hit=False,
            cache_type="none",
            similarity_score=1.0,
            latency_ms=latency_ms,
            sources=effective_sources,
            citations=effective_sources,
            retrieved_context=gen_result.retrieved_context,
            citation_validation_passed=gen_result.citation_validation_passed,
            knowledge_base_version=kb_ver,
            model_used=gen_result.model,
            is_supported=gen_result.is_supported,
            rerank_used=use_reranker,
            error=gen_result.error,
        )
