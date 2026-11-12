"""Coordinate retrieval, cited generation, answer caching and staged policy updates."""

import copy
import json
import hashlib
import time
from uuid import uuid4
from concurrent.futures import Future, TimeoutError as FutureTimeout
from pathlib import Path
from dotenv import load_dotenv
from threading import RLock, BoundedSemaphore
from backend.locking import synchronized
from backend.admission import generation_slot, positive_setting
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import logging

from backend.rag.retrieval import KnowledgeBaseManager, BM25Retriever, DenseRetriever, HybridRetriever, filter_results
from backend.rag.reranker import CrossEncoderReranker, select_passages
from backend.rag.generator import CustomerSupportGenerator, CustomerSupportAnswer
from backend.cache.redis_cache import CustomerSupportCache, RedisUnavailable

logger = logging.getLogger(__name__)
RETRIEVAL_VERSION = "hybrid-filtered-v4"
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


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
    cache_type: str = "none"  # "exact" or "none"
    is_supported: bool = True
    rerank_used: bool = True
    error: Optional[str] = None
    refusal_reason: Optional[str] = None
    cache_degraded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_references"] = self.sources
        return d


class CustomerSupportRAGPipeline:
    """Initialize retrieval once; reuse successful answers until policies change."""

    def __init__(
        self,
        kb_manager: Optional[KnowledgeBaseManager] = None,
        cache: Optional[CustomerSupportCache] = None,
        generator: Optional[CustomerSupportGenerator] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        lazy_dense: bool = False,
    ):
        self._lock = RLock()
        self._inflight = {}
        self._generation_slots = BoundedSemaphore(positive_setting("MAX_CONCURRENT_GENERATIONS", 2))
        logger.info("[Pipeline] Initializing Customer Support RAG Pipeline...")

        self.kb_manager = kb_manager or KnowledgeBaseManager()
        logger.info(f"[Pipeline] Loaded {len(self.kb_manager.chunks)} policy chunks. KB Revision: {self.kb_manager.get_collection_revision()}")

        self.cache = cache or CustomerSupportCache()

        self.generator = generator or CustomerSupportGenerator()

        self.bm25_retriever = BM25Retriever(self.kb_manager.chunks)

        if not lazy_dense:
            self.dense_retriever = DenseRetriever(self.kb_manager.chunks)
            self.hybrid_retriever = HybridRetriever(self.bm25_retriever, self.dense_retriever)
        else:
            self.dense_retriever = None
            self.hybrid_retriever = None

        self.reranker = reranker or CrossEncoderReranker(lazy_load=lazy_dense)

        logger.info("[Pipeline] Pipeline initialization complete and ready for queries.")

    @synchronized
    def get_knowledge_base(self) -> Dict[str, Any]:
        """Serve the active revision from Redis, rebuilding evicted snapshots."""
        revision = self.kb_manager.get_collection_revision()
        try:
            snapshot = self.cache.get_knowledge_base(revision)
            if snapshot is None:
                snapshot = self._knowledge_base_snapshot(self.kb_manager)
                self.cache.set_knowledge_base(revision, snapshot)
            return snapshot
        except RedisUnavailable:
            return self._knowledge_base_snapshot(self.kb_manager)

    @staticmethod
    def _knowledge_base_snapshot(manager: KnowledgeBaseManager) -> Dict[str, Any]:
        docs = {}
        for chunk in manager.chunks:
            docs.setdefault(chunk.doc_id, {"doc_id": chunk.doc_id, "title": chunk.title.split(" - ")[0]})
        return {"version_hash": manager.version_hash, "documents": list(docs.values()),
                "chunks": [chunk.to_dict() for chunk in manager.chunks]}

    @synchronized
    def stage_and_update_policy(self, chunk_id: str, new_text: str) -> Dict[str, Any]:
        """Stage indexes and write through Redis before committing a policy revision."""
        new_text = new_text.strip()
        if not new_text:
            raise ValueError("Policy text cannot be empty.")
        old_version = self.kb_manager.get_collection_revision()
        kb_path = self.kb_manager.kb_path

        raw_content = kb_path.read_text(encoding="utf-8")
        staged_data = json.loads(raw_content)

        updated = False
        target_doc_id = ""
        old_text = ""
        for doc in staged_data.get("documents", []):
            for c in doc.get("chunks", []):
                if c.get("chunk_id") == chunk_id:
                    old_text = c.get("text", "")
                    c["text"] = new_text
                    updated = True
                    target_doc_id = doc.get("doc_id", "")
                    break
            if updated:
                break

        if not updated:
            raise ValueError(f"Chunk '{chunk_id}' not found in knowledge base.")

        staged_manager = copy.copy(self.kb_manager)
        staged_manager.load_data(staged_data)
        staged_chunks = staged_manager.chunks

        new_bm25 = BM25Retriever(staged_chunks)
        new_dense = None
        new_hybrid = None
        if self.dense_retriever is not None:
            new_dense = DenseRetriever(staged_chunks, model_name=self.dense_retriever.model_name,
                                       model=self.dense_retriever.model, previous=self.dense_retriever)
            new_hybrid = HybridRetriever(new_bm25, new_dense)

        temp_path = kb_path.with_suffix(".json.tmp")
        try:
            temp_path.write_text(json.dumps(staged_data, indent=2), encoding="utf-8")
            # Cache by immutable revision before the atomic file commit. If either
            # write fails, the active revision remains intact. An uncommitted
            # snapshot is unreachable and expires normally, making retries safe.
            self.cache.set_knowledge_base(
                staged_manager.get_collection_revision(),
                self._knowledge_base_snapshot(staged_manager),
            )
            temp_path.replace(kb_path)
        finally:
            temp_path.unlink(missing_ok=True)

        self.kb_manager = staged_manager
        new_version = self.kb_manager.get_collection_revision()
        self.bm25_retriever = new_bm25
        if new_dense:
            self.dense_retriever = new_dense
            self.hybrid_retriever = new_hybrid

        logger.info(
            f"[Pipeline] Policy update activated safely for doc '{target_doc_id}', chunk '{chunk_id}'. "
            f"Version transition: {old_version} -> {new_version}"
        )

        return {
            "chunk_id": chunk_id,
            "doc_id": target_doc_id,
            "old_version": old_version,
            "new_version": new_version,
            "old_text": old_text,
            "new_text": new_text,
        }

    def answer_question(
        self,
        question: str,
        skip_cache: bool = False,
        use_reranker: bool = True,
        top_k_retrieve: int = 50,
        top_k_final: int = 4,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> SupportRAGResponse:
        """Share one in-flight execution among matching requests in this process."""
        if skip_cache:
            return self._answer_question(question, skip_cache, use_reranker,
                                         top_k_retrieve, top_k_final, metadata_filters)
        started = time.perf_counter()
        with self._lock:
            key = json.dumps([
                self.cache.normalize_query(question),
                self.kb_manager.get_collection_revision(), self.generator.model,
                use_reranker, top_k_retrieve, top_k_final, metadata_filters or {},
            ], sort_keys=True)
            pending = self._inflight.get(key)
            leader = pending is None
            if leader:
                if len(self._inflight) >= 256:
                    raise RuntimeError("Support is busy. Please try again shortly.")
                pending = Future()
                self._inflight[key] = pending
        if not leader:
            try:
                result = copy.deepcopy(pending.result(timeout=positive_setting("ANSWER_WAIT_SECONDS", 3000)))
            except FutureTimeout:
                return self._compute_answer(question, skip_cache, use_reranker,
                                            top_k_retrieve, top_k_final, metadata_filters)
            result.question = question.strip()
            result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return result
        try:
            result = self._answer_question(question, skip_cache, use_reranker,
                                           top_k_retrieve, top_k_final, metadata_filters)
            pending.set_result(result)
            return result
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _answer_question(self, question, skip_cache=False, use_reranker=True,
                         top_k_retrieve=50, top_k_final=4, metadata_filters=None):
        args = (question, skip_cache, use_reranker, top_k_retrieve, top_k_final, metadata_filters)
        if skip_cache:
            return self._compute_answer(*args)
        with self._lock:
            identity = json.dumps([
                self.cache.normalize_query(question), self.kb_manager.get_collection_revision(),
                self.generator.model, getattr(self.reranker, "model_name", ""),
                RETRIEVAL_VERSION, use_reranker, top_k_retrieve, top_k_final, metadata_filters or {},
            ], sort_keys=True)
        key = f"{self.cache.namespace}:lock:{hashlib.sha256(identity.encode()).hexdigest()}"
        token = uuid4().hex
        wait_seconds = positive_setting("ANSWER_WAIT_SECONDS", 3000)
        deadline = time.monotonic() + wait_seconds
        acquired = False
        try:
            while True:
                try:
                    acquired = self.cache.acquire_answer_lock(key, token, wait_seconds)
                except RedisUnavailable:
                    break
                if acquired or time.monotonic() >= deadline:
                    break
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            # Recheck the answer cache after acquiring the lease; the previous
            # owner may have populated it while this request was waiting.
            return self._compute_answer(*args)
        finally:
            if acquired:
                self.cache.release_answer_lock(key, token)

    def _compute_answer(
        self,
        question: str,
        skip_cache: bool = False,
        use_reranker: bool = True,
        top_k_retrieve: int = 50,
        top_k_final: int = 4,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> SupportRAGResponse:
        """Return a cached answer or retrieve, cite, and cache policy evidence."""
        start_time = time.perf_counter()
        clean_q = question.strip()

        # Snapshot the complete revision under the update lock. In-flight queries
        # keep their original indexes and cache namespace if a policy changes.
        with self._lock:
            kb_ver = self.kb_manager.get_collection_revision()
            dense = self.dense_retriever
            hybrid = self.hybrid_retriever
            bm25 = self.bm25_retriever
        model_name = self.generator.model
        # The builder's version and retrieval settings identify cache entries.
        # Validate filters even when an answer could be served from cache.
        filter_results([], metadata_filters)
        config = [RETRIEVAL_VERSION, metadata_filters or {}, use_reranker,
                  top_k_retrieve, top_k_final,
                  str(getattr(dense, "model_name", "bm25")),
                  str(getattr(self.reranker, "model_name", ""))]
        cache_model = model_name + ":" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

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

        # During a Redis outage, serve fresh grounded
        # answers without introducing a second cache or returning stale policies.
        cache_degraded = False

        if not skip_cache:
            try:
                # An exact match needs only one Redis GET.
                cached_data, is_hit = self.cache.get_answer(
                    query=clean_q,
                    kb_version=kb_ver,
                    model=cache_model,
                )
                if is_hit and cached_data:
                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.debug("Exact cache hit in %s ms", latency_ms)
                    cached_sources = cached_data.get("sources", [])
                    return SupportRAGResponse(
                        question=clean_q,
                        answer=cached_data.get("answer", ""),
                        cache_hit=True,
                        cache_type="exact",
                        latency_ms=latency_ms,
                        sources=cached_sources,
                        citations=cached_sources,
                        retrieved_context=cached_data.get("retrieved_context", cached_sources),
                        citation_validation_passed=True,
                        knowledge_base_version=kb_ver,
                        model_used=cached_data.get("model_used", model_name),
                        is_supported=cached_data.get("is_supported", True),
                        rerank_used=False,
                    )
            except RedisUnavailable:
                cache_degraded = True
            except Exception as e:
                logger.warning(f"[Pipeline] Cache lookup failed ({e}). Proceeding to normal RAG processing.")

        with generation_slot(self._generation_slots):
            if hybrid:
                candidates = hybrid.search(
                    query=clean_q,
                    candidate_pool_size=top_k_retrieve,
                    final_top_k=top_k_retrieve,
                    metadata_filters=metadata_filters,
                )
            else:
                bm25_res = filter_results(bm25.search(clean_q, top_k=top_k_retrieve), metadata_filters)
                candidates = [{"chunk": c, "rrf_score": s} for c, s in bm25_res]

            if use_reranker and self.reranker:
                ranked = self.reranker.rerank(
                    query=clean_q,
                    candidates=candidates,
                    top_n=None,
                )
                reranked_passages = select_passages(candidates, ranked, top_k_final)
            else:
                reranked_passages = candidates[:top_k_final]

            gen_result: CustomerSupportAnswer = self.generator.generate(
                question=clean_q,
                retrieved_passages=reranked_passages,
            )

            effective_sources = gen_result.citations
            answer = gen_result.answer
            supported = (
                gen_result.is_supported
                and bool(answer)
                and not gen_result.error
                and bool(effective_sources)
                and gen_result.citation_validation_passed
            )
            cache_ttl = None
            rerank_used = bool(reranked_passages) and all("rerank_score" in p for p in reranked_passages)
            if supported and not skip_cache and (not use_reranker or rerank_used):
                try:
                    stored = self.cache.set_answer(
                        query=clean_q,
                        kb_version=kb_ver,
                        model=cache_model,
                        answer=answer,
                        sources=effective_sources,
                        ttl=cache_ttl,
                        model_used=gen_result.model,
                        retrieved_context=gen_result.retrieved_context,
                    )
                    if stored:
                        logger.debug("[Pipeline] Cached cited answer.")
                except RedisUnavailable:
                    cache_degraded = True
                except Exception as e:
                    logger.warning(f"[Pipeline] Failed to cache answer in Redis: {e}")

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self.cache.record_miss_latency(latency_ms)
            return SupportRAGResponse(
                question=clean_q,
                answer=answer,
                cache_hit=False,
                cache_type="none",
                latency_ms=latency_ms,
                sources=effective_sources,
                citations=effective_sources,
                retrieved_context=gen_result.retrieved_context,
                citation_validation_passed=gen_result.citation_validation_passed,
                knowledge_base_version=kb_ver,
                model_used=gen_result.model,
                is_supported=gen_result.is_supported,
                rerank_used=rerank_used,
                error=gen_result.error,
                refusal_reason=gen_result.refusal_reason,
                cache_degraded=cache_degraded,
            )
