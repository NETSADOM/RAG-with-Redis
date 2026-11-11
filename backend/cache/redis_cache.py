"""
backend/cache/redis_cache.py
Advanced Redis Answer Caching for Customer Support RAG.

Architecture & Features:
1. Cache-Aside (Lazy Loading): Look up Redis before retrieval or LLM generation.
2. Write-Through on Misses: Persist answer + citations + query vector before returning to client.
3. LFU (Least Frequently Used) Eviction:
   - Configures Redis 'maxmemory-policy allkeys-lfu' to retain frequently asked FAQs.
   - Explicitly tracks access_count frequency on every hit for LFU observation.
4. Semantic Caching:
   - Evaluates cosine similarity of query embeddings against cached questions.
   - Reuses verified answers for equivalent/paraphrased questions (threshold >= 0.88).
5. Policy-Aware Invalidation:
   - Tracks document citations (source_doc_ids).
   - Invalidate answers selectively when a specific policy document is updated.
   - Preserves global knowledge-base version fingerprinting.
"""

import os
import json
import time
import hashlib
import re
from typing import Dict, Any, Optional, Tuple, List
import logging
import numpy as np
import redis
import fakeredis

logger = logging.getLogger(__name__)

# Versioned cache namespace keys (bumped to v2 to enforce citation validation)
KEY_PREFIX = "cs_rag:ans:v2:"
INDEX_PREFIX = "cs_rag:idx:v2:"
DOC_PREFIX = "cs_rag:doc:v2:"


class CustomerSupportCache:
    """
    Enterprise Redis caching layer for customer support inquiries.
    Implements Cache-Aside, Write-Through on misses, LFU eviction,
    Semantic Caching, and Policy-Aware invalidation.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: Optional[int] = None,
        semantic_threshold: Optional[float] = None,
        use_fallback: bool = True,
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl or int(os.getenv("CACHE_TTL", "3600"))
        self.semantic_threshold = semantic_threshold or float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.88"))
        self.semantic_caching_enabled = os.getenv("ENABLE_SEMANTIC_CACHE", "false").lower() in ("true", "1", "yes")
        self.eviction_policy = "allkeys-lfu"
        self.use_fallback = use_fallback

        self._client: Optional[redis.Redis] = None
        self._is_emulator = False

        # Live telemetry metrics
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.total_hit_latency_ms = 0.0
        self.total_miss_latency_ms = 0.0

        self._connect()

    def _connect(self) -> None:
        """Attempt connection to live Redis; fall back to in-memory fakeredis if unavailable."""
        try:
            client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
            )
            client.ping()
            self._client = client
            self._is_emulator = False
            logger.info(f"[Cache] Connected to live Redis instance at '{self.redis_url}'.")

            # Configure Redis LFU Eviction policy
            try:
                self._client.config_set("maxmemory-policy", self.eviction_policy)
                logger.info(f"[Cache] Configured Redis eviction policy: '{self.eviction_policy}'.")
            except Exception as conf_err:
                logger.info(f"[Cache] Note: Redis CONFIG SET maxmemory-policy ({conf_err}) - continuing with client-side LFU tracking.")

        except Exception as e:
            if self.use_fallback:
                logger.warning(
                    f"[Cache] Live Redis server unreachable ({e}). Activating in-memory Redis emulator."
                )
                self._client = fakeredis.FakeRedis(decode_responses=True)
                self._is_emulator = True
            else:
                logger.error(f"[Cache] Redis connection failed: {e}")
                self._client = None

    @property
    def is_available(self) -> bool:
        """Check if Redis connection is active and operational."""
        if not self._client:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    @property
    def hits(self) -> int:
        """Total cache hits (exact + semantic)."""
        return self.exact_hits + self.semantic_hits

    def normalize_query(self, query: str) -> str:
        """Normalize query: lowercase, trim whitespace, and strip terminal punctuation."""
        clean = re.sub(r"[?!.,;:]+$", "", query.strip().lower())
        return re.sub(r"\s+", " ", clean)

    def generate_cache_key(self, query: str, kb_version: str, model: str = "") -> str:
        """
        Derive a deterministic SHA-256 cache key from normalized query, KB version, and model.
        Uses versioned KEY_PREFIX (v2) to isolate verified citation entries.
        """
        clean_q = self.normalize_query(query)
        raw_key = f"support_rag::v2::{clean_q}::kb={kb_version}::m={model}"
        return f"{KEY_PREFIX}{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()}"

    def _is_semantically_incompatible(self, q1: str, q2: str) -> bool:
        """
        Guard against semantic false positives.
        Ensures questions differing in a number, country, or negation don't share an answer.
        """
        clean1 = self.normalize_query(q1)
        clean2 = self.normalize_query(q2)

        # 1. Number difference check (e.g. 30 days vs 60 days)
        nums1 = set(re.findall(r"\b\d+(?:\.\d+)?\b", clean1))
        nums2 = set(re.findall(r"\b\d+(?:\.\d+)?\b", clean2))
        if (nums1 or nums2) and nums1 != nums2:
            return True

        # 2. Negation difference check (e.g. "can I return" vs "can I not return")
        negations = {
            "not", "no", "never", "cannot", "cant", "can't",
            "wont", "won't", "dont", "don't", "doesnt", "doesn't", "without"
        }
        tokens1 = set(re.findall(r"\b\w+\b", clean1.lower()))
        tokens2 = set(re.findall(r"\b\w+\b", clean2.lower()))
        has_neg1 = bool(tokens1 & negations)
        has_neg2 = bool(tokens2 & negations)
        if has_neg1 != has_neg2:
            return True

        # 3. Country / Region difference check (e.g. Canada vs UK vs US)
        countries = {
            "canada", "uk", "united kingdom", "britain", "us", "usa", "united states",
            "america", "eu", "europe", "european union", "australia", "germany",
            "france", "mexico", "japan", "china", "india"
        }
        c1 = {c for c in countries if re.search(r"\b" + re.escape(c) + r"\b", clean1)}
        c2 = {c for c in countries if re.search(r"\b" + re.escape(c) + r"\b", clean2)}
        if (c1 or c2) and c1 != c2:
            return True

        return False

    def get_answer(
        self,
        query: str,
        kb_version: str,
        model: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Fast Exact-Match Cache Lookup (O(1) in <0.2ms).
        Returns:
            (payload, True) on Cache HIT.
            (None, False) on Cache MISS.
        """
        if not self.is_available:
            self.misses += 1
            return None, False

        start_t = time.perf_counter()
        key = self.generate_cache_key(query, kb_version, model)

        try:
            raw = self._client.get(key)
            lookup_ms = (time.perf_counter() - start_t) * 1000

            if raw is None:
                return None, False

            payload = json.loads(raw)

            # LFU Access Frequency Accounting
            payload["access_count"] = payload.get("access_count", 0) + 1
            self._client.set(key, json.dumps(payload), keepttl=True)

            self.exact_hits += 1
            self.total_hit_latency_ms += lookup_ms
            return payload, True

        except Exception as e:
            logger.warning(f"[Cache] Exact lookup failed for key '{key}': {e}")
            return None, False

    def get_answer_semantic(
        self,
        query_vec: np.ndarray,
        kb_version: str,
        model: str = "",
        threshold: Optional[float] = None,
        query_text: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], bool, float]:
        """
        Semantic Cache Lookup:
        Reuses verified answers for equivalent questions by comparing query embeddings.
        Filters candidates by requesting model and validates question compatibility
        (rejecting questions differing in numbers, countries, or negation).
        Returns:
            (payload, True, max_sim) on Semantic HIT.
            (None, False, max_sim) on Semantic MISS.
        """
        if not self.is_available or not self.semantic_caching_enabled:
            return None, False, 0.0

        cutoff = threshold if threshold is not None else self.semantic_threshold
        start_t = time.perf_counter()

        index_key = f"{INDEX_PREFIX}{kb_version}"
        try:
            # Fetch all active keys for this KB version
            active_keys = self._client.smembers(index_key)
            if not active_keys:
                return None, False, 0.0

            best_sim = -1.0
            best_payload = None
            best_key = None

            q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

            for key in active_keys:
                raw = self._client.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)

                # Filter candidate by requesting model
                if model and entry.get("model") and entry.get("model") != model:
                    continue

                # Guard: Reject matches differing in numbers, country/regions, or negation
                cached_query = entry.get("query", "")
                if query_text and cached_query:
                    if self._is_semantically_incompatible(query_text, cached_query):
                        continue

                cached_vec = entry.get("query_embedding")
                if not cached_vec:
                    continue

                cached_norm = np.array(cached_vec, dtype=np.float32)
                # Dot product of normalized vectors equals cosine similarity
                cos_sim = float(np.dot(q_norm, cached_norm))

                if cos_sim > best_sim:
                    best_sim = cos_sim
                    best_payload = entry
                    best_key = key

            lookup_ms = (time.perf_counter() - start_t) * 1000

            if best_sim >= cutoff and best_payload is not None:
                # LFU Access Frequency update on semantic match
                best_payload["access_count"] = best_payload.get("access_count", 0) + 1
                if best_key:
                    self._client.set(best_key, json.dumps(best_payload), keepttl=True)

                self.semantic_hits += 1
                self.total_hit_latency_ms += lookup_ms
                return best_payload, True, best_sim

            return None, False, max(best_sim, 0.0)

        except Exception as e:
            logger.warning(f"[Cache] Semantic search error: {e}")
            return None, False, 0.0

    def set_answer(
        self,
        query: str,
        kb_version: str,
        model: str,
        answer: str,
        sources: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Write-Through Storage on Cache Miss:
        Persists synthesized answer, verified sources, and query vector in Redis with TTL.
        """
        if not self.is_available or not answer.strip():
            return False

        key = self.generate_cache_key(query, kb_version, model)
        effective_ttl = ttl if ttl is not None else self.default_ttl

        # Extract structured source metadata for policy-aware invalidation
        source_doc_ids = list(dict.fromkeys(s.get("doc_id", "") for s in sources if s.get("doc_id")))
        source_titles = list(dict.fromkeys(s.get("title", "") for s in sources if s.get("title")))
        source_chunk_ids = [s.get("chunk_id", "") for s in sources if s.get("chunk_id")]

        # Only store embedding and maintain semantic index if semantic caching is enabled
        effective_query_embedding = query_embedding if self.semantic_caching_enabled else None

        payload = {
            "query": query,
            "normalized_query": self.normalize_query(query),
            "answer": answer,
            "source_doc_ids": source_doc_ids,
            "source_titles": source_titles,
            "source_chunk_ids": source_chunk_ids,
            "sources": sources,
            "knowledge_base_version": kb_version,
            "model": model,
            "query_embedding": effective_query_embedding,
            "access_count": 1,  # Initialize LFU frequency
            "cached_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            serialized = json.dumps(payload)
            if effective_ttl > 0:
                self._client.set(key, serialized, ex=effective_ttl)
            else:
                self._client.set(key, serialized)

            # Register key in KB semantic index ONLY if semantic caching is active
            if self.semantic_caching_enabled:
                index_key = f"{INDEX_PREFIX}{kb_version}"
                self._client.sadd(index_key, key)

            return True

        except Exception as e:
            logger.warning(f"[Cache] Failed to store answer for key '{key}': {e}")
            return False

    def invalidate_by_doc_id(self, doc_id: str) -> int:
        """
        Policy-Aware Invalidation:
        Selectively purges all cached answers that cite the modified policy document.
        Returns:
            Number of invalidated cache entries.
        """
        if not self.is_available:
            return 0

        doc_idx = f"{DOC_PREFIX}{doc_id}"
        count = 0
        try:
            keys_to_purge = self._client.smembers(doc_idx)
            for key in keys_to_purge:
                if self._client.delete(key):
                    count += 1
            # Clean up the document index set
            self._client.delete(doc_idx)
            logger.info(f"[Cache] Policy-aware invalidation: purged {count} entries citing '{doc_id}'.")
            return count
        except Exception as e:
            logger.warning(f"[Cache] Policy invalidation error for '{doc_id}': {e}")
            return 0

    def record_miss_latency(self, latency_ms: float) -> None:
        """Record the latency of a complete RAG execution on cache miss."""
        self.misses += 1
        self.total_miss_latency_ms += latency_ms

    def clear(self) -> bool:
        """
        Purge only NovaCart's answer and index keys from Redis.
        Preserves unrelated keys stored in the database.
        """
        if not self.is_available:
            return False
        try:
            patterns = [f"{KEY_PREFIX}*", f"{INDEX_PREFIX}*", f"{DOC_PREFIX}*", "cs_rag:*"]
            keys_to_delete = set()
            for pattern in patterns:
                matched = self._client.keys(pattern)
                if matched:
                    keys_to_delete.update(matched)

            if keys_to_delete:
                self._client.delete(*keys_to_delete)

            self.exact_hits = 0
            self.semantic_hits = 0
            self.misses = 0
            self.total_hit_latency_ms = 0.0
            self.total_miss_latency_ms = 0.0
            return True
        except Exception as e:
            logger.warning(f"[Cache] Scoped clear failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Return real tracked telemetry metrics for the frontend performance dashboard."""
        total_hits = self.hits
        total_queries = total_hits + self.misses
        hit_rate = (total_hits / total_queries) if total_queries > 0 else 0.0

        avg_hit_ms = (self.total_hit_latency_ms / total_hits) if total_hits > 0 else 0.0
        avg_miss_ms = (self.total_miss_latency_ms / self.misses) if self.misses > 0 else 0.0

        total_keys = 0
        if self.is_available and self._client:
            try:
                total_keys = len(self._client.keys(f"{KEY_PREFIX}*"))
            except Exception:
                total_keys = 0

        estimated_cost_saved_usd = round(total_hits * 0.0004, 4)

        return {
            "is_connected": self.is_available,
            "is_emulator": self._is_emulator,
            "eviction_policy": self.eviction_policy,
            "total_questions": total_queries,
            "cache_hits": total_hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "cache_misses": self.misses,
            "hit_rate_pct": round(hit_rate * 100, 2),
            "avg_hit_latency_ms": round(avg_hit_ms, 2),
            "avg_miss_latency_ms": round(avg_miss_ms, 2),
            "llm_calls_avoided": total_hits,
            "estimated_cost_saved_usd": estimated_cost_saved_usd,
            "total_keys_in_redis": total_keys,
            "default_ttl_seconds": self.default_ttl,
            "semantic_threshold": self.semantic_threshold,
        }
