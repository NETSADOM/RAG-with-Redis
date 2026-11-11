"""
backend/cache/redis_cache.py
Dedicated Redis answer caching module with exact-match SHA-256 keys,
knowledge-base-version invalidation, source preservation, and metrics tracking.
"""

import os
import json
import time
import hashlib
import re
from typing import Dict, Any, Optional, Tuple, List
import logging
import redis
import fakeredis

logger = logging.getLogger(__name__)


class CustomerSupportCache:
    """
    Manages Redis caching for customer support question answering.
    Features:
    - Knowledge-base version awareness: keys include KB hash so updates invalidate immediately.
    - Preserves complete source citations (doc IDs, titles, chunk IDs).
    - Robust fallback to in-memory FakeRedis if live Redis is unreachable.
    - Detailed latency and hit-rate telemetry for real-time performance evaluation.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: Optional[int] = None,
        use_fallback: bool = True,
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl or int(os.getenv("CACHE_TTL", "3600"))
        self.use_fallback = use_fallback

        self._client: Optional[redis.Redis] = None
        self._is_emulator = False

        # Live telemetry counters
        self.hits = 0
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

    def generate_cache_key(self, query: str, kb_version: str, model: str = "") -> str:
        """
        Derive a deterministic SHA-256 cache key from normalized query, KB version, and model.
        Changing the knowledge base version immediately changes the key, preventing stale answers.
        """
        # Normalize: strip leading/trailing spaces, lowercase, collapse whitespace, remove terminal punctuation
        clean_q = re.sub(r"[?!.,;]+$", "", query.strip().lower())
        clean_q = re.sub(r"\s+", " ", clean_q)
        raw_key = f"support_rag::{clean_q}::kb={kb_version}::m={model}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get_answer(
        self,
        query: str,
        kb_version: str,
        model: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Check Redis before running RAG retrieval or LLM generation.

        Returns:
            (cached_data_dict, True) on Cache HIT.
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
                self.misses += 1
                return None, False

            payload = json.loads(raw)
            self.hits += 1
            self.total_hit_latency_ms += lookup_ms
            return payload, True

        except Exception as e:
            logger.warning(f"[Cache] Lookup failed for key '{key}': {e}")
            self.misses += 1
            return None, False

    def set_answer(
        self,
        query: str,
        kb_version: str,
        model: str,
        answer: str,
        sources: List[Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store a successful, supported answer and its source references in Redis with TTL.
        """
        if not self.is_available or not answer.strip():
            return False

        key = self.generate_cache_key(query, kb_version, model)
        effective_ttl = ttl if ttl is not None else self.default_ttl

        # Extract structured metadata fields required by specification
        source_doc_ids = list(dict.fromkeys(s.get("doc_id", "") for s in sources if s.get("doc_id")))
        source_titles = list(dict.fromkeys(s.get("title", "") for s in sources if s.get("title")))
        source_chunk_ids = [s.get("chunk_id", "") for s in sources if s.get("chunk_id")]

        payload = {
            "query": query,
            "answer": answer,
            "source_doc_ids": source_doc_ids,
            "source_titles": source_titles,
            "source_chunk_ids": source_chunk_ids,
            "sources": sources,
            "knowledge_base_version": kb_version,
            "model": model,
            "cached_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        try:
            serialized = json.dumps(payload)
            if effective_ttl > 0:
                return bool(self._client.set(key, serialized, ex=effective_ttl))
            return bool(self._client.set(key, serialized))
        except Exception as e:
            logger.warning(f"[Cache] Failed to store answer for key '{key}': {e}")
            return False

    def clear(self) -> bool:
        """Purge all cached keys from Redis."""
        if not self.is_available:
            return False
        try:
            self._client.flushdb()
            self.hits = 0
            self.misses = 0
            self.total_hit_latency_ms = 0.0
            self.total_miss_latency_ms = 0.0
            return True
        except Exception as e:
            logger.warning(f"[Cache] Flush failed: {e}")
            return False

    def record_miss_latency(self, latency_ms: float) -> None:
        """Record the latency of a complete RAG execution on cache miss."""
        self.total_miss_latency_ms += latency_ms

    def get_metrics(self) -> Dict[str, Any]:
        """Return real tracked telemetry metrics for the frontend performance dashboard."""
        total_queries = self.hits + self.misses
        hit_rate = (self.hits / total_queries) if total_queries > 0 else 0.0

        avg_hit_ms = (self.total_hit_latency_ms / self.hits) if self.hits > 0 else 0.0
        avg_miss_ms = (self.total_miss_latency_ms / self.misses) if self.misses > 0 else 0.0

        total_keys = 0
        if self.is_available and self._client:
            try:
                total_keys = self._client.dbsize()
            except Exception:
                try:
                    total_keys = len(self._client.keys("*"))
                except Exception:
                    total_keys = 0

        # Estimated cost savings: assuming average $0.0004 per LLM generation call avoided
        estimated_cost_saved_usd = round(self.hits * 0.0004, 4)

        return {
            "is_connected": self.is_available,
            "is_emulator": self._is_emulator,
            "total_questions": total_queries,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate_pct": round(hit_rate * 100, 2),
            "avg_hit_latency_ms": round(avg_hit_ms, 2),
            "avg_miss_latency_ms": round(avg_miss_ms, 2),
            "llm_calls_avoided": self.hits,
            "estimated_cost_saved_usd": estimated_cost_saved_usd,
            "total_keys_in_redis": total_keys,
            "default_ttl_seconds": self.default_ttl,
        }

