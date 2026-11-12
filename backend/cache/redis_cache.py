import os
import json
import time
import hashlib
import re
from typing import Dict, Any, Optional, Tuple, List
import logging
import redis
from redis.backoff import NoBackoff
from redis.retry import Retry
from threading import RLock
from backend.locking import synchronized

logger = logging.getLogger(__name__)

# Answer schema namespace; Redis manages memory-pressure eviction.
KEY_PREFIX = "cs_rag:ans:v3:"


class RedisUnavailable(RuntimeError):
    """Redis is required at startup; runtime callers may serve uncached reads."""


class CustomerSupportCache:
    """Cache successful cited answers; revisions and TTL control freshness."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: Optional[int] = None,
        namespace: str = "cs_rag",
    ):
        if not re.fullmatch(r"[a-zA-Z0-9_:-]+", namespace):
            raise ValueError("Invalid cache namespace.")
        self.namespace = namespace
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl if default_ttl is not None else int(os.getenv("CACHE_TTL", "86400"))
        if self.default_ttl <= 0:
            raise ValueError("CACHE_TTL must be positive.")
        self.eviction_policy = "server-managed"
        self._lock = RLock()
        self._retry_after = 0.0

        self._client: Optional[redis.Redis] = None

        # Live telemetry metrics
        self.exact_hits = 0
        self.misses = 0
        self.total_hit_latency_ms = 0.0
        self.total_miss_latency_ms = 0.0

        self._connect()

    def _connect(self) -> None:
        """Connect to local or hosted Redis. rediss:// enables verified TLS."""
        client = None
        try:
            client = redis.Redis.from_url(
                self.redis_url, socket_connect_timeout=1, socket_timeout=1,
                decode_responses=True, max_connections=10,
                retry=Retry(NoBackoff(), 0),
            )
            try:
                client.ping()
            except redis.AuthenticationError:
                raise
            except (redis.ConnectionError, redis.TimeoutError):
                logger.warning("Redis connection interrupted at startup. Retrying once.")
                client.ping()
        except (redis.RedisError, ValueError):
            if client is not None:
                client.close()
            raise RedisUnavailable("Redis is required. Check REDIS_URL and database availability in your local .env file.") from None
        self._client = client
        logger.info("Connected to Redis.")

    def require_available(self):
        if not self.is_available:
            raise RedisUnavailable("Redis is temporarily unavailable. Please try again.")

    def _redis_call(self, command, *args, **kwargs):
        """After failure, skip Redis for 30 seconds; allow only one recovery probe."""
        with self._lock:
            now = time.monotonic()
            if now < self._retry_after:
                raise RedisUnavailable("Redis is temporarily unavailable.")
            recovering = self._retry_after > 0
            if recovering:
                self._retry_after = now + 30
        try:
            if self._client is None:
                raise redis.ConnectionError()
            result = getattr(self._client, command)(*args, **kwargs)
        except redis.RedisError:
            with self._lock:
                self._retry_after = time.monotonic() + 30
            raise RedisUnavailable("Redis is temporarily unavailable.") from None
        if recovering:
            with self._lock:
                self._retry_after = 0.0
            logger.info("Redis recovered; caching resumed.")
        return result

    def close(self):
        if self._client is not None:
            self._client.close()

    @property
    def is_available(self) -> bool:
        """Check if Redis connection is active and operational."""
        if not self._client:
            return False
        try:
            return bool(self._redis_call("ping"))
        except Exception:
            return False

    def normalize_query(self, query: str) -> str:
        """Normalize query: lowercase, trim whitespace, and strip terminal punctuation."""
        clean = re.sub(r"[?!.,;:]+$", "", query.strip().lower())
        return re.sub(r"\s+", " ", clean)

    def generate_cache_key(self, query: str, kb_version: str, model: str = "") -> str:
        """
        Derive a deterministic SHA-256 cache key from normalized query, KB version, and model.
        Includes a schema namespace to isolate incompatible entries.
        """
        clean_q = self.normalize_query(query)
        raw_key = f"support_rag::v3::{clean_q}::kb={kb_version}::m={model}"
        return f"{self.namespace}:ans:v3:{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()}"

    def get_answer(
        self,
        query: str,
        kb_version: str,
        model: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        'Exact-match lookup without rewriting the entry or extending its TTL.'
        if self._client is None:
            raise RedisUnavailable("Redis is unavailable.")

        start_t = time.perf_counter()
        key = self.generate_cache_key(query, kb_version, model)

        try:
            raw = self._redis_call("get", key)
            lookup_ms = (time.perf_counter() - start_t) * 1000

            if raw is None:
                return None, False

            payload = json.loads(raw)

            with self._lock:
                self.exact_hits += 1
                self.total_hit_latency_ms += lookup_ms
            return payload, True

        except RedisUnavailable:
            raise RedisUnavailable("Redis lookup failed. Please try again.") from None
        except Exception as e:
            logger.warning(f"[Cache] Exact lookup failed for key '{key}': {e}")
            return None, False

    def set_answer(
        self,
        query: str,
        kb_version: str,
        model: str,
        answer: str,
        sources: List[Dict[str, Any]],
        ttl: Optional[int] = None,
        model_used: Optional[str] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Write-Through Storage on Cache Miss:
        Store an answer and its citations with a TTL.
        """
        if self._client is None:
            raise RedisUnavailable("Redis is unavailable.")
        if not answer.strip():
            return False

        key = self.generate_cache_key(query, kb_version, model)
        effective_ttl = ttl if ttl is not None else self.default_ttl
        if effective_ttl <= 0:
            raise ValueError("Cache TTL must be positive.")

        payload = {
            "answer": answer,
            "sources": sources,
            "model_used": model_used or model,
            "retrieved_context": retrieved_context or [],
        }

        try:
            serialized = json.dumps(payload)
            self._redis_call("set", key, serialized, ex=effective_ttl)

            return True

        except RedisUnavailable:
            raise RedisUnavailable("Redis write failed. Please try again.") from None
        except Exception as e:
            logger.warning(f"[Cache] Failed to store answer for key '{key}': {e}")
            return False

    @synchronized
    def record_miss_latency(self, latency_ms: float) -> None:
        """Record the latency of a complete RAG execution on cache miss."""
        self.misses += 1
        self.total_miss_latency_ms += latency_ms

    def set_knowledge_base(self, revision: str, snapshot: Dict[str, Any]) -> None:
        """Synchronously cache an immutable policy revision before activation."""
        if self._client is None:
            raise RedisUnavailable("Redis is unavailable.")
        serialized = json.dumps(snapshot)
        try:
            self._redis_call("set", f"{self.namespace}:kb:v1:{revision}", serialized, ex=self.default_ttl)
        except RedisUnavailable:
            raise RedisUnavailable("Policy cache write failed. Please try again.") from None

    def get_knowledge_base(self, revision: str) -> Optional[Dict[str, Any]]:
        """Read a revision snapshot, returning None after expiration or eviction."""
        if self._client is None:
            raise RedisUnavailable("Redis is unavailable.")
        try:
            raw = self._redis_call("get", f"{self.namespace}:kb:v1:{revision}")
        except RedisUnavailable:
            raise RedisUnavailable("Policy cache lookup failed. Please try again.") from None
        return json.loads(raw) if raw is not None else None

    @synchronized
    def clear(self) -> bool:
        """
        Purge only NovaCart's answer and index keys from Redis.
        Preserves unrelated keys stored in the database.
        """
        if not self.is_available:
            return False
        try:
            for key in self._client.scan_iter(match=f"{self.namespace}:*", count=100):
                self._client.delete(key)

            self.exact_hits = 0
            self.misses = 0
            self.total_hit_latency_ms = 0.0
            self.total_miss_latency_ms = 0.0
            return True
        except Exception as e:
            logger.warning(f"[Cache] Scoped clear failed: {e}")
            return False

    @synchronized
    def get_metrics(self) -> Dict[str, Any]:
        """Return real tracked telemetry metrics for the frontend performance dashboard."""
        total_hits = self.exact_hits
        total_queries = total_hits + self.misses
        hit_rate = (total_hits / total_queries) if total_queries > 0 else 0.0

        avg_hit_ms = (self.total_hit_latency_ms / total_hits) if total_hits > 0 else 0.0
        avg_miss_ms = (self.total_miss_latency_ms / self.misses) if self.misses > 0 else 0.0

        return {
            "is_connected": self.is_available,
            "eviction_policy": self.eviction_policy,
            "total_questions": total_queries,
            "cache_hits": total_hits,
            "exact_hits": self.exact_hits,
            "cache_misses": self.misses,
            "hit_rate_pct": round(hit_rate * 100, 2),
            "avg_hit_latency_ms": round(avg_hit_ms, 2),
            "avg_miss_latency_ms": round(avg_miss_ms, 2),
            "generation_runs_avoided": total_hits,
            "default_ttl_seconds": self.default_ttl,
        }
