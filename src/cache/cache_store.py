"""
src/cache/cache_store.py
Standardized, robust key-value cache store with JSON serialization,
TTL support, pattern-based invalidation, and hit/miss telemetry.
"""

from typing import Optional, Dict, Any, List
import json
import time
import logging

from src.config import config
from src.cache.redis_client import RedisConnectionManager

logger = logging.getLogger(__name__)


class CacheStore:
    """
    High-level key-value caching interface over Redis.
    Preserves answer text, source document citations, and retrieval passages.
    """

    def __init__(
        self,
        connection_manager: Optional[RedisConnectionManager] = None,
        default_ttl: Optional[int] = None,
    ):
        self.manager = connection_manager or RedisConnectionManager()
        self.default_ttl = default_ttl if default_ttl is not None else config.redis_ttl
        
        # In-memory telemetry counters
        self.hits = 0
        self.misses = 0

    @property
    def client(self):
        return self.manager.client

    @property
    def is_available(self) -> bool:
        return self.manager.is_connected

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and deserialize cached payload by key.
        Returns None on cache miss or connection failure.
        """
        if not self.is_available or not self.client:
            self.misses += 1
            return None

        try:
            raw_val = self.client.get(key)
            if raw_val is None:
                self.misses += 1
                return None

            payload = json.loads(raw_val)
            self.hits += 1
            return payload
        except Exception as e:
            logger.warning(f"[CacheStore] Failed to read key '{key}': {e}")
            self.misses += 1
            return None

    def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Serialize and store payload with expiration TTL (in seconds).
        """
        if not self.is_available or not self.client:
            return False

        effective_ttl = ttl if ttl is not None else self.default_ttl
        try:
            # Handle objects with to_dict or standard dict
            if hasattr(data, "to_dict"):
                payload = data.to_dict()
            elif isinstance(data, dict):
                payload = dict(data)
            else:
                payload = data

            # Enforce metadata timestamp
            if isinstance(payload, dict) and "cached_at" not in payload:
                payload["cached_at"] = time.time()

            serialized = json.dumps(payload)
            if effective_ttl and effective_ttl > 0:
                return bool(self.client.set(key, serialized, ex=effective_ttl))
            else:
                return bool(self.client.set(key, serialized))
        except Exception as e:
            logger.warning(f"[CacheStore] Failed to store key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete an individual key from the cache."""
        if not self.is_available or not self.client:
            return False
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.warning(f"[CacheStore] Failed to delete key '{key}': {e}")
            return False

    def delete_by_pattern(self, pattern: str) -> int:
        """
        Scan and delete all keys matching a wildcard pattern (e.g. '*::ver=abc*').
        """
        if not self.is_available or not self.client:
            return 0
        try:
            keys = self.client.keys(pattern)
            if keys:
                return int(self.client.delete(*keys))
            return 0
        except Exception as e:
            logger.warning(f"[CacheStore] Pattern deletion failed for '{pattern}': {e}")
            return 0

    def ttl(self, key: str) -> int:
        """Return remaining time-to-live in seconds, or -2 if missing, -1 if no TTL."""
        if not self.is_available or not self.client:
            return -2
        try:
            return int(self.client.ttl(key))
        except Exception:
            return -2

    def exists(self, key: str) -> bool:
        """Check whether a key exists in cache."""
        if not self.is_available or not self.client:
            return False
        try:
            return bool(self.client.exists(key))
        except Exception:
            return False

    def clear(self) -> bool:
        """Flush all keys in the current database."""
        if not self.is_available or not self.client:
            return False
        try:
            self.client.flushdb()
            return True
        except Exception as e:
            logger.warning(f"[CacheStore] Flush failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Return cache health, hit rate, and memory metrics."""
        total_queries = self.hits + self.misses
        hit_rate = (self.hits / total_queries) if total_queries > 0 else 0.0

        info = self.manager.get_info()
        return {
            "is_connected": self.is_available,
            "is_emulator": self.manager.is_emulator,
            "hits": self.hits,
            "misses": self.misses,
            "total_reads": total_queries,
            "hit_rate_pct": round(hit_rate * 100, 2),
            "total_keys": info.get("total_keys", 0),
            "memory_used_bytes": info.get("memory_used_bytes", 0),
            "eviction_policy": info.get("eviction_policy", config.redis_eviction_policy),
            "evicted_keys": info.get("evicted_keys", 0),
        }

    def reset_metrics(self) -> None:
        """Reset in-memory hit/miss counters."""
        self.hits = 0
        self.misses = 0
