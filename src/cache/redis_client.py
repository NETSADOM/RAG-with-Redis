"""
src/cache/redis_client.py
Redis connection manager supporting both live Redis instances and
local in-memory emulation (via fakeredis) with graceful degradation.
"""

from typing import Optional, Dict, Any
import logging
import redis
import fakeredis

from src.config import config

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """
    Manages the Redis client lifecycle, health checks, and fallback mechanisms.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        use_fallback: Optional[bool] = None,
        eviction_policy: Optional[str] = None,
    ):
        self.redis_url = redis_url or config.redis_url
        self.use_fallback = use_fallback if use_fallback is not None else config.redis_use_fallback
        self.target_eviction_policy = (eviction_policy or config.redis_eviction_policy).lower()
        self._client: Optional[redis.Redis] = None
        self._is_emulator = False
        self._connect()

    def _connect(self) -> None:
        """Attempt connection to live Redis; fall back to fakeredis if enabled."""
        if not config.redis_enabled:
            logger.info("[RedisManager] Redis is disabled in configuration.")
            self._client = None
            return

        try:
            # Test live connection with 1-second timeout
            client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
            )
            client.ping()
            self._client = client
            self._is_emulator = False
            self._configure_eviction_policy(self.target_eviction_policy)
            logger.info(f"[RedisManager] Connected to live Redis at '{self.redis_url}'.")
        except Exception as e:
            if self.use_fallback:
                logger.warning(
                    f"[RedisManager] Live Redis unreachable ({e}). Activating in-memory Redis emulator."
                )
                self._client = fakeredis.FakeRedis(decode_responses=True)
                self._is_emulator = True
                self._configure_eviction_policy(self.target_eviction_policy)
            else:
                logger.error(f"[RedisManager] Live Redis connection failed: {e}")
                self._client = None

    def _configure_eviction_policy(self, policy: str) -> None:
        """Attempt to configure Redis maxmemory-policy."""
        if not self._client:
            return
        valid_policies = {
            "allkeys-lru", "allkeys-lfu", "volatile-lru", "volatile-lfu",
            "allkeys-random", "volatile-random", "volatile-ttl", "noeviction"
        }
        if policy not in valid_policies:
            policy = "allkeys-lru"

        try:
            if not self._is_emulator:
                self._client.config_set("maxmemory-policy", policy)
            self.target_eviction_policy = policy
        except Exception as e:
            logger.warning(f"[RedisManager] Could not set maxmemory-policy to {policy}: {e}")

    def set_eviction_policy(self, policy: str) -> bool:
        """Dynamically update the eviction policy for live Redis or emulator tracking."""
        self._configure_eviction_policy(policy)
        return True

    @property
    def client(self) -> Optional[redis.Redis]:
        """Returns the active redis.Redis (or FakeRedis) instance."""
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if Redis connection is currently operational."""
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    @property
    def is_emulator(self) -> bool:
        """Whether the active client is an in-memory emulator."""
        return self._is_emulator

    def get_info(self) -> Dict[str, Any]:
        """Retrieve operational stats, memory usage, and key counts."""
        if not self.is_connected or not self._client:
            return {
                "connected": False,
                "emulator": False,
                "memory_used_bytes": 0,
                "total_keys": 0,
                "eviction_policy": self.target_eviction_policy,
            }

        try:
            info = self._client.info()
            total_keys = self._client.dbsize()
            memory_bytes = info.get("used_memory", 0)
            evictions = info.get("evicted_keys", 0)
            return {
                "connected": True,
                "emulator": self._is_emulator,
                "redis_version": info.get("redis_version", "emulator-2.40"),
                "memory_used_bytes": memory_bytes,
                "memory_human": info.get("used_memory_human", f"{memory_bytes}B"),
                "total_keys": total_keys,
                "eviction_policy": self.target_eviction_policy,
                "evicted_keys": evictions,
            }
        except Exception:
            # For emulator fallback
            try:
                keys = self._client.keys("*")
                return {
                    "connected": True,
                    "emulator": True,
                    "memory_used_bytes": len(keys) * 1024,
                    "total_keys": len(keys),
                    "eviction_policy": self.target_eviction_policy,
                    "evicted_keys": 0,
                }
            except Exception:
                return {"connected": False, "emulator": False, "total_keys": 0}

    def close(self) -> None:
        """Close active connection pool."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
