"""
src/cache
Modular Redis caching layer for the RAG application.
Provides Redis connectivity, multiple caching strategies,
TTL expiration, version-based invalidation, and telemetry.
"""

from src.cache.redis_client import RedisConnectionManager
from src.cache.cache_store import CacheStore
from src.cache.cache_strategies import (
    BaseCacheStrategy,
    CacheAsideStrategy,
    ReadThroughStrategy,
    WriteThroughStrategy,
    WriteBehindStrategy,
    WriteAroundStrategy,
    get_cache_strategy,
)

__all__ = [
    "RedisConnectionManager",
    "CacheStore",
    "BaseCacheStrategy",
    "CacheAsideStrategy",
    "ReadThroughStrategy",
    "WriteThroughStrategy",
    "WriteBehindStrategy",
    "WriteAroundStrategy",
    "get_cache_strategy",
]

