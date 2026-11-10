"""
src/cache/cache_strategies.py
Implements five distinct caching patterns for comparison:
1. Cache-Aside (Lazy Loading)
2. Read-Through
3. Write-Through
4. Write-Behind (Write-Back)
5. Write-Around
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
import time
import queue
import threading
import logging

from src.cache.cache_store import CacheStore

logger = logging.getLogger(__name__)


class BaseCacheStrategy(ABC):
    """Abstract interface for RAG answer caching strategies."""

    def __init__(self, cache_store: CacheStore, name: str):
        self.cache_store = cache_store
        self.name = name

    @abstractmethod
    def get_or_compute_answer(
        self,
        question: str,
        document_ids: Optional[List[str]],
        compute_fn: Callable[[], Any],
        cache_key: str,
        ttl: Optional[int] = None,
    ) -> Any:
        """Retrieve answer using the concrete caching pattern."""
        pass

    def on_document_write(
        self,
        filename: str,
        persist_fn: Callable[[], Any],
        precompute_fn: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """Handle document ingestion/update according to strategy."""
        return persist_fn()


class CacheAsideStrategy(BaseCacheStrategy):
    """
    1. Cache-Aside (Lazy Loading):
    The application reads the cache first.
    On HIT: Returns cached answer without calling retrieval or generation.
    On MISS: Executes RAG pipeline and stores result in cache with TTL.
    """

    def __init__(self, cache_store: CacheStore):
        super().__init__(cache_store, name="cache_aside")

    def get_or_compute_answer(
        self,
        question: str,
        document_ids: Optional[List[str]],
        compute_fn: Callable[[], Any],
        cache_key: str,
        ttl: Optional[int] = None,
    ) -> Any:
        # 1. Lookup in Cache
        cached_payload = self.cache_store.get(cache_key)
        if cached_payload is not None:
            return cached_payload, True

        # 2. Compute on MISS
        computed_res = compute_fn()

        # 3. Store in Cache
        if computed_res is not None:
            self.cache_store.set(cache_key, computed_res, ttl=ttl)

        return computed_res, False


class ReadThroughStrategy(BaseCacheStrategy):
    """
    2. Read-Through:
    The application queries the cache layer directly as its primary interface.
    The cache layer encapsulates fetching from the RAG pipeline when data is missing.
    """

    def __init__(self, cache_store: CacheStore):
        super().__init__(cache_store, name="read_through")

    def get_or_compute_answer(
        self,
        question: str,
        document_ids: Optional[List[str]],
        compute_fn: Callable[[], Any],
        cache_key: str,
        ttl: Optional[int] = None,
    ) -> Any:
        # Cache layer itself handles read-through loading
        cached = self.cache_store.get(cache_key)
        if cached is not None:
            return cached, True

        fresh = compute_fn()
        self.cache_store.set(cache_key, fresh, ttl=ttl)
        return fresh, False


class WriteThroughStrategy(BaseCacheStrategy):
    """
    3. Write-Through:
    On ingestion or pre-computation, data is written synchronously to BOTH
    the persistent store (ChromaDB) and the cache.
    Avoids cache misses for anticipated or seeded document queries.
    """

    def __init__(self, cache_store: CacheStore):
        super().__init__(cache_store, name="write_through")

    def get_or_compute_answer(
        self,
        question: str,
        document_ids: Optional[List[str]],
        compute_fn: Callable[[], Any],
        cache_key: str,
        ttl: Optional[int] = None,
    ) -> Any:
        cached = self.cache_store.get(cache_key)
        if cached is not None:
            return cached, True

        fresh = compute_fn()
        self.cache_store.set(cache_key, fresh, ttl=ttl)
        return fresh, False

    def on_document_write(
        self,
        filename: str,
        persist_fn: Callable[[], Any],
        precompute_fn: Optional[Callable[[], Any]] = None,
    ) -> Any:
        # Synchronously write to ChromaDB
        result = persist_fn()
        # Synchronously warm/seed cache if precompute function provided
        if precompute_fn:
            try:
                precompute_fn()
            except Exception as e:
                logger.warning(f"[WriteThrough] Precompute cache seeding failed: {e}")
        return result


class WriteBehindStrategy(BaseCacheStrategy):
    """
    4. Write-Behind (Write-Back):
    Writes are acknowledged immediately in the cache/queue and persisted
    asynchronously to the underlying store by a background worker.
    """

    def __init__(self, cache_store: CacheStore):
        super().__init__(cache_store, name="write_behind")
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        while True:
            try:
                task = self._queue.get()
                if task is None:
                    break
                fn, key, data, ttl = task
                if fn:
                    fn()
                if key and data:
                    self.cache_store.set(key, data, ttl=ttl)
            except Exception as e:
                logger.error(f"[WriteBehind] Worker error: {e}")
            finally:
                self._queue.task_done()

    def get_or_compute_answer(
        self,
        question: str,
        document_ids: Optional[List[str]],
        compute_fn: Callable[[], Any],
        cache_key: str,
        ttl: Optional[int] = None,
    ) -> Any:
        cached = self.cache_store.get(cache_key)
        if cached is not None:
            return cached, True

        fresh = compute_fn()
        # Enqueue write to background worker (non-blocking)
        self._queue.put((None, cache_key, fresh, ttl))
        return fresh, False

    def on_document_write(
        self,
        filename: str,
        persist_fn: Callable[[], Any],
        precompute_fn: Optional[Callable[[], Any]] = None,
    ) -> Any:
        # In Write-Behind, persistence task is placed into background queue
        self._queue.put((persist_fn, None, None, None))
        return {"status": "queued", "filename": filename}


class WriteAroundStrategy(BaseCacheStrategy):
    """
    5. Write-Around:
    Document ingestion writes directly to persistent storage (ChromaDB),
    completely bypassing the cache. The cache is ONLY populated when
    specific questions are subsequently read.
    Prevents cache flooding from newly ingested, rarely queried documents.
    """

    def __init__(self, cache_store: CacheStore):
        super().__init__(cache_store, name="write_around")

    def get_or_compute_answer(
        self,
        question: str,
        document_ids: Optional[List[str]],
        compute_fn: Callable[[], Any],
        cache_key: str,
        ttl: Optional[int] = None,
    ) -> Any:
        cached = self.cache_store.get(cache_key)
        if cached is not None:
            return cached, True

        # Only cached after actual read query
        fresh = compute_fn()
        self.cache_store.set(cache_key, fresh, ttl=ttl)
        return fresh, False

    def on_document_write(
        self,
        filename: str,
        persist_fn: Callable[[], Any],
        precompute_fn: Optional[Callable[[], Any]] = None,
    ) -> Any:
        # Directly persist, strictly bypassing cache
        return persist_fn()


def get_cache_strategy(strategy_name: str, cache_store: CacheStore) -> BaseCacheStrategy:
    """Factory helper to instantiate caching strategies by name."""
    name = (strategy_name or "cache_aside").lower()
    if name == "read_through":
        return ReadThroughStrategy(cache_store)
    elif name == "write_through":
        return WriteThroughStrategy(cache_store)
    elif name == "write_behind":
        return WriteBehindStrategy(cache_store)
    elif name == "write_around":
        return WriteAroundStrategy(cache_store)
    else:
        return CacheAsideStrategy(cache_store)

