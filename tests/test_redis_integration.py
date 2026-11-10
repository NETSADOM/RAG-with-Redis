"""
tests/test_redis_integration.py
Phase 1 verification:
Automated tests for Redis connectivity, GET, SET, TTL expiration, and key deletion.
"""

import time
import pytest
from src.cache.redis_client import RedisConnectionManager
from src.cache.cache_store import CacheStore


def test_redis_connection_and_ping():
    """Verify that connection manager connects and pings successfully."""
    manager = RedisConnectionManager()
    assert manager.is_connected is True
    assert manager.client is not None
    info = manager.get_info()
    assert info["connected"] is True


def test_cache_set_get_and_delete():
    """Test basic GET, SET, and DELETE operations."""
    store = CacheStore()
    test_key = "test:query:123"
    test_data = {
        "question": "What is RAG?",
        "answer": "Retrieval-Augmented Generation.",
        "sources": [{"filename": "rag.pdf", "page": 1}],
    }

    # Clean before test
    store.delete(test_key)

    # SET
    success = store.set(test_key, test_data, ttl=30)
    assert success is True
    assert store.exists(test_key) is True

    # GET
    retrieved = store.get(test_key)
    assert retrieved is not None
    assert retrieved["question"] == "What is RAG?"
    assert retrieved["answer"] == "Retrieval-Augmented Generation."
    assert len(retrieved["sources"]) == 1

    # DELETE
    del_success = store.delete(test_key)
    assert del_success is True
    assert store.exists(test_key) is False
    assert store.get(test_key) is None


def test_cache_ttl():
    """Verify TTL assignment and countdown."""
    store = CacheStore()
    test_key = "test:ttl:key"
    store.set(test_key, {"val": 1}, ttl=5)

    remaining_ttl = store.ttl(test_key)
    assert 0 < remaining_ttl <= 5


def test_cache_pattern_deletion():
    """Verify wildcard pattern invalidation."""
    store = CacheStore()
    store.set("test:scopeA:q1", {"data": 1}, ttl=60)
    store.set("test:scopeA:q2", {"data": 2}, ttl=60)
    store.set("test:scopeB:q3", {"data": 3}, ttl=60)

    deleted_count = store.delete_by_pattern("test:scopeA:*")
    assert deleted_count >= 2
    assert store.exists("test:scopeA:q1") is False
    assert store.exists("test:scopeA:q2") is False
    assert store.exists("test:scopeB:q3") is True

    # Clean up
    store.delete("test:scopeB:q3")

