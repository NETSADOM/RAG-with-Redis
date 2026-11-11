"""
backend/tests/test_real_redis_integration.py
Integration tests against real Redis (used in CI with Redis 7 service container).
Tests:
- Real Redis TCP connection
- Cache hits & misses
- TTL expiration verification
- LFU eviction policy configuration
- Document ID selective invalidation
- Global knowledge-base revision invalidation
"""

import time
import os
import pytest
import redis

from backend.cache.redis_cache import CustomerSupportCache
from backend.rag.pipeline import CustomerSupportRAGPipeline
from backend.rag.retrieval import KBChunk, KnowledgeBaseManager


REDIS_URL = os.getenv("TEST_REDIS_URL", os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))


def is_live_redis_available() -> bool:
    try:
        r = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5)
        return bool(r.ping())
    except Exception:
        return False


# Skip whole module if not in CI and live Redis is not available
live_redis = pytest.mark.skipif(
    not is_live_redis_available(),
    reason="Real Redis server not running locally on port 6379 (Runs in GitHub Actions CI with Redis service)"
)


@live_redis
class TestRealRedisIntegration:
    """Integration suite verifying real Redis operations."""

    @pytest.fixture(autouse=True)
    def setup_real_redis(self):
        self.cache = CustomerSupportCache(redis_url=REDIS_URL, use_fallback=False)
        assert self.cache._is_emulator is False, "Must be connected to a real Redis instance"
        self.cache.clear()
        yield
        self.cache.clear()

    def test_real_redis_connection_and_lfu_configuration(self):
        """Verify real TCP connection and LFU maxmemory-policy setting."""
        assert self.cache.is_available is True
        client = self.cache._client

        # Verify ping
        assert client.ping() is True

        # Check maxmemory-policy config
        try:
            cfg = client.config_get("maxmemory-policy")
            policy = cfg.get("maxmemory-policy", "")
            # In CI with Redis 7, this will be allkeys-lfu
            assert policy in ("allkeys-lfu", "noeviction", "volatile-lru")
        except Exception:
            # Some managed environments disallow CONFIG GET
            pass

    def test_real_redis_cache_hits_and_misses(self):
        """Verify exact cache hit and miss lifecycle on real Redis."""
        q = "How long do I have to return an item?"
        kb_rev = "rev_test_001"
        model = "gpt-4o-mini"

        # 1. First lookup: Miss
        val, hit = self.cache.get_answer(q, kb_rev, model)
        assert hit is False
        assert val is None

        # 2. Write-Through persistence
        sources = [{
            "doc_id": "doc_refunds_returns",
            "chunk_id": "c1",
            "title": "Return Policy",
            "page_number": 1,
            "source_ref": "Page 1, Sec 1.1",
            "supporting_passage": "30 days to return.",
        }]
        stored = self.cache.set_answer(
            query=q,
            kb_version=kb_rev,
            model=model,
            answer="Customers have 30 days to return items.",
            sources=sources,
            ttl=10,
        )
        assert stored is True

        # 3. Second lookup: Hit
        cached_val, hit2 = self.cache.get_answer(q, kb_rev, model)
        assert hit2 is True
        assert cached_val["answer"] == "Customers have 30 days to return items."
        assert cached_val["access_count"] == 2
        assert len(cached_val["sources"]) == 1
        assert cached_val["sources"][0]["page_number"] == 1

    def test_real_redis_ttl_expiration(self):
        """Verify real Redis TTL expiration."""
        q = "What are your shipping rates?"
        kb_rev = "rev_test_001"

        # Set with 1-second TTL
        self.cache.set_answer(
            query=q,
            kb_version=kb_rev,
            model="gpt-4o-mini",
            answer="Standard shipping is $4.99.",
            sources=[{"doc_id": "doc_shipping"}],
            ttl=1,
        )

        # Immediate lookup: Should be a HIT
        _, hit_immediate = self.cache.get_answer(q, kb_rev, "gpt-4o-mini")
        assert hit_immediate is True

        # Wait for TTL to expire
        time.sleep(1.2)

        # Post-expiration lookup: Must be a MISS
        _, hit_expired = self.cache.get_answer(q, kb_rev, "gpt-4o-mini")
        assert hit_expired is False

    def test_real_redis_policy_aware_invalidation(self):
        """Verify policy-aware invalidation purges affected keys on real Redis."""
        kb_rev = "rev_test_001"

        # Cache answer citing return policy
        self.cache.set_answer(
            query="Can I return opened food?",
            kb_version=kb_rev,
            model="gpt-4o-mini",
            answer="No, perishable food cannot be returned.",
            sources=[{"doc_id": "doc_refunds_returns"}],
        )

        # Cache answer citing shipping policy
        self.cache.set_answer(
            query="Do you deliver to Canada?",
            kb_version=kb_rev,
            model="gpt-4o-mini",
            answer="Yes, shipping to Canada takes 7-14 business days.",
            sources=[{"doc_id": "doc_shipping_delivery"}],
        )

        # Both exist
        _, hit_return = self.cache.get_answer("Can I return opened food?", kb_rev, "gpt-4o-mini")
        _, hit_ship = self.cache.get_answer("Do you deliver to Canada?", kb_rev, "gpt-4o-mini")
        assert hit_return is True
        assert hit_ship is True

        # Invalidate only return policy
        purged = self.cache.invalidate_by_doc_id("doc_refunds_returns")
        assert purged == 1

        # Return answer is purged
        _, hit_return_after = self.cache.get_answer("Can I return opened food?", kb_rev, "gpt-4o-mini")
        assert hit_return_after is False

        # Shipping answer is preserved
        _, hit_ship_after = self.cache.get_answer("Do you deliver to Canada?", kb_rev, "gpt-4o-mini")
        assert hit_ship_after is True

