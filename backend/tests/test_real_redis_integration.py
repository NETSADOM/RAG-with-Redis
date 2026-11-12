"""Real Redis integration: connections, TTL, revision isolation and server configuration."""

import time
import os
import pytest
import redis
from uuid import uuid4

from backend.cache.redis_cache import CustomerSupportCache


REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


def is_live_redis_available() -> bool:
    try:
        cache = CustomerSupportCache(redis_url=REDIS_URL)
        cache.close()
        return True
    except Exception:
        return False


if os.getenv("REQUIRE_REAL_REDIS") == "1" and not is_live_redis_available():
    raise RuntimeError("Real Redis integration is required but TEST_REDIS_URL is unavailable.")

live_redis = pytest.mark.skipif(
    not is_live_redis_available(),
    reason="Real Redis server not running locally on port 6379 (Runs in GitHub Actions CI with Redis service)"
)


@live_redis
class TestRealRedisIntegration:
    """Integration suite verifying real Redis operations."""

    @pytest.fixture(autouse=True)
    def setup_real_redis(self):
        self.cache = CustomerSupportCache(redis_url=REDIS_URL, default_ttl=60,
                                          namespace="cs_rag:test:" + uuid4().hex)
        yield
        self.cache.clear()
        self.cache.close()

    def test_real_redis_connection_preserves_server_configuration(self):
        """The application must not change a shared Redis server's eviction policy."""
        assert self.cache.is_available is True
        client = self.cache._client

        assert client.ping() is True

        try:
            cfg = client.config_get("maxmemory-policy")
        except redis.ResponseError:
            pytest.skip("Redis does not allow CONFIG GET")
        another = CustomerSupportCache(redis_url=REDIS_URL)
        another.close()
        assert client.config_get("maxmemory-policy") == cfg

    def test_real_redis_cache_hits_and_misses(self):
        """Verify exact cache hit and miss lifecycle on real Redis."""
        q = "How long do I have to return an item?"
        kb_rev = "rev_test_001"
        model = "gpt-4o-mini"

        val, hit = self.cache.get_answer(q, kb_rev, model)
        assert hit is False
        assert val is None

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

        cached_val, hit2 = self.cache.get_answer(q, kb_rev, model)
        assert hit2 is True
        assert cached_val["answer"] == "Customers have 30 days to return items."
        assert len(cached_val["sources"]) == 1
        assert cached_val["sources"][0]["page_number"] == 1

    def test_real_redis_ttl_expiration(self):
        """Verify real Redis TTL expiration."""
        q = "What are your shipping rates?"
        kb_rev = "rev_test_001"

        self.cache.set_answer(
            query=q,
            kb_version=kb_rev,
            model="gpt-4o-mini",
            answer="Standard shipping is $4.99.",
            sources=[{"doc_id": "doc_shipping"}],
            ttl=1,
        )

        _, hit_immediate = self.cache.get_answer(q, kb_rev, "gpt-4o-mini")
        assert hit_immediate is True

        time.sleep(1.2)

        _, hit_expired = self.cache.get_answer(q, kb_rev, "gpt-4o-mini")
        assert hit_expired is False

    def test_real_redis_revision_invalidation(self):
        self.cache.set_answer("Returns?", "old", "model", "30 days", [])
        assert self.cache.get_answer("Returns?", "old", "model")[1]
        assert not self.cache.get_answer("Returns?", "new", "model")[1]
