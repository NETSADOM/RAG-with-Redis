"""
backend/tests/test_caching_enhancements.py
Automated tests for:
1. Cache-Aside + Write-Through on Misses
2. LFU Eviction Configuration and Access Count Tracking
3. Semantic Caching for Equivalent Customer Questions
4. Policy-Aware Invalidation by Document ID
"""

import numpy as np
import pytest
from backend.cache.redis_cache import CustomerSupportCache
from backend.rag.pipeline import CustomerSupportRAGPipeline


@pytest.fixture
def clean_cache():
    cache = CustomerSupportCache(use_fallback=True)
    cache.clear()
    return cache


def test_write_through_and_lfu_access_frequency(clean_cache):
    """Verify write-through persistence on miss and LFU access frequency increment on hit."""
    query = "What is the return window?"
    sources = [{"doc_id": "doc_returns_001", "chunk_id": "ret_01", "title": "Return Policy"}]

    # 1. Write-through storage
    stored = clean_cache.set_answer(
        query=query,
        kb_version="v1.0.0",
        model="test-llm",
        answer="You have 30 days to return.",
        sources=sources,
    )
    assert stored is True

    # 2. First hit -> access_count should become 2
    cached, hit = clean_cache.get_answer(query, "v1.0.0", "test-llm")
    assert hit is True
    assert cached["access_count"] == 2

    # 3. Second hit -> access_count should become 3 (reinforcing LFU retention)
    cached2, hit2 = clean_cache.get_answer(query, "v1.0.0", "test-llm")
    assert hit2 is True
    assert cached2["access_count"] == 3


def test_semantic_caching_for_equivalent_questions(clean_cache):
    """Verify semantic caching matches equivalent questions based on embedding cosine similarity."""
    q_original = "How long do I have to return an item?"
    # Synthetic 384-dim normalized vector
    np.random.seed(42)
    vec1 = np.random.randn(384).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)

    clean_cache.set_answer(
        query=q_original,
        kb_version="v1.0.0",
        model="test-llm",
        answer="Customers have 30 days to return eligible merchandise.",
        sources=[{"doc_id": "doc_returns_001", "title": "Return Policy"}],
        query_embedding=vec1.tolist(),
    )

    # Highly similar query vector (cosine similarity exactly 0.95)
    noise = np.random.randn(384).astype(np.float32)
    # Project out component along vec1 so noise is orthogonal
    noise -= np.dot(noise, vec1) * vec1
    noise /= np.linalg.norm(noise)
    vec_similar = 0.95 * vec1 + np.sqrt(1 - 0.95**2) * noise

    sem_payload, sem_hit, sim_score = clean_cache.get_answer_semantic(
        query_vec=vec_similar,
        kb_version="v1.0.0",
        model="test-llm",
        threshold=0.85,
    )

    assert sem_hit is True
    assert sim_score >= 0.85
    assert sem_payload["answer"] == "Customers have 30 days to return eligible merchandise."
    assert clean_cache.semantic_hits == 1

    # Unrelated query vector (orthogonal / low cosine similarity)
    vec_unrelated = np.random.randn(384).astype(np.float32)
    vec_unrelated /= np.linalg.norm(vec_unrelated)

    unrel_payload, unrel_hit, unrel_sim = clean_cache.get_answer_semantic(
        query_vec=vec_unrelated,
        kb_version="v1.0.0",
        model="test-llm",
        threshold=0.85,
    )
    assert unrel_hit is False
    assert unrel_payload is None


def test_policy_aware_invalidation_by_doc_id(clean_cache):
    """Verify policy-aware invalidation purges only the keys citing the updated policy document."""
    # Store return policy question
    clean_cache.set_answer(
        query="What is the return period?",
        kb_version="v1.0.0",
        model="test-llm",
        answer="30 days.",
        sources=[{"doc_id": "doc_returns_001", "title": "Returns"}],
    )

    # Store shipping policy question
    clean_cache.set_answer(
        query="What are shipping rates?",
        kb_version="v1.0.0",
        model="test-llm",
        answer="Standard shipping is $5.99.",
        sources=[{"doc_id": "doc_shipping_002", "title": "Shipping"}],
    )

    # Both are cached
    _, hit_ret = clean_cache.get_answer("What is the return period?", "v1.0.0", "test-llm")
    _, hit_ship = clean_cache.get_answer("What are shipping rates?", "v1.0.0", "test-llm")
    assert hit_ret is True
    assert hit_ship is True

    # Invalidate ONLY doc_returns_001
    purged = clean_cache.invalidate_by_doc_id("doc_returns_001")
    assert purged == 1

    # Return question should now be a MISS
    _, hit_ret_after = clean_cache.get_answer("What is the return period?", "v1.0.0", "test-llm")
    assert hit_ret_after is False

    # Shipping question remains a HIT (unaffected by return policy update)
    _, hit_ship_after = clean_cache.get_answer("What are shipping rates?", "v1.0.0", "test-llm")
    assert hit_ship_after is True


def test_lfu_eviction_policy_configured(clean_cache):
    """Verify eviction policy property is allkeys-lfu."""
    assert clean_cache.eviction_policy == "allkeys-lfu"
    metrics = clean_cache.get_metrics()
    assert metrics["eviction_policy"] == "allkeys-lfu"
