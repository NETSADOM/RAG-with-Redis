"""
tests/test_caching.py
Automated test suite verifying all Redis caching patterns, invalidation,
TTL expiration, citation preservation, and graceful failure fallbacks.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from src.config import config
from src.cache import CacheStore, RedisConnectionManager, get_cache_strategy
from src.cache.cache_strategies import (
    CacheAsideStrategy,
    ReadThroughStrategy,
    WriteThroughStrategy,
    WriteBehindStrategy,
    WriteAroundStrategy,
)
from src.rag_pipeline import RAGPipeline, RAGAnswer
from src.retriever import RetrievalResult, RetrievedChunk
from src.generator import GenerationResult


@pytest.fixture
def mock_pipeline():
    """Create a lightweight pipeline with mocked vector store and generator."""
    store = CacheStore()
    store.clear()

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.cache_store = store
    pipeline.cache_strategy = CacheAsideStrategy(store)

    # Mock chunk & retriever
    mock_chunk = RetrievedChunk(
        text="Attention is all you need research paper.",
        filename="attention.pdf",
        page_number=1,
        total_pages=15,
        chunk_id="chunk_1",
        chunk_index=0,
        distance=0.15,
        similarity_score=0.85,
        doc_id="attention.pdf",
        doc_name="attention.pdf",
    )
    mock_ret = MagicMock()
    mock_ret.retrieve.return_value = RetrievalResult(
        query="Who is the author?",
        chunks=[mock_chunk],
        has_sufficient_evidence=True,
        closest_distance=0.15,
        relevance_threshold=0.65,
    )
    pipeline.retriever = mock_ret

    # Mock generator
    mock_gen = MagicMock()
    mock_gen.model_name = "test-qwen-model"
    mock_gen.provider = "groq"
    mock_gen.generate.return_value = GenerationResult(
        answer="The author of the paper is Vaswani et al., published in 2017.",
        model="test-qwen-model",
        provider="groq",
        context_chunks=[mock_chunk],
        has_sufficient_context=True,
    )
    pipeline.generator = mock_gen

    # Mock vector store
    mock_vs = MagicMock()
    mock_vs.get_collection_fingerprint.return_value = "v1_abcd1234"
    mock_vs.list_documents.return_value = [
        {"filename": "attention.pdf", "doc_id": "attention.pdf", "chunks": 10}
    ]
    pipeline.vector_store = mock_vs

    return pipeline


def test_cache_miss_invokes_rag_pipeline(mock_pipeline):
    """Verify that a cold cache causes a cache miss and invokes retrieval and LLM."""
    query = "Who wrote the Attention paper?"
    res = mock_pipeline.answer_question(query)

    assert res.is_cached is False
    assert "Vaswani" in res.answer
    assert len(res.sources) == 1
    assert res.sources[0]["doc_name"] == "attention.pdf"
    assert res.sources[0]["page_number"] == 1
    assert mock_pipeline.generator.generate.call_count == 1
    assert mock_pipeline.retriever.retrieve.call_count == 1


def test_cache_hit_bypasses_rag_pipeline(mock_pipeline):
    """Verify that a warm cache returns the cached answer and skips retrieval and LLM."""
    query = "Who wrote the Attention paper?"

    # Cold run -> Miss
    first_res = mock_pipeline.answer_question(query)
    assert first_res.is_cached is False
    assert mock_pipeline.generator.generate.call_count == 1
    assert mock_pipeline.retriever.retrieve.call_count == 1

    # Second run -> Warm HIT
    # Make generator raise an exception if invoked to prove it is completely bypassed
    mock_pipeline.generator.generate.side_effect = RuntimeError("LLM should NOT be called on cache hit!")
    mock_pipeline.retriever.retrieve.side_effect = RuntimeError("Retriever should NOT be called on cache hit!")

    second_res = mock_pipeline.answer_question(query)
    assert second_res.is_cached is True
    assert second_res.answer == first_res.answer
    assert second_res.sources == first_res.sources
    assert second_res.retrieved_passages == first_res.retrieved_passages
    assert second_res.cache_key == first_res.cache_key


def test_source_metadata_preserved_in_cache(mock_pipeline):
    """Verify that citations, page numbers, similarity scores, and passages are preserved."""
    query = "What is the primary mechanism?"
    res1 = mock_pipeline.answer_question(query)
    res2 = mock_pipeline.answer_question(query)

    assert res2.is_cached is True
    assert len(res2.sources) == 1
    source = res2.sources[0]
    assert source["filename"] == "attention.pdf"
    assert source["page_number"] == 1
    assert source["similarity"] == 0.85
    assert len(res2.retrieved_passages) == 1
    assert res2.retrieved_passages[0]["text"] == "Attention is all you need research paper."


def test_ttl_expiration(mock_pipeline):
    """Verify that cached answers expire automatically after TTL."""
    query = "Will this query expire quickly?"
    key = mock_pipeline.compute_cache_key(query)

    # Set with 1-second TTL
    mock_pipeline.cache_store.set(key, {"answer": "Temporary value"}, ttl=1)
    assert mock_pipeline.cache_store.get(key) is not None

    # Wait for TTL expiration
    time.sleep(1.2)
    assert mock_pipeline.cache_store.get(key) is None


def test_invalidation_after_document_change(mock_pipeline):
    """Verify that changing collection version changes the cache key, triggering a miss."""
    query = "Who wrote the Attention paper?"

    # Initial query on v1
    res1 = mock_pipeline.answer_question(query)
    key_v1 = res1.cache_key

    # Simulate document upload or deletion changing collection version
    mock_pipeline.vector_store.get_collection_fingerprint.return_value = "v2_efgh5678"

    # Reset side-effects
    mock_pipeline.generator.generate.side_effect = None
    mock_pipeline.generator.generate.return_value = GenerationResult(
        answer="Updated answer for version 2.",
        model="test-qwen-model",
        provider="groq",
        context_chunks=[],
        has_sufficient_context=True,
    )
    mock_pipeline.retriever.retrieve.side_effect = None

    res2 = mock_pipeline.answer_question(query)
    assert res2.is_cached is False
    assert res2.cache_key != key_v1
    assert res2.collection_version == "v2_efgh5678"


def test_redis_failure_graceful_fallback(mock_pipeline):
    """Verify that pipeline falls back gracefully to direct RAG if Redis throws an error."""
    query = "What happens if Redis dies?"

    # Break cache store get and set
    mock_pipeline.cache_store.get = MagicMock(side_effect=Exception("Redis connection refused"))
    mock_pipeline.cache_store.set = MagicMock(side_effect=Exception("Redis connection refused"))

    res = mock_pipeline.answer_question(query)
    assert res.is_cached is False
    assert "Vaswani" in res.answer
    assert len(res.sources) == 1


def test_all_caching_strategies_contract():
    """Verify the 5 caching strategies execute and conform to the caching contract."""
    store = CacheStore()
    store.clear()

    strategies = [
        get_cache_strategy("cache_aside", store),
        get_cache_strategy("read_through", store),
        get_cache_strategy("write_through", store),
        get_cache_strategy("write_behind", store),
        get_cache_strategy("write_around", store),
    ]

    for strat in strategies:
        key = f"test_strat_{strat.name}"
        compute_mock = MagicMock(return_value={"answer": f"result for {strat.name}"})

        # 1. Cold call -> Miss
        res, is_cached = strat.get_or_compute_answer(
            question="test",
            document_ids=None,
            compute_fn=compute_mock,
            cache_key=key,
            ttl=60,
        )
        assert is_cached is False
        assert compute_mock.call_count == 1

        # Give write-behind thread a moment to persist if applicable
        if strat.name == "write_behind":
            time.sleep(0.1)

        # 2. Warm call -> Hit
        res2, is_cached2 = strat.get_or_compute_answer(
            question="test",
            document_ids=None,
            compute_fn=compute_mock,
            cache_key=key,
            ttl=60,
        )
        assert is_cached2 is True
        assert compute_mock.call_count == 1  # Not called again


def test_write_consistency(mock_pipeline):
    """Verify write consistency: payload read from cache exactly matches what was written."""
    query = "What is the transformer architecture?"
    computed_res = mock_pipeline.answer_question(query)

    # Fetch raw payload from cache
    cached_payload = mock_pipeline.cache_store.get(computed_res.cache_key)
    assert cached_payload is not None
    assert cached_payload["answer"] == computed_res.answer
    assert cached_payload["model_used"] == computed_res.model_used
    assert len(cached_payload["sources"]) == len(computed_res.sources)
    assert cached_payload["sources"][0]["doc_name"] == computed_res.sources[0]["doc_name"]
    assert cached_payload["sources"][0]["similarity"] == computed_res.sources[0]["similarity"]
    assert "cached_at" in cached_payload


def test_cache_key_scope_isolation(mock_pipeline):
    """Verify that scoped queries do not collide with global or differently scoped queries."""
    query = "Summarize key findings"

    key_global = mock_pipeline.compute_cache_key(query, document_ids=None)
    key_doc_a = mock_pipeline.compute_cache_key(query, document_ids=["paper_a.pdf"])
    key_doc_b = mock_pipeline.compute_cache_key(query, document_ids=["paper_b.pdf"])
    key_docs_ab = mock_pipeline.compute_cache_key(query, document_ids=["paper_b.pdf", "paper_a.pdf"])
    key_docs_ba = mock_pipeline.compute_cache_key(query, document_ids=["paper_a.pdf", "paper_b.pdf"])

    # All different scopes must produce different cache keys
    assert key_global != key_doc_a
    assert key_doc_a != key_doc_b
    assert key_doc_a != key_docs_ab

    # Order of selected documents in scope must NOT affect the cache key (deterministic sorting)
    assert key_docs_ab == key_docs_ba

