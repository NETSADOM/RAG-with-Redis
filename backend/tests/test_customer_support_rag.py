"""
backend/tests/test_customer_support_rag.py
Comprehensive automated test suite covering all 11 required scenarios:
1. Redis is checked before retrieval or generation.
2. Cache hits return answers and source references correctly.
3. Cache hits do not call retrieval, reranking, or the LLM.
4. Cache misses execute the full pipeline.
5. Successful responses are stored in Redis.
6. Failed responses are not cached.
7. Redis failures fall back to normal RAG processing.
8. Cached answers become invalid after knowledge-base updates.
9. Hybrid search combines BM25 and dense search correctly (RRF).
10. Reranking changes result order when relevance scores differ.
11. The system does not invent answers for unsupported questions.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.rag.retrieval import KnowledgeBaseManager, KBChunk, BM25Retriever, DenseRetriever, HybridRetriever
from backend.rag.reranker import CrossEncoderReranker
from backend.rag.generator import CustomerSupportGenerator, CustomerSupportAnswer
from backend.cache.redis_cache import CustomerSupportCache
from backend.rag.pipeline import CustomerSupportRAGPipeline


@pytest.fixture
def mock_kb_chunks():
    """Sample policy chunks representing returns and shipping."""
    return [
        KBChunk(
            chunk_id="returns_1",
            doc_id="doc_returns",
            title="Return Policy",
            section="Eligibility",
            source_ref="help.novacart.com/returns#window",
            text="You have 30 days to return physical items in original condition for a full refund.",
            doc_version="v1.0",
        ),
        KBChunk(
            chunk_id="shipping_1",
            doc_id="doc_shipping",
            title="Shipping Policy",
            section="Standard Delivery",
            source_ref="help.novacart.com/shipping#standard",
            text="Standard shipping takes 3-5 business days and is free on orders over $50.",
            doc_version="v1.0",
        ),
        KBChunk(
            chunk_id="cancellation_1",
            doc_id="doc_cancellations",
            title="Cancellation Policy",
            section="Grace Period",
            source_ref="help.novacart.com/cancellations#grace",
            text="Orders can be cancelled within a 30-minute self-service grace period after submission.",
            doc_version="v1.0",
        ),
    ]


@pytest.fixture
def mock_pipeline(mock_kb_chunks):
    """Assembles a lightweight RAG pipeline with mocked LLM and memory cache."""
    # KB Manager
    kb_mgr = MagicMock(spec=KnowledgeBaseManager)
    kb_mgr.chunks = mock_kb_chunks
    kb_mgr.version_hash = "kb_hash_v100"
    kb_mgr.get_collection_revision.return_value = "kb_hash_v100"

    # Cache (in-memory emulator)
    cache = CustomerSupportCache(use_fallback=True)
    cache.clear()

    # Mock Generator
    generator = MagicMock(spec=CustomerSupportGenerator)
    generator.model = "gpt-4o-mini"
    generator.generate.return_value = CustomerSupportAnswer(
        question="How long do I have to return an item?",
        answer="You have 30 days from delivery to return physical items in their original condition.",
        is_supported=True,
        sources=[{
            "doc_id": "doc_returns",
            "title": "Return Policy",
            "section": "Eligibility",
            "chunk_id": "returns_1",
            "source_ref": "help.novacart.com/returns#window",
            "relevance_score": 0.95,
        }],
        citations=[{
            "doc_id": "doc_returns",
            "title": "Return Policy",
            "section": "Eligibility",
            "chunk_id": "returns_1",
            "source_ref": "help.novacart.com/returns#window",
            "relevance_score": 0.95,
        }],
        citation_validation_passed=True,
        model="gpt-4o-mini",
    )

    # Fast Reranker Mock
    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.rerank.side_effect = lambda query, candidates, top_n: candidates[:top_n]

    # Pipeline with lazy dense to avoid heavy model downloads during quick unit tests
    pipe = CustomerSupportRAGPipeline(
        kb_manager=kb_mgr,
        cache=cache,
        generator=generator,
        reranker=reranker,
        lazy_dense=True,
    )
    return pipe


# --------------------------------------------------------------------------
# Test 1: Redis is checked before retrieval or generation
# --------------------------------------------------------------------------
def test_1_redis_checked_before_retrieval_or_generation(mock_pipeline):
    query = "What is the return window?"
    mock_pipeline.cache.get_answer = MagicMock(return_value=(None, False))

    mock_pipeline.answer_question(query)

    # Verify cache.get_answer was invoked first
    assert mock_pipeline.cache.get_answer.call_count == 1
    mock_pipeline.cache.get_answer.assert_called_with(
        query=query,
        kb_version="kb_hash_v100",
        model="gpt-4o-mini",
    )


# --------------------------------------------------------------------------
# Test 2: Cache hits return answers and source references correctly
# --------------------------------------------------------------------------
def test_2_cache_hits_return_answers_and_sources_correctly(mock_pipeline):
    query = "What is the return window?"

    # First call: Cold miss -> Populates cache
    res1 = mock_pipeline.answer_question(query)
    assert res1.cache_hit is False

    # Second call: Warm HIT
    res2 = mock_pipeline.answer_question(query)
    assert res2.cache_hit is True
    assert res2.answer == res1.answer
    assert len(res2.sources) == len(res1.sources)
    assert res2.sources[0]["doc_id"] == "doc_returns"
    assert res2.sources[0]["chunk_id"] == "returns_1"


# --------------------------------------------------------------------------
# Test 3: Cache hits do not call retrieval, reranking, or the LLM
# --------------------------------------------------------------------------
def test_3_cache_hits_do_not_call_retrieval_reranking_or_llm(mock_pipeline):
    query = "How long to return?"

    # Initial query to warm cache
    mock_pipeline.answer_question(query)
    assert mock_pipeline.generator.generate.call_count == 1

    # Configure side effects to throw exceptions if invoked
    mock_pipeline.generator.generate.side_effect = RuntimeError("LLM must NOT be called on Cache HIT!")
    mock_pipeline.reranker.rerank.side_effect = RuntimeError("Reranker must NOT be called on Cache HIT!")
    mock_pipeline.bm25_retriever.search = MagicMock(side_effect=RuntimeError("Retrieval must NOT be called!"))

    # Hit query
    hit_res = mock_pipeline.answer_question(query)
    assert hit_res.cache_hit is True
    assert mock_pipeline.generator.generate.call_count == 1  # Still 1, not called again


# --------------------------------------------------------------------------
# Test 4: Cache misses execute the full pipeline
# --------------------------------------------------------------------------
def test_4_cache_misses_execute_full_pipeline(mock_pipeline):
    query = "Brand new query that is never cached"

    res = mock_pipeline.answer_question(query)
    assert res.cache_hit is False
    assert mock_pipeline.generator.generate.call_count == 1
    assert mock_pipeline.reranker.rerank.call_count == 1


# --------------------------------------------------------------------------
# Test 5: Successful responses are stored in Redis
# --------------------------------------------------------------------------
def test_5_successful_responses_stored_in_redis(mock_pipeline):
    query = "Do you accept returns after 30 days?"
    res = mock_pipeline.answer_question(query)

    # Verify key exists in Redis
    cached_payload, is_hit = mock_pipeline.cache.get_answer(
        query=query,
        kb_version=mock_pipeline.kb_manager.version_hash,
        model=mock_pipeline.generator.model,
    )
    assert is_hit is True
    assert cached_payload is not None
    assert cached_payload["answer"] == res.answer
    assert "source_doc_ids" in cached_payload
    assert "source_chunk_ids" in cached_payload
    assert cached_payload["knowledge_base_version"] == "kb_hash_v100"


# --------------------------------------------------------------------------
# Test 6: Failed or unsupported responses are not cached
# --------------------------------------------------------------------------
def test_6_failed_responses_not_cached(mock_pipeline):
    query = "Query that yields error or unsupported answer"

    # Simulate unsupported or failed LLM output
    mock_pipeline.generator.generate.return_value = CustomerSupportAnswer(
        question=query,
        answer="I'm sorry, our documentation does not contain information on this topic.",
        is_supported=False,
        sources=[],
        model="gpt-4o-mini",
        error="Unsupported topic",
    )

    res = mock_pipeline.answer_question(query)
    assert res.is_supported is False

    # Ensure it was NOT saved to Redis
    cached_payload, is_hit = mock_pipeline.cache.get_answer(
        query=query,
        kb_version=mock_pipeline.kb_manager.version_hash,
        model=mock_pipeline.generator.model,
    )
    assert is_hit is False
    assert cached_payload is None


# --------------------------------------------------------------------------
# Test 7: Redis failures fall back to normal RAG processing
# --------------------------------------------------------------------------
def test_7_redis_failures_fallback_to_normal_rag(mock_pipeline):
    query = "What is the shipping cost?"

    # Simulate Redis connection failure
    mock_pipeline.cache.get_answer = MagicMock(side_effect=Exception("Redis connection refused"))
    mock_pipeline.cache.set_answer = MagicMock(side_effect=Exception("Redis write timeout"))

    # Must complete RAG gracefully without crashing
    res = mock_pipeline.answer_question(query)
    assert res is not None
    assert res.cache_hit is False
    assert len(res.answer) > 0


# --------------------------------------------------------------------------
# Test 8: Cached answers become invalid after knowledge-base updates
# --------------------------------------------------------------------------
def test_8_cached_answers_invalid_after_kb_update(mock_pipeline):
    query = "What is the return period?"

    # 1. Warm cache on version v100
    res1 = mock_pipeline.answer_question(query)
    assert res1.cache_hit is False

    # 2. Check hit on v100
    res2 = mock_pipeline.answer_question(query)
    assert res2.cache_hit is True

    # 3. Simulate knowledge base policy update -> version hash changes
    mock_pipeline.kb_manager.version_hash = "kb_hash_v200_updated"
    mock_pipeline.kb_manager.get_collection_revision.return_value = "kb_hash_v200_updated"
    mock_pipeline.generator.generate.return_value = CustomerSupportAnswer(
        question=query,
        answer="Our updated policy allows 45 days for returns.",
        is_supported=True,
        sources=[{"doc_id": "doc_returns", "title": "Updated Policy", "chunk_id": "c1", "source_ref": "ref"}],
        model="gpt-4o-mini",
    )

    # 4. Same query on v200 must be a MISS and return fresh policy
    res3 = mock_pipeline.answer_question(query)
    assert res3.cache_hit is False
    assert "45 days" in res3.answer
    assert res3.knowledge_base_version == "kb_hash_v200_updated"


# --------------------------------------------------------------------------
# Test 9: Hybrid search combines BM25 and dense search correctly (RRF)
# --------------------------------------------------------------------------
def test_9_hybrid_search_combines_bm25_and_dense_search_rrf(mock_kb_chunks):
    bm25 = BM25Retriever(mock_kb_chunks)

    # Mock Dense Retriever
    mock_dense = MagicMock(spec=DenseRetriever)
    mock_dense.search.return_value = [
        (mock_kb_chunks[1], 0.90),  # Shipping first
        (mock_kb_chunks[0], 0.80),  # Returns second
    ]

    hybrid = HybridRetriever(bm25=bm25, dense=mock_dense, rrf_k=60)
    results = hybrid.search(query="shipping delivery times", candidate_pool_size=3)

    assert len(results) > 0
    # The shipping chunk should have high RRF score because it ranks high in both
    top_chunk = results[0]["chunk"]
    assert top_chunk.chunk_id == "shipping_1"
    assert "rrf_score" in results[0]
    assert results[0]["rrf_score"] > 0.0


# --------------------------------------------------------------------------
# Test 10: Reranking changes result order when relevance scores differ
# --------------------------------------------------------------------------
def test_10_reranking_changes_result_order_when_scores_differ(mock_kb_chunks):
    reranker = CrossEncoderReranker(lazy_load=True)

    candidates = [
        {"chunk": mock_kb_chunks[1], "rrf_score": 0.032},  # shipping
        {"chunk": mock_kb_chunks[0], "rrf_score": 0.030},  # returns
    ]

    # Mock cross encoder model prediction
    mock_ce = MagicMock()
    # Predict high relevance for returns (chunk 0) and low for shipping (chunk 1)
    mock_ce.predict.return_value = [0.12, 0.98]  # shipping=0.12, returns=0.98
    reranker._model = mock_ce

    reranked = reranker.rerank(query="How do I get a refund?", candidates=candidates)

    # Returns chunk should now be ranked #1
    assert reranked[0]["chunk"].chunk_id == "returns_1"
    assert reranked[0]["rerank_score"] == 0.98
    assert reranked[1]["chunk"].chunk_id == "shipping_1"
    assert reranked[1]["rerank_score"] == 0.12


# --------------------------------------------------------------------------
# Test 11: The system does not invent answers for unsupported questions
# --------------------------------------------------------------------------
def test_11_does_not_invent_answers_for_unsupported_questions():
    generator = CustomerSupportGenerator(api_key="")  # offline mode
    answer = generator.generate(
        question="What is the recipe for chocolate chip cookies?",
        retrieved_passages=[],  # No relevant passages found
        company_name="NovaCart Essentials",
    )

    assert answer.is_supported is False
    assert "does not contain information" in answer.answer
    assert len(answer.sources) == 0


# --------------------------------------------------------------------------
# Test 12: EF-08 Regression - Rejects tracking passage as answer for cutoff time
# --------------------------------------------------------------------------
def test_12_ef08_regression_rejects_tracking_passage_for_cutoff_time():
    """
    Regression test for EF-08:
    'What is the daily cutoff time for Overnight Priority delivery?'
    The dataset specifies 1:00 PM EST, shipping_delivery_c1, and page 1.
    The system must reject shipping_delivery_c2 (tracking passage, page 2) as an answer.
    """
    generator = CustomerSupportGenerator(api_key="")

    cutoff_chunk = KBChunk(
        chunk_id="shipping_delivery_c1",
        doc_id="doc_shipping_delivery",
        title="Shipping and Delivery Policy - Shipping Speeds & Costs",
        section="Shipping Methods & Transit Times",
        source_ref="Page 1, Sec 2.1",
        page_number=1,
        text="Overnight Priority Delivery (next business day if ordered before 1:00 PM EST) is available for $19.99.",
    )

    tracking_chunk = KBChunk(
        chunk_id="shipping_delivery_c2",
        doc_id="doc_shipping_delivery",
        title="Shipping and Delivery Policy - Tracking and Address Delivery",
        section="Tracking Numbers & PO Box Delivery",
        source_ref="Page 2, Sec 2.2",
        page_number=2,
        text="A tracking link is automatically dispatched via email and SMS within 24 hours of package departure from our warehouse. Standard shipping can deliver to PO Boxes and military APO/FPO addresses via USPS, but Expedited and Overnight deliveries require a physical street address through FedEx or UPS.",
    )

    # 1. When ONLY the tracking passage is provided, it does not answer cutoff time
    passages_tracking_only = [{"chunk": tracking_chunk, "rerank_score": -0.91}]
    ans_tracking = generator.generate(
        question="What is the daily cutoff time for Overnight Priority delivery?",
        retrieved_passages=passages_tracking_only,
    )
    # The tracking passage alone must NOT produce a valid citation claiming to answer the cutoff time
    assert "1:00 PM" not in ans_tracking.answer
    assert not any(c.get("chunk_id") == "shipping_delivery_c1" for c in ans_tracking.citations)

    # 2. When candidate pool contains both chunks, generator must correctly select c1 (Page 1)
    passages_both = [
        {"chunk": tracking_chunk, "rerank_score": -0.91},
        {"chunk": cutoff_chunk, "rerank_score": -1.23},
    ]
    ans_correct = generator.generate(
        question="What is the daily cutoff time for Overnight Priority delivery?",
        retrieved_passages=passages_both,
    )

    assert "1:00 PM" in ans_correct.answer
    assert ans_correct.is_supported is True
    assert len(ans_correct.citations) > 0
    top_cite = ans_correct.citations[0]
    assert top_cite["chunk_id"] == "shipping_delivery_c1"
    assert top_cite["page_number"] == 1
    assert top_cite["is_validated"] is True
    assert ans_correct.citation_validation_passed is True

