import json
import pytest
from unittest.mock import MagicMock
from backend.rag.retrieval import KnowledgeBaseManager, KBChunk, BM25Retriever, DenseRetriever, HybridRetriever
from backend.rag.reranker import CrossEncoderReranker
from backend.rag.generator import CustomerSupportGenerator, CustomerSupportAnswer
from backend.cache.redis_cache import CustomerSupportCache, RedisUnavailable
from backend.rag.pipeline import CustomerSupportRAGPipeline
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.rag.retrieval import KnowledgeBaseManager
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock
from unittest.mock import Mock
import time
from backend.rag.pipeline import CustomerSupportRAGPipeline, SupportRAGResponse


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
    """Assembles a lightweight RAG pipeline with mocked answer builder and memory cache."""
    kb_mgr = MagicMock(spec=KnowledgeBaseManager)
    kb_mgr.chunks = mock_kb_chunks
    kb_mgr.version_hash = "kb_hash_v100"
    kb_mgr.get_collection_revision.return_value = "kb_hash_v100"

    cache = CustomerSupportCache()
    cache.clear()

    generator = MagicMock(spec=CustomerSupportGenerator)
    generator.model = "gpt-4o-mini"
    generator.generate.return_value = CustomerSupportAnswer(
        question="How long do I have to return an item?",
        answer="You have 30 days from delivery to return physical items in their original condition.",
        is_supported=True,
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

    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.rerank.side_effect = lambda query, candidates, top_n: [{**item, "rerank_score": 1.0} for item in candidates[:top_n]]

    pipe = CustomerSupportRAGPipeline(
        kb_manager=kb_mgr,
        cache=cache,
        generator=generator,
        reranker=reranker,
        lazy_dense=True,
    )
    return pipe


def test_1_redis_checked_before_retrieval_or_generation(mock_pipeline):
    query = "What is the return window?"
    mock_pipeline.cache.get_answer = MagicMock(return_value=(None, False))

    mock_pipeline.answer_question(query)

    assert mock_pipeline.cache.get_answer.call_count == 1
    kwargs = mock_pipeline.cache.get_answer.call_args.kwargs
    assert kwargs["query"] == query
    assert kwargs["kb_version"] == "kb_hash_v100"
    assert kwargs["model"].startswith("gpt-4o-mini:")


def test_2_cache_hits_return_answers_and_sources_correctly(mock_pipeline):
    query = "What is the return window?"

    res1 = mock_pipeline.answer_question(query)
    assert res1.cache_hit is False

    res2 = mock_pipeline.answer_question(query)
    assert res2.cache_hit is True
    assert res2.answer == res1.answer
    assert len(res2.sources) == len(res1.sources)
    assert res2.sources[0]["doc_id"] == "doc_returns"
    assert res2.sources[0]["chunk_id"] == "returns_1"


def test_3_cache_hits_do_not_call_retrieval_reranking_or_llm(mock_pipeline):
    query = "How long to return?"

    mock_pipeline.answer_question(query)
    assert mock_pipeline.generator.generate.call_count == 1

    mock_pipeline.generator.generate.side_effect = RuntimeError("Answer builder must not run on a cache hit")
    mock_pipeline.reranker.rerank.side_effect = RuntimeError("Reranker must NOT be called on Cache HIT!")
    mock_pipeline.bm25_retriever.search = MagicMock(side_effect=RuntimeError("Retrieval must NOT be called!"))

    hit_res = mock_pipeline.answer_question(query)
    assert hit_res.cache_hit is True
    assert mock_pipeline.generator.generate.call_count == 1  # Still 1, not called again


def test_4_cache_misses_execute_full_pipeline(mock_pipeline):
    query = "Brand new query that is never cached"

    res = mock_pipeline.answer_question(query)
    assert res.cache_hit is False
    assert mock_pipeline.generator.generate.call_count == 1
    assert mock_pipeline.reranker.rerank.call_count == 1


def test_5_successful_responses_stored_in_redis(mock_pipeline):
    query = "Do you accept returns after 30 days?"
    res = mock_pipeline.answer_question(query)

    keys = list(mock_pipeline.cache._client.scan_iter(match="cs_rag:ans:*"))
    assert len(keys) == 1
    cached_payload = json.loads(mock_pipeline.cache._client.get(keys[0]))
    assert cached_payload["answer"] == res.answer
    assert cached_payload["sources"] == res.citations


def test_6_failed_responses_not_cached(mock_pipeline):
    query = "Query that yields error or unsupported answer"

    mock_pipeline.generator.generate.return_value = CustomerSupportAnswer(
        question=query,
        answer="I'm sorry, our documentation does not contain information on this topic.",
        is_supported=False,
        model="gpt-4o-mini",
        error="Unsupported topic",
    )

    res = mock_pipeline.answer_question(query)
    assert res.is_supported is False

    assert not list(mock_pipeline.cache._client.scan_iter(match="cs_rag:ans:*"))


def test_7_redis_failure_uses_uncached_generation(mock_pipeline):
    query = "What is the shipping cost?"

    mock_pipeline.cache.get_answer = MagicMock(side_effect=RedisUnavailable("Redis connection refused"))

    response = mock_pipeline.answer_question(query)
    assert response.is_supported and response.cache_degraded
    assert not response.cache_hit
    mock_pipeline.generator.generate.assert_called_once()


def test_8_cached_answers_invalid_after_kb_update(mock_pipeline):
    query = "What is the return period?"

    res1 = mock_pipeline.answer_question(query)
    assert res1.cache_hit is False

    res2 = mock_pipeline.answer_question(query)
    assert res2.cache_hit is True

    mock_pipeline.kb_manager.version_hash = "kb_hash_v200_updated"
    mock_pipeline.kb_manager.get_collection_revision.return_value = "kb_hash_v200_updated"
    mock_pipeline.generator.generate.return_value = CustomerSupportAnswer(
        question=query,
        answer="Our updated policy allows 45 days for returns.",
        is_supported=True,
        citations=[{"doc_id": "doc_returns", "title": "Updated Policy", "chunk_id": "c1", "source_ref": "ref"}],
        model="gpt-4o-mini",
    )

    res3 = mock_pipeline.answer_question(query)
    assert res3.cache_hit is False
    assert "45 days" in res3.answer
    assert res3.knowledge_base_version == "kb_hash_v200_updated"


def test_9_hybrid_search_combines_bm25_and_dense_search_rrf(mock_kb_chunks):
    bm25 = BM25Retriever(mock_kb_chunks)

    mock_dense = MagicMock(spec=DenseRetriever)
    mock_dense.search.return_value = [
        (mock_kb_chunks[1], 0.90),  # Shipping first
        (mock_kb_chunks[0], 0.80),  # Returns second
    ]

    hybrid = HybridRetriever(bm25=bm25, dense=mock_dense, rrf_k=60)
    results = hybrid.search(query="shipping delivery times", candidate_pool_size=3)

    assert len(results) > 0
    top_chunk = results[0]["chunk"]
    assert top_chunk.chunk_id == "shipping_1"
    assert "rrf_score" in results[0]
    assert results[0]["rrf_score"] > 0.0


def test_10_reranking_changes_result_order_when_scores_differ(mock_kb_chunks):
    reranker = CrossEncoderReranker(lazy_load=True)

    candidates = [
        {"chunk": mock_kb_chunks[1], "rrf_score": 0.032},  # shipping
        {"chunk": mock_kb_chunks[0], "rrf_score": 0.030},  # returns
    ]

    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.12, 0.98]  # shipping=0.12, returns=0.98
    reranker._model = mock_ce

    reranked = reranker.rerank(query="How do I get a refund?", candidates=candidates)

    assert reranked[0]["chunk"].chunk_id == "returns_1"
    assert reranked[0]["rerank_score"] == 0.98
    assert reranked[1]["chunk"].chunk_id == "shipping_1"
    assert reranked[1]["rerank_score"] == 0.12


def test_11_does_not_invent_answers_for_unsupported_questions():
    generator = CustomerSupportGenerator()
    answer = generator.generate(
        question="What is the recipe for chocolate chip cookies?",
        retrieved_passages=[],  # No relevant passages found
    )

    assert answer.is_supported is False
    assert "do not contain information" in answer.answer
    assert len(answer.citations) == 0


def test_13_admin_endpoints_protected_with_auth_token(mock_pipeline, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app, get_pipeline
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    client = TestClient(app)
    try:
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        assert client.post("/api/cache/clear").status_code == 503
        monkeypatch.setenv("ADMIN_API_KEY", "test-private-token")
        assert client.post("/api/cache/clear").status_code == 401
        assert client.post("/api/cache/clear", headers={"X-Admin-Token": "wrong"}).status_code == 401
        assert client.post("/api/cache/clear", headers={"X-Admin-Token": "test-private-token"}).json()["success"]
        assert client.post("/api/kb/update-policy", json={"chunk_id": "c1", "new_policy_text": "text"}).status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_14_policy_updates_safe_to_retry_and_rebuilds_indexes(mock_pipeline, tmp_path):
    kb_file = tmp_path / "knowledge_base.json"
    kb_content = {
        "company_name": "NovaCart Test",
        "version": "1.0.0",
        "documents": [
            {
                "doc_id": "doc_test_1",
                "title": "Test Policy",
                "version": "v1.0",
                "source_ref": "Page 1, Doc T-1",
                "chunks": [
                    {
                        "chunk_id": "test_c1",
                        "title": "Test Chunk 1",
                        "section": "General",
                        "page_number": 1,
                        "source_ref": "Page 1",
                        "source_url": "",
                        "text": "Original return period is 30 days.",
                    }
                ],
            }
        ],
    }
    kb_file.write_text(json.dumps(kb_content), encoding="utf-8")

    real_kb_mgr = KnowledgeBaseManager(kb_path=str(kb_file))
    mock_pipeline.kb_manager = real_kb_mgr

    mock_pipeline.bm25_retriever = BM25Retriever(real_kb_mgr.chunks)

    with pytest.raises(ValueError):
        mock_pipeline.stage_and_update_policy(
            chunk_id="nonexistent_chunk_id",
            new_text="Some text",
        )
    assert "Original return period is 30 days." in kb_file.read_text(encoding="utf-8")

    res = mock_pipeline.stage_and_update_policy(
        chunk_id="test_c1",
        new_text="Updated return period is now 45 days.",
    )

    assert res["chunk_id"] == "test_c1"
    assert res["old_version"] != res["new_version"]
    assert "45 days" in res["new_text"]
    assert "30 days" in res["old_text"]
    assert "Updated return period is now 45 days." in kb_file.read_text(encoding="utf-8")
    assert mock_pipeline.bm25_retriever.chunks[0].text == "Updated return period is now 45 days."


def test_cache_separates_configuration_and_preserves_metadata(pipeline):
    first = pipeline.answer_question("Returns?")
    hit = pipeline.answer_question("Returns?")
    assert hit.cache_hit and hit.citations == first.citations
    assert hit.retrieved_context == first.retrieved_context
    assert hit.model_used == first.model_used
    assert not pipeline.answer_question("Returns?", use_reranker=False).cache_hit
    assert not pipeline.answer_question("Returns?", top_k_final=1).cache_hit
    pipeline.generator.model = "local-extractive-v2"
    assert not pipeline.answer_question("Returns?").cache_hit


def test_metadata_filters_scope_retrieval_and_cache(pipeline):
    pipeline.answer_question("Returns?")
    filters = {"doc_id": "doc", "page_number": 1}
    scoped = pipeline.answer_question("Returns?", metadata_filters=filters)
    assert not scoped.cache_hit
    assert pipeline.generator.generate.call_args.kwargs['retrieved_passages']
    assert pipeline.answer_question("Returns?", metadata_filters={"page_number": 1, "doc_id": "doc"}).cache_hit
    pipeline.generator = CustomerSupportGenerator()
    absent = pipeline.answer_question("Returns?", metadata_filters={"doc_id": "absent"})
    assert not absent.cache_hit and not absent.is_supported
    assert not absent.retrieved_context and not absent.citations


def test_chat_api_accepts_metadata_constraints_and_rejects_unknown_fields(pipeline):
    from backend.main import app, get_pipeline
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        response = client.post('/api/chat', json={'question': 'Returns?', 'metadata_filters': {'doc_id': 'doc'}})
        assert response.status_code == 200
        assert pipeline.generator.generate.call_args.kwargs['retrieved_passages']
        for filters in ({'unknown': 'value'}, {'page_number': 0}):
            assert client.post('/api/chat', json={'question': 'Returns?', 'metadata_filters': filters}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_failed_index_rebuild_preserves_disk_memory_and_revision(pipeline):
    manager = pipeline.kb_manager
    before = manager.kb_path.read_bytes()
    revision = manager.version_hash
    with patch("backend.rag.pipeline.BM25Retriever", side_effect=RuntimeError("index failed")):
        with pytest.raises(RuntimeError):
            pipeline.stage_and_update_policy("c1", "45 days")
    assert pipeline.kb_manager is manager
    assert manager.version_hash == revision
    assert manager.kb_path.read_bytes() == before
    result = pipeline.stage_and_update_policy("c1", "45 days")
    assert result["old_version"] == revision
    assert result["new_version"] != revision
    assert pipeline.kb_manager.chunks[0].title == "Returns"
    assert pipeline.stage_and_update_policy("c1", "45 days")["new_version"] == result["new_version"]


def test_failed_disk_write_does_not_activate_indexes(pipeline):
    manager = pipeline.kb_manager
    with patch("pathlib.Path.replace", side_effect=OSError("disk unavailable")):
        with pytest.raises(OSError):
            pipeline.stage_and_update_policy("c1", "45 days")
    assert pipeline.kb_manager is manager
    assert "30 days" in manager.kb_path.read_text()


def test_policy_update_writes_through_before_success(pipeline):
    pipeline.answer_question("Returns?")
    result = pipeline.stage_and_update_policy("c1", "45 days")
    snapshot = pipeline.cache.get_knowledge_base(result["new_version"])
    assert snapshot["chunks"][0]["text"] == "45 days"
    assert "45 days" in pipeline.kb_manager.kb_path.read_text()
    assert pipeline.get_knowledge_base() == snapshot
    assert not pipeline.answer_question("Returns?").cache_hit
    pipeline.cache.clear()
    assert pipeline.get_knowledge_base() == snapshot
    assert pipeline.cache.get_knowledge_base(result["new_version"]) == snapshot


def test_policy_cache_write_failure_preserves_active_revision(pipeline):
    manager = pipeline.kb_manager
    before = manager.kb_path.read_bytes()
    with patch.object(pipeline.cache, "set_knowledge_base", side_effect=RedisUnavailable("unavailable")):
        with pytest.raises(RedisUnavailable):
            pipeline.stage_and_update_policy("c1", "45 days")
    assert pipeline.kb_manager is manager
    assert manager.kb_path.read_bytes() == before
    assert not manager.kb_path.with_suffix(".json.tmp").exists()
    result = pipeline.stage_and_update_policy("c1", "45 days")
    assert pipeline.cache.get_knowledge_base(result["new_version"])["chunks"][0]["text"] == "45 days"


def test_cache_outage_runs_uncached_generation(pipeline):
    pipeline.cache._client = None
    result = pipeline.answer_question("Returns?")
    assert result.is_supported and result.cache_degraded
    assert not result.cache_hit
    pipeline.generator.generate.assert_called_once()
    assert pipeline.cache.misses == 1


def test_api_question_cache_static_and_validation(pipeline):
    from backend.main import app, get_pipeline
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    client = TestClient(app)
    try:
        assert client.get("/").status_code == 200
        assert client.get("/public/app.js").status_code == 200
        assert client.post("/api/chat", json={"question": " "}).status_code == 400
        assert client.post("/api/chat", json={"question": "x" * 2001}).status_code == 422
        first = client.post("/api/chat", json={"question": "Returns?"}).json()
        hit = client.post("/api/chat", json={"question": "Returns?"}).json()
        assert not first["cache_hit"] and hit["cache_hit"]
        assert client.get("/api/metrics").json()["total_questions"] == 2
        assert client.get("/api/kb").json()["chunks"][0]["source_url"] == ""
    finally:
        app.dependency_overrides.clear()


def test_evaluation_does_not_count_api_errors_as_correct_refusal():
    from benchmarks.evaluate_quality import score_response
    from backend.rag.pipeline import SupportRAGResponse
    response = SupportRAGResponse(question="Unknown?", answer="Unavailable", cache_hit=False,
        latency_ms=0, sources=[], knowledge_base_version="v1", model_used="model",
        is_supported=False, error="generation_failed")
    item = {"id": "UN-01", "category": "unsupported", "question": "Unknown?"}
    assert not score_response(item, response)["clean_refusal"]


@pytest.mark.parametrize("answer,page,expected", [("1 PM EST [1]", 1, True), ("1 PM PST [1]", 1, False), ("1 PM EST [1]", 2, False)])
def test_cutoff_evaluation_requires_timezone_and_correct_page(answer, page, expected):
    from benchmarks.evaluate_quality import score_response
    from backend.rag.pipeline import SupportRAGResponse
    response = SupportRAGResponse(question="Cutoff?", answer=answer, cache_hit=False,
        latency_ms=0, sources=[], knowledge_base_version="v1", model_used="model",
        citations=[{"chunk_id": "shipping_delivery_c1", "page_number": page}])
    from benchmarks.evaluate_quality import DATASET
    item = next(item for item in json.loads(DATASET.read_text()) if item["id"] == "EF-08")
    assert score_response(item, response)["supported_answer_check"] == expected


def make_pipeline():
    pipeline = CustomerSupportRAGPipeline.__new__(CustomerSupportRAGPipeline)
    pipeline._lock = RLock()
    pipeline._inflight = {}
    pipeline.cache = Mock()
    pipeline.cache.normalize_query.side_effect = lambda q: q.lower().strip(" ?")
    pipeline.kb_manager = Mock()
    pipeline.kb_manager.get_collection_revision.return_value = "v1"
    pipeline.generator = Mock(model="model")
    return pipeline


@pytest.mark.parametrize("fails", [False, True])
def test_matching_requests_share_work_and_cleanup(fails):
    pipeline = make_pipeline()
    entered, release = Event(), Event()
    response = SupportRAGResponse("Returns?", "30 days", False, 10, [], "v1", "model")

    def execute(*args):
        entered.set()
        assert release.wait(3)
        if fails:
            raise ValueError("provider failed")
        return response

    pipeline._answer_question = Mock(side_effect=execute)
    with ThreadPoolExecutor(max_workers=5) as pool:
        first = pool.submit(pipeline.answer_question, "Returns?")
        assert entered.wait(3)
        others = [pool.submit(pipeline.answer_question, "returns") for _ in range(4)]
        time.sleep(0.1)
        release.set()
        for future in [first, *others]:
            if fails:
                with pytest.raises(ValueError, match="provider failed"):
                    future.result(timeout=3)
            else:
                assert future.result(timeout=3).answer == "30 days"
    assert pipeline._answer_question.call_count == 1
    assert not pipeline._inflight
    if not fails:
        assert others[0].result() is not first.result()


def test_different_questions_execute_independently():
    pipeline = make_pipeline()
    pipeline._answer_question = Mock(return_value=None)
    pipeline.answer_question("Returns?")
    pipeline.answer_question("Shipping?")
    assert pipeline._answer_question.call_count == 2
