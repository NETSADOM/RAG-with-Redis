import json
from threading import BoundedSemaphore
from unittest.mock import Mock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.admission import RequestLimiter, SupportBusy, generation_slot
from backend.cache.redis_cache import CustomerSupportCache
from backend.locking import single_worker
from backend.main import app, get_pipeline
from backend.rag.generator import CustomerSupportGenerator
from backend.rag.pipeline import SupportRAGResponse
from backend.rag.retrieval import KBChunk, DenseRetriever
from benchmarks.evaluate_quality import DATASET, factual_checks, score_response, validate_dataset, quality_failures


def test_all_evaluation_cases_have_consistent_expectations():
    data = json.loads(DATASET.read_text())
    validate_dataset(data)
    crypto = next(item for item in data if item["id"] == "UN-33")
    assert crypto["category"] == "exact_fact"
    assert crypto["expected_chunk_ids"] == ["payments_billing_c1"]


@pytest.mark.parametrize("answer", ["Returns are accepted for 900 days. [1]",
                                    "Returns are accepted for 30 days from delivery, or 900 days on request. [1]",
                                    "There is a return policy. [1]"])
def test_evaluator_rejects_wrong_or_missing_facts_despite_valid_citation(answer):
    item = next(item for item in json.loads(DATASET.read_text()) if item["id"] == "EF-01")
    response = SupportRAGResponse(question=item["question"], answer=answer, cache_hit=False, latency_ms=1,
                                  sources=[], knowledge_base_version="v1", model_used="test",
                                  citations=[{"chunk_id": "refunds_returns_c1", "page_number": 1}])
    result = score_response(item, response)
    assert result["expected_citation_coverage"] == 1
    assert result["supported_answer_check"] is False


def test_quality_gate_rejects_errors_and_low_scores():
    good = dict(errors=0, supported_answer_check_pct=100, clean_refusal_pct=100,
                retrieval_recall_pct=100, expected_citation_coverage_pct=100)
    assert not quality_failures(good)
    assert quality_failures({**good, "errors": 1})
    for metric in good.keys() - {"errors"}:
        assert quality_failures({**good, metric: 0})


def test_auditor_receives_exclusions_and_conflicting_documents():
    generator = CustomerSupportGenerator()
    passages = [{"chunk": KBChunk("c1", "returns", "Returns", "Window", "Page 1",
                                 "Returns are accepted for 30 days. Opened hygiene goods are excluded."),
                 "keyword_rank": 1, "semantic_score": 0.8, "rrf_score": 0.03},
                {"chunk": KBChunk("c2", "exceptions", "Exceptions", "Exceptions", "Page 2", "Different policy applies.")}]
    claim = {"text": "Returns are accepted for 30 days.", "chunk_id": "c1", "quote": "Returns are accepted for 30 days."}
    with patch.object(generator, "_complete", side_effect=[{"claims": [claim]}, {"supported": False}]) as complete:
        result = generator.generate("Can I return opened hygiene goods?", passages)
    audit = complete.call_args_list[1].args[1]
    assert "hygiene goods are excluded" in audit["documents"][0]["text"]
    assert len(audit["documents"]) == 2
    assert result.retrieved_context[0]["keyword_rank"] == 1
    assert not result.is_supported


def test_numeric_verdict_is_not_a_boolean_true():
    generator = CustomerSupportGenerator()
    chunk = KBChunk("c", "d", "Returns", "Window", "Page 1", "30 days")
    with patch.object(generator, "_complete", side_effect=[
        {"claims": [{"text": "30 days", "quote": "30 days", "chunk_id": "c"}]}, {"supported": 1, "omitted_conditions": [], "unanswered_parts": [], "conflicts": []}
    ]):
        assert not generator.generate("Returns?", [{"chunk": chunk}]).is_supported


def test_invented_number_is_rejected_without_trusting_auditor():
    generator = CustomerSupportGenerator()
    chunk = KBChunk("c", "d", "Returns", "Window", "Page 1", "Returns within 30 days.")
    with patch.object(generator, "_complete", return_value={"claims": [
        {"text": "Returns within 900 days.", "quote": chunk.text, "chunk_id": "c"}
    ]}) as complete:
        assert not generator.generate("Returns?", [{"chunk": chunk}]).is_supported
    assert complete.call_count == 1


@pytest.mark.parametrize("verdict", [
    {"supported": True},
    {"supported": True, "omitted_conditions": ["48-hour reporting deadline"], "unanswered_parts": [], "conflicts": []},
    {"supported": True, "omitted_conditions": [], "unanswered_parts": ["Fees"], "conflicts": []},
    {"supported": True, "omitted_conditions": [], "unanswered_parts": [], "conflicts": ["Unresolved policy conflict"]},
])
def test_audit_cannot_approve_an_answer_with_unresolved_issues(verdict):
    generator = CustomerSupportGenerator()
    chunk = KBChunk("c", "d", "Returns", "Window", "Page 1", "30 days")
    with patch.object(generator, "_complete", side_effect=[
        {"claims": [{"text": "30 days", "quote": "30 days", "chunk_id": "c"}]}, verdict
    ]):
        assert not generator.generate("Returns?", [{"chunk": chunk}]).is_supported


def test_equivalent_policy_citation_covers_the_expected_evidence_group():
    item = next(item for item in json.loads(DATASET.read_text()) if item["id"] == "CD-22")
    response = SupportRAGResponse(question=item["question"], answer=item["ground_truth"], cache_hit=False,
                                  latency_ms=1, sources=[], knowledge_base_version="v1", model_used="test",
                                  citations=[{"chunk_id": "order_cancellation_c2", "page_number": 1}])
    assert score_response(item, response)["supported_answer_check"] is True
    response.citations = [{"chunk_id": "unrelated", "page_number": 1}]
    assert score_response(item, response)["supported_answer_check"] is False


def test_missing_configuration_fails_before_redis_or_models(monkeypatch):
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with patch("backend.main.CustomerSupportCache") as cache:
        with pytest.raises(ValueError, match="API key and model"):
            with TestClient(app):
                pass
        cache.assert_not_called()


def test_readiness_and_provider_error_http_status(pipeline):
    generator = CustomerSupportGenerator()
    generator.api_key = ""
    pipeline.generator = generator
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        assert client.get("/api/health").status_code == 503
        with patch.object(generator, "_complete", side_effect=TimeoutError("private response")):
            result = client.post("/api/chat", json={"question": "Returns?"})
        assert result.status_code == 503
        assert result.json()["error"]
        assert pipeline.cache.exact_hits == 0
    finally:
        app.dependency_overrides.clear()


def test_rate_limit_and_admin_only_bypass(pipeline, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "private-test-key")
    app.state.limiter = RequestLimiter(limit=2)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        assert client.post("/api/chat", json={"question": "Returns?", "skip_cache": True}).status_code == 401
        assert client.post("/api/chat", json={"question": "Returns?", "skip_cache": True}, headers={"X-Admin-Token": "private-test-key"}).status_code == 200
        assert client.post("/api/chat", json={"question": "Returns?"}).status_code == 200
        limited = client.post("/api/chat", json={"question": "Returns?"}, headers={"X-Forwarded-For": "different-client"})
        assert limited.status_code == 429
        assert limited.headers["retry-after"] == "60"
    finally:
        app.dependency_overrides.clear()


def test_limiter_window_expiration_and_bounded_client_storage():
    now = [0]
    limiter = RequestLimiter(limit=1, max_clients=1, clock=lambda: now[0])
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert not limiter.allow("b")
    now[0] = 61
    assert limiter.allow("b")
    assert len(limiter._clients) == 1


def test_generation_capacity_applies_to_bypass_and_releases_after_failure(pipeline):
    pipeline._generation_slots = BoundedSemaphore(1)
    with generation_slot(pipeline._generation_slots):
        with pytest.raises(SupportBusy):
            pipeline.answer_question("Returns?", skip_cache=True)
    with patch.object(pipeline.generator, "generate", side_effect=ValueError("failure")):
        with pytest.raises(ValueError):
            pipeline.answer_question("Returns?", skip_cache=True)
    assert pipeline.answer_question("Returns?", skip_cache=True).is_supported


def test_cache_hits_work_while_generation_capacity_is_full(pipeline):
    pipeline.answer_question("Returns?")
    pipeline._generation_slots = BoundedSemaphore(1)
    with generation_slot(pipeline._generation_slots):
        assert pipeline.answer_question("Returns?").cache_hit


def test_degraded_reranking_is_not_cached(pipeline):
    pipeline.reranker.rerank.side_effect = lambda query, candidates, top_n: candidates[:top_n]
    assert not pipeline.answer_question("Returns?").rerank_used
    assert not pipeline.answer_question("Returns?").cache_hit
    pipeline.reranker.rerank.side_effect = lambda query, candidates, top_n: [{**c, "rerank_score": 1} for c in candidates[:top_n]]
    assert pipeline.answer_question("Returns?").rerank_used
    assert pipeline.answer_question("Returns?").cache_hit


def test_single_worker_lock_rejects_second_instance_and_releases(tmp_path):
    path = tmp_path / "server.lock"
    with single_worker(path):
        with pytest.raises(RuntimeError, match="one server worker"):
            with single_worker(path):
                pass
    with single_worker(path):
        pass


def test_policy_update_encodes_only_changed_passages():
    def chunk(cid, text):
        return KBChunk(cid, "d", "Policy", "Section", "Page 1", text)
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: np.array([[len(t), 1] for t in texts])
    first = DenseRetriever([chunk("a", "unchanged"), chunk("b", "old")], model=model)
    original = first.embeddings.copy()
    model.encode.reset_mock()
    second = DenseRetriever([chunk("a", "unchanged"), chunk("b", "new")], model=model, previous=first)
    assert model.encode.call_args.args[0] == ["Policy: new"]
    assert np.array_equal(first.embeddings, original)
    assert np.array_equal(second.embeddings[0], first.embeddings[0])
    assert first._encode_lock is second._encode_lock


def test_test_namespace_cleanup_preserves_application_data():
    application = CustomerSupportCache()
    isolated = CustomerSupportCache(namespace="cs_rag:test:isolated")
    isolated._client = application._client
    application.set_answer("Returns?", "v1", "model", "30 days", [])
    isolated.set_answer("Returns?", "v1", "model", "test answer", [])
    isolated.set_knowledge_base("v1", {"chunks": []})
    assert isolated.clear()
    assert application.get_answer("Returns?", "v1", "model")[1]
    assert not isolated.get_answer("Returns?", "v1", "model")[1]
    assert isolated.get_knowledge_base("v1") is None
    with pytest.raises(ValueError):
        CustomerSupportCache(namespace="cs_rag:*")


def test_benchmark_does_not_treat_failures_as_fast_successes():
    from benchmarks.benchmark import summarize
    failure = SupportRAGResponse("Q", "Unavailable", False, 1, [], "v1", "test", is_supported=False, error="failed")
    result = summarize([None, failure], 1)
    assert result["busy_rejections"] == 1 and result["errors"] == 1
    assert result["successful_answers_per_second"] == 0
    assert result["mean_success_latency_ms"] is None


def test_context_selection_preserves_best_hybrid_match_and_scores():
    from backend.rag.reranker import select_passages
    candidates = [{"chunk": KBChunk(str(i), "d", "Policy", "Section", "Page 1", "Text"),
                   "rrf_score": 1 / (i + 1)} for i in range(5)]
    ranked = [{**item, "rerank_score": i} for i, item in enumerate(reversed(candidates))]
    selected = select_passages(candidates, ranked, 4)
    assert len(selected) == 4
    assert len({p["chunk"].chunk_id for p in selected}) == 4
    assert candidates[0]["chunk"] in [p["chunk"] for p in selected]
    assert all("rerank_score" in p and "rrf_score" in p for p in selected)
    assert select_passages(candidates, ranked, 1) == ranked[:1]
    assert select_passages([], [], 4) == []


def test_malformed_output_is_not_counted_as_a_correct_refusal():
    generator = CustomerSupportGenerator()
    chunk = KBChunk("c", "d", "Returns", "Window", "Page 1", "30 days")
    with patch.object(generator, "_complete", return_value={"unexpected": "output"}):
        generated = generator.generate("Unrelated?", [{"chunk": chunk}])
    assert generated.refusal_reason == "invalid_claim"
    response = SupportRAGResponse("Unrelated?", generated.answer, False, 1, [], "v1", "test",
                                  is_supported=False, refusal_reason=generated.refusal_reason)
    assert not score_response({"id": "unsupported", "category": "unsupported", "question": "Unrelated?"}, response)["clean_refusal"]
