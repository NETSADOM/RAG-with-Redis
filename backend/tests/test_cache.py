import pytest
from unittest.mock import MagicMock, patch
import redis
from fastapi.testclient import TestClient
from backend.cache.redis_cache import CustomerSupportCache, RedisUnavailable
from backend.main import app, get_pipeline


@pytest.fixture
def clean_cache():
    cache = CustomerSupportCache()
    cache.clear()
    return cache


def test_hit_does_not_rewrite_answer_or_extend_ttl(clean_cache):
    clean_cache.set_answer("Returns?", "v1", "model", "30 days", [], ttl=30)
    key = clean_cache.generate_cache_key("Returns?", "v1", "model")
    original = clean_cache._client.get(key)
    before = clean_cache._client.pttl(key)
    assert clean_cache.get_answer("Returns?", "v1", "model")[1]
    assert clean_cache._client.get(key) == original
    assert 0 < clean_cache._client.pttl(key) <= before


def test_revision_change_bypasses_all_old_answers(clean_cache):
    for query in ("Returns?", "Shipping?"):
        clean_cache.set_answer(query, "old", "model", "answer", [])
        assert clean_cache.get_answer(query, "old", "model")[1]
        assert not clean_cache.get_answer(query, "new", "model")[1]


def test_clear_cache_preserves_unrelated_redis_keys(clean_cache):
    """
    Verify CustomerSupportCache.clear() removes only NovaCart keys (cs_rag:*),
    preserving unrelated application keys stored in the same Redis database.
    """
    clean_cache.set_answer(
        query="How long do I have to return an item?",
        kb_version="v1.0.0",
        model="gpt-4o-mini",
        answer="You have 30 days to return items.",
        sources=[{"doc_id": "doc_returns", "title": "Return Policy"}],
    )

    unrelated_key = "user_session:9948:auth"
    clean_cache._client.set(unrelated_key, "active_user_jwt_token_payload")

    _, hit = clean_cache.get_answer("How long do I have to return an item?", "v1.0.0", "gpt-4o-mini")
    assert hit is True
    assert clean_cache._client.get(unrelated_key) == "active_user_jwt_token_payload"

    cleared = clean_cache.clear()
    assert cleared is True

    _, hit_after = clean_cache.get_answer("How long do I have to return an item?", "v1.0.0", "gpt-4o-mini")
    assert hit_after is False

    unrelated_val = clean_cache._client.get(unrelated_key)
    assert unrelated_val == "active_user_jwt_token_payload"


CONNECT = CustomerSupportCache._connect


def test_startup_requires_redis_and_hides_credentials(monkeypatch):
    monkeypatch.setattr(CustomerSupportCache, "_connect", CONNECT)
    with patch("backend.cache.redis_cache.redis.Redis.from_url", side_effect=redis.ConnectionError("secret-password")):
        with pytest.raises(RedisUnavailable) as error:
            CustomerSupportCache(redis_url="rediss://default:secret-password@example.test:6379")
    assert "secret-password" not in str(error.value)


@pytest.mark.parametrize("ping_results", [[True], [redis.TimeoutError(), True]])
def test_hosted_tls_url_and_bounded_connection_pool(monkeypatch, ping_results):
    monkeypatch.setattr(CustomerSupportCache, "_connect", CONNECT)
    client = MagicMock()
    client.ping.side_effect = ping_results
    with patch("backend.cache.redis_cache.redis.Redis.from_url", return_value=client) as factory:
        cache = CustomerSupportCache(redis_url="rediss://default:password@example.test:6379")
    assert factory.call_args.args[0].startswith("rediss://")
    assert factory.call_args.kwargs["max_connections"] == 10
    assert client.ping.call_count == len(ping_results)
    cache.close()
    client.close.assert_called_once()


def test_redis_read_and_write_errors_are_not_cache_misses():
    cache = CustomerSupportCache()
    cache._client = MagicMock()
    cache._client.get.side_effect = redis.ConnectionError()
    with pytest.raises(RedisUnavailable):
        cache.get_answer("Returns?", "v1")
    cache._client.set.side_effect = redis.ConnectionError()
    with pytest.raises(RedisUnavailable):
        cache.set_answer("Returns?", "v1", "model", "30 days", [])


def test_health_reports_outage_and_recovery():
    from backend.main import app, get_pipeline
    pipeline = MagicMock()
    pipeline.kb_manager.version_hash = "v1"
    pipeline.kb_manager.chunks = []
    pipeline.generator.model = "grounded-test"
    pipeline.generator.configured = True
    pipeline.generator.last_error = None
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        pipeline.cache.is_available = False
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["cache_mode"] == "uncached"
        assert response.json()["redis_connected"] is False
        pipeline.cache.is_available = True
        assert client.get("/api/health").status_code == 200
        pipeline.answer_question.side_effect = RedisUnavailable("Redis is temporarily unavailable.")
        assert client.post("/api/chat", json={"question": "Returns?"}).status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_outage_cooldown_skips_repeated_network_calls_and_recovers():
    cache = CustomerSupportCache()
    client = MagicMock()
    cache._client = client
    client.get.side_effect = redis.TimeoutError()
    with patch("backend.cache.redis_cache.time.monotonic", return_value=100):
        for _ in range(50):
            with pytest.raises(RedisUnavailable):
                cache.get_answer("Returns?", "v1")
        assert client.get.call_count == 1
        assert not cache.is_available
        client.ping.assert_not_called()
    client.get.side_effect = None
    client.get.return_value = '{"answer":"30 days","sources":[]}'
    with patch("backend.cache.redis_cache.time.monotonic", return_value=131):
        assert cache.get_answer("Returns?", "v1")[1]
    assert cache._retry_after == 0


def test_chat_and_policy_browsing_work_during_runtime_outage(pipeline):
    pipeline.cache._client = MagicMock()
    pipeline.cache._client.get.side_effect = redis.ConnectionError()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        result = client.post("/api/chat", json={"question": "Returns?"})
        assert result.status_code == 200
        assert result.json()["cache_degraded"] and result.json()["citations"]
        assert not result.json()["cache_hit"]
        assert client.get("/api/kb").json()["chunks"][0]["text"] == "30 days"
        assert all(":lock:" in call.args[0] for call in pipeline.cache._client.set.call_args_list)
        assert pipeline.cache._client.get.call_count == 1
    finally:
        app.dependency_overrides.clear()


def test_failed_cache_write_does_not_discard_a_valid_answer(pipeline):
    pipeline.cache._client = MagicMock()
    pipeline.cache._client.get.return_value = None
    pipeline.cache._client.set.side_effect = redis.TimeoutError()
    result = pipeline.answer_question("Returns?")
    assert result.is_supported and result.citations and not result.error
    assert result.cache_degraded


def test_admin_policy_updates_still_require_redis(pipeline):
    manager = pipeline.kb_manager
    before = manager.kb_path.read_bytes()
    pipeline.cache._client = MagicMock()
    pipeline.cache._client.set.side_effect = redis.ConnectionError()
    with pytest.raises(RedisUnavailable):
        pipeline.stage_and_update_policy("c1", "45 days")
    assert pipeline.kb_manager is manager
    assert manager.kb_path.read_bytes() == before


def test_explicit_cache_bypass_does_not_wait_for_redis(pipeline):
    pipeline.cache._client = MagicMock()
    result = pipeline.answer_question("Returns?", skip_cache=True)
    assert result.is_supported
    assert not pipeline.cache._client.mock_calls


def test_optional_startup_recovers_after_cooldown(monkeypatch):
    monkeypatch.setattr(CustomerSupportCache, "_connect", CONNECT)
    client = MagicMock()
    with patch("backend.cache.redis_cache.redis.Redis.from_url", side_effect=[redis.ConnectionError(), client]):
        cache = CustomerSupportCache(require_connection=False)
        assert not cache.is_available
        cache._retry_after = 1
        assert cache.is_available
        assert cache._client is client
        assert cache._retry_after == 0


def test_distributed_lock_ownership(clean_cache):
    key = "cs_rag:lock:test"
    assert clean_cache.acquire_answer_lock(key, "owner", 30)
    assert not clean_cache.acquire_answer_lock(key, "other", 30)
    clean_cache.release_answer_lock(key, "other")
    assert clean_cache._client.get(key) == "owner"
    clean_cache.release_answer_lock(key, "owner")
    assert clean_cache.acquire_answer_lock(key, "other", 30)


def test_distributed_wait_timeout_runs_pipeline(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline.cache, "acquire_answer_lock", lambda *args: False)
    monkeypatch.setattr("backend.rag.pipeline.time.sleep", lambda seconds: None)
    with patch("backend.rag.pipeline.time.monotonic", side_effect=[0, 3001]):
        result = pipeline.answer_question("Returns?")
    assert result.is_supported
    pipeline.generator.generate.assert_called_once()
