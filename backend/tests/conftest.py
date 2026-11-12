import json
from unittest.mock import MagicMock
from backend.rag.generator import CustomerSupportAnswer
from backend.rag.pipeline import CustomerSupportRAGPipeline
from backend.rag.retrieval import KnowledgeBaseManager
import fakeredis
import pytest
from backend.cache.redis_cache import CustomerSupportCache
from backend.admission import RequestLimiter


@pytest.fixture(autouse=True)
def isolated_admission():
    from backend.main import app
    app.state.limiter = RequestLimiter()

@pytest.fixture(autouse=True)
def isolated_cache(request, monkeypatch):
    if request.node.path.name == "test_real_redis_integration.py":
        return
    def connect(self):
        self._client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(CustomerSupportCache, "_connect", connect)


@pytest.fixture
def pipeline(tmp_path):
    kb_path = tmp_path / "kb.json"
    kb_path.write_text(json.dumps({"documents": [{"doc_id": "doc", "title": "Returns", "chunks": [{"chunk_id": "c1", "text": "30 days"}]}]}))
    generator = MagicMock()
    generator.model = "test-model"
    generator.generate.return_value = CustomerSupportAnswer(question="Returns?", answer="30 days [1]", is_supported=True, citations=[{"chunk_id": "c1", "citation_number": 1}], model="test-model", retrieved_context=[{"chunk_id": "c1"}])
    reranker = MagicMock()
    reranker.model_name = "test-reranker"
    reranker.rerank.side_effect = lambda query, candidates, top_n: [{**item, "rerank_score": 1.0} for item in candidates[:top_n]]
    return CustomerSupportRAGPipeline(kb_manager=KnowledgeBaseManager(kb_path), generator=generator, reranker=reranker, cache=CustomerSupportCache(), lazy_dense=True)

