"""
test_web.py
Automated tests for the FastAPI web frontend API.
Tests UI serving, status endpoint, documents listing, and query handling with mocks.
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

from src.web.app import app
from src.rag_pipeline import RAGAnswer

client = TestClient(app)


def test_serve_index_html():
    """GET / should serve the frontend index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "PDF RAG" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_get_status():
    """GET /api/status should return health and model metadata."""
    with patch("src.web.app.pipeline.generator.check_connection", return_value=True):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "llm_model" in data
        assert "total_records" in data


def test_get_documents():
    """GET /api/documents should return currently indexed documents."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "count" in data


def test_query_endpoint_success():
    """POST /api/query should process queries and return structured answers."""
    mock_rag_answer = RAGAnswer(
        question="Who is Dr. Elena Vance?",
        answer="Dr. Elena Vance is an AI researcher [paper.pdf, Page 1].",
        has_evidence=True,
        sources=[{"filename": "paper.pdf", "page_number": 1, "distance": 0.45, "similarity": 0.55}],
        retrieved_passages=[{"chunk_id": "c1", "filename": "paper.pdf", "page_number": 1, "text": "researcher", "distance": 0.45}],
        model_used="qwen/qwen3.8-27b",
    )

    with patch("src.web.app.pipeline.answer_question", return_value=mock_rag_answer):
        response = client.post("/api/query", json={"question": "Who is Dr. Elena Vance?"})
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "Who is Dr. Elena Vance?"
        assert "Elena Vance" in data["answer"]
        assert data["has_evidence"] is True
        assert len(data["sources"]) == 1


def test_query_endpoint_empty_question():
    """POST /api/query with empty string should return HTTP 400."""
    response = client.post("/api/query", json={"question": "   "})
    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]

