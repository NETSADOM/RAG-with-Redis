"""
test_generator.py
Automated tests for LLMGenerator.
Verifies prompt construction, defense against prompt injection,
response parsing for both Groq and Ollama, connection error handling, and insufficient-context behavior.
Uses mocks to test external HTTP service interactions cleanly.
"""

from unittest.mock import patch, MagicMock
import pytest
import requests

from src.retriever import RetrievedChunk, RetrievalResult
from src.generator import LLMGenerator, GenerationResult


@pytest.fixture
def sample_retrieval_result() -> RetrievalResult:
    """Fixture providing a mock successful retrieval result."""
    chunk1 = RetrievedChunk(
        text="Dr. Elena Vance published research on October 14, 2024.",
        filename="rag_paper.pdf",
        page_number=1,
        total_pages=3,
        chunk_id="rag_paper.pdf_p1_c0",
        chunk_index=0,
        distance=0.42,
        similarity_score=0.58,
    )
    return RetrievalResult(
        query="Who is the author?",
        chunks=[chunk1],
        has_sufficient_evidence=True,
        closest_distance=0.42,
        relevance_threshold=0.80,
        message="Retrieved 1 chunk.",
    )


def test_build_prompt_structure(sample_retrieval_result: RetrievalResult):
    """Prompt must strictly enclose context within <context> tags and clearly label the question."""
    generator = LLMGenerator()
    prompt = generator.build_prompt("Who is the author?", sample_retrieval_result.chunks)

    assert "<context>" in prompt
    assert "</context>" in prompt
    assert "--- Reference Passage 1 [Source: rag_paper.pdf, Page: 1] ---" in prompt
    assert "Dr. Elena Vance" in prompt
    assert "User Question: Who is the author?" in prompt


def test_generate_groq_success(sample_retrieval_result: RetrievalResult):
    """When Groq responds with HTTP 200, generator should parse the OpenAI-compatible choices."""
    generator = LLMGenerator(provider="groq", model_name="qwen/qwen3.8-27b", api_key="gsk_mock_test_key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "The paper was written by Dr. Elena Vance [rag_paper.pdf, Page 1]."
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = generator.generate("Who is the author?", sample_retrieval_result)

        assert mock_post.called
        assert result.has_sufficient_context is True
        assert result.error is None
        assert "Dr. Elena Vance" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0]["filename"] == "rag_paper.pdf"
        assert result.citations[0]["page_number"] == 1


def test_generate_ollama_success(sample_retrieval_result: RetrievalResult):
    """When Ollama responds with HTTP 200, generator should parse the answer and retain citations."""
    generator = LLMGenerator(provider="ollama", model_name="llama3.2:1b")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "The author is Dr. Elena Vance [rag_paper.pdf, Page 1]."
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = generator.generate("Who is the author?", sample_retrieval_result)

        assert mock_post.called
        assert result.has_sufficient_context is True
        assert result.error is None
        assert "Dr. Elena Vance" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0]["filename"] == "rag_paper.pdf"
        assert result.citations[0]["page_number"] == 1


def test_generate_insufficient_evidence():
    """When retrieval result has insufficient evidence, LLM call must be skipped."""
    generator = LLMGenerator()
    empty_retrieval = RetrievalResult(
        query="Unrelated question",
        chunks=[],
        has_sufficient_evidence=False,
        closest_distance=1.05,
        relevance_threshold=0.80,
        message="Closest distance 1.05 exceeds threshold 0.80.",
    )

    with patch("requests.post") as mock_post:
        result = generator.generate("Unrelated question", empty_retrieval)

        # requests.post must NOT be called when evidence is missing
        assert not mock_post.called
        assert result.has_sufficient_context is False
        assert "not enough information" in result.answer.lower()


def test_generate_connection_error(sample_retrieval_result: RetrievalResult):
    """When Ollama daemon is offline, generator must return a friendly instructions message."""
    generator = LLMGenerator(provider="ollama", base_url="http://localhost:11434")

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        result = generator.generate("Who is the author?", sample_retrieval_result)

        assert result.error is not None
        assert "Cannot connect to Ollama" in result.error
        assert "ollama serve" in result.error


def test_generate_model_not_found(sample_retrieval_result: RetrievalResult):
    """When model is not downloaded (HTTP 404), return instructions on how to pull it."""
    generator = LLMGenerator(provider="ollama", model_name="llama3.2:1b")

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "model 'llama3.2:1b' not found"

    with patch("requests.post", return_value=mock_response):
        result = generator.generate("Who is the author?", sample_retrieval_result)

        assert result.error is not None
        assert "was not found in your Ollama installation" in result.error
        assert "ollama pull llama3.2:1b" in result.error
