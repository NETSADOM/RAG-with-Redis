"""
test_rag_pipeline.py
Automated end-to-end integration tests for the RAGPipeline.
Verifies the complete pipeline flow:
PDF ingestion -> chunking -> vector storage -> semantic retrieval -> answer synthesis.
"""

from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path

from src.pdf_loader import DocumentPage
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline, RAGAnswer


@pytest.fixture(scope="module")
def embedding_gen() -> EmbeddingGenerator:
    """Shared embedding generator fixture."""
    return EmbeddingGenerator()


@pytest.fixture
def test_pipeline(tmp_path: Path, embedding_gen: EmbeddingGenerator) -> RAGPipeline:
    """Fixture creating an isolated RAGPipeline connected to a temporary ChromaDB."""
    db_dir = tmp_path / "pipeline_chroma_db"
    store = VectorStore(
        persist_dir=db_dir,
        collection_name="pipeline_test_col",
        embedding_generator=embedding_gen,
    )
    pipeline = RAGPipeline(
        embedding_generator=embedding_gen,
        vector_store=store,
    )
    return pipeline


def test_pipeline_ingestion_and_question_answering(test_pipeline: RAGPipeline):
    """Verify that ingested documents can be queried and answered with citations."""
    # Mock PDFLoader to return controlled pages without reading actual disk files
    mock_pages = [
        DocumentPage(
            text="The Apollo 11 mission landed humans on the Moon on July 20, 1969. Commander Neil Armstrong was the first to step onto the lunar surface.",
            filename="apollo.pdf",
            page_number=1,
            total_pages=1,
        )
    ]

    with patch.object(test_pipeline.loader, "load_directory", return_value=mock_pages):
        summary = test_pipeline.ingest_documents()
        assert summary["pages_extracted"] == 1
        assert summary["chunks_stored"] >= 1
        assert test_pipeline.vector_store.count() >= 1

    # Mock LLM generate endpoint (compatible with both Groq choices and Ollama response)
    answer_text = "Neil Armstrong stepped onto the lunar surface on July 20, 1969 [apollo.pdf, Page 1]."
    mock_llm_response = MagicMock()
    mock_llm_response.status_code = 200
    mock_llm_response.json.return_value = {
        "response": answer_text,
        "choices": [{"message": {"content": answer_text}}],
    }

    with patch("requests.post", return_value=mock_llm_response):
        rag_answer = test_pipeline.answer_question("When did Apollo 11 land on the Moon and who was the commander?")

        assert isinstance(rag_answer, RAGAnswer)
        assert rag_answer.has_evidence is True
        assert "Neil Armstrong" in rag_answer.answer
        assert len(rag_answer.sources) >= 1
        assert rag_answer.sources[0]["filename"] == "apollo.pdf"
        assert rag_answer.sources[0]["page_number"] == 1
        assert len(rag_answer.retrieved_passages) >= 1
        assert "Apollo 11 mission" in rag_answer.retrieved_passages[0]["text"]


def test_pipeline_insufficient_context_handling(test_pipeline: RAGPipeline):
    """Querying about a topic not in the documents must not call LLM and return transparent message."""
    mock_pages = [
        DocumentPage(
            text="Quantum computing utilizes qubits and superposition to perform complex calculations.",
            filename="quantum.pdf",
            page_number=1,
            total_pages=1,
        )
    ]

    with patch.object(test_pipeline.loader, "load_directory", return_value=mock_pages):
        test_pipeline.ingest_documents()

    with patch("requests.post") as mock_post:
        # Ask completely unrelated question
        rag_answer = test_pipeline.answer_question(
            "What is the best recipe for baking chocolate chip cookies in an oven?",
            relevance_threshold=0.80,
        )

        # Must NOT call the LLM
        assert not mock_post.called
        assert rag_answer.has_evidence is False
        assert "not enough information" in rag_answer.answer.lower()
        assert len(rag_answer.sources) == 0


def test_pipeline_empty_query(test_pipeline: RAGPipeline):
    """Passing an empty question should return an instructional message."""
    rag_answer = test_pipeline.answer_question("   ")
    assert rag_answer.has_evidence is False
    assert "non-empty" in rag_answer.answer.lower()

