"""
test_retriever.py
Automated tests for the Retriever module.
Verifies semantic retrieval, relevance distance threshold filtering,
empty database handling, empty query handling, and metadata preservation.
"""

import pytest
from pathlib import Path

from src.embeddings import EmbeddingGenerator
from src.text_splitter import TextChunk
from src.vector_store import VectorStore
from src.retriever import Retriever


@pytest.fixture(scope="module")
def embedding_gen() -> EmbeddingGenerator:
    """Shared embedding generator fixture."""
    return EmbeddingGenerator()


@pytest.fixture
def populated_vector_store(tmp_path: Path, embedding_gen: EmbeddingGenerator) -> VectorStore:
    """Fixture providing a vector store pre-populated with controlled test documents."""
    db_dir = tmp_path / "chroma_retriever_test"
    store = VectorStore(
        persist_dir=db_dir,
        collection_name="retriever_test_col",
        embedding_generator=embedding_gen,
    )

    chunks = [
        TextChunk(
            text="Dr. Elena Vance published research on Retrieval-Augmented Generation on October 14, 2024.",
            chunk_id="rag_paper.pdf_p1_c0",
            chunk_index=0,
            filename="rag_paper.pdf",
            page_number=1,
            total_pages=3,
            char_count=92,
        ),
        TextChunk(
            text="Vector embeddings map words and sentences into 384-dimensional dense vectors using all-MiniLM-L6-v2.",
            chunk_id="rag_paper.pdf_p2_c0",
            chunk_index=0,
            filename="rag_paper.pdf",
            page_number=2,
            total_pages=3,
            char_count=101,
        ),
        TextChunk(
            text="Photosynthesis is the biological process used by plants to convert light energy into chemical energy.",
            chunk_id="biology.pdf_p1_c0",
            chunk_index=0,
            filename="biology.pdf",
            page_number=1,
            total_pages=1,
            char_count=101,
        ),
    ]

    store.add_chunks(chunks)
    return store


def test_relevant_query_retrieval(populated_vector_store: VectorStore, embedding_gen: EmbeddingGenerator):
    """A relevant question about the author of the RAG paper should retrieve page 1 of rag_paper.pdf."""
    retriever = Retriever(
        vector_store=populated_vector_store,
        embedding_generator=embedding_gen,
        top_k=3,
        relevance_threshold=0.80,
    )

    result = retriever.retrieve("Who is the author of the RAG paper and when was it published?")

    assert result.has_sufficient_evidence is True
    assert len(result.chunks) >= 1

    top_chunk = result.chunks[0]
    assert top_chunk.filename == "rag_paper.pdf"
    assert top_chunk.page_number == 1
    assert "Elena Vance" in top_chunk.text
    assert top_chunk.distance <= 0.80
    assert top_chunk.similarity_score >= 0.20


def test_irrelevant_query_filtered_by_threshold(populated_vector_store: VectorStore, embedding_gen: EmbeddingGenerator):
    """An entirely unrelated query should be rejected by the relevance threshold."""
    retriever = Retriever(
        vector_store=populated_vector_store,
        embedding_generator=embedding_gen,
        top_k=3,
        # Threshold at 0.85 will easily discard queries with distance >= 0.95
        relevance_threshold=0.85,
    )

    # Completely unrelated query (distances to all chunks are >= 0.95)
    result = retriever.retrieve("What is the best recipe for baking chocolate chip cookies in an oven?")

    # Even though KNN finds 'nearest' items, distance will exceed threshold!
    assert result.has_sufficient_evidence is False
    assert len(result.chunks) == 0
    assert "exceeds the relevance threshold" in result.message


def test_configurable_top_k(populated_vector_store: VectorStore, embedding_gen: EmbeddingGenerator):
    """Top-k parameter should limit the maximum number of chunks returned."""
    retriever = Retriever(
        vector_store=populated_vector_store,
        embedding_generator=embedding_gen,
        top_k=1,
        relevance_threshold=1.0,  # Permissive to test top_k capping
    )

    result = retriever.retrieve("science research", top_k=1)
    assert len(result.chunks) == 1

    result_2 = retriever.retrieve("science research", top_k=2)
    assert len(result_2.chunks) == 2


def test_empty_database_handling(tmp_path: Path, embedding_gen: EmbeddingGenerator):
    """Querying an empty database should return has_sufficient_evidence = False without error."""
    empty_store = VectorStore(
        persist_dir=tmp_path / "empty_db",
        collection_name="empty_col",
        embedding_generator=embedding_gen,
    )
    retriever = Retriever(vector_store=empty_store, embedding_generator=embedding_gen)

    result = retriever.retrieve("Any question?")
    assert result.has_sufficient_evidence is False
    assert len(result.chunks) == 0
    assert "empty" in result.message.lower()


def test_empty_query_handling(populated_vector_store: VectorStore, embedding_gen: EmbeddingGenerator):
    """Passing an empty or whitespace query should return gracefully."""
    retriever = Retriever(vector_store=populated_vector_store, embedding_generator=embedding_gen)

    result = retriever.retrieve("   \n\t  ")
    assert result.has_sufficient_evidence is False
    assert len(result.chunks) == 0
    assert "empty" in result.message.lower()


def test_metadata_and_citations(populated_vector_store: VectorStore, embedding_gen: EmbeddingGenerator):
    """Verify that citations and source documents are properly populated."""
    retriever = Retriever(vector_store=populated_vector_store, embedding_generator=embedding_gen)
    result = retriever.retrieve("How do vector embeddings work with all-MiniLM-L6-v2?")

    assert result.has_sufficient_evidence is True
    assert "rag_paper.pdf" in result.source_documents

    citations = result.citations
    assert len(citations) > 0
    first_citation = citations[0]
    assert "filename" in first_citation
    assert "page_number" in first_citation
    assert "distance" in first_citation
    assert "similarity" in first_citation



def test_hybrid_retrieval_and_rrf(populated_vector_store: VectorStore, embedding_gen: EmbeddingGenerator):
    """Hybrid mode combines dense and sparse signals, boosting exact keyword matches."""
    retriever = Retriever(
        vector_store=populated_vector_store,
        embedding_generator=embedding_gen,
        mode="hybrid",
        relevance_threshold=0.85,
    )

    # "384-dimensional dense vectors" is an exact keyword in page 2 of rag_paper.pdf
    result = retriever.retrieve("384-dimensional dense vectors", mode="hybrid")

    assert result.has_sufficient_evidence is True
    assert len(result.chunks) >= 1
    top_chunk = result.chunks[0]
    assert top_chunk.page_number == 2
    assert "384-dimensional" in top_chunk.text
    assert top_chunk.rrf_score is not None
    assert top_chunk.search_method == "hybrid"
