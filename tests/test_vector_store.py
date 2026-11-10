"""
test_vector_store.py
Automated tests for EmbeddingGenerator and VectorStore.
Verifies embedding generation, vector dimensions, normalization,
persistent ChromaDB storage, duplicate prevention, updates, and deletion.
"""

import math
import pytest
from pathlib import Path

from src.embeddings import EmbeddingGenerator
from src.text_splitter import TextChunk
from src.vector_store import VectorStore


# ============================================================================
# Embedding Generator Tests
# ============================================================================

@pytest.fixture(scope="module")
def embedding_gen() -> EmbeddingGenerator:
    """Shared embedding generator fixture."""
    return EmbeddingGenerator()


def test_embedding_dimensions_and_normalization(embedding_gen: EmbeddingGenerator):
    """Test that generated vectors have dimension 384 and L2 norm equal to 1.0."""
    text = "Artificial intelligence and vector search fundamentals."
    vec = embedding_gen.embed_text(text)

    # MiniLM-L6-v2 dimension must be 384
    assert len(vec) == 384
    assert embedding_gen.dimension == 384

    # Calculate L2 norm: sqrt(sum(x^2))
    l2_norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(l2_norm, rel=1e-3) == 1.0


def test_embed_documents_batch(embedding_gen: EmbeddingGenerator):
    """Test batch embedding generation for multiple text snippets."""
    texts = [
        "First document text about neural networks.",
        "Second document text regarding database indexes.",
        "Third document text covering information retrieval.",
    ]
    vectors = embedding_gen.embed_documents(texts)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384
        l2_norm = math.sqrt(sum(x * x for x in v))
        assert pytest.approx(l2_norm, rel=1e-3) == 1.0


def test_embed_empty_text_raises_error(embedding_gen: EmbeddingGenerator):
    """Attempting to embed empty or whitespace-only text should raise ValueError."""
    with pytest.raises(ValueError):
        embedding_gen.embed_text("   \n\t  ")


# ============================================================================
# Vector Store Tests
# ============================================================================

@pytest.fixture
def temp_vector_store(tmp_path: Path, embedding_gen: EmbeddingGenerator) -> VectorStore:
    """Creates a temporary, isolated VectorStore instance."""
    db_dir = tmp_path / "chroma_test_db"
    return VectorStore(
        persist_dir=db_dir,
        collection_name="test_collection",
        embedding_generator=embedding_gen,
    )


def test_add_chunks_and_count(temp_vector_store: VectorStore):
    """Test adding chunks to ChromaDB and checking the total count."""
    chunks = [
        TextChunk(
            text="Chunk one about machine learning models.",
            chunk_id="doc1.pdf_p1_c0",
            chunk_index=0,
            filename="doc1.pdf",
            page_number=1,
            total_pages=2,
            char_count=40,
        ),
        TextChunk(
            text="Chunk two about vector indexing in ChromaDB.",
            chunk_id="doc1.pdf_p2_c0",
            chunk_index=0,
            filename="doc1.pdf",
            page_number=2,
            total_pages=2,
            char_count=44,
        ),
    ]

    added = temp_vector_store.add_chunks(chunks)
    assert added == 2
    assert temp_vector_store.count() == 2
    assert temp_vector_store.is_document_ingested("doc1.pdf") is True
    assert temp_vector_store.is_document_ingested("other.pdf") is False


def test_duplicate_ingestion_prevention(temp_vector_store: VectorStore):
    """Re-ingesting an unchanged document should NOT create duplicate records."""
    chunks = [
        TextChunk(
            text="Initial text for document A.",
            chunk_id="docA.pdf_p1_c0",
            chunk_index=0,
            filename="docA.pdf",
            page_number=1,
            total_pages=1,
            char_count=28,
        ),
    ]

    # First ingestion
    first_add = temp_vector_store.add_chunks(chunks)
    assert first_add == 1
    assert temp_vector_store.count() == 1

    # Second ingestion of the same file without force_reindex
    second_add = temp_vector_store.add_chunks(chunks, force_reindex=False)
    assert second_add == 0  # Should skip redundant ingestion
    assert temp_vector_store.count() == 1  # Record count must stay 1, no duplicates!


def test_document_deletion(temp_vector_store: VectorStore):
    """Deleting a document should remove all of its chunks while keeping other documents."""
    chunks_doc1 = [
        TextChunk(text="Doc 1 chunk", chunk_id="doc1.pdf_p1_c0", chunk_index=0, filename="doc1.pdf", page_number=1, total_pages=1, char_count=11),
    ]
    chunks_doc2 = [
        TextChunk(text="Doc 2 chunk", chunk_id="doc2.pdf_p1_c0", chunk_index=0, filename="doc2.pdf", page_number=1, total_pages=1, char_count=11),
    ]

    temp_vector_store.add_chunks(chunks_doc1)
    temp_vector_store.add_chunks(chunks_doc2)
    assert temp_vector_store.count() == 2

    # Delete doc1
    deleted = temp_vector_store.delete_document("doc1.pdf")
    assert deleted == 1
    assert temp_vector_store.count() == 1
    assert temp_vector_store.is_document_ingested("doc1.pdf") is False
    assert temp_vector_store.is_document_ingested("doc2.pdf") is True


def test_document_update(temp_vector_store: VectorStore):
    """Updating a document replaces old chunks with the new set."""
    old_chunks = [
        TextChunk(text="Old chunk 1", chunk_id="doc.pdf_p1_c0", chunk_index=0, filename="doc.pdf", page_number=1, total_pages=1, char_count=11),
    ]
    new_chunks = [
        TextChunk(text="New updated chunk 1", chunk_id="doc.pdf_p1_c0", chunk_index=0, filename="doc.pdf", page_number=1, total_pages=1, char_count=19),
        TextChunk(text="New updated chunk 2", chunk_id="doc.pdf_p1_c1", chunk_index=1, filename="doc.pdf", page_number=1, total_pages=1, char_count=19),
    ]

    temp_vector_store.add_chunks(old_chunks)
    assert temp_vector_store.count() == 1

    temp_vector_store.update_document("doc.pdf", new_chunks)
    assert temp_vector_store.count() == 2

    stored = temp_vector_store.get_document_chunks("doc.pdf")
    texts = [s["text"] for s in stored]
    assert "New updated chunk 1" in texts
    assert "New updated chunk 2" in texts


def test_persistence_across_restarts(tmp_path: Path, embedding_gen: EmbeddingGenerator):
    """Verify that records persist when closing and reopening ChromaDB."""
    db_dir = tmp_path / "persistent_chroma_dir"

    # Instance 1: write data
    store1 = VectorStore(persist_dir=db_dir, collection_name="persist_col", embedding_generator=embedding_gen)
    chunks = [
        TextChunk(text="Persistent text to test storage.", chunk_id="p.pdf_p1_c0", chunk_index=0, filename="p.pdf", page_number=1, total_pages=1, char_count=32),
    ]
    store1.add_chunks(chunks)
    assert store1.count() == 1
    del store1  # Close instance 1

    # Instance 2: reopen same directory
    store2 = VectorStore(persist_dir=db_dir, collection_name="persist_col", embedding_generator=embedding_gen)
    assert store2.count() == 1
    assert store2.is_document_ingested("p.pdf") is True
    chunks_reloaded = store2.get_document_chunks("p.pdf")
    assert len(chunks_reloaded) == 1
    assert chunks_reloaded[0]["text"] == "Persistent text to test storage."

