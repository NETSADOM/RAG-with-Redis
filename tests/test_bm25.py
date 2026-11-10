"""
test_bm25.py
Automated tests for BM25Index.
Verifies tokenization, term frequency scoring, inverse document frequency (IDF),
document removal, and index persistence across saves and loads.
"""

import pytest
from pathlib import Path
from src.bm25 import BM25Index


def test_bm25_tokenization():
    """Tokenization should convert to lowercase and extract alphanumeric words."""
    tokens = BM25Index.tokenize("Dr. Elena Vance published all-MiniLM-L6-v2 in 2024!")
    assert "elena" in tokens
    assert "vance" in tokens
    assert "minilm" in tokens
    assert "l6" in tokens
    assert "v2" in tokens
    assert "2024" in tokens


def test_bm25_search_scoring():
    """Documents containing exact query keywords must score higher than unrelated ones."""
    index = BM25Index()
    docs = [
        {"chunk_id": "c1", "text": "Dr. Elena Vance is an AI researcher in vector databases.", "metadata": {}},
        {"chunk_id": "c2", "text": "Photosynthesis produces oxygen and glucose in green leaves.", "metadata": {}},
    ]
    index.add_documents(docs)

    results = index.search("Elena Vance researcher", top_k=2)
    assert len(results) == 1  # Only c1 has any query terms
    assert results[0][0] == "c1"
    assert results[0][1] > 0.0

    bio_results = index.search("oxygen leaves", top_k=2)
    assert len(bio_results) == 1
    assert bio_results[0][0] == "c2"
    assert bio_results[0][1] > 0.0


def test_bm25_remove_document():
    """Removing a document should delete it from the corpus and inverted index."""
    index = BM25Index()
    docs = [
        {"chunk_id": "doc1", "text": "UniqueKeywordAlpha exists here.", "metadata": {}},
        {"chunk_id": "doc2", "text": "Standard common sentence.", "metadata": {}},
    ]
    index.add_documents(docs)

    assert len(index.search("UniqueKeywordAlpha")) == 1

    index.remove_document("doc1")
    assert len(index.search("UniqueKeywordAlpha")) == 0
    assert "doc1" not in index.corpus


def test_bm25_save_and_load(tmp_path: Path):
    """Saving and loading an index must preserve search results identically."""
    index1 = BM25Index()
    docs = [
        {"chunk_id": "c1", "text": "First chunk discussing neural search.", "metadata": {}},
        {"chunk_id": "c2", "text": "Second chunk describing information retrieval.", "metadata": {}},
    ]
    index1.add_documents(docs)
    original_results = index1.search("neural search", top_k=2)

    save_file = tmp_path / "bm25.json"
    index1.save(save_file)

    index2 = BM25Index.load(save_file)
    loaded_results = index2.search("neural search", top_k=2)

    assert len(original_results) == len(loaded_results)
    assert original_results[0][0] == loaded_results[0][0]
    assert pytest.approx(original_results[0][1], rel=1e-4) == loaded_results[0][1]

