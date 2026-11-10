"""
test_text_splitter.py
Automated tests for text chunking, overlap verification,
metadata preservation, and edge cases (short pages, long pages).
"""

import pytest
from src.pdf_loader import DocumentPage
from src.text_splitter import TextSplitter, TextChunk


def test_short_page_no_split():
    """A short page smaller than chunk_size should stay as a single chunk."""
    page = DocumentPage(
        text="This is a short paragraph that easily fits in a single chunk.",
        filename="short.pdf",
        page_number=1,
        total_pages=1,
    )
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_page(page)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == page.text
    assert chunk.chunk_id == "short.pdf_p1_c0"
    assert chunk.chunk_index == 0
    assert chunk.filename == "short.pdf"
    assert chunk.page_number == 1
    assert chunk.total_pages == 1
    assert chunk.metadata["page_number"] == 1


def test_long_page_chunking_and_overlap():
    """A long page must be split into multiple chunks, respecting size and overlap."""
    sentences = [
        f"Sentence number {i} provides detailed academic facts about vector databases and retrieval."
        for i in range(1, 20)
    ]
    full_text = " ".join(sentences)

    page = DocumentPage(
        text=full_text,
        filename="long_doc.pdf",
        page_number=2,
        total_pages=5,
    )

    # Use smaller chunk_size to test splitting behavior deterministically
    chunk_size = 250
    chunk_overlap = 60
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_page(page)

    assert len(chunks) > 1

    # Verify each chunk has metadata and valid length
    for idx, chunk in enumerate(chunks):
        assert chunk.filename == "long_doc.pdf"
        assert chunk.page_number == 2
        assert chunk.chunk_index == idx
        assert chunk.chunk_id == f"long_doc.pdf_p2_c{idx}"
        assert len(chunk.text) > 0

    # Verify overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        chunk_a = chunks[i].text
        chunk_b = chunks[i + 1].text
        # Check that some words from the end of chunk_a appear in chunk_b
        words_a = chunk_a.split()
        last_few_words = " ".join(words_a[-3:])
        assert last_few_words in chunk_b, f"Overlap missing between chunk {i} and {i+1}."


def test_no_text_loss():
    """Every sentence in the original document must appear in at least one chunk."""
    sentences = [
        "First sentence introduces the primary thesis.",
        "Second sentence describes the empirical methodology.",
        "Third sentence details the mathematical formulation of loss.",
        "Fourth sentence presents experimental results across five benchmarks.",
        "Fifth sentence concludes with implications for future research.",
    ]
    full_text = " ".join(sentences)

    page = DocumentPage(
        text=full_text,
        filename="thesis.pdf",
        page_number=1,
        total_pages=1,
    )

    splitter = TextSplitter(chunk_size=120, chunk_overlap=30)
    chunks = splitter.split_page(page)

    for sentence in sentences:
        found = any(sentence in chunk.text for chunk in chunks)
        assert found, f"Sentence '{sentence}' was lost during chunking!"


def test_empty_or_whitespace_page():
    """Empty or whitespace-only pages should produce zero chunks."""
    page = DocumentPage(
        text="   \n\n   \t  ",
        filename="empty.pdf",
        page_number=1,
        total_pages=1,
    )
    splitter = TextSplitter()
    chunks = splitter.split_page(page)
    assert len(chunks) == 0


def test_invalid_overlap_raises_error():
    """Overlap greater than or equal to chunk size should raise a ValueError."""
    with pytest.raises(ValueError) as exc:
        TextSplitter(chunk_size=200, chunk_overlap=200)
    assert "strictly less than" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        TextSplitter(chunk_size=200, chunk_overlap=250)
    assert "strictly less than" in str(exc.value)


def test_split_multiple_pages():
    """Test splitting multiple DocumentPages preserves their respective page numbers."""
    pages = [
        DocumentPage(text="Page one text content.", filename="book.pdf", page_number=1, total_pages=2),
        DocumentPage(text="Page two text content.", filename="book.pdf", page_number=2, total_pages=2),
    ]
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_pages(pages)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == "book.pdf_p1_c0"
    assert chunks[1].page_number == 2
    assert chunks[1].chunk_id == "book.pdf_p2_c0"

