"""
tests/test_multi_doc_scenarios.py
Automated tests covering all 8 multi-document requirements:
1. Upload one PDF and ask a question.
2. Upload five PDFs and ask a question across all of them.
3. Select one PDF and ask a document-specific question (verifying filtering).
4. Select multiple PDFs and ask a question across them.
5. Delete one document and verify its chunks are no longer retrieved.
6. Upload another PDF without affecting previously uploaded documents (incremental updates).
7. Verify that citations reference the correct document and page.
8. Verify that document filtering strictly prevents leaking outside the selected scope.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import pypdf

from src.config import AppConfig
from src.pdf_loader import PDFLoader
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import LLMGenerator
from src.rag_pipeline import RAGPipeline, RAGAnswer


def create_test_pdf(filepath: Path, pages_text: list[str]) -> Path:
    """Helper to create a real valid PDF with text on each page."""
    writer = pypdf.PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        # Note: in pypdf add_blank_page doesn't have native text draw without canvas/reportlab,
        # but we can set up annotations or text streams, or use PDFLoader mocking.
    return filepath


@pytest.fixture
def multi_doc_pipeline(tmp_path: Path):
    """Fixture initializing an isolated RAGPipeline with a temporary ChromaDB directory."""
    chroma_dir = tmp_path / "test_chroma_db"
    docs_dir = tmp_path / "test_docs"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(
        project_root=tmp_path,
        documents_dir=docs_dir,
        chroma_db_dir=chroma_dir,
        collection_name="multi_doc_test",
        relevance_threshold=0.85,
    )

    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(
        persist_dir=chroma_dir,
        collection_name="multi_doc_test",
        embedding_generator=embedder,
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=embedder,
        relevance_threshold=0.85,
    )
    # LLM generator with mocked responses for deterministic tests
    generator = LLMGenerator(provider="groq", model_name="qwen/qwen3.8-27b", api_key="test_key")

    pipeline = RAGPipeline(
        loader=PDFLoader(documents_dir=docs_dir),
        splitter=TextSplitter(chunk_size=1000, chunk_overlap=200),
        embedding_generator=embedder,
        vector_store=vector_store,
        retriever=retriever,
        generator=generator,
    )
    return pipeline


def test_scenario_1_upload_one_pdf_and_query(multi_doc_pipeline: RAGPipeline):
    """Scenario 1: Upload one PDF and ask a question."""
    from src.pdf_loader import DocumentPage

    pages = [
        DocumentPage(
            text="Artificial Intelligence ethics focuses on fairness, transparency, and accountability in algorithmic decisions.",
            filename="ai_ethics.pdf",
            page_number=1,
            total_pages=1,
        )
    ]

    with patch.object(multi_doc_pipeline.loader, "load_file", return_value=pages):
        summary = multi_doc_pipeline.ingest_file("ai_ethics.pdf")
        assert summary["chunks"] >= 1
        assert multi_doc_pipeline.vector_store.count() >= 1

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "AI ethics focuses on fairness, transparency, and accountability [ai_ethics.pdf, Page 1]."}}]
    }

    with patch("requests.post", return_value=mock_resp):
        ans = multi_doc_pipeline.answer_question("What does AI ethics focus on?")
        assert ans.has_evidence is True
        assert len(ans.sources) >= 1
        assert ans.sources[0]["filename"] == "ai_ethics.pdf"
        assert ans.sources[0]["page_number"] == 1
        assert "fairness" in ans.answer


def test_scenario_2_upload_five_pdfs_and_global_query(multi_doc_pipeline: RAGPipeline):
    """Scenario 2: Upload five PDFs and ask a question across all of them."""
    from src.pdf_loader import DocumentPage

    docs_data = [
        ("doc1_ethics.pdf", "AI ethics requires fairness and transparent models.", 1),
        ("doc2_quantum.pdf", "Quantum computing uses qubits and quantum superposition.", 1),
        ("doc3_solar.pdf", "Solar photovoltaic cells convert sunlight directly into clean electricity.", 1),
        ("doc4_neural.pdf", "Convolutional neural networks extract hierarchical spatial image features.", 1),
        ("doc5_space.pdf", "The Artemis mission aims to establish a permanent human outpost on the Moon.", 1),
    ]

    for fname, text, page in docs_data:
        pages = [DocumentPage(text=text, filename=fname, page_number=page, total_pages=1)]
        with patch.object(multi_doc_pipeline.loader, "load_file", return_value=pages):
            multi_doc_pipeline.ingest_file(fname)

    assert multi_doc_pipeline.vector_store.count() == 5
    docs_summary = multi_doc_pipeline.list_documents()
    assert len(docs_summary) == 5

    # Ask a question across all documents (document_ids=None)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Solar cells convert sunlight into electricity [doc3_solar.pdf, Page 1]."}}]
    }

    with patch("requests.post", return_value=mock_resp):
        ans = multi_doc_pipeline.answer_question("How do solar cells work?", document_ids=None)
        assert ans.has_evidence is True
        assert ans.sources[0]["filename"] == "doc3_solar.pdf"


def test_scenario_3_and_8_document_scope_filtering_strictness(multi_doc_pipeline: RAGPipeline):
    """
    Scenario 3 & 8: Select one PDF and ask a document-specific question.
    Verify that document filtering strictly excludes other documents even if they share terms.
    """
    from src.pdf_loader import DocumentPage

    # Ingest two documents discussing 'systems'
    d1_pages = [
        DocumentPage(
            text="Quantum computing systems utilize entangled qubits for exponential parallelism.",
            filename="quantum.pdf",
            page_number=1,
            total_pages=1,
        )
    ]
    d2_pages = [
        DocumentPage(
            text="Renewable energy systems utilize solar panels and battery storage for power.",
            filename="solar.pdf",
            page_number=1,
            total_pages=1,
        )
    ]

    with patch.object(multi_doc_pipeline.loader, "load_file", side_effect=[d1_pages, d2_pages]):
        multi_doc_pipeline.ingest_file("quantum.pdf")
        multi_doc_pipeline.ingest_file("solar.pdf")

    # Filter retrieval strictly to quantum.pdf
    res = multi_doc_pipeline.retriever.retrieve(
        query="What do modern systems utilize?",
        document_ids=["quantum.pdf"],
    )

    assert res.has_sufficient_evidence is True
    # All retrieved chunks MUST strictly belong to quantum.pdf
    for chunk in res.chunks:
        assert chunk.filename == "quantum.pdf"
        assert "solar.pdf" not in chunk.filename

    # Now filter retrieval strictly to solar.pdf
    res_solar = multi_doc_pipeline.retriever.retrieve(
        query="What do modern systems utilize?",
        document_ids=["solar.pdf"],
    )
    assert res_solar.has_sufficient_evidence is True
    for chunk in res_solar.chunks:
        assert chunk.filename == "solar.pdf"
        assert "quantum.pdf" not in chunk.filename


def test_scenario_4_select_multiple_pdfs(multi_doc_pipeline: RAGPipeline):
    """Scenario 4: Select multiple PDFs and ask a question across the chosen subset."""
    from src.pdf_loader import DocumentPage

    docs = [
        DocumentPage(text="Deep learning optimizes weight matrices via gradient descent.", filename="dl.pdf", page_number=1, total_pages=1),
        DocumentPage(text="Reinforcement learning uses reward functions and policy gradients.", filename="rl.pdf", page_number=1, total_pages=1),
        DocumentPage(text="A completely unrelated astronomy text about distant galaxies.", filename="astronomy.pdf", page_number=1, total_pages=1),
    ]

    for d in docs:
        with patch.object(multi_doc_pipeline.loader, "load_file", return_value=[d]):
            multi_doc_pipeline.ingest_file(d.filename)

    # Scope search to dl.pdf and rl.pdf only
    scope = ["dl.pdf", "rl.pdf"]
    retrieval = multi_doc_pipeline.retriever.retrieve("What algorithms optimize models?", document_ids=scope)

    assert retrieval.has_sufficient_evidence is True
    filenames = {c.filename for c in retrieval.chunks}
    # Must only include chunks from the selected scope
    assert filenames.issubset(set(scope))
    assert "astronomy.pdf" not in filenames


def test_scenario_5_delete_document_and_verify_removal(multi_doc_pipeline: RAGPipeline):
    """Scenario 5: Delete one document and verify its chunks are no longer retrieved."""
    from src.pdf_loader import DocumentPage

    page = DocumentPage(
        text="SpecialUniqueKeyword999 exists only in this temporary document.",
        filename="temp_doc.pdf",
        page_number=1,
        total_pages=1,
    )
    with patch.object(multi_doc_pipeline.loader, "load_file", return_value=[page]):
        multi_doc_pipeline.ingest_file("temp_doc.pdf")

    assert multi_doc_pipeline.vector_store.count() == 1

    # Verify it can be retrieved before deletion
    pre_del = multi_doc_pipeline.retriever.retrieve("SpecialUniqueKeyword999")
    assert pre_del.has_sufficient_evidence is True

    # Delete the document
    deleted = multi_doc_pipeline.delete_document("temp_doc.pdf")
    assert deleted >= 1
    assert multi_doc_pipeline.vector_store.count() == 0

    # Verify BM25 and ChromaDB no longer return it
    post_del = multi_doc_pipeline.retriever.retrieve("SpecialUniqueKeyword999")
    assert post_del.has_sufficient_evidence is False
    assert len(post_del.chunks) == 0


def test_scenario_6_incremental_upload_without_affecting_existing(multi_doc_pipeline: RAGPipeline):
    """Scenario 6: Upload another PDF without affecting or duplicating previously uploaded documents."""
    from src.pdf_loader import DocumentPage

    p1 = DocumentPage(text="First initial document text.", filename="first.pdf", page_number=1, total_pages=1)
    with patch.object(multi_doc_pipeline.loader, "load_file", return_value=[p1]):
        multi_doc_pipeline.ingest_file("first.pdf")

    assert multi_doc_pipeline.vector_store.count() == 1

    # Upload second document
    p2 = DocumentPage(text="Second added document text.", filename="second.pdf", page_number=1, total_pages=1)
    with patch.object(multi_doc_pipeline.loader, "load_file", return_value=[p2]):
        multi_doc_pipeline.ingest_file("second.pdf")

    assert multi_doc_pipeline.vector_store.count() == 2
    docs = multi_doc_pipeline.list_documents()
    doc_names = [d["filename"] for d in docs]
    assert "first.pdf" in doc_names
    assert "second.pdf" in doc_names


def test_scenario_7_citations_match_correct_document_and_page(multi_doc_pipeline: RAGPipeline):
    """Scenario 7: Verify that citations reference the exact document and page number."""
    from src.pdf_loader import DocumentPage

    p1 = DocumentPage(text="Introduction and background.", filename="multi_page.pdf", page_number=1, total_pages=2)
    p2 = DocumentPage(text="The breakthrough experiment succeeded on page two.", filename="multi_page.pdf", page_number=2, total_pages=2)

    with patch.object(multi_doc_pipeline.loader, "load_file", return_value=[p1, p2]):
        multi_doc_pipeline.ingest_file("multi_page.pdf")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "The experiment succeeded on page two [multi_page.pdf, Page 2]."}}]
    }

    with patch("requests.post", return_value=mock_resp):
        ans = multi_doc_pipeline.answer_question("Where did the experiment succeed?")
        assert ans.has_evidence is True
        # Verify citation metadata
        p2_citations = [s for s in ans.sources if s["page_number"] == 2]
        assert len(p2_citations) >= 1
        assert p2_citations[0]["filename"] == "multi_page.pdf"


def test_scenario_redis_cache_key_generation(multi_doc_pipeline: RAGPipeline):
    """Scenario: Redis Cache Readiness - verify deterministic cache key derivation."""
    key1 = multi_doc_pipeline.compute_cache_key("What is AI?", document_ids=["doc1.pdf", "doc2.pdf"])
    key2 = multi_doc_pipeline.compute_cache_key("What is AI?", document_ids=["doc2.pdf", "doc1.pdf"])
    key3 = multi_doc_pipeline.compute_cache_key("What is AI?", document_ids=["doc1.pdf"])

    # Key must be deterministic regardless of scope order
    assert key1 == key2
    # Key must differ when document scope changes
    assert key1 != key3
    assert len(key1) == 64  # SHA256 hex string

