"""
test_pdf_loader.py
Automated tests for PDF text extraction, page-number preservation,
empty page handling, and scanned PDF detection.
"""

import io
import pytest
from pathlib import Path
from src.pdf_loader import PDFLoader, DocumentPage, ScannedPDFError, PDFExtractionError


def create_test_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate a structurally compliant PDF with specified pages of text."""
    num_pages = len(pages_text)
    font_obj_id = 3 + num_pages
    next_obj_id = font_obj_id + 1

    page_objs = []
    stream_objs = []

    for i, text in enumerate(pages_text):
        page_id = 3 + i
        if text.strip():
            stream_id = next_obj_id
            next_obj_id += 1
            escaped_text = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            stream_content = f"BT /F1 12 Tf 50 750 Td ({escaped_text}) Tj ET".encode('latin-1')
            stream_obj = (
                f"{stream_id} 0 obj\n<< /Length {len(stream_content)} >>\nstream\n".encode('latin-1')
                + stream_content
                + b"\nendstream\nendobj\n"
            )
            stream_objs.append((stream_id, stream_obj))
            page_obj = (
                f"{page_id} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {stream_id} 0 R /Resources << /Font << /F1 {font_obj_id} 0 R >> >> >>\n"
                f"endobj\n"
            ).encode('latin-1')
        else:
            page_obj = (
                f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            ).encode('latin-1')
        page_objs.append((page_id, page_obj))

    kids_str = " ".join(f"{3+i} 0 R" for i in range(num_pages))
    pages_root_obj = f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n".encode('latin-1')
    catalog_obj = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    font_obj = f"{font_obj_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode('latin-1')

    all_objs = {1: catalog_obj, 2: pages_root_obj, font_obj_id: font_obj}
    for pid, pbytes in page_objs:
        all_objs[pid] = pbytes
    for sid, sbytes in stream_objs:
        all_objs[sid] = sbytes

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    total_objs = max(all_objs.keys()) + 1
    offsets = {}
    for oid in sorted(all_objs.keys()):
        offsets[oid] = out.tell()
        out.write(all_objs[oid])

    xref_offset = out.tell()
    out.write(f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode('latin-1'))
    for oid in range(1, total_objs):
        if oid in offsets:
            out.write(f"{offsets[oid]:010d} 00000 n \n".encode('latin-1'))
        else:
            out.write(b"0000000000 65535 f \n")

    out.write(
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode('latin-1')
    )
    return out.getvalue()


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Fixture creating a 3-page test PDF where page 2 is blank."""
    pdf_bytes = create_test_pdf_bytes([
        "Introduction to RAG pipelines and vector search.",
        "",  # Blank page
        "Detailed architecture of ChromaDB and embedding stores."
    ])
    file_path = tmp_path / "sample_doc.pdf"
    file_path.write_bytes(pdf_bytes)
    return file_path


@pytest.fixture
def scanned_pdf_path(tmp_path: Path) -> Path:
    """Fixture creating a 2-page PDF with no text (simulating scanned images)."""
    pdf_bytes = create_test_pdf_bytes(["", ""])
    file_path = tmp_path / "scanned_invoice.pdf"
    file_path.write_bytes(pdf_bytes)
    return file_path


def test_load_file_success(sample_pdf_path: Path):
    """Test extracting text page by page with correct page numbers and metadata."""
    loader = PDFLoader()
    pages = loader.load_file(sample_pdf_path)

    # 3 total pages in PDF, but page 2 is blank, so 2 DocumentPage objects are returned
    assert len(pages) == 2

    page_1 = pages[0]
    assert page_1.filename == "sample_doc.pdf"
    assert page_1.page_number == 1
    assert page_1.total_pages == 3
    assert "Introduction to RAG pipelines" in page_1.text
    assert page_1.metadata["page_number"] == 1
    assert page_1.metadata["total_pages"] == 3
    assert page_1.metadata["filename"] == "sample_doc.pdf"

    page_3 = pages[1]
    assert page_3.filename == "sample_doc.pdf"
    assert page_3.page_number == 3  # Correctly identifies as page 3, skipping page 2
    assert page_3.total_pages == 3
    assert "Detailed architecture of ChromaDB" in page_3.text


def test_scanned_pdf_detection(scanned_pdf_path: Path):
    """Test that scanned/image-only PDFs raise ScannedPDFError mentioning OCR."""
    loader = PDFLoader()
    with pytest.raises(ScannedPDFError) as exc_info:
        loader.load_file(scanned_pdf_path)

    assert "OCR" in str(exc_info.value)
    assert "scanned image" in str(exc_info.value)
    assert "NO extractable text was found" in str(exc_info.value)


def test_nonexistent_file():
    """Test error handling when the requested PDF file does not exist."""
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file("documents/non_existent_file.pdf")


def test_invalid_file_extension(tmp_path: Path):
    """Test error handling when a non-PDF file is provided."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Hello world")

    loader = PDFLoader()
    with pytest.raises(ValueError) as exc_info:
        loader.load_file(txt_file)
    assert "Only .pdf is supported" in str(exc_info.value)


def test_load_directory(tmp_path: Path):
    """Test scanning a directory and aggregating pages from multiple PDFs."""
    # Create two valid PDFs and one scanned PDF in directory
    doc1 = tmp_path / "doc1.pdf"
    doc1.write_bytes(create_test_pdf_bytes(["Document 1 Page 1", "Document 1 Page 2"]))

    doc2 = tmp_path / "doc2.pdf"
    doc2.write_bytes(create_test_pdf_bytes(["Document 2 Page 1"]))

    scanned = tmp_path / "scanned.pdf"
    scanned.write_bytes(create_test_pdf_bytes([""]))

    loader = PDFLoader(documents_dir=tmp_path)
    all_pages = loader.load_directory(tmp_path)

    # doc1 has 2 pages, doc2 has 1 page -> 3 pages extracted
    assert len(all_pages) == 3

    filenames = [p.filename for p in all_pages]
    assert filenames.count("doc1.pdf") == 2
    assert filenames.count("doc2.pdf") == 1
