"""
pdf_loader.py
Extracts text from PDF documents page by page while preserving metadata
such as the original filename, page number, and total page count.
Handles empty pages, extraction errors, and detects scanned/image-only PDFs.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pypdf
from pypdf.errors import PdfReadError

from src.config import config


@dataclass
class DocumentPage:
    """Represents the extracted text and metadata of a single PDF page."""

    text: str
    filename: str
    page_number: int  # 1-indexed (Page 1, Page 2, ...)
    total_pages: int
    doc_id: str = ""  # Unique identifier for the document
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.doc_id:
            # Deterministic, clean doc_id derived from filename
            self.doc_id = Path(self.filename).name
        if not self.metadata:
            self.metadata = {
                "source": self.filename,
                "filename": self.filename,
                "doc_id": self.doc_id,
                "doc_name": self.filename,
                "page_number": self.page_number,
                "total_pages": self.total_pages,
            }
        else:
            self.metadata.setdefault("doc_id", self.doc_id)
            self.metadata.setdefault("doc_name", self.filename)
            self.metadata.setdefault("filename", self.filename)


class PDFExtractionError(Exception):
    """Base exception for PDF loading errors."""
    pass


class ScannedPDFError(PDFExtractionError):
    """Raised when a PDF contains pages but zero extractable text (image-only/scanned)."""
    pass


class PDFLoader:
    """
    Loads and extracts text from individual PDF files or an entire directory.
    Preserves page boundaries, source file names, and detects scanned/empty documents.
    """

    def __init__(self, documents_dir: Optional[Path | str] = None):
        self.documents_dir = Path(documents_dir) if documents_dir else config.documents_dir

    def load_file(self, file_path: Path | str) -> List[DocumentPage]:
        """
        Extract text page by page from a single PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list of DocumentPage objects for all pages containing text.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a PDF.
            PDFExtractionError: If the PDF is corrupted, password-protected, or unreadable.
            ScannedPDFError: If the PDF has pages but contains zero extractable text (OCR needed).
        """
        path = Path(file_path).resolve()
        filename = path.name

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file format '{path.suffix}'. Only .pdf is supported: {filename}")

        try:
            reader = pypdf.PdfReader(str(path))
        except PdfReadError as e:
            raise PDFExtractionError(f"Failed to read corrupted or malformed PDF '{filename}': {e}") from e
        except Exception as e:
            raise PDFExtractionError(f"Unexpected error opening PDF '{filename}': {e}") from e

        if reader.is_encrypted:
            try:
                # Try decrypting with empty password
                decrypted = reader.decrypt("")
                if not decrypted:
                    raise PDFExtractionError(
                        f"PDF '{filename}' is password-protected and cannot be extracted without the password."
                    )
            except Exception as e:
                raise PDFExtractionError(
                    f"PDF '{filename}' is password-protected: {e}"
                ) from e

        total_pages = len(reader.pages)
        if total_pages == 0:
            print(f"[Warning] PDF '{filename}' contains 0 pages.")
            return []

        extracted_pages: List[DocumentPage] = []
        empty_page_numbers: List[int] = []
        total_extracted_characters = 0

        for page_idx, page in enumerate(reader.pages):
            page_number = page_idx + 1  # 1-indexed for human readability
            try:
                text = page.extract_text() or ""
                cleaned_text = text.strip()
            except Exception as e:
                print(f"[Warning] Extraction failed on page {page_number} of '{filename}': {e}")
                cleaned_text = ""

            if not cleaned_text:
                empty_page_numbers.append(page_number)
                continue

            total_extracted_characters += len(cleaned_text)
            extracted_pages.append(
                DocumentPage(
                    text=cleaned_text,
                    filename=filename,
                    page_number=page_number,
                    total_pages=total_pages,
                )
            )

        # Inform the user about empty or non-text pages
        if empty_page_numbers:
            print(
                f"[Notice] '{filename}': Pages {empty_page_numbers} contained no extractable text "
                f"(blank page, image, or graphic)."
            )

        # If document has pages but literally 0 extractable text across all pages
        if total_pages > 0 and total_extracted_characters == 0:
            error_msg = (
                f"PDF '{filename}' contains {total_pages} page(s), but NO extractable text was found. "
                "This document appears to be a scanned image or photo PDF. "
                "Optical Character Recognition (OCR) is required to extract text from scanned documents."
            )
            print(f"[Error] {error_msg}")
            raise ScannedPDFError(error_msg)

        return extracted_pages

    def load_directory(self, dir_path: Optional[Path | str] = None) -> List[DocumentPage]:
        """
        Scan a directory and load all supported PDF files.

        Args:
            dir_path: Optional directory path. Defaults to configured documents_dir.

        Returns:
            A combined list of DocumentPage objects from all successfully processed PDFs.
        """
        target_dir = Path(dir_path) if dir_path else self.documents_dir

        if not target_dir.exists():
            print(f"[Error] Documents directory does not exist: {target_dir}")
            return []

        pdf_files = sorted(
            [p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        )

        if not pdf_files:
            print(f"[Notice] No PDF documents found in: {target_dir}")
            return []

        print(f"\nFound {len(pdf_files)} PDF document(s) in {target_dir}:")
        for f in pdf_files:
            print(f"  - {f.name}")

        all_pages: List[DocumentPage] = []
        success_count = 0
        failed_files: Dict[str, str] = {}

        for pdf_file in pdf_files:
            print(f"\nProcessing '{pdf_file.name}'...")
            try:
                pages = self.load_file(pdf_file)
                all_pages.extend(pages)
                success_count += 1
                print(f"  ✓ Successfully extracted {len(pages)} page(s) with text from '{pdf_file.name}'.")
            except (ScannedPDFError, PDFExtractionError, Exception) as e:
                failed_files[pdf_file.name] = str(e)
                print(f"  ✗ Failed to process '{pdf_file.name}': {e}")

        print("\n--- PDF Ingestion Summary ---")
        print(f"Total files discovered : {len(pdf_files)}")
        print(f"Successfully processed : {success_count}")
        print(f"Failed files           : {len(failed_files)}")
        print(f"Total text pages loaded: {len(all_pages)}")

        if failed_files:
            print("\nDetails of failed files:")
            for fname, reason in failed_files.items():
                print(f"  - {fname}: {reason}")

        return all_pages

