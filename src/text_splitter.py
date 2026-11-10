"""
text_splitter.py
Splits extracted document pages into overlapping text chunks with preserved
metadata (source filename, page number, chunk index, and stable chunk IDs).
Preserves natural sentence boundaries and punctuation without dropping text.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from src.pdf_loader import DocumentPage
from src.config import config


@dataclass
class TextChunk:
    """Represents a discrete chunk of text extracted from a document page."""

    text: str
    chunk_id: str
    chunk_index: int
    filename: str
    page_number: int
    total_pages: int
    char_count: int
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = Path(self.filename).name
        if not self.metadata:
            self.metadata = {
                "chunk_id": self.chunk_id,
                "chunk_index": self.chunk_index,
                "filename": self.filename,
                "doc_name": self.filename,
                "doc_id": self.doc_id,
                "source": self.filename,
                "page_number": self.page_number,
                "total_pages": self.total_pages,
                "char_count": self.char_count,
            }
        else:
            self.metadata.setdefault("chunk_id", self.chunk_id)
            self.metadata.setdefault("chunk_index", self.chunk_index)
            self.metadata.setdefault("filename", self.filename)
            self.metadata.setdefault("doc_name", self.filename)
            self.metadata.setdefault("doc_id", self.doc_id)
            self.metadata.setdefault("page_number", self.page_number)
            self.metadata.setdefault("total_pages", self.total_pages)


class TextSplitter:
    """
    Splits text from DocumentPage objects into chunks of configurable size
    and overlap, using recursive natural language delimiters.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Initialize TextSplitter.

        Args:
            chunk_size: Target maximum chunk size in characters (default from config).
                        1 token ≈ 4 characters in English. (500 tokens ≈ 2000 chars).
            chunk_overlap: Overlap between consecutive chunks in characters (default from config).
                           (100 tokens ≈ 400 chars).
        """
        self.chunk_size = chunk_size if chunk_size is not None else config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})."
            )

    def _split_into_units(self, text: str) -> List[str]:
        """
        Break text into natural language units (paragraphs -> sentences -> words)
        while preserving terminal punctuation.
        """
        paragraphs = re.split(r'\n{2,}', text)
        units: List[str] = []

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # Split into sentences using lookbehind for sentence-ending punctuation
            sentences = re.split(r'(?<=[.?!])\s+', p)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue

                if len(s) <= self.chunk_size:
                    units.append(s)
                else:
                    # If an individual sentence exceeds chunk_size, split by words
                    words = s.split(' ')
                    word_buf: List[str] = []
                    buf_len = 0
                    for w in words:
                        added = len(w) + (1 if word_buf else 0)
                        if buf_len + added <= self.chunk_size:
                            word_buf.append(w)
                            buf_len += added
                        else:
                            if word_buf:
                                units.append(' '.join(word_buf))
                            # Handle an individual word longer than chunk_size
                            if len(w) > self.chunk_size:
                                for i in range(0, len(w), self.chunk_size):
                                    units.append(w[i:i + self.chunk_size])
                                word_buf = []
                                buf_len = 0
                            else:
                                word_buf = [w]
                                buf_len = len(w)
                    if word_buf:
                        units.append(' '.join(word_buf))

        return units

    def split_text(self, text: str) -> List[str]:
        """
        Split a block of text into overlapping chunks up to chunk_size.
        """
        raw = text.strip()
        if not raw:
            return []

        if len(raw) <= self.chunk_size:
            return [raw]

        units = self._split_into_units(raw)
        if not units:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit) + (1 if current else 0)

            if current_len + unit_len <= self.chunk_size:
                current.append(unit)
                current_len += unit_len
            else:
                if current:
                    chunk_text = ' '.join(current).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                    # Build overlap buffer from the end of current
                    overlap_units: List[str] = []
                    overlap_len = 0
                    for item in reversed(current):
                        item_len = len(item) + (1 if overlap_units else 0)
                        if overlap_len + item_len <= self.chunk_overlap:
                            overlap_units.insert(0, item)
                            overlap_len += item_len
                        else:
                            break

                    current = overlap_units
                    current_len = sum(len(x) for x in current) + max(0, len(current) - 1)

                current.append(unit)
                current_len += len(unit) + (1 if len(current) > 1 else 0)

        if current:
            chunk_text = ' '.join(current).strip()
            if chunk_text and (not chunks or chunk_text != chunks[-1]):
                chunks.append(chunk_text)

        return chunks

    def split_page(self, page: DocumentPage) -> List[TextChunk]:
        """
        Split a single DocumentPage into one or more TextChunk objects.
        Preserves page number, filename, and assigns deterministic chunk IDs.
        """
        raw_text = page.text.strip()
        if not raw_text:
            return []

        chunks_text = self.split_text(raw_text)

        result_chunks: List[TextChunk] = []
        for idx, text_block in enumerate(chunks_text):
            cleaned = text_block.strip()
            if not cleaned:
                continue

            # Deterministic, unique chunk ID: <filename>_p<page_number>_c<chunk_index>
            chunk_id = f"{page.filename}_p{page.page_number}_c{idx}"

            chunk = TextChunk(
                text=cleaned,
                chunk_id=chunk_id,
                chunk_index=idx,
                filename=page.filename,
                page_number=page.page_number,
                total_pages=page.total_pages,
                char_count=len(cleaned),
                doc_id=page.doc_id,
            )
            result_chunks.append(chunk)

        return result_chunks

    def split_pages(self, pages: List[DocumentPage]) -> List[TextChunk]:
        """
        Split a list of DocumentPage objects into TextChunk objects.
        """
        all_chunks: List[TextChunk] = []
        for page in pages:
            page_chunks = self.split_page(page)
            all_chunks.extend(page_chunks)

        return all_chunks
