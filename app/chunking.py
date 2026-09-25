"""Text extraction and chunking.

Strategy: chunk per page (so every chunk carries an exact page citation), packing
whole sentences up to ~chunk_size characters with ~chunk_overlap characters of
trailing sentences repeated at the start of the next chunk. Sentence-aligned
boundaries keep chunks readable when shown to the user as evidence.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n\s*\n")


class ExtractionError(Exception):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int  # 1-based; text files are treated as a single page


def extract_pages(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ExtractionError(f"unsupported file type: {suffix}")
    try:
        if suffix == ".pdf":
            reader = PdfReader(path)
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        else:
            pages = [(1, path.read_text(encoding="utf-8", errors="replace"))]
    except Exception as exc:  # corrupt / encrypted PDFs raise a variety of errors
        raise ExtractionError(f"could not read {path.name}: {exc}") from exc

    pages = [(n, text) for n, text in pages if text.strip()]
    if not pages:
        # Most likely a scanned PDF; would need OCR (e.g. Amazon Textract).
        raise ExtractionError(f"no extractable text in {path.name}")
    return pages


def _sentences(text: str) -> list[str]:
    parts = (re.sub(r"\s+", " ", p).strip() for p in _SENTENCE_SPLIT.split(text))
    return [p for p in parts if p]


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in pages:
        current: list[str] = []
        length = 0
        for sentence in _sentences(text):
            if current and length + len(sentence) > chunk_size:
                chunks.append(Chunk(" ".join(current), page_no))
                # Carry trailing sentences forward as overlap.
                carried: list[str] = []
                carried_len = 0
                for prev in reversed(current):
                    if carried_len + len(prev) > overlap:
                        break
                    carried.insert(0, prev)
                    carried_len += len(prev) + 1
                current, length = carried, carried_len
            current.append(sentence)
            length += len(sentence) + 1
        if current:
            chunks.append(Chunk(" ".join(current), page_no))
    return chunks
