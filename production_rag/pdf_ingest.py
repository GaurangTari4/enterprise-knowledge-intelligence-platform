from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pymupdf as fitz

from .text import normalize_whitespace, truncate_text


@dataclass(slots=True)
class PDFPageText:
    page_number: int
    text: str
    char_count: int


@dataclass(slots=True)
class PDFExtractionResult:
    path: Path
    page_count: int
    pages: list[PDFPageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


def extract_pdf_text(pdf_path: str | Path) -> PDFExtractionResult:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.name}")

    document = fitz.open(path)
    pages: list[PDFPageText] = []
    try:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text = normalize_whitespace(page.get_text("text"))
            pages.append(
                PDFPageText(
                    page_number=page_index + 1,
                    text=text,
                    char_count=len(text),
                )
            )
    finally:
        document.close()

    return PDFExtractionResult(path=path, page_count=len(pages), pages=pages)


def iter_page_summaries(result: PDFExtractionResult, max_length: int = 800) -> Iterable[str]:
    for page in result.pages:
        if not page.text:
            yield f"Page {page.page_number}: [no extractable text]"
            continue
        yield f"Page {page.page_number}: {truncate_text(page.text, max_length=max_length)}"
