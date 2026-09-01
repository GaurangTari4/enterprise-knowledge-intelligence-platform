from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self._capture_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._capture_title = True

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._capture_title = False
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3", "h4"}:
            self.body_parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        text = data.strip()
        if not text or self._skip_depth > 0:
            return
        if self._capture_title:
            self.title_parts.append(text)
        else:
            self.body_parts.append(text)

    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    def text(self) -> str:
        text = " ".join(self.body_parts)
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines = [line for line in lines if line]
        return "\n".join(cleaned_lines).strip()


@dataclass(slots=True)
class WebpageText:
    url: str
    title: str
    text: str
    html_path: Path | None = None


def fetch_webpage_text(url: str, *, timeout: int = 30) -> WebpageText:
    request = Request(url, headers={"User-Agent": "production-rag/0.1"})
    with urlopen(request, timeout=timeout) as response:  # nosec: B310 - user-supplied content fetch by design
        raw = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"
        html = raw.decode(encoding, errors="replace")

    parser = _TextExtractor()
    parser.feed(html)
    title = parser.title() or url
    text = parser.text()
    return WebpageText(url=url, title=title, text=text)
