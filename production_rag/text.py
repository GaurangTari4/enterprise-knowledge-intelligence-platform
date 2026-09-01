from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "more",
    "most",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "should",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text or ""):
        token = match.group(0).lower()
        if token.endswith("'s") and len(token) > 2:
            token = token[:-2]
        tokens.append(token)
    return tokens


def meaningful_terms(text: str, max_terms: int | None = None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if max_terms is not None and len(terms) >= max_terms:
            break
    return terms


def split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\r", "\n").strip()
    if not cleaned:
        return []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", cleaned) if paragraph.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs or [cleaned]:
        normalized = normalize_whitespace(paragraph)
        parts = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", normalized)
        chunks.extend(part.strip() for part in parts if part.strip())
    return chunks


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text or "")


def truncate_text(text: str, max_length: int = 240) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1].rstrip() + "…"


def best_sentence_for_query(text: str, query_terms: Iterable[str]) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return normalize_whitespace(text)

    query_term_set = {term.lower() for term in query_terms if term}
    query_numbers = set(extract_numbers(" ".join(query_term_set)))
    best_sentence = sentences[0]
    best_score = -1.0

    for sentence in sentences:
        sentence_terms = set(meaningful_terms(sentence))
        overlap = len(query_term_set & sentence_terms)
        numeric_overlap = len(set(extract_numbers(sentence)) & query_numbers)
        score = overlap + (0.5 * numeric_overlap)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence.strip()


def top_terms(text: str, limit: int = 8) -> list[str]:
    counts = Counter(term for term in meaningful_terms(text) if term not in STOPWORDS)
    return [term for term, _ in counts.most_common(limit)]
