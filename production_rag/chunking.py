from __future__ import annotations

from dataclasses import dataclass

from .text import normalize_whitespace, split_sentences, tokenize


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    token_count: int
    sentence_count: int
    overlap_from_previous_tokens: int = 0


def _token_count(text: str) -> int:
    return len(tokenize(text))


def _join_sentences(sentences: list[str]) -> str:
    return normalize_whitespace(" ".join(sentence.strip() for sentence in sentences if sentence.strip()))


def chunk_text(
    text: str,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 75,
) -> list[Chunk]:
    if min_tokens <= 0 or max_tokens <= 0:
        raise ValueError("min_tokens and max_tokens must be positive")
    if min_tokens > max_tokens:
        raise ValueError("min_tokens must be less than or equal to max_tokens")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")

    sentences = split_sentences(text)
    if not sentences:
        cleaned = normalize_whitespace(text)
        if not cleaned:
            return []
        return [Chunk(index=1, text=cleaned, token_count=_token_count(cleaned), sentence_count=1)]

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    current_overlap_from_previous = 0
    current_index = 1
    sentence_index = 0

    while sentence_index < len(sentences):
        sentence = normalize_whitespace(sentences[sentence_index])
        if not sentence:
            sentence_index += 1
            continue

        sentence_tokens = _token_count(sentence)

        if not current_sentences:
            current_sentences.append(sentence)
            current_tokens = sentence_tokens
            sentence_index += 1
            continue

        projected_tokens = current_tokens + sentence_tokens
        if projected_tokens <= max_tokens:
            current_sentences.append(sentence)
            current_tokens = projected_tokens
            sentence_index += 1
            continue

        if current_tokens < min_tokens and sentence_tokens <= max_tokens:
            current_sentences.append(sentence)
            current_tokens = projected_tokens
            sentence_index += 1
            continue

        chunk_text_value = _join_sentences(current_sentences)
        if chunk_text_value:
            chunks.append(
                Chunk(
                    index=current_index,
                    text=chunk_text_value,
                    token_count=_token_count(chunk_text_value),
                    sentence_count=len(current_sentences),
                    overlap_from_previous_tokens=current_overlap_from_previous,
                )
            )
            current_index += 1

        overlap_sentences: list[str] = []
        overlap_count = 0
        if overlap_tokens > 0:
            for previous_sentence in reversed(current_sentences):
                prev_tokens = _token_count(previous_sentence)
                if overlap_sentences and overlap_count + prev_tokens > overlap_tokens:
                    break
                overlap_sentences.insert(0, previous_sentence)
                overlap_count += prev_tokens
                if overlap_count >= overlap_tokens:
                    break

        current_sentences = overlap_sentences[:] if overlap_sentences else []
        current_tokens = _token_count(_join_sentences(current_sentences)) if current_sentences else 0
        current_overlap_from_previous = overlap_count if current_sentences else 0

        if not current_sentences:
            # The single sentence is too large; keep it as its own chunk.
            chunks.append(
                Chunk(
                    index=current_index,
                    text=sentence,
                    token_count=sentence_tokens,
                    sentence_count=1,
                    overlap_from_previous_tokens=current_overlap_from_previous,
                )
            )
            current_index += 1
            sentence_index += 1
            current_sentences = []
            current_tokens = 0
            current_overlap_from_previous = 0
            continue

    if current_sentences:
        chunk_text_value = _join_sentences(current_sentences)
        if chunk_text_value:
            chunks.append(
                Chunk(
                    index=current_index,
                    text=chunk_text_value,
                    token_count=_token_count(chunk_text_value),
                    sentence_count=len(current_sentences),
                    overlap_from_previous_tokens=current_overlap_from_previous,
                )
            )

    return chunks
