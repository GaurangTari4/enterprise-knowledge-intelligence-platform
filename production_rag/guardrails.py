from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import FaithfulnessReport, HallucinationReport, RetrievedChunk, SourceCitation
from .text import extract_numbers, meaningful_terms


NO_ANSWER_PHRASES = (
    "i don't have enough information",
    "i do not have enough information",
    "not have enough information",
)

FAITHFULNESS_STOPWORDS = {
    "answer",
    "context",
    "provided",
    "information",
    "information",
    "enough",
    "please",
    "using",
    "only",
    "question",
    "section",
    "source",
    "sources",
    "cite",
    "cited",
}

MIN_FAITHFULNESS_COVERAGE = 0.7
MIN_HALLUCINATION_COVERAGE = 0.7


@dataclass(slots=True)
class _AnswerSupport:
    coverage: float
    unsupported_terms: list[str]
    unsupported_numbers: list[str]
    total_items: int
    supported_items: int
    declined: bool


def _collect_answer_support(
    answer: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    insufficient_information: bool = False,
) -> _AnswerSupport:
    lowered_answer = answer.lower()
    declined = insufficient_information or any(phrase in lowered_answer for phrase in NO_ANSWER_PHRASES)
    if declined:
        return _AnswerSupport(
            coverage=1.0,
            unsupported_terms=[],
            unsupported_numbers=[],
            total_items=0,
            supported_items=0,
            declined=True,
        )

    if not retrieved_chunks:
        answer_terms = [term for term in meaningful_terms(answer) if term not in FAITHFULNESS_STOPWORDS]
        answer_numbers = extract_numbers(answer)
        return _AnswerSupport(
            coverage=0.0,
            unsupported_terms=answer_terms,
            unsupported_numbers=answer_numbers,
            total_items=len(answer_terms) + len(answer_numbers),
            supported_items=0,
            declined=False,
        )

    context_text = " ".join(chunk.chunk.text for chunk in retrieved_chunks)
    context_terms = set(meaningful_terms(context_text))
    context_numbers = set(extract_numbers(context_text))
    answer_terms = [term for term in meaningful_terms(answer) if term not in FAITHFULNESS_STOPWORDS]
    answer_numbers = extract_numbers(answer)

    unsupported_terms: list[str] = []
    unsupported_numbers: list[str] = []
    supported_count = 0
    for term in answer_terms:
        if term in context_terms:
            supported_count += 1
        else:
            unsupported_terms.append(term)

    for number in answer_numbers:
        if number in context_numbers:
            supported_count += 1
        else:
            unsupported_numbers.append(number)

    total_items = len(answer_terms) + len(answer_numbers)
    coverage = supported_count / total_items if total_items else 1.0
    return _AnswerSupport(
        coverage=coverage,
        unsupported_terms=unsupported_terms,
        unsupported_numbers=unsupported_numbers,
        total_items=total_items,
        supported_items=supported_count,
        declined=False,
    )


def assess_faithfulness(
    answer: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    insufficient_information: bool = False,
) -> FaithfulnessReport:
    support = _collect_answer_support(answer, retrieved_chunks, insufficient_information=insufficient_information)
    if support.declined:
        return FaithfulnessReport(
            passed=True,
            coverage=1.0,
            unsupported_terms=[],
            reason="The answer correctly declined to answer from the provided context.",
        )

    if not retrieved_chunks:
        return FaithfulnessReport(
            passed=False,
            coverage=support.coverage,
            unsupported_terms=support.unsupported_terms + support.unsupported_numbers,
            reason="No retrieved context was available to support the answer.",
        )

    unsupported_items = support.unsupported_terms + support.unsupported_numbers
    passed = support.coverage >= MIN_FAITHFULNESS_COVERAGE and not unsupported_items
    reason = (
        "Answer terms are supported by the retrieved context."
        if passed
        else f"Unsupported terms detected: {', '.join(unsupported_items[:8])}"
    )
    return FaithfulnessReport(
        passed=passed,
        coverage=support.coverage,
        unsupported_terms=unsupported_items,
        reason=reason,
    )


def detect_hallucination(
    answer: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    insufficient_information: bool = False,
) -> HallucinationReport:
    support = _collect_answer_support(answer, retrieved_chunks, insufficient_information=insufficient_information)
    if support.declined:
        return HallucinationReport(
            detected=False,
            severity=0.0,
            coverage=1.0,
            unsupported_terms=[],
            unsupported_numbers=[],
            reason="The answer declined to answer instead of hallucinating unsupported facts.",
        )

    unsupported_items = support.unsupported_terms + support.unsupported_numbers
    detected = support.coverage < MIN_HALLUCINATION_COVERAGE or bool(unsupported_items)
    if not retrieved_chunks:
        detected = True

    severity = 1.0 - support.coverage
    if unsupported_items:
        severity = min(1.0, severity + 0.2)
    if not retrieved_chunks:
        severity = 1.0

    reason = (
        "No hallucination signal detected."
        if not detected
        else f"Unsupported claims detected: {', '.join(unsupported_items[:8])}"
    )
    return HallucinationReport(
        detected=detected,
        severity=max(0.0, min(1.0, severity)),
        coverage=support.coverage,
        unsupported_terms=support.unsupported_terms,
        unsupported_numbers=support.unsupported_numbers,
        reason=reason,
    )


def render_source_citations(citations: Sequence[SourceCitation]) -> list[str]:
    lines: list[str] = []
    for index, citation in enumerate(citations, start=1):
        page_number = citation.metadata.get("page_number", "?")
        lines.append(f"[{index}] {citation.document_title} | page {page_number} | {citation.chunk_id}")
        lines.append(f"    {citation.excerpt}")
    return lines
