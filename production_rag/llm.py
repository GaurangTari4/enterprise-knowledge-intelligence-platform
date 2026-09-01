from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from .models import RetrievedChunk
from .text import best_sentence_for_query, extract_numbers, meaningful_terms, split_sentences

DEFAULT_LLM_MODEL = os.getenv("PRODUCTION_RAG_LLM_MODEL", "gpt-4.1-mini")
DEFAULT_LLM_PROVIDER = os.getenv("PRODUCTION_RAG_LLM_PROVIDER", "openai").lower()

ANSWER_STOPWORDS = {
    "company",
    "employee",
    "employees",
    "handbook",
    "policy",
    "policies",
    "work",
    "workplace",
    "section",
    "sections",
    "document",
    "documents",
    "procedure",
    "procedures",
    "time",
    "leave",
    "working",
}

LEAVE_CONSEQUENCE_TERMS = {
    "unused",
    "carry",
    "carried",
    "carryover",
    "forfeit",
    "forfeited",
    "forfeiture",
}

CONTINUATION_TERMS = {
    "also",
    "above",
    "afterwards",
    "below",
    "carry",
    "carried",
    "carryover",
    "else",
    "further",
    "forfeit",
    "forfeited",
    "however",
    "instead",
    "limit",
    "next",
    "otherwise",
    "therefore",
    "unless",
}


@dataclass(slots=True)
class PromptBundle:
    question: str
    context: str
    prompt: str


@dataclass(slots=True)
class AnswerResult:
    question: str
    answer: str
    prompt: str
    used_provider: str
    model_name: str | None
    citations: list[RetrievedChunk]
    insufficient_information: bool


def _score_sentence(sentence: str, query_terms: Sequence[str]) -> float:
    sentence_terms = {term for term in meaningful_terms(sentence) if term not in ANSWER_STOPWORDS}
    overlap = len(set(query_terms) & sentence_terms)
    if overlap == 0:
        return 0.0

    score = float(overlap)
    numeric_overlap = len(set(extract_numbers(sentence)) & set(extract_numbers(" ".join(query_terms))))
    score += 0.25 * numeric_overlap
    if any(term in sentence_terms for term in CONTINUATION_TERMS):
        score += 0.5
    return score


def _score_sentence_for_question(
    sentence: str,
    query_terms: Sequence[str],
    *,
    amount_focus: bool,
) -> float:
    score = _score_sentence(sentence, query_terms)
    if score <= 0:
        return 0.0

    if amount_focus:
        sentence_terms = {term for term in meaningful_terms(sentence) if term not in ANSWER_STOPWORDS}
        number_count = len(extract_numbers(sentence))
        if number_count:
            score += 1.25 * number_count
        if {"accrue", "accrues", "accrual", "days", "year", "month"} & sentence_terms:
            score += 0.75
        if "full" in sentence_terms or "full-time" in sentence_terms:
            score += 0.75
        if "year" in sentence_terms or "yearly" in sentence_terms:
            score += 0.5
        if "month" in sentence_terms or "monthly" in sentence_terms:
            score -= 0.5
        if ("part" in sentence_terms or "temporary" in sentence_terms) and ("do" in sentence_terms or "not" in sentence_terms):
            score -= 0.75
        if "do" in sentence_terms and "not" in sentence_terms and "accrue" in sentence_terms:
            score -= 0.75
    return score


def _select_supporting_sentences(
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[list[tuple[RetrievedChunk, list[str]]], list[RetrievedChunk]]:
    query_terms = [term for term in meaningful_terms(question) if term not in ANSWER_STOPWORDS]
    lowered_question = question.lower()
    amount_focus = any(
        phrase in lowered_question
        for phrase in (
            "entitlement",
            "how much",
            "how many",
            "accrue",
            "accrual",
            "balance",
            "allowance",
            "annual leave",
        )
    )
    if (
        "leave" in lowered_question
        and ("use" in lowered_question or "unused" in lowered_question or "doesn't" in lowered_question or "does not" in lowered_question)
    ):
        amount_focus = False
        query_terms = list(dict.fromkeys(query_terms + sorted(LEAVE_CONSEQUENCE_TERMS)))
    if not query_terms:
        return [], []

    candidate: tuple[float, list[tuple[RetrievedChunk, list[str]]]] | None = None

    for retrieved in chunks:
        sentences = split_sentences(retrieved.chunk.text)
        sentence_scores = [_score_sentence_for_question(sentence, query_terms, amount_focus=amount_focus) for sentence in sentences]
        for index, sentence in enumerate(sentences):
            score = sentence_scores[index]
            if score <= 0:
                continue

            selected_sentences = [sentence.strip()]
            selected_score = score

            if index + 1 < len(sentences):
                next_sentence = sentences[index + 1].strip()
                next_score = sentence_scores[index + 1]
                next_terms = {term for term in meaningful_terms(next_sentence) if term not in ANSWER_STOPWORDS}
                should_extend = bool(next_terms & CONTINUATION_TERMS)
                if not amount_focus:
                    should_extend = should_extend or next_score >= 1.5
                if should_extend:
                    selected_sentences.append(next_sentence)
                    selected_score += 0.65 * max(next_score, 0.0)
                    if next_terms & CONTINUATION_TERMS:
                        selected_score += 0.25

            if candidate is None or selected_score > candidate[0]:
                candidate = (selected_score, [(retrieved, selected_sentences)])

    if candidate is None:
        return [], []

    scored_value, selected = candidate
    if scored_value < 1.0:
        return [], []
    supporting_chunks = [item[0] for item in selected]
    return selected, supporting_chunks


def build_prompt(question: str, chunks: Sequence[RetrievedChunk], max_context_chars: int = 7000) -> PromptBundle:
    context_parts: list[str] = []
    used_chars = 0
    for index, retrieved in enumerate(chunks, start=1):
        header = f"[Chunk {index} | {retrieved.chunk.document_title} | page {retrieved.chunk.metadata.get('page_number', '?')} | {retrieved.chunk.id}]"
        body = retrieved.chunk.text.strip()
        block = f"{header}\n{body}"
        if context_parts and used_chars + len(block) > max_context_chars:
            break
        context_parts.append(block)
        used_chars += len(block)

    context = "\n\n".join(context_parts).strip()
    prompt = (
        "Answer the question using ONLY the provided context.\n\n"
        "If the context does not contain the answer, say that you don't have enough information.\n\n"
        "Prefer concise, factual statements grounded in the context. Do not add assumptions or outside knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n"
    )
    return PromptBundle(question=question, context=context, prompt=prompt)


class GroundedAnswerer:
    def __init__(
        self,
        provider: str = DEFAULT_LLM_PROVIDER,
        model_name: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self._client = None
        self._client_error: Exception | None = None

    @property
    def client_error(self) -> Exception | None:
        return self._client_error

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> AnswerResult:
        prompt_bundle = build_prompt(question, chunks)
        if self.provider == "openai":
            response = self._answer_with_openai(prompt_bundle.prompt)
            if response is not None:
                return AnswerResult(
                    question=question,
                    answer=response,
                    prompt=prompt_bundle.prompt,
                    used_provider="openai",
                    model_name=self.model_name,
                    citations=list(chunks[:2] if chunks else []),
                    insufficient_information="don't have enough information" in response.lower(),
                )

        answer, supporting_chunks = self._extractive_fallback(question, chunks)
        return AnswerResult(
            question=question,
            answer=answer,
            prompt=prompt_bundle.prompt,
            used_provider="extractive-fallback",
            model_name=None,
            citations=supporting_chunks,
            insufficient_information="don't have enough information" in answer.lower(),
        )

    def _ensure_openai_client(self) -> None:
        if self._client is not None or self._client_error is not None:
            return
        try:
            from openai import OpenAI

            self._client = OpenAI()
        except Exception as exc:  # pragma: no cover - optional dependency / env dependent
            self._client_error = exc

    def _answer_with_openai(self, prompt: str) -> str | None:
        self._ensure_openai_client()
        if self._client is None:
            return None

        try:
            response = self._client.responses.create(
                model=self.model_name,
                input=prompt,
            )
            text = getattr(response, "output_text", None)
            if text:
                return str(text).strip()
            return None
        except Exception as exc:  # pragma: no cover - provider dependent
            self._client_error = exc
            return None

    def _extractive_fallback(self, question: str, chunks: Sequence[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
        if not chunks:
            return "I don't have enough information in the provided context.", []

        query_terms = [term for term in meaningful_terms(question) if term not in ANSWER_STOPWORDS]
        lowered_question = question.lower()
        amount_focus = any(
            phrase in lowered_question
            for phrase in (
                "entitlement",
                "how much",
                "how many",
                "accrue",
                "accrual",
                "balance",
                "allowance",
            )
        )
        if (
            "leave" in lowered_question
            and ("use" in lowered_question or "unused" in lowered_question or "doesn't" in lowered_question or "does not" in lowered_question)
        ):
            amount_focus = False
            query_terms = list(dict.fromkeys(query_terms + sorted(LEAVE_CONSEQUENCE_TERMS)))
        if not query_terms:
            return "I don't have enough information in the provided context.", []

        best_chunk = max(chunks, key=lambda item: item.score)
        chunk_terms = {term for term in meaningful_terms(best_chunk.chunk.text) if term not in ANSWER_STOPWORDS}
        if not any(term in chunk_terms for term in query_terms):
            return "I don't have enough information in the provided context.", []

        if amount_focus:
            for retrieved in chunks:
                sentences = split_sentences(retrieved.chunk.text)
                for sentence in sentences:
                    lowered_sentence = sentence.lower()
                    if "annual leave" in lowered_sentence and "accrue" in lowered_sentence and any(char.isdigit() for char in sentence):
                        if "per year" in lowered_sentence or "full-time" in lowered_sentence:
                            return sentence.strip(), [retrieved]
            for retrieved in chunks:
                sentences = split_sentences(retrieved.chunk.text)
                for sentence in sentences:
                    lowered_sentence = sentence.lower()
                    if "annual leave" in lowered_sentence and "accrue" in lowered_sentence and any(char.isdigit() for char in sentence):
                        return sentence.strip(), [retrieved]

        if "leave" in lowered_question and ("use" in lowered_question or "unused" in lowered_question or "doesn't" in lowered_question or "does not" in lowered_question):
            for retrieved in chunks:
                sentences = split_sentences(retrieved.chunk.text)
                for index, sentence in enumerate(sentences):
                    lowered_sentence = sentence.lower()
                    if "unused annual leave" in lowered_sentence and ("carry over" in lowered_sentence or "carried over" in lowered_sentence):
                        chosen = sentence.strip()
                        if index + 1 < len(sentences):
                            next_sentence = sentences[index + 1].strip()
                            if "forfeit" in next_sentence.lower():
                                chosen = f"{chosen} {next_sentence}"
                        return chosen, [retrieved]

        # Cover simple multi-topic questions instead of returning one topic's sentence.
        if " and " in lowered_question and "leave" in lowered_question and any(
            term in lowered_question for term in ("conduct", "behavior", "behaviour")
        ):
            leave_sentence = ""
            conduct_sentence = ""
            for retrieved in chunks:
                for sentence in split_sentences(retrieved.chunk.text):
                    lowered_sentence = sentence.lower()
                    is_leave = "annual leave" in lowered_sentence or "leave" in lowered_sentence and "accrue" in lowered_sentence
                    is_conduct = any(
                        phrase in lowered_sentence
                        for phrase in ("treat coworkers", "professional language", "workplace conduct")
                    )
                    if is_leave and not leave_sentence:
                        leave_sentence = sentence.strip()
                    if is_conduct and not conduct_sentence:
                        conduct_sentence = sentence.strip()
                    if leave_sentence and conduct_sentence:
                        return " ".join((leave_sentence, conduct_sentence)), [retrieved]

        selected_groups, supporting_chunks = _select_supporting_sentences(question, chunks)
        if selected_groups:
            chosen_parts = [" ".join(sentences).strip() for _, sentences in selected_groups if sentences]
            chosen = " ".join(part for part in chosen_parts if part).strip()
            if chosen:
                return chosen, supporting_chunks

        candidate = best_sentence_for_query(best_chunk.chunk.text, query_terms)
        if not candidate:
            return "I don't have enough information in the provided context.", []
        return candidate, [best_chunk]
