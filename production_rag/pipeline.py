from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .guardrails import assess_faithfulness, detect_hallucination, render_source_citations
from .llm import GroundedAnswerer
from .memory import ConversationStore, rewrite_question
from .models import ChatResponse, RetrievedChunk, SourceCitation
from .qdrant_index import QdrantVectorIndex
from .retrieval import HybridRetriever, HybridRetrievalConfig
from .text import meaningful_terms, split_sentences


def _append_citation_markers(sentence: str, citation_indexes: Sequence[int]) -> str:
    sentence = sentence.strip()
    if not sentence:
        return sentence

    markers = "".join(f"[{index}]" for index in citation_indexes)
    if not markers:
        return sentence

    match = re.match(r"^(.*?)([.?!]+)?$", sentence)
    if not match:
        return f"{sentence} {markers}".strip()

    body = match.group(1).rstrip()
    punctuation = match.group(2) or ""
    return f"{body} {markers}{punctuation}".strip()


def _rank_sentence_citations(sentence: str, retrieved_chunks: Sequence[RetrievedChunk], limit: int = 2) -> list[int]:
    sentence_terms = set(meaningful_terms(sentence))
    if not sentence_terms:
        return []

    ranked: list[tuple[float, int]] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        chunk_terms = set(meaningful_terms(chunk.chunk.text))
        overlap = len(sentence_terms & chunk_terms)
        if overlap == 0:
            continue
        score = overlap + (chunk.score * 0.25)
        ranked.append((score, index))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in ranked[:limit]]


def render_answer_with_citations(answer: str, citations: Sequence[SourceCitation], retrieved_chunks: Sequence[RetrievedChunk]) -> str:
    if not citations:
        return answer

    sentences = split_sentences(answer)
    if sentences:
        annotated_sentences = []
        for sentence in sentences:
            citation_indexes = _rank_sentence_citations(sentence, retrieved_chunks)
            annotated_sentences.append(_append_citation_markers(sentence, citation_indexes))
        rendered_answer = " ".join(annotated_sentences)
    else:
        rendered_answer = answer

    lines = [rendered_answer, "", "Sources"]
    lines.extend(render_source_citations(citations))
    return "\n".join(lines).strip()


@dataclass(slots=True)
class RagPipeline:
    index: QdrantVectorIndex
    retriever: HybridRetriever
    answerer: GroundedAnswerer
    conversation_store: ConversationStore | None = None
    top_k: int = 5
    candidate_k: int = 20
    conversation_history_limit: int = 12

    @classmethod
    def from_storage_path(
        cls,
        storage_path,
        collection_name: str = "production_rag_chunks",
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        conversation_store_path: str | None = "data/production_rag.sqlite3",
    ) -> "RagPipeline":
        index = QdrantVectorIndex(storage_path=storage_path, collection_name=collection_name)
        retriever = HybridRetriever(index, config=HybridRetrievalConfig())
        answerer = GroundedAnswerer()
        if conversation_store_path:
            conversation_path = Path(conversation_store_path)
            if not conversation_path.is_absolute():
                conversation_path = Path(__file__).resolve().parent.parent / conversation_path
            conversation_store = ConversationStore(conversation_path)
        else:
            conversation_store = None
        return cls(
            index=index,
            retriever=retriever,
            answerer=answerer,
            conversation_store=conversation_store,
            top_k=top_k,
            candidate_k=candidate_k,
        )

    def retrieve(
        self,
        question: str,
        *,
        limit: int | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        return self.retriever.search(
            question,
            limit=limit or self.top_k,
            candidate_k=self.candidate_k,
            metadata_filters=metadata_filters,
        )

    def answer(
        self,
        question: str,
        *,
        limit: int | None = None,
        conversation_id: str | None = None,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> ChatResponse:
        start = perf_counter()
        retrieval_limit = limit or self.top_k
        resolved_conversation_id = conversation_id
        rewritten_query = question
        history = []
        if self.conversation_store is not None:
            resolved_conversation_id = self.conversation_store.ensure_conversation(resolved_conversation_id)
            history = self.conversation_store.list_recent_turns(resolved_conversation_id, limit=self.conversation_history_limit)
            rewritten_query = rewrite_question(question, history)
            self.conversation_store.add_turn(
                resolved_conversation_id,
                "user",
                question,
                metadata={"rewritten_query": rewritten_query, "history_size": len(history)},
            )

        retrieved = self.retrieve(rewritten_query, limit=retrieval_limit, metadata_filters=metadata_filters)
        answer_result = self.answerer.answer(rewritten_query, retrieved)
        faithfulness = assess_faithfulness(
            answer_result.answer,
            retrieved,
            insufficient_information=answer_result.insufficient_information,
        )

        final_answer = answer_result.answer
        final_insufficient_information = answer_result.insufficient_information
        if not final_insufficient_information and not faithfulness.passed:
            final_answer = "I don't have enough information in the provided context."
            final_insufficient_information = True
            faithfulness = assess_faithfulness(final_answer, retrieved, insufficient_information=True)

        hallucination = detect_hallucination(
            final_answer,
            retrieved,
            insufficient_information=final_insufficient_information,
        )
        supporting_chunks = answer_result.citations or retrieved
        source_citations = [chunk.to_citation() for chunk in supporting_chunks]
        latency_ms = (perf_counter() - start) * 1000.0

        if self.conversation_store is not None:
            self.conversation_store.add_turn(
                resolved_conversation_id,
                "assistant",
                final_answer,
                metadata={
                    "question": question,
                    "rewritten_query": rewritten_query,
                    "provider": answer_result.used_provider,
                    "model_name": answer_result.model_name,
                    "faithfulness": asdict(faithfulness),
                    "hallucination": asdict(hallucination),
                },
            )

        return ChatResponse(
            conversation_id=resolved_conversation_id,
            question=question,
            rewritten_query=rewritten_query,
            answer=render_answer_with_citations(final_answer, source_citations, supporting_chunks),
            prompt=answer_result.prompt,
            sources=source_citations,
            retrieval=source_citations,
            faithfulness=faithfulness,
            hallucination=hallucination,
            latency_ms=latency_ms,
            answer_provider=answer_result.used_provider,
            answer_model_name=answer_result.model_name,
            insufficient_information=final_insufficient_information,
        )
