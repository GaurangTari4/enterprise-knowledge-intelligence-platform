from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class DocumentRecord:
    id: str
    title: str
    text: str
    source_type: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class DocumentSummary:
    id: str
    title: str
    source_type: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    text_length: int
    chunk_count: int


@dataclass(slots=True)
class ChunkRecord:
    id: str
    document_id: str
    document_title: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class ConversationTurn:
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class SourceCitation:
    document_id: str
    document_title: str
    chunk_id: str
    chunk_index: int
    score: float
    excerpt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    bm25_score: float
    vector_score: float
    reasons: list[str] = field(default_factory=list)

    def to_citation(self) -> SourceCitation:
        excerpt = self.chunk.text if len(self.chunk.text) <= 260 else self.chunk.text[:257].rstrip() + "..."
        return SourceCitation(
            document_id=self.chunk.document_id,
            document_title=self.chunk.document_title,
            chunk_id=self.chunk.id,
            chunk_index=self.chunk.chunk_index,
            score=self.score,
            excerpt=excerpt,
            metadata=self.chunk.metadata,
        )


@dataclass(slots=True)
class FaithfulnessReport:
    passed: bool
    coverage: float
    unsupported_terms: list[str]
    reason: str


@dataclass(slots=True)
class HallucinationReport:
    detected: bool
    severity: float
    coverage: float
    unsupported_terms: list[str]
    unsupported_numbers: list[str]
    reason: str


@dataclass(slots=True)
class ChatResponse:
    conversation_id: str
    question: str
    rewritten_query: str
    answer: str
    prompt: str
    sources: list[SourceCitation]
    retrieval: list[SourceCitation]
    faithfulness: FaithfulnessReport
    hallucination: HallucinationReport
    latency_ms: float
    answer_provider: str = ""
    answer_model_name: str | None = None
    insufficient_information: bool = False
