from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence

from .bm25 import BM25Index
from .models import ChunkRecord, RetrievedChunk
from .qdrant_index import QdrantVectorIndex
from .text import best_sentence_for_query, meaningful_terms


RETRIEVAL_STOPWORDS = {
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


@dataclass(slots=True)
class HybridRetrievalConfig:
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rerank_overlap_weight: float = 0.08
    rerank_sentence_weight: float = 0.05
    rerank_phrase_bonus: float = 0.25
    scroll_page_size: int = 256


class HybridRetriever:
    def __init__(self, index: QdrantVectorIndex, config: HybridRetrievalConfig | None = None) -> None:
        self.index = index
        self.config = config or HybridRetrievalConfig()
        self._chunk_cache: list[ChunkRecord] | None = None

    def _load_chunks(
        self,
        refresh: bool = False,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[ChunkRecord]:
        if self._chunk_cache is not None and not refresh:
            if metadata_filters:
                return [chunk for chunk in self._chunk_cache if self._metadata_matches(chunk.metadata, metadata_filters)]
            return self._chunk_cache

        if not self.index.client.collection_exists(self.index.collection_name):
            self._chunk_cache = []
            return self._chunk_cache

        chunks: list[ChunkRecord] = []
        offset = None
        while True:
            points, offset = self.index.client.scroll(
                collection_name=self.index.collection_name,
                limit=self.config.scroll_page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                text = str(payload.get("chunk_text", ""))
                chunks.append(
                    ChunkRecord(
                        id=str(payload.get("chunk_id", point.id)),
                        document_id=str(payload.get("document_id", "")),
                        document_title=str(payload.get("document_title", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        text=text,
                        metadata={key: value for key, value in payload.items() if key != "chunk_text"},
                    )
                )
            if offset is None:
                break

        self._chunk_cache = chunks
        if metadata_filters:
            return [chunk for chunk in chunks if self._metadata_matches(chunk.metadata, metadata_filters)]
        return chunks

    @staticmethod
    def _metadata_matches(metadata: Mapping[str, object], metadata_filters: Mapping[str, object] | None) -> bool:
        if not metadata_filters:
            return True

        for key, expected in metadata_filters.items():
            actual = metadata.get(key)
            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def refresh(self) -> None:
        self._chunk_cache = None

    def search(
        self,
        query_text: str,
        *,
        limit: int = 5,
        candidate_k: int = 20,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        chunks = self._load_chunks(metadata_filters=metadata_filters)
        if not chunks:
            return []

        candidate_k = max(limit, candidate_k)
        bm25 = BM25Index([chunk.text for chunk in chunks])
        bm25_scores = bm25.score_all(query_text)
        bm25_map = {chunks[index].id: score for index, score in enumerate(bm25_scores) if score > 0}

        vector_results = self.index.query(
            query_text,
            limit=min(candidate_k, len(chunks)),
            metadata_filters=metadata_filters,
        )
        vector_map = {result.chunk_id: result.score for result in vector_results}

        candidate_ids = set(vector_map) | set(sorted(bm25_map, key=bm25_map.get, reverse=True)[:candidate_k])
        chunk_by_id = {chunk.id: chunk for chunk in chunks}

        vector_max = max(vector_map.values(), default=0.0) or 1.0
        bm25_max = max(bm25_map.values(), default=0.0) or 1.0
        query_terms = [term for term in meaningful_terms(query_text) if term not in RETRIEVAL_STOPWORDS]
        query_term_set = set(query_terms)
        query_phrase = " ".join(query_terms[:4]).strip()

        scored: list[RetrievedChunk] = []
        for chunk_id in candidate_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue

            vector_score = vector_map.get(chunk_id, 0.0)
            bm25_score = bm25_map.get(chunk_id, 0.0)
            combined_score = 0.0
            if vector_score > 0:
                combined_score += self.config.vector_weight * (vector_score / vector_max)
            if bm25_score > 0:
                combined_score += self.config.bm25_weight * (bm25_score / bm25_max)

            reranked_score = self._rerank_chunk(
                query_term_set=query_term_set,
                query_phrase=query_phrase,
                chunk=chunk,
                base_score=combined_score,
            )
            reasons = []
            if vector_score > 0:
                reasons.append("vector")
            if bm25_score > 0:
                reasons.append("bm25")
            if reranked_score != combined_score:
                reasons.append("reranked")

            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=reranked_score,
                    bm25_score=bm25_score,
                    vector_score=vector_score,
                    reasons=reasons,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _rerank_chunk(self, *, query_term_set: set[str], query_phrase: str, chunk: ChunkRecord, base_score: float) -> float:
        if not query_term_set:
            return base_score

        chunk_terms = {term for term in meaningful_terms(chunk.text) if term not in RETRIEVAL_STOPWORDS}
        overlap = len(query_term_set & chunk_terms)

        best_sentence = best_sentence_for_query(chunk.text, query_term_set)
        sentence_terms = {term for term in meaningful_terms(best_sentence) if term not in RETRIEVAL_STOPWORDS}
        sentence_overlap = len(query_term_set & sentence_terms)

        score = base_score
        score += self.config.rerank_overlap_weight * overlap
        score += self.config.rerank_sentence_weight * sentence_overlap

        lowered = chunk.text.lower()
        if query_phrase and query_phrase in lowered:
            score += self.config.rerank_phrase_bonus

        return score
