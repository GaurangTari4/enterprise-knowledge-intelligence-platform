from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence
import uuid

from qdrant_client import QdrantClient, models

from .chunking import Chunk, chunk_text
from .embeddings import BgeM3Embedder
from .models import ChunkRecord, RetrievedChunk


@dataclass(slots=True)
class QdrantSearchResult:
    chunk_id: str
    document_title: str
    page_number: int | None
    chunk_index: int
    score: float
    text: str
    metadata: dict


def _metadata_matches(metadata: Mapping[str, object], metadata_filters: Mapping[str, object] | None) -> bool:
    if not metadata_filters:
        return True

    for key, expected in metadata_filters.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set, frozenset)):
            if actual not in expected:
                return False
        elif isinstance(expected, dict):
            if actual != expected:
                return False
        else:
            if actual != expected:
                return False
    return True


class QdrantVectorIndex:
    def __init__(
        self,
        storage_path: str | Path = "data/qdrant",
        collection_name: str = "production_rag_chunks",
        embedder: BgeM3Embedder | None = None,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.embedder = embedder or BgeM3Embedder()
        self.client = QdrantClient(path=str(self.storage_path))
        self.embedding_dimension = len(self.embedder.embed_one("dimension probe"))
        self._point_namespace = uuid.UUID("00000000-0000-0000-0000-000000000042")

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.embedding_dimension, distance=models.Distance.COSINE),
        )

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_title: str,
        *,
        page_number: int = 1,
        min_tokens: int = 500,
        max_tokens: int = 800,
        overlap_tokens: int = 75,
    ) -> list[ChunkRecord]:
        chunks = chunk_text(text, min_tokens=min_tokens, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        return [
            ChunkRecord(
                id=f"{document_id}_chunk_{chunk.index:04d}",
                document_id=document_id,
                document_title=document_title,
                chunk_index=chunk.index,
                text=chunk.text,
                metadata={
                    "page_number": page_number,
                    "token_count": chunk.token_count,
                    "sentence_count": chunk.sentence_count,
                    "overlap_from_previous_tokens": chunk.overlap_from_previous_tokens,
                },
            )
            for chunk in chunks
        ]

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        self.ensure_collection()
        vectors = self.embedder.embed_many([chunk.text for chunk in chunks])
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.text,
                **chunk.metadata,
            }
            point_id = str(uuid.uuid5(self._point_namespace, chunk.id))
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    def index_document(
        self,
        text: str,
        document_id: str,
        document_title: str,
        *,
        page_number: int = 1,
        min_tokens: int = 500,
        max_tokens: int = 800,
        overlap_tokens: int = 75,
    ) -> list[ChunkRecord]:
        chunk_records = self.chunk_document(
            text,
            document_id,
            document_title,
            page_number=page_number,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        self.upsert_chunks(chunk_records)
        return chunk_records

    def query(
        self,
        query_text: str,
        limit: int = 5,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[QdrantSearchResult]:
        self.ensure_collection()
        query_vector = self.embedder.embed_one(query_text)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=max(limit * 5, limit),
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)
        results: list[QdrantSearchResult] = []
        for point in points:
            payload = point.payload or {}
            metadata = {key: value for key, value in payload.items() if key not in {"chunk_text"}}
            if not _metadata_matches(metadata, metadata_filters):
                continue
            results.append(
                QdrantSearchResult(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    document_title=str(payload.get("document_title", "")),
                    page_number=payload.get("page_number"),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    score=float(point.score),
                    text=str(payload.get("chunk_text", "")),
                    metadata=metadata,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def query_as_retrieved_chunks(
        self,
        query_text: str,
        limit: int = 5,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        results = self.query(query_text, limit=limit, metadata_filters=metadata_filters)
        retrieved: list[RetrievedChunk] = []
        for result in results:
            chunk = ChunkRecord(
                id=result.chunk_id,
                document_id=str(result.metadata.get("document_id", "")),
                document_title=result.document_title,
                chunk_index=result.chunk_index,
                text=result.text,
                metadata=result.metadata,
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=result.score,
                    bm25_score=0.0,
                    vector_score=result.score,
                )
            )
        return retrieved


def chunks_to_preview(chunks: Sequence[ChunkRecord], limit: int = 5) -> list[str]:
    previews: list[str] = []
    for chunk in chunks[:limit]:
        page_number = chunk.metadata.get("page_number", "?")
        previews.append(
            f"{chunk.id} | page={page_number} | chunk={chunk.chunk_index} | tokens={chunk.metadata.get('token_count', '?')}"
        )
        previews.append(chunk.text)
        previews.append("")
    return previews
