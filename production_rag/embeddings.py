from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Sequence

from .text import tokenize

DEFAULT_EMBEDDING_MODEL = os.getenv("PRODUCTION_RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_EMBEDDING_DIMENSION = int(os.getenv("PRODUCTION_RAG_EMBEDDING_DIMENSION", "1024"))
PREFER_OFFLINE_MODEL = os.getenv("PRODUCTION_RAG_PREFER_OFFLINE_MODEL", "1") != "0"
ALLOW_MODEL_DOWNLOAD = os.getenv("PRODUCTION_RAG_ALLOW_MODEL_DOWNLOAD", "0") == "1"


@dataclass(slots=True)
class EmbeddingResult:
    text: str
    vector: list[float]
    model_name: str


class HashingEmbedder:
    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        if dimension < 32:
            raise ValueError("dimension must be at least 32")
        self.dimension = dimension

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            index = self._index(token)
            sign = 1.0 if self._sign(token) else -1.0
            vector[index] += sign

            if len(token) >= 6:
                subtoken = f"{token[:3]}:{token[-3:]}"
                vector[self._index(subtoken)] += 0.5 * sign

        return _normalize(vector)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def _sign(self, token: str) -> bool:
        digest = hashlib.blake2b((token + "!").encode("utf-8"), digest_size=1).digest()
        return bool(digest[0] % 2)


class BgeM3Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        fallback_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        self.model_name = model_name
        self.fallback_dimension = fallback_dimension
        self._model = None
        self._load_error: Exception | None = None

    @property
    def is_ready(self) -> bool:
        return self._load_error is None

    @property
    def load_error(self) -> Exception | None:
        return self._load_error

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            try:
                if PREFER_OFFLINE_MODEL:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                else:
                    self._model = SentenceTransformer(self.model_name)
            except Exception:
                if not ALLOW_MODEL_DOWNLOAD:
                    raise
                self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # pragma: no cover - network/device dependent
            self._load_error = exc

    def embed_one(self, text: str) -> list[float]:
        self._ensure_model()
        if self._model is None:
            return HashingEmbedder(self.fallback_dimension).embed_one(text)

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _coerce_vector(vector)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_model()
        if self._model is None:
            return HashingEmbedder(self.fallback_dimension).embed_many(texts)

        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [_coerce_vector(vector) for vector in vectors]


def _coerce_vector(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in list(vector)]  # type: ignore[arg-type]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value

    if not left_norm or not right_norm:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)
