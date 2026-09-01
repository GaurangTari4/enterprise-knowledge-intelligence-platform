from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(slots=True)
class Settings:
    database_path: Path = field(
        default_factory=lambda: Path(os.getenv("PRODUCTION_RAG_DATABASE", "data/production_rag.sqlite3"))
    )
    host: str = field(default_factory=lambda: os.getenv("PRODUCTION_RAG_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PRODUCTION_RAG_PORT", "8000")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("PRODUCTION_RAG_CHUNK_SIZE", "900")))
    chunk_overlap_sentences: int = field(
        default_factory=lambda: int(os.getenv("PRODUCTION_RAG_CHUNK_OVERLAP_SENTENCES", "1"))
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("PRODUCTION_RAG_TOP_K", "5")))
    candidate_k: int = field(default_factory=lambda: int(os.getenv("PRODUCTION_RAG_CANDIDATE_K", "20")))
    vector_weight: float = field(default_factory=lambda: float(os.getenv("PRODUCTION_RAG_VECTOR_WEIGHT", "0.55")))
    no_answer_threshold: float = field(
        default_factory=lambda: float(os.getenv("PRODUCTION_RAG_NO_ANSWER_THRESHOLD", "0.25"))
    )
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv("PRODUCTION_RAG_EMBED_DIM", "256")))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()

