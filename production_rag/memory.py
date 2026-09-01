from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Sequence
import uuid

from .text import meaningful_terms


def _now_ts() -> float:
    return time()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


@dataclass(slots=True)
class ConversationTurnRecord:
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: float


class ConversationStore:
    def __init__(self, database_path: str | Path = "data/production_rag.sqlite3") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation_created
                ON conversation_turns(conversation_id, created_at ASC);
                """
            )
            connection.commit()

    def create_conversation(self, title: str | None = None) -> str:
        conversation_id = str(uuid.uuid4())
        now = _now_ts()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title or "Untitled conversation", now, now),
            )
            connection.commit()
        return conversation_id

    def ensure_conversation(self, conversation_id: str | None = None, title: str | None = None) -> str:
        if conversation_id:
            try:
                uuid.UUID(conversation_id)
            except (ValueError, AttributeError):
                raise ValueError(f"Invalid conversation ID: {conversation_id}") from None
            with self._connect() as connection:
                row = connection.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
                if row is not None:
                    return conversation_id
            raise ValueError(f"Conversation not found: {conversation_id}")
        return self.create_conversation(title=title)

    def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurnRecord:
        turn_id = str(uuid.uuid4())
        now = _now_ts()
        metadata_json = _safe_json(metadata or {})
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_turns (id, conversation_id, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, conversation_id, role, content, metadata_json, now),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            connection.commit()
        return ConversationTurnRecord(
            id=turn_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata or {},
            created_at=now,
        )

    def list_recent_turns(self, conversation_id: str, limit: int = 12) -> list[ConversationTurnRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, metadata_json, created_at
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        records = [
            ConversationTurnRecord(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                metadata=_parse_json(row["metadata_json"], {}),
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]
        records.reverse()
        return records

    def list_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        conversations: list[dict[str, Any]] = []
        for row in rows:
            conversations.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                    "turn_count": len(self.list_recent_turns(row["id"], limit=1000)),
                }
            )
        return conversations

    def summarize_history(self, conversation_id: str, limit: int = 12) -> str:
        turns = self.list_recent_turns(conversation_id, limit=limit)
        if not turns:
            return ""
        lines = []
        for turn in turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)


def rewrite_question(question: str, history: Sequence[ConversationTurnRecord] | None = None) -> str:
    history = list(history or [])
    if not history:
        return question

    lowered = question.strip().lower()
    ambiguous_starts = (
        "what about",
        "and ",
        "how about",
        "what is it",
        "what does it",
        "how does it",
        "what happens next",
        "what happens if",
        "how does that",
        "what about it",
    )
    pronoun_terms = {"it", "they", "that", "those", "this", "these", "them", "there", "their"}
    question_terms = set(meaningful_terms(question))
    is_ambiguous = lowered.startswith(ambiguous_starts) or bool(pronoun_terms & question_terms) or len(question_terms) < 4
    if not is_ambiguous:
        return question

    last_user = next((turn for turn in reversed(history) if turn.role == "user"), None)
    if last_user is None:
        return question

    topic_terms = meaningful_terms(last_user.content, max_terms=6)
    if not topic_terms:
        return question

    topic = " ".join(topic_terms)
    if topic.lower() in lowered:
        return question
    return f"{topic}: {question}"
