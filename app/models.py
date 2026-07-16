from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal


def utc_or_local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(slots=True)
class Message:
    id: int
    role: str
    text: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not self.metadata:
            result.pop("metadata")
        return result


@dataclass(slots=True)
class Session:
    id: str
    agent: str
    started_at: str
    ended_at: str | None = None
    status: Literal["active", "finalizing", "finalized"] = "active"
    message_count: int = 0
    processed_until_message: int = 0
    summary_revision: int = 0
    new_token_estimate: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Session":
        allowed = cls.__dataclass_fields__
        return cls(**{k: v for k, v in value.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceRange:
    session_id: str
    from_message: int
    to_message: int

    def to_pair(self) -> list[int]:
        return [self.from_message, self.to_message]


@dataclass(slots=True)
class Topic:
    id: str
    session_id: str
    title: str
    description: str
    problem: str
    status: str
    keywords: list[str]
    source_ranges: list[SourceRange]
    created_at: str
    updated_at: str
    summary: str
    path: str = ""
    session_started_at: str = ""
    session_ended_at: str | None = None

    def card(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "title": self.title,
            "description": self.description,
            "problem": self.problem,
            "status": self.status,
            "keywords": self.keywords,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "session_started_at": self.session_started_at,
            "session_ended_at": self.session_ended_at,
            "path": self.path,
            "source_ranges": [
                {
                    "session_id": item.session_id,
                    "from": item.from_message,
                    "to": item.to_message,
                }
                for item in self.source_ranges
            ],
        }


@dataclass(slots=True)
class SearchHit:
    topic: Topic
    score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None
    lexical_relevance: float = 0.0
    vector_similarity: float | None = None
    rrf_score: float = 0.0
    is_latest: bool = True
    global_topic_id: str | None = None
    related_older_topic_ids: list[str] = field(default_factory=list)
    source_fragments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_summary: bool = True) -> dict[str, Any]:
        value = self.topic.card()
        value["score"] = round(self.score, 6)
        value["relevance"] = {
            "lexical": round(self.lexical_relevance, 6),
            "vector": (
                round(self.vector_similarity, 6)
                if self.vector_similarity is not None
                else None
            ),
            "rrf": round(self.rrf_score, 6),
        }
        value["is_latest"] = self.is_latest
        if self.global_topic_id:
            value["global_topic_id"] = self.global_topic_id
            value["global_projection_available"] = True
        if self.related_older_topic_ids:
            value["related_older_topic_ids"] = self.related_older_topic_ids
        if include_summary:
            value["summary"] = self.topic.summary
        if self.source_fragments:
            value["source_fragments"] = self.source_fragments
        return value


@dataclass(slots=True)
class ExplicitMemory:
    memory_id: str
    idempotency_key: str
    verbatim: str
    normalized: str
    kind: str
    scope_type: str
    scope_id: str | None
    author: str
    source_session_id: str
    source_message_id: int
    created_at: str
    status: str = "active"
    event: str = "remember"
    supersedes: str | None = None
    valid_to: str | None = None
    requires_confirmation: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExplicitMemory":
        scope = value.get("scope", {})
        if isinstance(scope, str):
            scope_type, _, scope_id = scope.partition(":")
        elif isinstance(scope, dict):
            scope_type = str(scope.get("type", "global"))
            raw_scope_id = scope.get("id")
            scope_id = str(raw_scope_id) if raw_scope_id not in (None, "") else ""
        else:
            scope_type, scope_id = "global", ""
        return cls(
            memory_id=str(value["memory_id"]),
            idempotency_key=str(value["idempotency_key"]),
            verbatim=str(value["verbatim"]),
            normalized=str(value.get("normalized") or value["verbatim"]),
            kind=str(value["kind"]),
            scope_type=scope_type,
            scope_id=scope_id or None,
            author=str(value.get("author", "user")),
            source_session_id=str(value["source_session_id"]),
            source_message_id=int(value["source_message_id"]),
            created_at=str(value["created_at"]),
            status=str(value.get("status", "active")),
            event=str(value.get("event", "remember")),
            supersedes=(
                str(value["supersedes"]) if value.get("supersedes") else None
            ),
            valid_to=str(value["valid_to"]) if value.get("valid_to") else None,
            requires_confirmation=bool(value.get("requires_confirmation", False)),
        )

    def scope(self) -> dict[str, str]:
        result = {"type": self.scope_type}
        if self.scope_id:
            result["id"] = self.scope_id
        return result

    def to_event_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "memory_id": self.memory_id,
            "idempotency_key": self.idempotency_key,
            "verbatim": self.verbatim,
            "normalized": self.normalized,
            "kind": self.kind,
            "scope": self.scope(),
            "author": self.author,
            "source_session_id": self.source_session_id,
            "source_message_id": self.source_message_id,
            "created_at": self.created_at,
            "status": "active",
            "supersedes": self.supersedes,
            "requires_confirmation": self.requires_confirmation,
        }

    def to_search_dict(self, score: float = 1.0) -> dict[str, Any]:
        return {
            "type": "explicit_memory",
            "memory_id": self.memory_id,
            "text": self.normalized,
            "verbatim": self.verbatim,
            "kind": self.kind,
            "scope": self.scope(),
            "author": self.author,
            "status": self.status,
            "supersedes": self.supersedes,
            "valid_to": self.valid_to,
            "source_session_id": self.source_session_id,
            "source_message_id": self.source_message_id,
            "created_at": self.created_at,
            "score": round(score, 6),
        }
