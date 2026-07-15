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

    def card(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "title": self.title,
            "description": self.description,
            "problem": self.problem,
            "status": self.status,
            "keywords": self.keywords,
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
    source_fragments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_summary: bool = True) -> dict[str, Any]:
        value = self.topic.card()
        value["score"] = round(self.score, 6)
        if include_summary:
            value["summary"] = self.topic.summary
        if self.source_fragments:
            value["source_fragments"] = self.source_fragments
        return value
