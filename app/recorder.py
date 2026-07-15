from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import Catalog
from .config import AppConfig
from .models import Message, Session, utc_or_local_now
from .storage import (
    append_message,
    atomic_write_json,
    atomic_write_text,
    estimate_tokens,
    exclusive_lock,
    read_messages,
    read_last_message,
    read_session,
    validate_id,
    write_session,
)


class SessionNotFoundError(FileNotFoundError):
    pass


class SessionRecorder:
    def __init__(self, config: AppConfig, catalog: Catalog):
        self.config = config
        self.catalog = catalog
        self.config.storage.sessions_dir.mkdir(parents=True, exist_ok=True)

    def start_session(
        self,
        agent: str,
        session_id: str | None = None,
        started_at: str | None = None,
    ) -> Session:
        started_at = started_at or utc_or_local_now()
        session_id = validate_id(session_id or f"session-{uuid.uuid4().hex[:12]}", "session id")
        timestamp = datetime.fromisoformat(started_at)
        session_path = (
            self.config.storage.sessions_dir
            / f"{timestamp.year:04d}"
            / f"{timestamp.month:02d}"
            / session_id
        )
        if session_path.exists() or self.catalog.get_session(session_id):
            raise FileExistsError(f"session already exists: {session_id}")
        (session_path / "topics").mkdir(parents=True, exist_ok=False)
        session = Session(id=session_id, agent=agent, started_at=started_at)
        write_session(session_path, session)
        atomic_write_text(session_path / "transcript.jsonl", "")
        atomic_write_json(
            session_path / "index.json",
            {"session_id": session_id, "overview": "", "topics": []},
        )
        self.catalog.upsert_session(session, session_path)
        return session

    def locate(self, session_id: str) -> Path:
        validate_id(session_id, "session id")
        indexed = self.catalog.session_path(session_id)
        if indexed and (indexed / "session.json").exists():
            return indexed
        matches = list(self.config.storage.sessions_dir.glob(f"*/*/{session_id}/session.json"))
        if not matches:
            raise SessionNotFoundError(session_id)
        return matches[0].parent

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        if not role or not content:
            raise ValueError("role and content must be non-empty")
        path = self.locate(session_id)
        with exclusive_lock(path / ".session.lock"):
            session = read_session(path)
            if session.status != "active":
                raise RuntimeError(f"cannot append to {session.status} session")
            tail = read_last_message(path / "transcript.jsonl")
            if tail is not None and tail.id > session.message_count:
                recovered = read_messages(
                    path / "transcript.jsonl", session.message_count + 1, tail.id
                )
                session.message_count = tail.id
                session.new_token_estimate += sum(
                    estimate_tokens(item.text) for item in recovered
                )
            message = Message(
                id=session.message_count + 1,
                role=role,
                text=content,
                created_at=created_at or utc_or_local_now(),
                metadata=metadata or {},
            )
            # The append and fsync happen before any derived metadata is changed.
            append_message(path / "transcript.jsonl", message)
            session.message_count = message.id
            session.new_token_estimate += estimate_tokens(content)
            write_session(path, session)
            self.catalog.upsert_session(session, path)
            self._schedule_if_needed(session)
            return message

    def end_session(self, session_id: str, ended_at: str | None = None) -> Session:
        path = self.locate(session_id)
        with exclusive_lock(path / ".session.lock"):
            session = read_session(path)
            if session.status == "finalized":
                return session
            session.ended_at = ended_at or utc_or_local_now()
            session.status = "finalizing"
            if session.message_count == session.processed_until_message:
                session.status = "finalized"
            write_session(path, session)
            self.catalog.upsert_session(session, path)
            if session.status == "finalizing" and not self.catalog.has_open_job(session.id):
                self.catalog.enqueue_job(
                    session.id,
                    session.processed_until_message + 1,
                    session.message_count,
                    session.ended_at,
                )
            return session

    def read_turns(
        self, session_id: str, from_turn: int = 1, to_turn: int | None = None
    ) -> list[Message]:
        path = self.locate(session_id)
        return read_messages(path / "transcript.jsonl", from_turn, to_turn)

    def schedule_idle_sessions(self, now: datetime | None = None) -> int:
        now = now or datetime.now().astimezone()
        scheduled = 0
        for session_file in self.config.storage.sessions_dir.glob("*/*/*/session.json"):
            session = read_session(session_file.parent)
            if session.status != "active" or session.message_count <= session.processed_until_message:
                continue
            messages = read_messages(session_file.parent / "transcript.jsonl", session.message_count)
            if not messages:
                continue
            last_at = datetime.fromisoformat(messages[-1].created_at)
            if (now - last_at).total_seconds() < self.config.summarization.idle_minutes * 60:
                continue
            if not self.catalog.has_open_job(session.id):
                job_id = self.catalog.enqueue_job(
                    session.id,
                    session.processed_until_message + 1,
                    session.message_count,
                    now.isoformat(timespec="seconds"),
                )
                scheduled += int(job_id is not None)
        return scheduled

    def _schedule_if_needed(self, session: Session) -> None:
        new_messages = session.message_count - session.processed_until_message
        config = self.config.summarization
        threshold_reached = (
            new_messages >= config.messages_per_batch
            or session.new_token_estimate >= config.token_threshold
        )
        if threshold_reached and not self.catalog.has_open_job(session.id):
            self.catalog.enqueue_job(
                session.id,
                session.processed_until_message + 1,
                session.message_count,
                utc_or_local_now(),
            )
