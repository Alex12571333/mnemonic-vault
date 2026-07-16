from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any

from .catalog import Catalog
from .config import AppConfig
from .embeddings import Embedder, cosine_similarity, pack_vector, unpack_vector
from .models import ExplicitMemory, utc_or_local_now
from .recorder import SessionNotFoundError, SessionRecorder
from .retriever import fts_query, relevance_query_tokens, tokenize_query
from .storage import (
    append_jsonl,
    exclusive_lock,
    read_explicit_memories,
    validate_id,
)


logger = logging.getLogger(__name__)

MEMORY_KINDS = {
    "fact",
    "preference",
    "decision",
    "configuration",
    "identity",
    "constraint",
    "task",
    "correction",
}
SCOPE_TYPES = {"global", "agent", "project", "session"}
SCOPE_MODES = {"boost", "strict"}
SCOPE_BOOSTS = {
    "project": 0.12,
    "session": 0.10,
    "agent": 0.08,
}
GLOBAL_SCOPE_BOOST = 0.05
FOREIGN_SESSION_PENALTY = 0.18


class ExplicitMemoryStore:
    """Append-only user-directed memory with an immediately searchable index."""

    def __init__(
        self,
        config: AppConfig,
        catalog: Catalog,
        recorder: SessionRecorder,
        embedder: Embedder | None = None,
    ):
        self.config = config
        self.catalog = catalog
        self.recorder = recorder
        self.embedder = embedder
        self.path = config.storage.root / "explicit-memory.jsonl"
        self.lock_path = config.storage.root / ".explicit-memory.lock"

    def remember(
        self,
        verbatim: str,
        *,
        normalized: str | None = None,
        kind: str = "fact",
        scope_type: str = "global",
        scope_id: str | None = None,
        source_session_id: str | None = None,
        source_message_id: int | None = None,
        idempotency_key: str | None = None,
        supersedes: str | None = None,
        created_at: str | None = None,
        author: str = "user",
    ) -> dict[str, Any]:
        verbatim = verbatim.strip()
        normalized = (normalized or verbatim).strip()
        if not verbatim or not normalized:
            raise ValueError("verbatim and normalized must be non-empty")
        if len(verbatim) > self.config.api.max_message_chars:
            raise ValueError("verbatim is too large")
        if len(normalized) > self.config.api.max_message_chars:
            raise ValueError("normalized is too large")
        if kind not in MEMORY_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(sorted(MEMORY_KINDS))}")
        if scope_type not in SCOPE_TYPES:
            raise ValueError(
                f"scope.type must be one of: {', '.join(sorted(SCOPE_TYPES))}"
            )
        if scope_type == "global":
            if scope_id:
                raise ValueError("global scope must not have an id")
            scope_id = None
        elif not scope_id:
            raise ValueError(f"{scope_type} scope requires an id")
        elif len(scope_id) > 128:
            raise ValueError("scope.id is too long")
        if author != "user":
            raise ValueError("v0.5 only accepts explicit user-authored memories")
        if source_message_id is not None and not source_session_id:
            raise ValueError("source_message_id requires source_session_id")
        if supersedes:
            validate_id(supersedes, "superseded memory id")
        created_at = created_at or utc_or_local_now()
        # Validate timestamps before any durable write.
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        idempotency_key = validate_id(
            idempotency_key or self._generated_idempotency_key(
                verbatim,
                normalized,
                kind,
                scope_type,
                scope_id,
                source_session_id,
                source_message_id,
                supersedes,
            ),
            "idempotency key",
        )

        with exclusive_lock(self.lock_path):
            existing = self._find_existing(idempotency_key)
            if existing is not None:
                self._validate_retry(
                    existing, verbatim, normalized, kind, scope_type, scope_id, supersedes
                )
                return self._receipt(existing, self._enqueue_priority_summary(existing))

            old: ExplicitMemory | None = None
            if supersedes:
                old = self._memory_from_row(
                    self.catalog.get_explicit_memory(supersedes)
                )
                if old is None:
                    # Recover SQLite from the file before rejecting a valid reference.
                    self._replay_file(with_embeddings=False)
                    old = self._memory_from_row(
                        self.catalog.get_explicit_memory(supersedes)
                    )
                if old is None:
                    raise ValueError(f"superseded memory does not exist: {supersedes}")
                if old.status != "active":
                    raise ValueError(f"memory is already superseded: {supersedes}")
                if (old.scope_type, old.scope_id) != (scope_type, scope_id):
                    raise ValueError("a correction must use the same scope as supersedes")
                old_created = datetime.fromisoformat(
                    old.created_at.replace("Z", "+00:00")
                )
                new_created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if new_created.timestamp() < old_created.timestamp():
                    raise ValueError("a correction cannot predate the superseded memory")

            source_session_id, source_message_id = self._resolve_source(
                verbatim,
                idempotency_key,
                created_at,
                source_session_id,
                source_message_id,
            )
            memory = ExplicitMemory(
                memory_id=self._memory_id(idempotency_key),
                idempotency_key=idempotency_key,
                verbatim=verbatim,
                normalized=normalized,
                kind=kind,
                scope_type=scope_type,
                scope_id=scope_id,
                author="user",
                source_session_id=source_session_id,
                source_message_id=source_message_id,
                created_at=created_at,
                event="supersede" if supersedes else "remember",
                supersedes=supersedes,
            )
            # The JSONL fsync is the commit point. SQLite and embeddings are derived.
            append_jsonl(self.path, memory.to_event_dict())
            self.catalog.upsert_explicit_memory(memory)
            return self._receipt(memory, self._enqueue_priority_summary(memory))

    def get(self, memory_id: str) -> ExplicitMemory:
        validate_id(memory_id, "memory id")
        memory = self._memory_from_row(self.catalog.get_explicit_memory(memory_id))
        if memory is None:
            raise FileNotFoundError(memory_id)
        return memory

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        include_superseded: bool = False,
        scope_type: str | None = None,
        scope_id: str | None = None,
        context_scopes: list[tuple[str, str | None]] | None = None,
        scope_mode: str = "boost",
        include_all_scopes: bool = False,
    ) -> list[dict[str, Any]]:
        if scope_mode not in SCOPE_MODES:
            raise ValueError("scope_mode must be boost or strict")
        scopes = set(context_scopes or [])
        strict_scopes: set[tuple[str, str | None]] = set()
        if scope_type:
            primary_scope = (scope_type, scope_id)
            scopes.add(primary_scope)
            strict_scopes.add(primary_scope)
        else:
            strict_scopes.update(scopes)
        for context_type, context_id in scopes:
            if context_type not in SCOPE_TYPES:
                raise ValueError(f"unknown context scope type: {context_type}")
            if context_type == "global" and context_id:
                raise ValueError("global context scope must not have an id")
            if context_type != "global" and not context_id:
                raise ValueError(f"{context_type} context scope requires an id")
        if scope_mode == "strict" and not strict_scopes:
            raise ValueError("strict scope search requires scope or context_scopes")

        expression = fts_query(query)
        lexical_rows = (
            self.catalog.lexical_search_explicit(
                expression, max(limit * 6, 30), include_superseded
            )
            if expression
            else []
        )
        query_tokens = set(relevance_query_tokens(query))
        lexical_scores: dict[str, float] = {}
        for row in lexical_rows:
            memory_tokens = set(
                tokenize_query(
                    " ".join(
                        str(row[key] or "")
                        for key in (
                            "verbatim",
                            "normalized",
                            "kind",
                            "scope_type",
                            "scope_id",
                        )
                    )
                )
            )
            overlap = len(query_tokens & memory_tokens)
            if overlap:
                lexical_scores[str(row["memory_id"])] = (
                    overlap / len(query_tokens) if query_tokens else 0.0
                )
        # A lexical hit must remain fast even when the optional embedding endpoint
        # is offline. Semantic fallback is attempted only when FTS found nothing.
        vector_scores = (
            {} if lexical_scores else self._vector_scores(query, include_superseded)
        )
        ranked: list[tuple[float, ExplicitMemory]] = []
        for row in self.catalog.list_explicit_memory_rows():
            memory_id = str(row["memory_id"])
            if memory_id not in lexical_scores and memory_id not in vector_scores:
                continue
            memory = self._memory_from_row(row)
            if memory is None or (memory.status != "active" and not include_superseded):
                continue
            memory_scope = (memory.scope_type, memory.scope_id)
            if scope_mode == "strict" and memory_scope not in strict_scopes:
                continue
            base = max(lexical_scores.get(memory_id, 0.0), vector_scores.get(memory_id, 0.0))
            if base <= 0.0:
                continue
            boost = 0.0
            if memory.status == "active" and not include_superseded:
                boost += 0.08
            elif memory.status == "superseded" and include_superseded:
                boost += 0.08
            if memory.author == "user":
                boost += 0.05
            if memory.scope_type == "global":
                boost += GLOBAL_SCOPE_BOOST
            elif memory_scope in scopes:
                boost += SCOPE_BOOSTS[memory.scope_type]
            elif memory.scope_type == "session" and not include_all_scopes:
                # Session facts remain discoverable across the shared Vault, but
                # automatic recall should not casually inject another chat's
                # temporary instructions into the current prompt.
                boost -= FOREIGN_SESSION_PENALTY
            # A deterministic recency tie-breaker comes from the final tuple sort.
            score = max(0.0, min(1.0, base + boost))
            if score > 0.0:
                ranked.append((score, memory))
        ranked.sort(key=lambda item: (item[0], item[1].created_at, item[1].memory_id), reverse=True)
        return [memory.to_search_dict(score) for score, memory in ranked[:limit]]

    def rebuild_index(self, with_embeddings: bool = False) -> dict[str, int]:
        with exclusive_lock(self.lock_path):
            return self._replay_file(with_embeddings)

    def reembed_all(self) -> dict[str, int]:
        if self.embedder is None:
            raise RuntimeError("embedding endpoint is not configured")
        succeeded = 0
        failed = 0
        for row in self.catalog.list_explicit_memory_rows():
            memory = self._memory_from_row(row)
            if memory is None:
                failed += 1
                continue
            try:
                self.index_embedding(memory, raise_errors=True)
                succeeded += 1
            except Exception:
                logger.exception("could not re-embed explicit memory %s", memory.memory_id)
                failed += 1
        return {"embedded": succeeded, "failures": failed}

    def index_embedding(
        self, memory: ExplicitMemory, *, raise_errors: bool = False
    ) -> bool:
        if self.embedder is None:
            return False
        text = "\n".join(
            (
                memory.normalized,
                memory.verbatim,
                f"Kind: {memory.kind}",
                f"Scope: {memory.scope_type}:{memory.scope_id or ''}",
            )
        )
        try:
            vector = self.embedder.embed([text])[0]
            self.catalog.save_explicit_embedding(
                memory.memory_id,
                self.embedder.model,
                pack_vector(vector),
                len(vector),
                utc_or_local_now(),
            )
            return True
        except Exception as exc:
            if raise_errors:
                raise
            logger.warning("could not embed explicit memory %s: %s", memory.memory_id, exc)
            return False

    def _resolve_source(
        self,
        verbatim: str,
        idempotency_key: str,
        created_at: str,
        source_session_id: str | None,
        source_message_id: int | None,
    ) -> tuple[str, int]:
        if source_session_id:
            messages = self.recorder.read_turns(source_session_id)
            candidates = [
                message
                for message in messages
                if message.role == "user"
                and message.text == verbatim
                and (source_message_id is None or message.id == source_message_id)
            ]
            if not candidates:
                raise ValueError(
                    "source must reference a user transcript message with exact verbatim text"
                )
            source = candidates[-1]
            return source_session_id, source.id

        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        session_id = f"session-explicit-{timestamp.year:04d}-{timestamp.month:02d}"
        try:
            self.recorder.locate(session_id)
        except SessionNotFoundError:
            self.recorder.start_session("explicit-memory", session_id, created_at)
        external_event_id = "event-" + hashlib.sha256(
            f"explicit-source\0{idempotency_key}".encode("utf-8")
        ).hexdigest()[:40]
        metadata = {"source": "memory_remember", "idempotency_key": idempotency_key}
        try:
            message = self.recorder.append(
                session_id,
                "user",
                verbatim,
                created_at,
                metadata=metadata,
                external_event_id=external_event_id,
            )
        except RuntimeError as exc:
            if "cannot append to" not in str(exc):
                raise
            # A monthly source session should normally remain active. If it was
            # manually finalized, use a deterministic per-memory recovery session.
            suffix = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:12]
            session_id = f"session-explicit-recovery-{suffix}"
            try:
                self.recorder.locate(session_id)
            except SessionNotFoundError:
                self.recorder.start_session("explicit-memory", session_id, created_at)
            message = self.recorder.append(
                session_id,
                "user",
                verbatim,
                created_at,
                metadata=metadata,
                external_event_id=external_event_id,
            )
        indexed = self.catalog.message_by_external_event_global(external_event_id)
        actual_session_id = str(indexed["session_id"]) if indexed is not None else session_id
        return actual_session_id, message.id

    def _enqueue_priority_summary(self, memory: ExplicitMemory) -> str:
        row = self.catalog.get_session(memory.source_session_id)
        if row is None:
            return "unavailable"
        processed = int(row["processed_until_message"])
        message_count = int(row["message_count"])
        if memory.source_message_id <= processed:
            return "already_processed"
        job_id = self.catalog.enqueue_or_extend_priority_summary(
            memory.source_session_id,
            processed + 1,
            message_count,
            memory.created_at,
        )
        return "pending" if job_id is not None else "already_queued"

    def _find_existing(self, idempotency_key: str) -> ExplicitMemory | None:
        memory = self._memory_from_row(
            self.catalog.explicit_memory_by_idempotency(idempotency_key)
        )
        if memory is not None:
            return memory
        if not self.path.exists():
            return None
        events = read_explicit_memories(self.path)
        logged = next(
            (item for item in events if item.idempotency_key == idempotency_key), None
        )
        if logged is not None:
            self._replay_events(events, with_embeddings=False)
            return self._memory_from_row(
                self.catalog.explicit_memory_by_idempotency(idempotency_key)
            )
        return None

    def _replay_file(self, with_embeddings: bool) -> dict[str, int]:
        events = read_explicit_memories(self.path)
        return self._replay_events(events, with_embeddings)

    def _replay_events(
        self, events: list[ExplicitMemory], with_embeddings: bool
    ) -> dict[str, int]:
        failures = 0
        indexed = 0
        for memory in events:
            try:
                self.catalog.upsert_explicit_memory(memory)
                self._enqueue_priority_summary(memory)
                if with_embeddings:
                    self.index_embedding(memory)
                indexed += 1
            except Exception:
                logger.exception("could not replay explicit memory %s", memory.memory_id)
                failures += 1
        return {"explicit_memories": indexed, "explicit_memory_failures": failures}

    def _vector_scores(self, query: str, include_superseded: bool) -> dict[str, float]:
        if self.embedder is None:
            return {}
        try:
            query_vector = self.embedder.embed([query])[0]
        except Exception:
            return {}
        statuses = {
            str(row["memory_id"]): str(row["status"])
            for row in self.catalog.list_explicit_memory_rows()
        }
        scored: list[tuple[float, str]] = []
        for row in self.catalog.list_explicit_embeddings():
            memory_id = str(row["memory_id"])
            if not include_superseded and statuses.get(memory_id) != "active":
                continue
            if row["model"] != self.embedder.model:
                continue
            if int(row["dimension"]) != len(query_vector):
                continue
            vector = unpack_vector(row["embedding"], int(row["dimension"]))
            similarity = cosine_similarity(query_vector, vector)
            if similarity >= self.config.retrieval.vector_min_similarity:
                scored.append((similarity, memory_id))
        scored.sort(reverse=True)
        return dict((memory_id, score) for score, memory_id in scored)

    @staticmethod
    def _memory_from_row(row: Any | None) -> ExplicitMemory | None:
        if row is None:
            return None
        return ExplicitMemory(
            memory_id=str(row["memory_id"]),
            idempotency_key=str(row["idempotency_key"]),
            event=str(row["event"]),
            verbatim=str(row["verbatim"]),
            normalized=str(row["normalized"]),
            kind=str(row["kind"]),
            scope_type=str(row["scope_type"]),
            scope_id=str(row["scope_id"]) if row["scope_id"] else None,
            author=str(row["author"]),
            source_session_id=str(row["source_session_id"]),
            source_message_id=int(row["source_message_id"]),
            created_at=str(row["created_at"]),
            status=str(row["status"]),
            supersedes=str(row["supersedes"]) if row["supersedes"] else None,
            valid_to=str(row["valid_to"]) if row["valid_to"] else None,
            requires_confirmation=bool(row["requires_confirmation"]),
        )

    @staticmethod
    def _memory_id(idempotency_key: str) -> str:
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:24]
        return f"mem-{digest}"

    @staticmethod
    def _generated_idempotency_key(*values: Any) -> str:
        digest = hashlib.sha256(
            "\0".join("" if value is None else str(value) for value in values).encode(
                "utf-8"
            )
        ).hexdigest()[:40]
        return f"event-{digest}"

    @staticmethod
    def _validate_retry(
        memory: ExplicitMemory,
        verbatim: str,
        normalized: str,
        kind: str,
        scope_type: str,
        scope_id: str | None,
        supersedes: str | None,
    ) -> None:
        expected = (verbatim, normalized, kind, scope_type, scope_id, supersedes)
        actual = (
            memory.verbatim,
            memory.normalized,
            memory.kind,
            memory.scope_type,
            memory.scope_id,
            memory.supersedes,
        )
        if actual != expected:
            raise ValueError("idempotency_key is already bound to different memory")

    @staticmethod
    def _receipt(memory: ExplicitMemory, summary_job: str) -> dict[str, Any]:
        return {
            "stored": True,
            "memory_id": memory.memory_id,
            "available_for_recall": True,
            "summary_job": summary_job,
            "source": {
                "session_id": memory.source_session_id,
                "message_id": memory.source_message_id,
            },
        }
