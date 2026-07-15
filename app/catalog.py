from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from .models import Session, Topic


logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    processed_until_message INTEGER NOT NULL DEFAULT 0,
    summary_revision INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    problem TEXT NOT NULL,
    keywords TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_sources (
    topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    from_message INTEGER NOT NULL,
    to_message INTEGER NOT NULL,
    PRIMARY KEY (topic_id, session_id, from_message, to_message),
    CHECK (from_message > 0 AND to_message >= from_message)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    from_message INTEGER NOT NULL,
    to_message INTEGER NOT NULL,
    type TEXT NOT NULL DEFAULT 'summary',
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, from_message, to_message, type)
);

CREATE TABLE IF NOT EXISTS topic_embeddings (
    topic_id TEXT PRIMARY KEY REFERENCES topics(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vector_index_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS topics_fts USING fts5(
    topic_id UNINDEXED,
    title,
    description,
    problem,
    keywords,
    summary,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE INDEX IF NOT EXISTS topics_session_idx ON topics(session_id);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);
"""


class Catalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            import sqlite_vec

            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
        except (ImportError, AttributeError, sqlite3.Error):
            # The regular embedding table supports an exact cosine fallback.
            try:
                connection.enable_load_extension(False)
            except (AttributeError, sqlite3.Error):
                pass
        return connection

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)

    def _store_path(self, path: str | Path) -> str:
        value = Path(path).resolve()
        try:
            return str(value.relative_to(self.path.parent.resolve()))
        except ValueError:
            return str(value)

    def _resolve_path(self, path: str) -> Path:
        value = Path(path)
        return value if value.is_absolute() else self.path.parent / value

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def upsert_session(self, session: Session, path: str | Path) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, path, agent, status, started_at, ended_at, message_count,
                    processed_until_message, summary_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path=excluded.path, agent=excluded.agent, status=excluded.status,
                    started_at=excluded.started_at, ended_at=excluded.ended_at,
                    message_count=excluded.message_count,
                    processed_until_message=excluded.processed_until_message,
                    summary_revision=excluded.summary_revision
                """,
                (
                    session.id,
                    self._store_path(path),
                    session.agent,
                    session.status,
                    session.started_at,
                    session.ended_at,
                    session.message_count,
                    session.processed_until_message,
                    session.summary_revision,
                ),
            )

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        with self.connection() as connection:
            return connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()

    def session_path(self, session_id: str) -> Path | None:
        row = self.get_session(session_id)
        return self._resolve_path(row["path"]) if row else None

    def upsert_topic(self, topic: Topic, path: str | Path) -> None:
        stored_path = self._store_path(path)
        keywords = json.dumps(topic.keywords, ensure_ascii=False)
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO topics (
                    id, session_id, path, title, description, problem, keywords,
                    summary, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id=excluded.session_id, path=excluded.path,
                    title=excluded.title, description=excluded.description,
                    problem=excluded.problem, keywords=excluded.keywords,
                    summary=excluded.summary, status=excluded.status,
                    created_at=excluded.created_at, updated_at=excluded.updated_at
                """,
                (
                    topic.id,
                    topic.session_id,
                    stored_path,
                    topic.title,
                    topic.description,
                    topic.problem,
                    keywords,
                    topic.summary,
                    topic.status,
                    topic.created_at,
                    topic.updated_at,
                ),
            )
            connection.execute("DELETE FROM topic_sources WHERE topic_id = ?", (topic.id,))
            connection.executemany(
                """
                INSERT INTO topic_sources(topic_id, session_id, from_message, to_message)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (topic.id, item.session_id, item.from_message, item.to_message)
                    for item in topic.source_ranges
                ],
            )
            connection.execute("DELETE FROM topics_fts WHERE topic_id = ?", (topic.id,))
            connection.execute(
                """
                INSERT INTO topics_fts(topic_id, title, description, problem, keywords, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    topic.id,
                    topic.title,
                    topic.description,
                    topic.problem,
                    " ".join(topic.keywords),
                    topic.summary,
                ),
            )

    def delete_all_index_data(self) -> None:
        with self.transaction(immediate=True) as connection:
            if self._has_vec(connection):
                connection.execute("DROP TABLE IF EXISTS topic_vec")
            connection.execute("DELETE FROM vector_index_meta")
            connection.execute("DELETE FROM topics_fts")
            connection.execute("DELETE FROM topic_embeddings")
            connection.execute("DELETE FROM topic_sources")
            connection.execute("DELETE FROM topics")
            connection.execute("DELETE FROM jobs")
            connection.execute("DELETE FROM sessions")

    def delete_session_topics(self, session_id: str) -> None:
        with self.transaction(immediate=True) as connection:
            topic_ids = [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM topics WHERE session_id = ?", (session_id,)
                ).fetchall()
            ]
            for topic_id in topic_ids:
                connection.execute(
                    "DELETE FROM topics_fts WHERE topic_id = ?", (topic_id,)
                )
                if self._has_vec(connection) and self._table_exists(
                    connection, "topic_vec"
                ):
                    connection.execute(
                        "DELETE FROM topic_vec WHERE topic_id = ?", (topic_id,)
                    )
            connection.execute("DELETE FROM topics WHERE session_id = ?", (session_id,))

    def delete_session_jobs(self, session_id: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM jobs WHERE session_id = ?", (session_id,))

    def get_topic_row(self, topic_id: str) -> sqlite3.Row | None:
        with self.connection() as connection:
            return connection.execute(
                "SELECT * FROM topics WHERE id = ?", (topic_id,)
            ).fetchone()

    def topic_path(self, topic_id: str) -> Path | None:
        row = self.get_topic_row(topic_id)
        return self._resolve_path(row["path"]) if row else None

    def list_topic_rows(self, session_id: str | None = None) -> list[sqlite3.Row]:
        with self.connection() as connection:
            if session_id:
                return connection.execute(
                    "SELECT * FROM topics WHERE session_id = ? ORDER BY updated_at DESC",
                    (session_id,),
                ).fetchall()
            return connection.execute(
                "SELECT * FROM topics ORDER BY updated_at DESC"
            ).fetchall()

    def lexical_search(self, fts_query: str, limit: int) -> list[sqlite3.Row]:
        with self.connection() as connection:
            return connection.execute(
                """
                SELECT t.*, bm25(topics_fts, 0.0, 5.0, 3.0, 3.0, 2.0, 1.0) AS bm25_score
                FROM topics_fts
                JOIN topics AS t ON t.id = topics_fts.topic_id
                WHERE topics_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

    def enqueue_job(
        self,
        session_id: str,
        from_message: int,
        to_message: int,
        created_at: str,
        job_type: str = "summary",
    ) -> int | None:
        if to_message < from_message:
            return None
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    session_id, from_message, to_message, type, status,
                    attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?)
                """,
                (session_id, from_message, to_message, job_type, created_at, created_at),
            )
            return int(cursor.lastrowid) if cursor.rowcount else None

    def has_open_job(self, session_id: str) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE session_id = ? AND status IN ('pending', 'running')
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            return row is not None

    def claim_job(self, now: str) -> sqlite3.Row | None:
        with self.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = ?
                WHERE id = ?
                """,
                (now, row["id"]),
            )
            return connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (row["id"],)
            ).fetchone()

    def complete_job(self, job_id: int, now: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'completed', error = NULL, updated_at = ? WHERE id = ?",
                (now, job_id),
            )

    def fail_job(self, job_id: int, error: str, now: str, retry: bool = True) -> None:
        status = "pending" if retry else "failed"
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, error[-4000:], now, job_id),
            )

    def list_embeddings(self, topic_ids: Sequence[str] | None = None) -> list[sqlite3.Row]:
        with self.connection() as connection:
            if topic_ids:
                placeholders = ",".join("?" for _ in topic_ids)
                return connection.execute(
                    f"SELECT * FROM topic_embeddings WHERE topic_id IN ({placeholders})",
                    tuple(topic_ids),
                ).fetchall()
            return connection.execute("SELECT * FROM topic_embeddings").fetchall()

    def save_embedding(
        self,
        topic_id: str,
        model: str,
        vector: bytes,
        dimension: int,
        updated_at: str,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO topic_embeddings(topic_id, model, dimension, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(topic_id) DO UPDATE SET model=excluded.model,
                    dimension=excluded.dimension, embedding=excluded.embedding,
                    updated_at=excluded.updated_at
                """,
                (topic_id, model, dimension, vector, updated_at),
            )
        try:
            self._save_vec_embedding(topic_id, model, vector, dimension)
        except sqlite3.Error as exc:
            logger.warning("sqlite-vec indexing failed for %s: %s", topic_id, exc)

    def vector_search(
        self, model: str, query_vector: bytes, dimension: int, limit: int
    ) -> list[str]:
        try:
            with self.connection() as connection:
                if not self._has_vec(connection) or not self._table_exists(
                    connection, "topic_vec"
                ):
                    return []
                meta = connection.execute(
                    "SELECT model, dimension FROM vector_index_meta WHERE singleton = 1"
                ).fetchone()
                if (
                    meta is None
                    or meta["model"] != model
                    or int(meta["dimension"]) != dimension
                ):
                    return []
                rows = connection.execute(
                    """
                    SELECT topic_id, distance
                    FROM topic_vec
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance
                    """,
                    (query_vector, limit),
                ).fetchall()
                return [str(row["topic_id"]) for row in rows]
        except sqlite3.Error as exc:
            logger.warning("sqlite-vec search failed, using exact fallback: %s", exc)
            return []

    def _save_vec_embedding(
        self, topic_id: str, model: str, vector: bytes, dimension: int
    ) -> bool:
        if dimension <= 0:
            return False
        with self.transaction(immediate=True) as connection:
            if not self._has_vec(connection):
                return False
            meta = connection.execute(
                "SELECT model, dimension FROM vector_index_meta WHERE singleton = 1"
            ).fetchone()
            mismatch = meta is not None and (
                meta["model"] != model or int(meta["dimension"]) != dimension
            )
            if mismatch:
                connection.execute("DROP TABLE IF EXISTS topic_vec")
                connection.execute("DELETE FROM vector_index_meta")
                meta = None
            if meta is None:
                if self._table_exists(connection, "topic_vec"):
                    connection.execute("DROP TABLE topic_vec")
                connection.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS topic_vec USING vec0(
                        topic_id TEXT PRIMARY KEY,
                        embedding float[{int(dimension)}] distance_metric=cosine
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO vector_index_meta(singleton, model, dimension)
                    VALUES (1, ?, ?)
                    """,
                    (model, dimension),
                )
            connection.execute("DELETE FROM topic_vec WHERE topic_id = ?", (topic_id,))
            connection.execute(
                "INSERT INTO topic_vec(topic_id, embedding) VALUES (?, ?)",
                (topic_id, vector),
            )
            return True

    @staticmethod
    def _has_vec(connection: sqlite3.Connection) -> bool:
        try:
            connection.execute("SELECT vec_version()").fetchone()
            return True
        except sqlite3.Error:
            return False

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
            ).fetchone()
            is not None
        )

    def reset_running_jobs(self, now: str) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status='pending', updated_at=? WHERE status='running'",
                (now,),
            )
            return cursor.rowcount
