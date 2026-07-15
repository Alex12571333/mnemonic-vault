from __future__ import annotations

import logging
from pathlib import Path

from .catalog import Catalog
from .config import AppConfig
from .embeddings import Embedder, pack_vector, topic_embedding_text
from .models import Topic, utc_or_local_now
from .storage import read_session, read_topic


logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        config: AppConfig,
        catalog: Catalog,
        embedder: Embedder | None = None,
    ):
        self.config = config
        self.catalog = catalog
        self.embedder = embedder

    def index_topic(self, path: str | Path, with_embedding: bool = True) -> Topic:
        path = Path(path)
        topic = read_topic(path)
        topic.path = str(path)
        self.catalog.upsert_topic(topic, path)
        if with_embedding and self.embedder is not None:
            try:
                vector = self.embedder.embed(
                    [
                        topic_embedding_text(
                            topic.title,
                            topic.description,
                            topic.problem,
                            topic.keywords,
                            topic.summary,
                        )
                    ]
                )[0]
                self.catalog.save_embedding(
                    topic.id,
                    self.embedder.model,
                    pack_vector(vector),
                    len(vector),
                    utc_or_local_now(),
                )
            except Exception as exc:
                # Lexical retrieval remains available when the optional endpoint is down.
                logger.warning("could not embed topic %s: %s", topic.id, exc)
        return topic

    def rebuild(self, with_embeddings: bool = False) -> dict[str, int]:
        self.catalog.delete_all_index_data()
        sessions = 0
        topics = 0
        failures = 0
        recovered_jobs = 0
        for session_file in sorted(
            self.config.storage.sessions_dir.glob("*/*/*/session.json")
        ):
            try:
                session = read_session(session_file.parent)
                self.catalog.upsert_session(session, session_file.parent)
                sessions += 1
                if session.message_count > session.processed_until_message:
                    job_id = self.catalog.enqueue_job(
                        session.id,
                        session.processed_until_message + 1,
                        session.message_count,
                        utc_or_local_now(),
                    )
                    recovered_jobs += int(job_id is not None)
            except Exception:
                logger.exception("could not index session at %s", session_file)
                failures += 1
                continue
            for topic_file in sorted((session_file.parent / "topics").glob("*.md")):
                try:
                    self.index_topic(topic_file, with_embedding=with_embeddings)
                    topics += 1
                except Exception:
                    logger.exception("could not index topic at %s", topic_file)
                    failures += 1
        return {
            "sessions": sessions,
            "topics": topics,
            "recovered_jobs": recovered_jobs,
            "failures": failures,
        }

    def reembed_all(self) -> dict[str, int]:
        if self.embedder is None:
            raise RuntimeError("embedding endpoint is not configured")
        succeeded = 0
        failed = 0
        for row in self.catalog.list_topic_rows():
            path = self.catalog.topic_path(row["id"])
            if path is None:
                failed += 1
                continue
            try:
                self.index_topic(path, with_embedding=True)
                succeeded += 1
            except Exception:
                logger.exception("could not re-embed topic %s", row["id"])
                failed += 1
        return {"embedded": succeeded, "failures": failed}
