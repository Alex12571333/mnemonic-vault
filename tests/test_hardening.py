from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import create_app
from app.catalog import Catalog
from app.config import AppConfig
from app.evaluation import evaluate_retrieval
from app.indexer import Indexer
from app.models import SourceRange, Topic, utc_or_local_now
from app.recorder import SessionRecorder
from app.retriever import ContextBuilder, Retriever, rank_messages
from app.session_aliases import migrate_session_ids, stable_session_id
from app.service import Services
from app.storage import estimate_tokens, read_messages, read_session, write_topic
from app.summarizer import JobRunner, MemorySummarizer
from integrations.hermes.mnemonic_vault import MnemonicVaultMemoryProvider
from integrations.hermes.mnemonic_vault.client import (
    VaultHttpError,
    deterministic_event_id,
)
from run import validate_bind_security


class AxisEmbedder:
    model = "axis-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "dflash" in text.lower() else [0.0, 1.0]
            for text in texts
        ]


class OneTopicLLM:
    def __init__(self, callback=None):
        self.callback = callback

    def summarize(self, messages, current_topics, finalizing, topic_cards=None):
        if self.callback:
            self.callback()
        return {
            "operations": [
                {
                    "action": "create_topic",
                    "title": "Durable topic",
                    "description": "A durable test topic",
                    "problem": "Reliability",
                    "keywords": ["durable"],
                    "source_ranges": [[messages[0].id, messages[-1].id]],
                    "summary": "## Итог\nDurable summary.",
                }
            ]
        }


class InvalidSecondOperationLLM:
    def summarize(self, messages, current_topics, finalizing, topic_cards=None):
        return {
            "operations": [
                {
                    "action": "create_topic",
                    "title": "Must not leak",
                    "source_ranges": [[messages[0].id, messages[-1].id]],
                    "summary": "## Итог\nThis must remain staged only.",
                },
                {"action": "delete_everything"},
            ]
        }


class HardeningTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = AppConfig.from_mapping(
            {
                "storage": {
                    "root": "data",
                    "sessions_dir": "data/sessions",
                    "catalog_db": "data/catalog.sqlite",
                },
                "summarization": {
                    "messages_per_batch": 1,
                    "token_threshold": 10_000,
                    "idle_minutes": 30,
                },
                "retrieval": {
                    "lexical_top_k": 30,
                    "vector_top_k": 30,
                    "final_top_k": 5,
                    "auto_open_summaries": 2,
                    "total_context_budget_tokens": 400,
                    "card_budget_tokens": 260,
                    "summary_budget_tokens": 70,
                    "source_budget_tokens": 50,
                    "minimum_score": 0.35,
                    "vector_min_similarity": 0.8,
                    "lexical_min_query_coverage": 0.34,
                    "rrf_k": 60,
                },
                "embeddings": {"base_url": "", "model": ""},
            },
            self.root,
        )
        self.catalog = Catalog(self.config.storage.catalog_db)
        self.recorder = SessionRecorder(self.config, self.catalog)

    def tearDown(self):
        self.temporary.cleanup()

    def _services(self, llm=None, embedder=None) -> Services:
        indexer = Indexer(self.config, self.catalog, embedder)
        retriever = Retriever(self.config, self.catalog, self.recorder, embedder)
        summarizer = MemorySummarizer(
            self.config,
            self.catalog,
            self.recorder,
            indexer,
            llm or OneTopicLLM(),
        )
        runner = JobRunner(summarizer, self.recorder)
        return Services(
            config=self.config,
            catalog=self.catalog,
            recorder=self.recorder,
            indexer=indexer,
            retriever=retriever,
            context_builder=ContextBuilder(self.config, retriever),
            summarizer=summarizer,
            job_runner=runner,
        )

    def _write_topic(self, session_id: str, title: str = "DFlash launch") -> Topic:
        session_path = self.recorder.locate(session_id)
        now = utc_or_local_now()
        topic = Topic(
            id=f"topic-{session_id}-{title.lower().replace(' ', '-')}",
            session_id=session_id,
            title=title,
            description="CUDA inference configuration",
            problem="Stable inference",
            status="active",
            keywords=["DFlash", "CUDA"],
            source_ranges=[SourceRange(session_id, 1, 1)],
            created_at=now,
            updated_at=now,
            summary="## Итог\nUse the durable DFlash configuration.",
        )
        path = session_path / "topics" / f"{topic.id}.md"
        write_topic(path, topic)
        return topic

    def test_idempotent_append_and_rebuildable_transcript_fts(self):
        session = self.recorder.start_session("openclaw", "session-idempotent")
        first = self.recorder.append(
            session.id, "user", "Exact DFlash error 123", external_event_id="evt-1"
        )
        repeated = self.recorder.append(
            session.id, "user", "Exact DFlash error 123", external_event_id="evt-1"
        )
        self.assertEqual(first.id, repeated.id)
        path = self.recorder.locate(session.id)
        self.assertEqual(len(read_messages(path / "transcript.jsonl")), 1)
        with self.assertRaisesRegex(ValueError, "different content"):
            self.recorder.append(
                session.id, "user", "changed", external_event_id="evt-1"
            )

        retriever = Retriever(self.config, self.catalog, self.recorder)
        self.assertEqual(
            retriever.search_transcript("DFlash error")[0]["session_id"], session.id
        )
        self.assertEqual(retriever.search_transcript("totally unrelated"), [])

        rebuilt = Indexer(self.config, self.catalog).rebuild()
        self.assertEqual(rebuilt["messages"], 1)
        self.assertTrue(retriever.search_transcript("DFlash error"))

    def test_deterministic_adapter_event_is_idempotent_across_sessions(self):
        first_session = self.recorder.start_session("openclaw", "session-event-a")
        second_session = self.recorder.start_session("openclaw", "session-event-b")
        event_id = "event-" + "a" * 40
        first = self.recorder.append(
            first_session.id,
            "user",
            "one durable agent turn",
            external_event_id=event_id,
        )
        repeated = self.recorder.append(
            second_session.id,
            "user",
            "one durable agent turn",
            external_event_id=event_id,
        )
        self.assertEqual(repeated.id, first.id)
        self.assertEqual(
            read_messages(self.recorder.locate(second_session.id) / "transcript.jsonl"),
            [],
        )
        with self.assertRaisesRegex(ValueError, "different content"):
            self.recorder.append(
                second_session.id,
                "user",
                "collision",
                external_event_id=event_id,
            )

    def test_legacy_session_aliases_preserve_files_and_scope_transcript_search(self):
        external_id = "agent:main:telegram:direct:42"
        session_ids = ["session-openclaw-pid-100", "session-openclaw-pid-200"]
        before: dict[str, bytes] = {}
        for index, session_id in enumerate(session_ids, 1):
            session = self.recorder.start_session(
                "openclaw", session_id, f"2026-0{index}-01T10:00:00+09:00"
            )
            self.recorder.append(
                session.id,
                "user",
                f"legacy DFlash fragment {index}",
                metadata={"external_session_id": external_id},
            )
            transcript = self.recorder.locate(session.id) / "transcript.jsonl"
            before[session.id] = transcript.read_bytes()

        dry_run = migrate_session_ids(self.config, dry_run=True)
        self.assertEqual(dry_run["alias_groups"], 1)
        self.assertFalse((self.config.storage.root / "session-aliases.json").exists())
        report = migrate_session_ids(self.config)
        canonical = stable_session_id(external_id, "openclaw", "openclaw-main")
        self.assertEqual(report["aliases"][0]["canonical_session_id"], canonical)
        self.assertEqual(
            report["aliases"][0]["member_session_ids"], session_ids
        )
        for session_id in session_ids:
            transcript = self.recorder.locate(session_id) / "transcript.jsonl"
            self.assertEqual(transcript.read_bytes(), before[session_id])

        retriever = Retriever(self.config, self.catalog, self.recorder)
        hits = retriever.search_transcript("legacy DFlash fragment", canonical)
        self.assertEqual({hit["session_id"] for hit in hits}, set(session_ids))
        client = TestClient(create_app(services=self._services()))
        response = client.get(f"/v1/sessions/{canonical}/aliases")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["member_session_ids"], session_ids)

    def test_idempotent_retry_recovers_session_metadata_after_index_crash(self):
        session = self.recorder.start_session("openclaw", "session-index-crash")
        original = self.catalog.index_message
        attempts = 0

        def fail_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("simulated crash after transcript fsync")
            return original(*args, **kwargs)

        with patch.object(self.catalog, "index_message", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "after transcript fsync"):
                self.recorder.append(
                    session.id,
                    "user",
                    "durable exactly once",
                    external_event_id="recover-event-1",
                )
            repeated = self.recorder.append(
                session.id,
                "user",
                "durable exactly once",
                external_event_id="recover-event-1",
            )

        stored = read_session(self.recorder.locate(session.id))
        self.assertEqual(repeated.id, 1)
        self.assertEqual(stored.message_count, 1)
        self.assertEqual(
            len(read_messages(self.recorder.locate(session.id) / "transcript.jsonl")),
            1,
        )

    def test_absolute_relevance_gate_rejects_nearest_unrelated_vector(self):
        session = self.recorder.start_session("test", "session-relevance")
        self.recorder.append(session.id, "user", "DFlash CUDA launch")
        topic = self._write_topic(session.id)
        indexer = Indexer(self.config, self.catalog, AxisEmbedder())
        indexer.index_topic(self.recorder.locate(session.id) / "topics" / f"{topic.id}.md")
        retriever = Retriever(self.config, self.catalog, self.recorder, AxisEmbedder())
        self.assertEqual(retriever.search("gardening recipe"), [])
        self.assertEqual(retriever.search("DFlash")[0].topic.id, topic.id)

    def test_summarizer_uses_the_same_absolute_topic_relevance_gate(self):
        session = self.recorder.start_session("test", "session-summary-gate")
        self.recorder.append(session.id, "user", "DFlash CUDA launch")
        topic = self._write_topic(session.id)
        services = self._services(embedder=AxisEmbedder())
        services.indexer.index_topic(
            self.recorder.locate(session.id) / "topics" / f"{topic.id}.md"
        )
        unrelated = [
            SimpleNamespace(text="gardening recipe")
        ]
        relevant = [SimpleNamespace(text="DFlash launch")]
        self.assertEqual(
            services.summarizer._select_existing_topics(unrelated, [topic]), []
        )
        self.assertEqual(
            services.summarizer._select_existing_topics(relevant, [topic])[0].id,
            topic.id,
        )

    def test_rank_messages_never_returns_unmatched_recent_turns(self):
        session = self.recorder.start_session("test", "session-rank")
        self.recorder.append(session.id, "user", "alpha only")
        messages = self.recorder.read_turns(session.id)
        self.assertEqual(rank_messages(messages, "omega"), [])

    def test_near_duplicate_topics_prefer_newer_without_deleting_history(self):
        old_session = self.recorder.start_session(
            "test", "session-old", "2026-01-01T10:00:00+09:00"
        )
        self.recorder.append(old_session.id, "user", "DFlash launch")
        old = self._write_topic(old_session.id, "DFlash launch")
        old.updated_at = "2026-01-01T10:00:00+09:00"
        old_path = self.recorder.locate(old_session.id) / "topics" / f"{old.id}.md"
        write_topic(old_path, old)

        new_session = self.recorder.start_session(
            "test", "session-new", "2027-01-01T10:00:00+09:00"
        )
        self.recorder.append(new_session.id, "user", "DFlash launch")
        new = self._write_topic(new_session.id, "DFlash launch")
        new.updated_at = "2027-01-01T10:00:00+09:00"
        new_path = self.recorder.locate(new_session.id) / "topics" / f"{new.id}.md"
        write_topic(new_path, new)

        indexer = Indexer(self.config, self.catalog)
        indexer.index_topic(old_path, with_embedding=False)
        indexer.index_topic(new_path, with_embedding=False)
        hits = Retriever(self.config, self.catalog, self.recorder).search("DFlash launch")
        self.assertEqual([hit.topic.id for hit in hits], [new.id])
        self.assertEqual(hits[0].related_older_topic_ids, [old.id])
        historical = Retriever(self.config, self.catalog, self.recorder).search(
            "DFlash launch раньше"
        )
        self.assertEqual({hit.topic.id for hit in historical}, {old.id, new.id})
        dated = Retriever(self.config, self.catalog, self.recorder).search(
            "Какой DFlash launch мы использовали в 2026 году?"
        )
        self.assertEqual([hit.topic.id for hit in dated], [old.id])
        missing_date = Retriever(self.config, self.catalog, self.recorder).search(
            "Какой DFlash launch мы использовали в 2025 году?"
        )
        self.assertEqual(missing_date, [])
        self.assertTrue(old_path.exists())

    def test_token_estimate_is_conservative_for_russian_fallback(self):
        self.assertEqual(estimate_tokens("абвгдежзий"), 4)

    def test_context_builder_enforces_one_total_budget_and_exposes_dates(self):
        session = self.recorder.start_session(
            "test", "session-budget", "2026-07-15T10:00:00+09:00"
        )
        self.recorder.append(session.id, "user", "DFlash exact command")
        topic = self._write_topic(session.id)
        topic.summary = "## Итог\n" + ("DFlash details " * 500)
        write_topic(
            self.recorder.locate(session.id) / "topics" / f"{topic.id}.md",
            topic,
        )
        Indexer(self.config, self.catalog).index_topic(
            self.recorder.locate(session.id) / "topics" / f"{topic.id}.md",
            with_embedding=False,
        )
        retriever = Retriever(self.config, self.catalog, self.recorder)
        result = ContextBuilder(self.config, retriever).build(
            "DFlash", include_sources="always"
        )
        self.assertLessEqual(result["used_tokens"], result["budget_tokens"])
        self.assertEqual(
            sum(result["budget_breakdown"].values()), result["used_tokens"]
        )
        self.assertEqual(result["topics"][0]["session_started_at"],
                         "2026-07-15T10:00:00+09:00")
        self.assertIn("created_at", result["topics"][0])
        self.assertIn("[truncated to context budget]", result["topics"][0]["summary"])

    def test_invalid_operation_batch_writes_no_topic(self):
        session = self.recorder.start_session("test", "session-atomic")
        self.recorder.append(session.id, "user", "one durable turn")
        services = self._services(InvalidSecondOperationLLM())
        with self.assertRaisesRegex(ValueError, "unknown summary action"):
            services.summarizer.process_next_job()
        self.assertEqual(list((self.recorder.locate(session.id) / "topics").glob("*.md")), [])

    def test_staged_batch_replays_after_catalog_commit_failure_without_llm(self):
        session = self.recorder.start_session("test", "session-stage-replay")
        self.recorder.append(session.id, "user", "one durable staged turn")
        calls = 0

        def count_call():
            nonlocal calls
            calls += 1

        services = self._services(OneTopicLLM(count_call))
        original = self.catalog.upsert_topics
        commit_attempts = 0

        def fail_once(topics):
            nonlocal commit_attempts
            commit_attempts += 1
            if commit_attempts == 1:
                raise OSError("simulated catalog commit failure")
            return original(topics)

        with patch.object(self.catalog, "upsert_topics", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "simulated catalog"):
                services.summarizer.process_next_job()
            manifests = list((self.config.storage.root / "jobs").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            result = services.summarizer.process_next_job()

        self.assertEqual(result["processed_until_message"], 1)
        self.assertEqual(calls, 1)
        self.assertEqual(commit_attempts, 2)
        self.assertEqual(
            list((self.recorder.locate(session.id) / "topics").glob("*.md")) != [],
            True,
        )
        self.assertEqual(list((self.config.storage.root / "jobs").glob("*/manifest.json")), [])

    def test_concurrent_append_is_not_overwritten_by_summary_commit(self):
        session = self.recorder.start_session("test", "session-race")
        self.recorder.append(session.id, "user", "first turn")

        def append_during_llm():
            self.recorder.append(session.id, "assistant", "concurrent second turn")

        services = self._services(OneTopicLLM(append_during_llm))
        services.summarizer.process_next_job()
        stored = read_session(self.recorder.locate(session.id))
        self.assertEqual(stored.message_count, 2)
        self.assertEqual(stored.processed_until_message, 1)

    def test_failed_jobs_can_be_requeued(self):
        session = self.recorder.start_session("test", "session-retry")
        job_id = self.catalog.enqueue_job(session.id, 1, 1, utc_or_local_now())
        self.catalog.fail_job(job_id, "boom", utc_or_local_now(), retry=False)
        self.assertEqual(self.catalog.retry_failed_jobs(utc_or_local_now()), 1)
        claimed = self.catalog.claim_job(utc_or_local_now())
        self.assertEqual(claimed["id"], job_id)
        self.assertEqual(claimed["attempts"], 1)

    def test_api_auth_limits_and_safe_bind(self):
        self.config.api.max_request_bytes = 200
        self.config.api.max_message_chars = 5
        self.config.api.max_query_chars = 5
        services = self._services()
        with patch.dict(os.environ, {"MNEMONIC_VAULT_API_TOKEN": "secret"}):
            app = create_app(services=services, start_worker=False)
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.post(
                    "/v1/sessions/start", json={"session_id": "session-auth"}
                ).status_code, 401)
                response = client.post(
                    "/v1/sessions/start",
                    headers={"authorization": "Bearer secret"},
                    json={"session_id": "session-auth"},
                )
                self.assertEqual(response.status_code, 201)
                self.assertEqual(client.post(
                    "/v1/sessions/session-auth/messages",
                    headers={"authorization": "Bearer secret"},
                    json={"role": "user", "content": "123456"},
                ).status_code, 413)
                self.assertEqual(client.post(
                    "/v1/memory/search",
                    headers={"authorization": "Bearer secret"},
                    json={"query": "123456"},
                ).status_code, 413)
                self.assertEqual(
                    client.post(
                        "/v1/sessions/session-auth/messages",
                        headers={"authorization": "Bearer secret"},
                        json={"role": "user", "content": "1", "metadata": {"x": "y" * 300}},
                    ).status_code,
                    413,
                )
                chunked = client.post(
                    "/v1/sessions/session-auth/messages",
                    headers={"authorization": "Bearer secret"},
                    content=iter([b'{"role":"user","content":"', b"x" * 300, b'"}']),
                )
                self.assertEqual(chunked.status_code, 413)
                first = client.post(
                    "/v1/sessions/session-auth/messages",
                    headers={"authorization": "Bearer secret"},
                    json={
                        "role": "user",
                        "content": "12345",
                        "external_event_id": "http-event-1",
                    },
                )
                repeated = client.post(
                    "/v1/sessions/session-auth/messages",
                    headers={"authorization": "Bearer secret"},
                    json={
                        "role": "user",
                        "content": "12345",
                        "external_event_id": "http-event-1",
                    },
                )
                self.assertEqual(first.status_code, 201)
                self.assertEqual(first.json()["id"], repeated.json()["id"])
                self.assertEqual(
                    client.post(
                        "/v1/sessions/session-auth/end",
                        headers={"authorization": "Bearer secret"},
                        json={},
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    client.post(
                        "/v1/sessions/session-auth/messages",
                        headers={"authorization": "Bearer secret"},
                        json={"role": "user", "content": "later"},
                    ).status_code,
                    409,
                )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                validate_bind_security("0.0.0.0", self.config)
            validate_bind_security("127.0.0.1", self.config)

    def test_russian_retrieval_evaluation_reports_recall_and_rejection(self):
        class FakeRetriever:
            def search(self, query, max_topics=5):
                if "DFlash" in query:
                    return [SimpleNamespace(topic=SimpleNamespace(id="topic-dflash"))]
                return []

        dataset = self.root / "evaluation.jsonl"
        dataset.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "query": "Как запускали DFlash?",
                            "kind": "ru-paraphrase",
                            "relevant_topic_ids": ["topic-dflash"],
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "query": "Рецепт яблочного пирога",
                            "kind": "negative",
                            "relevant_topic_ids": [],
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = evaluate_retrieval(FakeRetriever(), dataset, top_k=5)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["negative_rejection_rate"], 1.0)


class AlwaysFailClient:
    def start_session(self, session_id, agent):
        return None

    def append_message(self, *args, **kwargs):
        raise OSError("offline")


class RecordingClient:
    def __init__(self):
        self.events = []

    def start_session(self, session_id, agent):
        self.events.append(("start", session_id))

    def append_message(self, session_id, role, content, metadata=None,
                       external_event_id=None):
        self.events.append(("message", role, content, external_event_id))
        return {}

    def end_session(self, session_id):
        self.events.append(("end", session_id))
        return {}


class StatusClient(RecordingClient):
    def __init__(self, status: int, poison: str = "poison"):
        super().__init__()
        self.status = status
        self.poison = poison

    def append_message(self, session_id, role, content, metadata=None,
                       external_event_id=None):
        if content == self.poison:
            raise VaultHttpError(self.status, "rejected")
        return super().append_message(
            session_id, role, content, metadata, external_event_id
        )


class FinalizedThenRecoveryClient(RecordingClient):
    def append_message(self, session_id, role, content, metadata=None,
                       external_event_id=None):
        if not session_id.startswith("session-recovery-"):
            raise VaultHttpError(409, "finalized")
        self.events.append(
            ("message", session_id, content, metadata, external_event_id)
        )
        return {}


class SpoolRecoveryTest(unittest.TestCase):
    def test_deterministic_hermes_event_identity_survives_reemission(self):
        first = deterministic_event_id(
            "hermes-main", "chat-42", "user", None, 12, "same turn"
        )
        self.assertEqual(
            first,
            deterministic_event_id(
                "hermes-main", "chat-42", "user", None, 12, "same turn"
            ),
        )
        self.assertNotEqual(
            first,
            deterministic_event_id(
                "hermes-main", "chat-42", "user", None, 14, "same turn"
            ),
        )

    def test_hermes_replays_spool_after_process_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary) / "hermes.jsonl"
            failing = MnemonicVaultMemoryProvider(
                client=AlwaysFailClient(), spool_path=spool
            )
            failing.initialize("restart-session", agent_context="primary")
            failing.sync_turn("durable user", "durable assistant")
            time.sleep(0.05)
            failing.shutdown()
            self.assertEqual(len(failing._spool.pending()), 2)

            client = RecordingClient()
            recovered = MnemonicVaultMemoryProvider(client=client, spool_path=spool)
            recovered.initialize("new-process", agent_context="primary")
            deadline = time.monotonic() + 2
            while recovered._spool.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            recovered.shutdown()
            self.assertEqual(recovered._spool.pending(), [])
            messages = [event for event in client.events if event[0] == "message"]
            self.assertEqual([event[2] for event in messages],
                             ["durable user", "durable assistant"])
            self.assertTrue(all(event[3] for event in messages))

    def test_permanent_poison_event_is_dead_lettered_and_next_event_delivers(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = MnemonicVaultMemoryProvider(
                client=StatusClient(413),
                spool_path=Path(temporary) / "hermes.jsonl",
            )
            provider.initialize("poison-session", agent_context="primary")
            provider.sync_turn("poison", "valid next")
            deadline = time.monotonic() + 2
            while provider._spool.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            provider.shutdown()
            self.assertEqual(provider._spool.pending(), [])
            dead = provider._spool.dead_letters()
            self.assertEqual(dead[0]["status"], 413)
            self.assertEqual(dead[0]["event"]["content"], "poison")
            self.assertTrue(any(event[0] == "message" for event in provider._client.events))

    def test_corrupt_complete_spool_record_is_quarantined_and_does_not_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = RecordingClient()
            provider = MnemonicVaultMemoryProvider(
                client=client,
                spool_path=Path(temporary) / "hermes.jsonl",
            )
            provider._spool.path.write_text("{corrupt json\n", encoding="utf-8")
            provider._spool.append(
                {
                    "kind": "message",
                    "session_id": "session-valid",
                    "external_session_id": "external-valid",
                    "agent": "hermes",
                    "role": "user",
                    "content": "valid after corruption",
                }
            )
            provider.initialize("corrupt-session", agent_context="primary")
            deadline = time.monotonic() + 2
            while provider._spool.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            provider.shutdown()
            self.assertEqual(provider._spool.pending(), [])
            self.assertEqual(
                provider._spool.dead_letters()[0]["event"]["metadata"]["raw_record"],
                "{corrupt json",
            )
            self.assertTrue(
                any(
                    event[0] == "message" and event[2] == "valid after corruption"
                    for event in client.events
                )
            )

    def test_auth_failure_stops_delivery_without_discarding_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = MnemonicVaultMemoryProvider(
                client=StatusClient(401),
                spool_path=Path(temporary) / "hermes.jsonl",
            )
            provider.initialize("auth-session", agent_context="primary")
            provider.sync_turn("poison", "")
            deadline = time.monotonic() + 2
            while provider._worker and provider._worker.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            provider.shutdown()
            self.assertEqual(len(provider._spool.pending()), 1)
            self.assertEqual(provider._spool.dead_letters(), [])

    def test_finalized_session_recovers_without_losing_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = FinalizedThenRecoveryClient()
            provider = MnemonicVaultMemoryProvider(
                client=client,
                spool_path=Path(temporary) / "hermes.jsonl",
            )
            provider.initialize("recovery-session", agent_context="primary")
            provider.sync_turn("recover me", "")
            deadline = time.monotonic() + 2
            while provider._spool.pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            provider.shutdown()
            messages = [event for event in client.events if event[0] == "message"]
            self.assertEqual(len(messages), 1)
            self.assertTrue(messages[0][1].startswith("session-recovery-"))
            self.assertIn("recovered_from_session", messages[0][3])

    def test_recovery_redirect_survives_restart_and_closes_recovery_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool_path = Path(temporary) / "hermes.jsonl"
            client = FinalizedThenRecoveryClient()
            provider = MnemonicVaultMemoryProvider(
                client=client, spool_path=spool_path
            )
            message = {
                "event_id": "event-" + "b" * 40,
                "kind": "message",
                "session_id": "session-finalized",
                "external_session_id": "external-a",
                "agent": "hermes",
                "role": "user",
                "content": "late durable turn",
            }
            provider._deliver(message)
            recovery_id = provider._spool.redirect_for("session-finalized")
            self.assertIsNotNone(recovery_id)
            provider._spool.compact()
            provider.shutdown()

            restarted = MnemonicVaultMemoryProvider(
                client=client, spool_path=spool_path
            )
            self.assertEqual(
                restarted._spool.redirect_for("session-finalized"), recovery_id
            )
            restarted._deliver(
                {
                    "event_id": "event-" + "c" * 40,
                    "kind": "end",
                    "session_id": "session-finalized",
                    "external_session_id": "external-a",
                    "agent": "hermes",
                }
            )
            restarted.shutdown()
            self.assertIn(("end", recovery_id), client.events)


if __name__ == "__main__":
    unittest.main()
