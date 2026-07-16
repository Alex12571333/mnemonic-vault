from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.catalog import Catalog
from app.config import AppConfig
from app.explicit_memory import ExplicitMemoryStore
from app.indexer import Indexer
from app.recorder import SessionRecorder
from app.retriever import ContextBuilder, Retriever
from app.service import Services
from app.storage import read_messages
from app.summarizer import JobRunner, MemorySummarizer


class NoopLLM:
    def summarize(self, messages, current_topics, finalizing, topic_cards=None):
        return {"operations": []}


class MustNotEmbed:
    model = "offline"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        raise RuntimeError("embedding service is offline")


class SemanticEmbedder:
    model = "semantic-explicit-v1"

    def embed(self, texts):
        return [
            [1.0, 0.0]
            if "production" in text.lower() or "боевой" in text.lower()
            else [0.0, 1.0]
            for text in texts
        ]


class ExplicitMemoryTest(unittest.TestCase):
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
                    "messages_per_batch": 20,
                    "token_threshold": 12000,
                    "idle_minutes": 30,
                },
                "retrieval": {
                    "lexical_top_k": 30,
                    "vector_top_k": 30,
                    "final_top_k": 5,
                    "auto_open_summaries": 2,
                    "total_context_budget_tokens": 1000,
                    "card_budget_tokens": 300,
                    "summary_budget_tokens": 500,
                    "source_budget_tokens": 300,
                    "minimum_score": 0.35,
                    "vector_min_similarity": 0.35,
                    "lexical_min_query_coverage": 0.34,
                    "rrf_k": 60,
                    "explicit_memory_top_k": 5,
                    "explicit_memory_budget_tokens": 500,
                },
                "embeddings": {"base_url": "", "model": ""},
            },
            self.root,
        )
        self.catalog = Catalog(self.config.storage.catalog_db)
        self.recorder = SessionRecorder(self.config, self.catalog)
        self.offline_embedder = MustNotEmbed()
        self.store = ExplicitMemoryStore(
            self.config, self.catalog, self.recorder, self.offline_embedder
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_remember_is_immediately_durable_searchable_and_priority_queued(self):
        receipt = self.store.remember(
            "Запомни: production Mnemonic Vault работает на 192.168.0.14",
            normalized="Production Mnemonic Vault работает на 192.168.0.14",
            kind="configuration",
            scope_type="project",
            scope_id="mnemonic-vault",
            idempotency_key="event-production-14",
            created_at="2026-07-16T20:30:00+09:00",
        )
        self.assertTrue(receipt["stored"])
        self.assertTrue(receipt["available_for_recall"])
        self.assertEqual(receipt["summary_job"], "pending")
        self.assertEqual(self.offline_embedder.calls, 0)

        events = (self.config.storage.root / "explicit-memory.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(events), 1)
        event = json.loads(events[0])
        self.assertEqual(event["verbatim"], "Запомни: production Mnemonic Vault работает на 192.168.0.14")
        self.assertEqual(event["scope"], {"type": "project", "id": "mnemonic-vault"})

        source = receipt["source"]
        messages = read_messages(
            self.recorder.locate(source["session_id"]) / "transcript.jsonl"
        )
        self.assertEqual(messages[source["message_id"] - 1].text, event["verbatim"])

        retriever = Retriever(
            self.config, self.catalog, self.recorder, explicit_memory=self.store
        )
        result = ContextBuilder(self.config, retriever).build(
            "где production Mnemonic Vault",
            scope_type="project",
            scope_id="mnemonic-vault",
        )
        self.assertEqual(result["topics"], [])
        self.assertEqual(result["explicit_memories"][0]["memory_id"], receipt["memory_id"])
        self.assertEqual(result["explicit_memories"][0]["text"], event["normalized"])
        self.assertEqual(result["results"][0]["type"], "explicit_memory")
        self.assertEqual(self.offline_embedder.calls, 0)

        claimed = self.catalog.claim_job("2026-07-16T20:31:00+09:00")
        self.assertEqual(claimed["type"], "explicit-memory-summary")

    def test_idempotency_source_validation_and_crash_recovery_from_file(self):
        session = self.recorder.start_session("openclaw", "session-source")
        source = self.recorder.append(
            session.id, "user", "Запомни точную команду --safe-mode"
        )
        first = self.store.remember(
            source.text,
            normalized="Точная команда запуска: --safe-mode",
            kind="configuration",
            scope_type="session",
            scope_id=session.id,
            source_session_id=session.id,
            source_message_id=source.id,
            idempotency_key="event-safe-mode",
        )
        repeated = self.store.remember(
            source.text,
            normalized="Точная команда запуска: --safe-mode",
            kind="configuration",
            scope_type="session",
            scope_id=session.id,
            source_session_id=session.id,
            source_message_id=source.id,
            idempotency_key="event-safe-mode",
        )
        self.assertEqual(first["memory_id"], repeated["memory_id"])
        self.assertEqual(
            len((self.config.storage.root / "explicit-memory.jsonl").read_text().splitlines()),
            1,
        )
        with self.assertRaisesRegex(ValueError, "different memory"):
            self.store.remember(
                "different",
                kind="configuration",
                scope_type="session",
                scope_id=session.id,
                idempotency_key="event-safe-mode",
            )
        with self.assertRaisesRegex(ValueError, "exact verbatim"):
            self.store.remember(
                "not the source text",
                source_session_id=session.id,
                source_message_id=source.id,
                idempotency_key="event-wrong-source",
            )

        with self.catalog.connection() as connection:
            connection.execute("DELETE FROM explicit_memories_fts")
            connection.execute("DELETE FROM explicit_memories")
        recovered = self.store.remember(
            source.text,
            normalized="Точная команда запуска: --safe-mode",
            kind="configuration",
            scope_type="session",
            scope_id=session.id,
            source_session_id=session.id,
            source_message_id=source.id,
            idempotency_key="event-safe-mode",
        )
        self.assertEqual(recovered["memory_id"], first["memory_id"])
        self.assertIsNotNone(self.catalog.get_explicit_memory(first["memory_id"]))

    def test_each_remember_extends_or_follows_a_running_priority_job(self):
        first = self.store.remember(
            "Запомни первый факт",
            idempotency_key="event-first-explicit",
            created_at="2026-07-16T21:00:00+09:00",
        )
        second = self.store.remember(
            "Запомни второй факт",
            idempotency_key="event-second-explicit",
            created_at="2026-07-16T21:01:00+09:00",
        )
        self.assertEqual(first["source"]["session_id"], second["source"]["session_id"])
        running = self.catalog.claim_job("2026-07-16T21:02:00+09:00")
        self.assertEqual((running["from_message"], running["to_message"]), (1, 2))

        third = self.store.remember(
            "Запомни третий факт",
            idempotency_key="event-third-explicit",
            created_at="2026-07-16T21:03:00+09:00",
        )
        self.assertEqual(third["summary_job"], "pending")
        with self.catalog.connection() as connection:
            followup = connection.execute(
                "SELECT * FROM jobs WHERE status='pending' ORDER BY id"
            ).fetchone()
        self.assertIsNotNone(followup)
        self.assertEqual(followup["type"], "explicit-memory-summary")
        self.assertEqual((followup["from_message"], followup["to_message"]), (3, 3))

    def test_supersede_preserves_history_and_rebuilds_materialized_state(self):
        old = self.store.remember(
            "Production находится на .14",
            kind="configuration",
            scope_type="project",
            scope_id="vault",
            idempotency_key="event-server-14",
            created_at="2026-07-01T10:00:00+09:00",
        )
        new = self.store.remember(
            "Production перенесён на .15",
            kind="correction",
            scope_type="project",
            scope_id="vault",
            idempotency_key="event-server-15",
            supersedes=old["memory_id"],
            created_at="2026-09-01T10:00:00+09:00",
        )
        active = self.store.search("Production", scope_type="project", scope_id="vault")
        self.assertEqual([item["memory_id"] for item in active], [new["memory_id"]])
        historical = self.store.search(
            "где Production раньше",
            include_superseded=True,
            scope_type="project",
            scope_id="vault",
        )
        self.assertEqual(
            {item["memory_id"] for item in historical},
            {old["memory_id"], new["memory_id"]},
        )
        old_memory = self.store.get(old["memory_id"])
        self.assertEqual(old_memory.status, "superseded")
        self.assertEqual(old_memory.valid_to, "2026-09-01T10:00:00+09:00")

        rebuilt = Indexer(self.config, self.catalog).rebuild()
        self.assertEqual(rebuilt["explicit_memories"], 2)
        self.assertEqual(self.store.get(old["memory_id"]).status, "superseded")

    def test_reembed_all_adds_optional_semantic_recall_after_the_commit(self):
        receipt = self.store.remember(
            "Production Mnemonic Vault работает на 192.168.0.14",
            kind="configuration",
            idempotency_key="event-semantic-production",
        )
        self.assertEqual(self.catalog.list_explicit_embeddings(), [])
        embedder = SemanticEmbedder()
        report = Indexer(self.config, self.catalog, embedder).reembed_all()
        self.assertEqual(report["explicit_memory_embeddings"], 1)
        semantic_store = ExplicitMemoryStore(
            self.config, self.catalog, self.recorder, embedder
        )
        hits = semantic_store.search("боевой узел")
        self.assertEqual(hits[0]["memory_id"], receipt["memory_id"])

    def test_api_remember_is_direct_and_user_only(self):
        indexer = Indexer(self.config, self.catalog)
        retriever = Retriever(
            self.config, self.catalog, self.recorder, explicit_memory=self.store
        )
        summarizer = MemorySummarizer(
            self.config,
            self.catalog,
            self.recorder,
            indexer,
            NoopLLM(),
        )
        services = Services(
            config=self.config,
            catalog=self.catalog,
            recorder=self.recorder,
            indexer=indexer,
            retriever=retriever,
            context_builder=ContextBuilder(self.config, retriever),
            summarizer=summarizer,
            job_runner=JobRunner(summarizer, self.recorder),
            explicit_memory=self.store,
        )
        with TestClient(create_app(services=services, start_worker=False)) as client:
            response = client.post(
                "/v1/memory/remember",
                json={
                    "verbatim": "Я предпочитаю ответы на русском",
                    "kind": "preference",
                    "scope": {"type": "global"},
                    "idempotency_key": "event-language-russian",
                },
            )
            self.assertEqual(response.status_code, 201)
            memory_id = response.json()["memory_id"]
            opened = client.get(f"/v1/memory/explicit/{memory_id}")
            self.assertEqual(opened.json()["author"], "user")
            searched = client.post(
                "/v1/memory/search",
                json={"query": "ответы на русском", "scope": {"type": "global"}},
            )
            self.assertEqual(searched.status_code, 200)
            self.assertEqual(searched.json()["explicit_memories"][0]["memory_id"], memory_id)


if __name__ == "__main__":
    unittest.main()
