from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.catalog import Catalog
from app.config import AppConfig
from app.global_topics import GlobalTopicStore
from app.indexer import Indexer
from app.models import SourceRange, Topic
from app.recorder import SessionRecorder
from app.retriever import ContextBuilder, Retriever
from app.service import Services
from app.storage import (
    atomic_write_json,
    estimate_tokens,
    read_json,
    read_session,
    write_session,
    write_topic,
)
from app.summarizer import JobRunner, MemorySummarizer


class NoopLLM:
    def summarize(self, *args, **kwargs):
        return {"operations": []}


class GlobalTopicProjectionTest(unittest.TestCase):
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
                "global_topics": {
                    "minimum_versions": 2,
                    "title_similarity_threshold": 0.5,
                    "lexical_similarity_threshold": 0.72,
                },
                "embeddings": {"base_url": "", "model": ""},
            },
            self.root,
        )
        self.catalog = Catalog(self.config.storage.catalog_db)
        self.recorder = SessionRecorder(self.config, self.catalog)
        self.indexer = Indexer(self.config, self.catalog)

    def tearDown(self):
        self.temporary.cleanup()

    def _topic(
        self,
        year: int,
        summary: str,
        *,
        title: str = "DFlash on DGX Spark",
        description: str | None = None,
        problem: str = "Stable fast inference on DGX Spark",
        keywords: list[str] | None = None,
        updated_at: str | None = None,
        finalized: bool = True,
    ) -> tuple[Topic, Path]:
        session_id = f"session-dflash-{year}"
        session = self.recorder.start_session(
            "openclaw", session_id, f"{year}-01-10T10:00:00+09:00"
        )
        self.recorder.append(session.id, "user", f"DFlash state in {year}")
        topic = Topic(
            id=f"topic-dflash-{year}",
            session_id=session.id,
            title=title,
            description=(
                f"DFlash inference configuration in {year}"
                if description is None
                else description
            ),
            problem=problem,
            status="active",
            keywords=keywords or ["DFlash", "DGX Spark", "vLLM"],
            source_ranges=[SourceRange(session.id, 1, 1)],
            created_at=f"{year}-01-10T10:05:00+09:00",
            updated_at=updated_at or f"{year}-01-10T10:05:00+09:00",
            summary=summary,
        )
        path = self.recorder.locate(session.id) / "topics" / f"{topic.id}.md"
        write_topic(path, topic)
        self.indexer.index_topic(path, with_embedding=False)
        if finalized:
            session = read_session(self.recorder.locate(session.id))
            session.processed_until_message = session.message_count
            session.status = "finalized"
            session.ended_at = f"{year}-01-10T10:10:00+09:00"
            write_session(self.recorder.locate(session.id), session)
            self.catalog.upsert_session(session, self.recorder.locate(session.id))
        return topic, path

    def _services(self) -> Services:
        retriever = Retriever(self.config, self.catalog, self.recorder)
        summarizer = MemorySummarizer(
            self.config,
            self.catalog,
            self.recorder,
            self.indexer,
            NoopLLM(),
        )
        return Services(
            config=self.config,
            catalog=self.catalog,
            recorder=self.recorder,
            indexer=self.indexer,
            retriever=retriever,
            context_builder=ContextBuilder(self.config, retriever),
            summarizer=summarizer,
            job_runner=JobRunner(summarizer, self.recorder),
        )

    def test_rebuild_creates_current_timeline_sources_without_touching_originals(self):
        old, old_path = self._topic(2026, "## Итог\nGemma was used first.")
        current, current_path = self._topic(2027, "## Итог\nQwen is the current model.")
        original_bytes = {
            old.id: old_path.read_bytes(),
            current.id: current_path.read_bytes(),
        }
        store = GlobalTopicStore(self.config)

        dry_run = store.rebuild(dry_run=True)
        self.assertEqual(dry_run["projected_topics"], 1)
        json.dumps(dry_run)
        self.assertFalse(store.root.exists())

        report = store.rebuild()
        self.assertEqual(report["projected_source_topics"], 2)
        entry = store.list()[0]
        global_id = entry["id"]
        opened = store.get(global_id)
        self.assertEqual(opened["current_topic_id"], current.id)
        self.assertIn("Qwen is the current model", opened["current"])
        self.assertIn("2026-01-10", opened["timeline"])
        self.assertIn("2027-01-10", opened["timeline"])
        self.assertIn("session started", opened["timeline"])
        self.assertIn("latest-session-snapshot", opened["current"])
        self.assertIn("не синтез всех", opened["current"])
        self.assertEqual(opened["sources"]["topic_ids"], [old.id, current.id])
        self.assertEqual(old_path.read_bytes(), original_bytes[old.id])
        self.assertEqual(current_path.read_bytes(), original_bytes[current.id])

        stale_index = read_json(store.index_path)
        stale_index["version"] = 1
        atomic_write_json(store.index_path, stale_index)
        self.assertEqual(store.list(), [])
        migration_preview = store.rebuild(dry_run=True)
        self.assertEqual(migration_preview["topics"][0]["id"], global_id)
        store.rebuild()

        bounded = store.get(global_id, max_timeline_entries=1)
        self.assertEqual(bounded["topic_ids"], [current.id])
        self.assertEqual(bounded["older_topic_ids_omitted"], 1)
        self.assertNotIn(old.id, bounded["timeline"])

        token_bounded = store.get(global_id, total_token_budget=300)
        serialized = json.dumps(token_bounded, ensure_ascii=False)
        self.assertLessEqual(estimate_tokens(serialized), 300)
        self.assertLessEqual(token_bounded["used_tokens"], 300)
        self.assertTrue(token_bounded["truncated_to_token_budget"])

        previous = store.root.parent / ".global-topics.previous"
        store.root.replace(previous)
        stale_stage = store.root.parent / ".global-topics.stage-interrupted"
        stale_stage.mkdir()
        recovered = store.rebuild()
        self.assertEqual(recovered["topics"][0]["id"], global_id)
        self.assertFalse(previous.exists())
        self.assertFalse(stale_stage.exists())

        shutil.rmtree(store.root)
        rebuilt = store.rebuild()
        self.assertEqual(rebuilt["topics"][0]["id"], global_id)

    def test_projection_id_survives_newer_versions_and_is_exposed_by_search_api(self):
        old, _ = self._topic(2026, "## Итог\nGemma was used first.")
        self._topic(2027, "## Итог\nQwen is the current model.")
        store = GlobalTopicStore(self.config)
        first_id = store.rebuild()["topics"][0]["id"]
        latest, _ = self._topic(2028, "## Итог\nQwen with new DFlash flags.")
        second = store.rebuild()
        self.assertEqual(second["topics"][0]["id"], first_id)
        self.assertEqual(second["topics"][0]["current_topic_id"], latest.id)

        retriever = Retriever(self.config, self.catalog, self.recorder)
        hits = retriever.search("DFlash DGX Spark")
        self.assertEqual(hits[0].global_topic_id, first_id)
        self.assertIn(old.id, hits[0].related_older_topic_ids)

        client = TestClient(create_app(services=self._services(), start_worker=False))
        listed = client.get("/v1/memory/global-topics")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["topics"][0]["id"], first_id)
        self.assertNotIn("topic_ids", listed.json()["topics"][0])
        opened = client.get(f"/v1/memory/global-topics/{first_id}")
        self.assertEqual(opened.status_code, 200)
        self.assertEqual(opened.json()["sources"]["current_topic_id"], latest.id)

        bounded_api = client.get(
            f"/v1/memory/global-topics/{first_id}",
            params={"total_token_budget": 300},
        )
        self.assertEqual(bounded_api.status_code, 200)
        self.assertLessEqual(bounded_api.json()["used_tokens"], 300)

    def test_latest_snapshot_uses_topic_update_and_ignores_active_sessions(self):
        updated_late, _ = self._topic(
            2026,
            "## Итог\nJanuary session was updated in December 2028.",
            updated_at="2028-12-20T18:00:00+09:00",
        )
        self._topic(
            2027,
            "## Итог\nNewer session started later but was updated earlier.",
            updated_at="2027-07-01T12:00:00+09:00",
        )
        self._topic(
            2029,
            "## Итог\nIncomplete active session must not become current.",
            updated_at="2029-01-01T12:00:00+09:00",
            finalized=False,
        )

        store = GlobalTopicStore(self.config)
        report = store.rebuild()
        self.assertEqual(report["scanned_topics"], 2)
        opened = store.get(report["topics"][0]["id"])
        self.assertEqual(opened["current_topic_id"], updated_late.id)
        self.assertIn("2028-12-20T18:00:00+09:00", opened["timeline"])
        self.assertIn("2026-01-10T10:00:00+09:00", opened["timeline"])
        self.assertNotIn("topic-dflash-2029", opened["sources"]["topic_ids"])

    def test_complete_link_clustering_blocks_similarity_chain_drift(self):
        self._topic(
            2026,
            "A",
            title="Infrastructure",
            description="",
            problem="",
            keywords=["alpha", "beta"],
        )
        self._topic(
            2027,
            "B",
            title="Infrastructure",
            description="",
            problem="",
            keywords=["alpha", "beta", "gamma", "delta"],
        )
        self._topic(
            2028,
            "C",
            title="Infrastructure",
            description="",
            problem="",
            keywords=["gamma", "delta"],
        )
        store = GlobalTopicStore(self.config)
        sources = {source["topic"].id: source for source in store._scan_sources()}
        self.assertTrue(store._matches(sources["topic-dflash-2026"], sources["topic-dflash-2027"]))
        self.assertTrue(store._matches(sources["topic-dflash-2027"], sources["topic-dflash-2028"]))
        self.assertFalse(store._matches(sources["topic-dflash-2026"], sources["topic-dflash-2028"]))
        report = store.rebuild(minimum_versions=3, dry_run=True)
        self.assertEqual(report["projected_topics"], 0)

    def test_same_generic_title_does_not_merge_unrelated_cards(self):
        self._topic(
            2026,
            "## Итог\nPrune roses in spring.",
            title="General notes",
            description="Rose gardening and soil care",
            problem="Healthy garden plants",
            keywords=["roses", "soil", "garden"],
        )
        self._topic(
            2027,
            "## Итог\nTune CUDA kernel occupancy.",
            title="General notes",
            description="CUDA kernels and GPU profiling",
            problem="Fast GPU inference",
            keywords=["CUDA", "GPU", "kernel"],
        )
        report = GlobalTopicStore(self.config).rebuild(dry_run=True)
        self.assertEqual(report["projected_topics"], 0)


if __name__ == "__main__":
    unittest.main()
