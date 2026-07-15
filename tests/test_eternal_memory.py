from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from app.catalog import Catalog
from app.api import create_app
from app.config import (
    AppConfig,
    EmbeddingsConfig,
    MemoryLLMConfig,
    SummarizationConfig,
)
from app.embeddings import OpenAICompatibleEmbedder, pack_vector
from app.indexer import Indexer
from app.models import Message, Topic
from app.recorder import SessionRecorder
from app.retriever import ContextBuilder, Retriever
from app.service import Services
from app.storage import append_message, estimate_tokens, read_session, read_topic
from app.summarizer import (
    JobRunner,
    MemorySummarizer,
    OpenAICompatibleMemoryLLM,
    parse_json_object,
)
from fastapi.testclient import TestClient


class FakeEmbedder:
    model = "fake-embedding-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            lowered = text.lower()
            result.append(
                [
                    float(lowered.count("dflash")),
                    float(lowered.count("cuda")),
                    float(lowered.count("памят")),
                    1.0,
                ]
            )
        return result


class FakeMemoryLLM:
    def summarize(self, messages, current_topics, finalizing, topic_cards=None):
        source_range = [[messages[0].id, messages[-1].id]]
        if current_topics:
            existing = current_topics[0]
            return {
                "overview": "Обсуждались запуск и точные параметры DFlash.",
                "operations": [
                    {
                        "action": "update_topic",
                        "topic_id": existing.id,
                        "title": "DFlash на DGX Spark",
                        "description": "Запуск DFlash и CUDA graphs с рабочими параметрами.",
                        "problem": "Стабильный inference на Spark.",
                        "keywords": ["DFlash", "DGX Spark", "CUDA graphs", "vLLM"],
                        "source_ranges": source_range,
                        "summary": "## Итог\nИспользовался флаг `--disable-cuda-graphs` после ошибки OOM.\n\n## Нерешённые вопросы\nНет.",
                    }
                ],
            }
        return {
            "overview": "Обсуждался запуск DFlash на DGX Spark.",
            "operations": [
                {
                    "action": "create_topic",
                    "title": "DFlash на DGX Spark",
                    "description": "Запуск vLLM, DFlash и CUDA graphs.",
                    "problem": "Стабильный быстрый inference.",
                    "keywords": ["DFlash", "DGX Spark", "CUDA", "vLLM"],
                    "source_ranges": source_range,
                    "summary": "## Итог\nДля запуска исследовались DFlash и CUDA graphs.\n\n## Нерешённые вопросы\nНужен точный флаг.",
                }
            ],
        }


class EternalMemoryTest(unittest.TestCase):
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
                    "messages_per_batch": 2,
                    "token_threshold": 10000,
                    "idle_minutes": 30,
                    "max_input_tokens": 16000,
                    "max_output_tokens": 2500,
                },
                "retrieval": {
                    "lexical_top_k": 30,
                    "vector_top_k": 30,
                    "final_top_k": 5,
                    "auto_open_summaries": 2,
                    "summary_budget_tokens": 1800,
                    "source_budget_tokens": 1800,
                    "minimum_score": 0.35,
                    "rrf_k": 60,
                },
                "memory_llm": {},
                "embeddings": {"base_url": "", "model": ""},
            },
            self.root,
        )
        self.catalog = Catalog(self.config.storage.catalog_db)
        self.recorder = SessionRecorder(self.config, self.catalog)
        self.embedder = FakeEmbedder()
        self.indexer = Indexer(self.config, self.catalog, self.embedder)
        self.summarizer = MemorySummarizer(
            self.config,
            self.catalog,
            self.recorder,
            self.indexer,
            FakeMemoryLLM(),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_append_summary_update_retrieve_and_rebuild(self):
        session = self.recorder.start_session(
            "openclaw", "session-test", "2026-07-15T17:20:00+09:00"
        )
        first = self.recorder.append(
            session.id,
            "user",
            "Как запустить DFlash на DGX Spark?",
            "2026-07-15T17:21:00+09:00",
        )
        second = self.recorder.append(
            session.id,
            "assistant",
            "Проверим vLLM и CUDA graphs.",
            "2026-07-15T17:22:00+09:00",
        )
        self.assertEqual((first.id, second.id), (1, 2))
        session_path = self.recorder.locate(session.id)
        lines = (session_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["text"], "Как запустить DFlash на DGX Spark?")

        first_result = self.summarizer.process_next_job()
        self.assertEqual(first_result["processed_until_message"], 2)
        topic_path = next((session_path / "topics").glob("*.md"))
        topic = read_topic(topic_path)
        self.assertEqual(topic.source_ranges[0].to_pair(), [1, 2])

        self.recorder.append(
            session.id,
            "user",
            "Какая точная команда помогла после OOM?",
            "2026-07-15T17:23:00+09:00",
        )
        self.recorder.append(
            session.id,
            "assistant",
            "Добавили --disable-cuda-graphs и перезапустили vLLM.",
            "2026-07-15T17:24:00+09:00",
        )
        second_result = self.summarizer.process_next_job()
        self.assertEqual(second_result["processed_until_message"], 4)
        topic = read_topic(topic_path)
        self.assertEqual(topic.source_ranges[0].to_pair(), [1, 4])
        self.assertIn("--disable-cuda-graphs", topic.summary)

        retriever = Retriever(self.config, self.catalog, self.recorder, self.embedder)
        builder = ContextBuilder(self.config, retriever)
        context = builder.build(
            "какая команда для DFlash", include_sources="auto"
        )
        self.assertEqual(context["topics"][0]["id"], topic.id)
        self.assertTrue(context["topics"][0]["source_fragments"])

        row = self.catalog.get_topic_row(topic.id)
        self.assertFalse(Path(row["path"]).is_absolute())
        vec_hits = self.catalog.vector_search(
            self.embedder.model,
            pack_vector(self.embedder.embed(["DFlash CUDA"])[0]),
            4,
            5,
        )
        self.assertEqual(vec_hits[0]["topic_id"], topic.id)
        self.assertGreaterEqual(vec_hits[0]["cosine_similarity"], -1.0)
        rebuilt = self.indexer.rebuild(with_embeddings=False)
        self.assertEqual(
            rebuilt,
            {
                "sessions": 1,
                "topics": 1,
                "messages": 4,
                "recovered_jobs": 0,
                "failures": 0,
            },
        )
        lexical_retriever = Retriever(self.config, self.catalog, self.recorder)
        self.assertEqual(lexical_retriever.search("DFlash")[0].topic.id, topic.id)

    def test_session_end_processes_tail_and_finalizes(self):
        session = self.recorder.start_session(
            "hermes", "session-tail", "2026-07-15T18:00:00+09:00"
        )
        self.recorder.append(session.id, "user", "Вопрос про DFlash")
        ended = self.recorder.end_session(session.id)
        self.assertEqual(ended.status, "finalizing")
        result = self.summarizer.process_next_job()
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(read_session(self.recorder.locate(session.id)).status, "finalized")

    def test_json_response_parser(self):
        value = parse_json_object('```json\n{"operations": []}\n```')
        self.assertEqual(value, {"operations": []})

    def test_recorder_recovers_fsynced_tail_after_metadata_crash(self):
        session = self.recorder.start_session(
            "openclaw", "session-recover", "2026-07-15T19:00:00+09:00"
        )
        self.recorder.append(session.id, "user", "message one")
        path = self.recorder.locate(session.id)
        append_message(
            path / "transcript.jsonl",
            Message(
                id=2,
                role="assistant",
                text="durable but metadata was not updated",
                created_at="2026-07-15T19:01:00+09:00",
            ),
        )
        third = self.recorder.append(session.id, "user", "message three")
        self.assertEqual(third.id, 3)
        self.assertEqual(read_session(path).message_count, 3)

    def test_fastapi_recording_and_reading_endpoints(self):
        retriever = Retriever(self.config, self.catalog, self.recorder, self.embedder)
        builder = ContextBuilder(self.config, retriever)
        runner = JobRunner(self.summarizer, self.recorder)
        services = Services(
            config=self.config,
            catalog=self.catalog,
            recorder=self.recorder,
            indexer=self.indexer,
            retriever=retriever,
            context_builder=builder,
            summarizer=self.summarizer,
            job_runner=runner,
        )
        app = create_app(services=services, start_worker=False)
        with TestClient(app) as client:
            started = client.post(
                "/v1/sessions/start",
                json={"agent": "openclaw", "session_id": "session-api"},
            )
            self.assertEqual(started.status_code, 201)
            written = client.post(
                "/v1/sessions/session-api/messages",
                json={"role": "user", "content": "API message"},
            )
            self.assertEqual(written.status_code, 201)
            turns = client.get("/v1/sessions/session-api/turns?from=1&to=1")
            self.assertEqual(turns.json()["turns"][0]["text"], "API message")


class OpenAICompatibleClientsTest(unittest.TestCase):
    def test_local_chat_and_embedding_protocols(self):
        requests = []

        class Response(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def urlopen(request, timeout):
            payload = json.loads(request.data)
            requests.append((request.full_url, payload))
            if request.full_url.endswith("/embeddings"):
                response = {
                    "data": [
                        {"index": index, "embedding": [float(index), 1.0]}
                        for index, _ in enumerate(payload["input"])
                    ]
                }
            else:
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": "```json\n{\"operations\": []}\n```"
                            }
                        }
                    ]
                }
            return Response(json.dumps(response).encode("utf-8"))

        base_url = "http://models.test/v1"
        with patch("urllib.request.urlopen", side_effect=urlopen):
            embedder = OpenAICompatibleEmbedder(
                EmbeddingsConfig(base_url=base_url, model="embed-test")
            )
            self.assertEqual(embedder.embed(["a", "b"]), [[0.0, 1.0], [1.0, 1.0]])
            llm = OpenAICompatibleMemoryLLM(
                MemoryLLMConfig(base_url=base_url, model="memory-test"),
                input_budgets=SummarizationConfig(
                    topic_cards_budget_tokens=120,
                    existing_summaries_budget_tokens=140,
                    existing_summaries_top_k=1,
                ),
            )
            topic = Topic(
                id="topic-budget",
                session_id="session-budget",
                title="Bounded topic",
                description="D" * 120,
                problem="P" * 80,
                status="active",
                keywords=["budget"],
                source_ranges=[],
                created_at="2026-07-15T00:00:00+09:00",
                updated_at="2026-07-15T00:00:00+09:00",
                summary="S" * 4_000,
            )
            self.assertEqual(
                llm.summarize(
                    [Message(1, "user", "test", "2026-07-15T00:00:00+09:00")],
                    [topic],
                    False,
                    topic_cards=[topic],
                ),
                {"operations": []},
            )
        self.assertEqual(
            [url for url, _ in requests],
            [
                "http://models.test/v1/embeddings",
                "http://models.test/v1/chat/completions",
            ],
        )
        llm_payload = json.loads(requests[-1][1]["messages"][1]["content"])
        self.assertLessEqual(
            estimate_tokens(json.dumps(llm_payload["topic_cards"], ensure_ascii=False)),
            120,
        )
        self.assertLessEqual(
            estimate_tokens(json.dumps(llm_payload["current_topics"], ensure_ascii=False)),
            140,
        )
        self.assertLessEqual(len(llm_payload["current_topics"]), 1)


if __name__ == "__main__":
    unittest.main()
