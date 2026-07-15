from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .catalog import Catalog
from .config import AppConfig, MemoryLLMConfig
from .indexer import Indexer
from .models import Message, SourceRange, Topic, utc_or_local_now
from .recorder import SessionRecorder
from .storage import (
    atomic_write_json,
    estimate_tokens,
    merge_ranges,
    read_json,
    read_messages,
    read_session,
    read_topic,
    validate_id,
    write_session,
    write_topic,
)


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You maintain derived topic summaries for an append-only conversation archive.
Return one JSON object only. Never invent facts. Every operation must be supported by the
new message IDs. Update an existing topic when possible; create a topic only for a distinct,
durable subject. A summary is a complete revised Markdown body, not a patch.

Schema:
{
  "overview": "short session overview, primarily useful when finalizing",
  "operations": [
    {
      "action": "create_topic | update_topic",
      "topic_id": "required for update; optional for create",
      "title": "concise title",
      "description": "one-sentence description",
      "problem": "problem or goal addressed",
      "keywords": ["search terms", "models", "commands"],
      "source_ranges": [[1, 4], [8, 9]],
      "status": "active | resolved | archived",
      "summary": "complete Markdown with sections: Итог, Какую проблему решает, Принятые решения, Что пробовали, Нерешённые вопросы"
    }
  ]
}

For update_topic, retain still-valid details from CURRENT TOPICS and incorporate the new
evidence. source_ranges in the operation must refer only to NEW MESSAGES; the application
merges them with prior ranges. Do not create topics for greetings or transient chatter."""


class MemoryLLM(Protocol):
    def summarize(
        self,
        messages: list[Message],
        current_topics: list[Topic],
        finalizing: bool,
    ) -> dict[str, Any]: ...


class OpenAICompatibleMemoryLLM:
    def __init__(self, config: MemoryLLMConfig, max_output_tokens: int = 2_500):
        self.config = config
        self.max_output_tokens = max_output_tokens

    def summarize(
        self,
        messages: list[Message],
        current_topics: list[Topic],
        finalizing: bool,
    ) -> dict[str, Any]:
        if not self.config.base_url or not self.config.model:
            raise RuntimeError("memory_llm endpoint is not configured")
        current = [
            {
                **topic.card(),
                "summary": topic.summary[:6_000],
            }
            for topic in current_topics[:30]
        ]
        incoming = [message.to_dict() for message in messages]
        user_payload = {
            "finalizing": finalizing,
            "current_topics": current,
            "new_messages": incoming,
        }
        request_body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if not self.config.enable_thinking:
            request_body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"memory LLM returned {exc.code}: {detail}") from exc
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("memory LLM returned an invalid chat response") from exc
        return parse_json_object(str(content))


def parse_json_object(content: str) -> dict[str, Any]:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("memory LLM response contains no JSON object")
        value = json.loads(content[start : end + 1])
    if not isinstance(value, dict) or not isinstance(value.get("operations", []), list):
        raise ValueError("memory LLM response does not match the operations schema")
    return value


class MemorySummarizer:
    def __init__(
        self,
        config: AppConfig,
        catalog: Catalog,
        recorder: SessionRecorder,
        indexer: Indexer,
        llm: MemoryLLM,
    ):
        self.config = config
        self.catalog = catalog
        self.recorder = recorder
        self.indexer = indexer
        self.llm = llm

    def process_next_job(self) -> dict[str, Any] | None:
        now = utc_or_local_now()
        job = self.catalog.claim_job(now)
        if job is None:
            return None
        try:
            result = self.process_job(dict(job))
            self.catalog.complete_job(job["id"], utc_or_local_now())
            self._schedule_followup_if_needed(
                self.recorder.locate(str(job["session_id"]))
            )
            return result
        except Exception as exc:
            retry = int(job["attempts"]) < 3
            self.catalog.fail_job(job["id"], str(exc), utc_or_local_now(), retry=retry)
            logger.exception("summary job %s failed", job["id"])
            raise

    def process_job(self, job: dict[str, Any]) -> dict[str, Any]:
        session_path = self.recorder.locate(str(job["session_id"]))
        session = read_session(session_path)
        first = max(int(job["from_message"]), session.processed_until_message + 1)
        last = min(int(job["to_message"]), session.message_count)
        if first > last:
            return {"job_id": job["id"], "operations": 0, "already_processed": True}
        messages = read_messages(session_path / "transcript.jsonl", first, last)
        if not messages or messages[0].id != first or messages[-1].id != last:
            raise RuntimeError(f"transcript range {first}-{last} is incomplete")

        applied = 0
        latest_overview = ""
        for batch in self._partition(messages):
            topics = self._load_session_topics(session_path)
            is_final_batch = batch[-1].id == session.message_count
            response = self.llm.summarize(
                batch,
                topics,
                finalizing=session.status == "finalizing" and is_final_batch,
            )
            applied += self._apply_operations(session_path, batch, topics, response)
            if response.get("overview"):
                latest_overview = str(response["overview"]).strip()

            # A completed sub-batch is durable even if a later LLM call needs retrying.
            session = read_session(session_path)
            session.processed_until_message = batch[-1].id
            session.summary_revision += 1
            remaining = read_messages(
                session_path / "transcript.jsonl",
                session.processed_until_message + 1,
                session.message_count,
            )
            session.new_token_estimate = sum(estimate_tokens(item.text) for item in remaining)
            if (
                session.status == "finalizing"
                and session.processed_until_message == session.message_count
            ):
                session.status = "finalized"
            write_session(session_path, session)
            self.catalog.upsert_session(session, session_path)
            self._write_session_index(session_path, latest_overview)

        return {
            "job_id": job["id"],
            "session_id": session.id,
            "processed_until_message": session.processed_until_message,
            "operations": applied,
            "status": session.status,
        }

    def _partition(self, messages: list[Message]) -> list[list[Message]]:
        # Reserve room for the system prompt and current topic summaries.
        budget = max(1_000, int(self.config.summarization.max_input_tokens * 0.65))
        result: list[list[Message]] = []
        current: list[Message] = []
        used = 0
        for message in messages:
            cost = estimate_tokens(message.text) + 30
            if current and used + cost > budget:
                result.append(current)
                current = []
                used = 0
            if cost > budget:
                # The full turn remains in transcript; only its LLM view is bounded.
                max_chars = max(1_000, budget * 4 - 400)
                message = Message(
                    id=message.id,
                    role=message.role,
                    text=message.text[:max_chars]
                    + "\n[Memory summarizer input truncated; open source turn for full text.]",
                    created_at=message.created_at,
                    metadata=message.metadata,
                )
                cost = estimate_tokens(message.text) + 30
            current.append(message)
            used += cost
        if current:
            result.append(current)
        return result

    def _apply_operations(
        self,
        session_path: Path,
        batch: list[Message],
        current_topics: list[Topic],
        response: dict[str, Any],
    ) -> int:
        by_id = {topic.id: topic for topic in current_topics}
        applied = 0
        for raw in response.get("operations", []):
            if not isinstance(raw, dict):
                raise ValueError("each summary operation must be an object")
            action = str(raw.get("action", ""))
            if action not in {"create_topic", "update_topic"}:
                raise ValueError(f"unknown summary action: {action!r}")
            ranges = self._validated_ranges(raw.get("source_ranges", []), batch)
            for item in ranges:
                item.session_id = session_path.name
            if not ranges:
                raise ValueError("summary operation has no valid source_ranges")
            now = utc_or_local_now()
            summary = raw.get("summary", raw.get("summary_update", ""))
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError("summary operation has no summary text")
            summary = normalize_summary(summary)

            if action == "update_topic":
                topic_id = validate_id(str(raw.get("topic_id", "")), "topic id")
                if topic_id not in by_id:
                    raise ValueError(f"cannot update unknown topic: {topic_id}")
                existing = by_id[topic_id]
                topic = Topic(
                    id=existing.id,
                    session_id=existing.session_id,
                    title=str(raw.get("title") or existing.title),
                    description=str(raw.get("description") or existing.description),
                    problem=str(raw.get("problem") or existing.problem),
                    status=str(raw.get("status") or existing.status),
                    keywords=clean_keywords(raw.get("keywords") or existing.keywords),
                    source_ranges=merge_ranges(existing.source_ranges + ranges),
                    created_at=existing.created_at,
                    updated_at=now,
                    summary=summary,
                )
            else:
                title = str(raw.get("title", "")).strip()
                if not title:
                    raise ValueError("create_topic requires a title")
                proposed = str(raw.get("topic_id", "")).strip()
                topic_id = (
                    validate_id(proposed, "topic id")
                    if proposed
                    else stable_topic_id(session_path.name, title)
                )
                indexed = self.catalog.get_topic_row(topic_id)
                if indexed is not None and indexed["session_id"] != session_path.name:
                    topic_id = stable_topic_id(session_path.name, title)
                if topic_id in by_id:
                    raise ValueError(f"create_topic conflicts with existing topic: {topic_id}")
                topic = Topic(
                    id=topic_id,
                    session_id=session_path.name,
                    title=title,
                    description=str(raw.get("description", "")).strip(),
                    problem=str(raw.get("problem", "")).strip(),
                    status=str(raw.get("status", "active")),
                    keywords=clean_keywords(raw.get("keywords", [])),
                    source_ranges=merge_ranges(ranges),
                    created_at=now,
                    updated_at=now,
                    summary=summary,
                )
            topic_path = session_path / "topics" / f"{topic.id}.md"
            topic.path = str(topic_path)
            write_topic(topic_path, topic)
            self.indexer.index_topic(topic_path, with_embedding=True)
            by_id[topic.id] = topic
            applied += 1
        return applied

    def _validated_ranges(
        self, value: Any, batch: list[Message]
    ) -> list[SourceRange]:
        first, last = batch[0].id, batch[-1].id
        if not isinstance(value, list):
            raise ValueError("source_ranges must be a list")
        result: list[SourceRange] = []
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"invalid source range: {pair!r}")
            start, end = int(pair[0]), int(pair[1])
            if start < first or end > last or start > end:
                raise ValueError(
                    f"source range {start}-{end} is outside new messages {first}-{last}"
                )
            result.append(SourceRange("", start, end))
        return result

    def _load_session_topics(self, session_path: Path) -> list[Topic]:
        result: list[Topic] = []
        for path in sorted((session_path / "topics").glob("*.md")):
            result.append(read_topic(path))
        return result

    def _write_session_index(self, session_path: Path, overview: str = "") -> None:
        index_path = session_path / "index.json"
        old: dict[str, Any] = {}
        if index_path.exists():
            old = read_json(index_path)
        topics = self._load_session_topics(session_path)
        if not overview:
            overview = str(old.get("overview", ""))
        if not overview and topics:
            overview = "Темы сессии: " + "; ".join(topic.title for topic in topics[:8])
        atomic_write_json(
            index_path,
            {
                "session_id": session_path.name,
                "overview": overview,
                "topics": [
                    {
                        "id": topic.id,
                        "title": topic.title,
                        "description": topic.description,
                        "problem": topic.problem,
                        "path": f"topics/{topic.id}.md",
                        "source_ranges": [item.to_pair() for item in topic.source_ranges],
                    }
                    for topic in topics
                ],
            },
        )

    def _schedule_followup_if_needed(self, session_path: Path) -> None:
        session = read_session(session_path)
        if session.message_count <= session.processed_until_message:
            return
        remaining = session.message_count - session.processed_until_message
        threshold = self.config.summarization
        should_schedule = (
            session.status == "finalizing"
            or remaining >= threshold.messages_per_batch
            or session.new_token_estimate >= threshold.token_threshold
        )
        if should_schedule and not self.catalog.has_open_job(session.id):
            self.catalog.enqueue_job(
                session.id,
                session.processed_until_message + 1,
                session.message_count,
                utc_or_local_now(),
            )


class JobRunner:
    def __init__(
        self,
        summarizer: MemorySummarizer,
        recorder: SessionRecorder,
        poll_seconds: float = 2.0,
    ):
        self.summarizer = summarizer
        self.recorder = recorder
        self.poll_seconds = poll_seconds

    def recover(self) -> int:
        return self.summarizer.catalog.reset_running_jobs(utc_or_local_now())

    def run_once(self) -> dict[str, Any] | None:
        self.recorder.schedule_idle_sessions()
        return self.summarizer.process_next_job()

    def run_forever(self, stop_event: Any) -> None:
        self.recover()
        while not stop_event.is_set():
            try:
                result = self.run_once()
                if result is None:
                    stop_event.wait(self.poll_seconds)
            except Exception:
                stop_event.wait(min(30.0, self.poll_seconds * 5))


def stable_topic_id(session_id: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    if not slug:
        slug = "memory"
    digest = hashlib.sha256(f"{session_id}\0{title.lower()}".encode("utf-8")).hexdigest()[:8]
    return f"topic-{slug}-{digest}"


def clean_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("keywords must be a list")
    result: list[str] = []
    for item in value:
        keyword = str(item).strip()
        if keyword and keyword not in result:
            result.append(keyword)
    return result[:40]


def normalize_summary(value: str) -> str:
    value = value.strip()
    if value.startswith("## "):
        return value
    return f"## Итог\n{value}"
