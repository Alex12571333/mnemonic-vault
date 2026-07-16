from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import AppConfig
from .models import Topic, utc_or_local_now
from .storage import (
    atomic_write_json,
    atomic_write_text,
    exclusive_lock,
    estimate_tokens,
    read_json,
    read_session,
    read_topic,
    validate_id,
)


WORD = re.compile(r"[\w.+#/-]+", re.UNICODE)
INDEX_VERSION = 2
STOP_WORDS = {
    "a",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "в",
    "для",
    "и",
    "на",
    "о",
    "по",
}


def projection_tokens(value: str) -> set[str]:
    return {
        token.strip("/.-")
        for token in WORD.findall(value.lower())
        if len(token.strip("/.-")) > 1 and token not in STOP_WORDS
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class GlobalTopicStore:
    """Rebuildable projections over immutable session topic summaries."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.root = config.storage.root / "global-topics"
        self.index_path = self.root / "index.json"
        self.lock_path = config.storage.root / ".global-topics.lock"
        self._mapping_mtime_ns = -1
        self._mapping_cache: dict[str, str] = {}

    def list(self) -> list[dict[str, Any]]:
        return list(self._read_index().get("topics", []))

    def list_cards(self, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        return [
            {key: value for key, value in entry.items() if key != "topic_ids"}
            for entry in self.list()[:limit]
        ]

    def get(
        self,
        global_topic_id: str,
        max_timeline_entries: int = 50,
        total_token_budget: int | None = None,
    ) -> dict[str, Any]:
        validate_id(global_topic_id, "global topic id")
        if not 1 <= max_timeline_entries <= 500:
            raise ValueError("max_timeline_entries must be between 1 and 500")
        token_budget = (
            self.config.global_topics.total_token_budget
            if total_token_budget is None
            else total_token_budget
        )
        if not 300 <= token_budget <= 32_000:
            raise ValueError("total_token_budget must be between 300 and 32000")
        entry = next(
            (
                item
                for item in self.list()
                if item.get("id") == global_topic_id
            ),
            None,
        )
        if entry is None:
            raise FileNotFoundError(global_topic_id)
        path = self.root / global_topic_id
        timeline = (path / "timeline.md").read_text(encoding="utf-8")
        sources = read_json(path / "sources.json")
        timeline_lines = timeline.splitlines()
        heading = [line for line in timeline_lines if not line.startswith("- **")]
        entries = [line for line in timeline_lines if line.startswith("- **")]
        maximum = min(
            max_timeline_entries,
            len(entries),
            len(sources.get("topic_ids", [])),
            len(sources.get("sources", [])),
        )
        current = (path / "current.md").read_text(encoding="utf-8")
        selected = maximum
        response = self._open_response(
            entry, sources, heading, entries, current, selected
        )
        # Reserve deterministic allowance for usage metadata and truncation flags.
        content_budget = max(1, token_budget - 64)
        while selected > 0 and self._response_cost(response) > content_budget:
            selected -= 1
            response = self._open_response(
                entry, sources, heading, entries, current, selected
            )
        if self._response_cost(response) > content_budget:
            current = self._truncate_current_to_budget(
                entry,
                sources,
                heading,
                entries,
                current,
                selected,
                content_budget,
            )
            response = self._open_response(
                entry, sources, heading, entries, current, selected
            )
            response["truncated_to_token_budget"] = True
        if selected < maximum:
            response["truncated_to_token_budget"] = True
        response = self._add_usage(response, token_budget)
        if self._response_cost(response) > token_budget:
            response = self._compact_budget_response(entry, token_budget)
        return response

    def _add_usage(
        self, response: dict[str, Any], token_budget: int
    ) -> dict[str, Any]:
        response["total_token_budget"] = token_budget
        for _ in range(3):
            response["used_tokens"] = self._response_cost(response)
        return response

    def _compact_budget_response(
        self, entry: dict[str, Any], token_budget: int
    ) -> dict[str, Any]:
        source_count = int(entry.get("source_count", 0))
        response: dict[str, Any] = {
            "id": entry["id"],
            "current_topic_id": entry["current_topic_id"],
            "source_count": source_count,
            "topic_ids": [],
            "older_topic_ids_omitted": source_count,
            "current": "",
            "timeline": "",
            "sources": {
                "global_topic_id": entry["id"],
                "current_topic_id": entry["current_topic_id"],
                "topic_ids": [],
                "sources": [],
                "older_sources_omitted": source_count,
            },
            "truncated_to_token_budget": True,
        }
        response = self._add_usage(response, token_budget)
        if self._response_cost(response) <= token_budget:
            return response
        minimal = {
            "id": entry["id"],
            "current_topic_id": entry["current_topic_id"],
            "source_count": source_count,
            "current": "",
            "timeline": "",
            "sources": {},
            "truncated_to_token_budget": True,
        }
        return self._add_usage(minimal, token_budget)

    def _open_response(
        self,
        entry: dict[str, Any],
        sources: dict[str, Any],
        heading: list[str],
        timeline_entries: list[str],
        current: str,
        selected: int,
    ) -> dict[str, Any]:
        total = len(timeline_entries)
        omitted = max(0, total - selected)
        selected_entries = timeline_entries[-selected:] if selected else []
        timeline = "\n".join(
            [
                *heading,
                *(([f"- _{omitted} older entries omitted_"]) if omitted else []),
                *selected_entries,
            ]
        ).rstrip() + "\n"
        bounded_sources = dict(sources)
        bounded_sources["topic_ids"] = (
            list(sources.get("topic_ids", []))[-selected:] if selected else []
        )
        bounded_sources["sources"] = (
            list(sources.get("sources", []))[-selected:] if selected else []
        )
        bounded_entry = dict(entry)
        bounded_entry["topic_ids"] = (
            list(entry.get("topic_ids", []))[-selected:] if selected else []
        )
        if omitted:
            bounded_sources["older_sources_omitted"] = omitted
            bounded_entry["older_topic_ids_omitted"] = omitted
        return {
            **bounded_entry,
            "current": current,
            "timeline": timeline,
            "sources": bounded_sources,
        }

    def _truncate_current_to_budget(
        self,
        entry: dict[str, Any],
        sources: dict[str, Any],
        heading: list[str],
        timeline_entries: list[str],
        current: str,
        selected: int,
        token_budget: int,
    ) -> str:
        marker = "\n[latest snapshot truncated to token budget]\n"
        low, high = 0, len(current)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = current[:middle] + marker
            response = self._open_response(
                entry,
                sources,
                heading,
                timeline_entries,
                candidate,
                selected,
            )
            if self._response_cost(response) <= token_budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _response_cost(response: dict[str, Any]) -> int:
        return estimate_tokens(json.dumps(response, ensure_ascii=False))

    def topic_mapping(self) -> dict[str, str]:
        if not self.index_path.exists():
            self._mapping_mtime_ns = -1
            self._mapping_cache = {}
            return {}
        mtime_ns = self.index_path.stat().st_mtime_ns
        if mtime_ns == self._mapping_mtime_ns:
            return self._mapping_cache
        result: dict[str, str] = {}
        for entry in self.list():
            global_id = str(entry.get("id", ""))
            for topic_id in entry.get("topic_ids", []):
                result[str(topic_id)] = global_id
        self._mapping_mtime_ns = mtime_ns
        self._mapping_cache = result
        return result

    def rebuild(
        self,
        *,
        minimum_versions: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        minimum = (
            self.config.global_topics.minimum_versions
            if minimum_versions is None
            else minimum_versions
        )
        if minimum < 2:
            raise ValueError("minimum_versions must be at least 2")
        sources = self._scan_sources()
        clusters = self._cluster(sources)
        eligible = [cluster for cluster in clusters if len(cluster) >= minimum]
        if dry_run:
            projections = self._build_projection_records(
                eligible, self._previous_groups()
            )
            return self._report(sources, clusters, projections, minimum, True)
        with exclusive_lock(self.lock_path):
            self._recover_interrupted_swap()
            projections = self._build_projection_records(
                eligible, self._previous_groups()
            )
            self._write_projection_tree(projections)
        return self._report(sources, clusters, projections, minimum, False)

    def _report(
        self,
        sources: list[dict[str, Any]],
        clusters: list[list[dict[str, Any]]],
        projections: list[dict[str, Any]],
        minimum: int,
        dry_run: bool,
    ) -> dict[str, Any]:
        return {
            "scanned_topics": len(sources),
            "candidate_clusters": len(clusters),
            "projected_topics": len(projections),
            "projected_source_topics": sum(
                len(item["topic_ids"]) for item in projections
            ),
            "minimum_versions": minimum,
            "dry_run": dry_run,
            "path": str(self.root),
            "topics": [
                {key: value for key, value in item.items() if key != "_sources"}
                for item in projections
            ],
        }

    def _scan_sources(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for session_file in sorted(
            self.config.storage.sessions_dir.glob("*/*/*/session.json")
        ):
            session = read_session(session_file.parent)
            if (
                session.status != "finalized"
                or session.processed_until_message != session.message_count
            ):
                continue
            for topic_file in sorted((session_file.parent / "topics").glob("*.md")):
                topic = read_topic(topic_file)
                result.append(
                    {
                        "topic": topic,
                        "path": self._portable_path(topic_file),
                        "session_started_at": session.started_at,
                        "session_ended_at": session.ended_at,
                        "title_tokens": projection_tokens(topic.title),
                        "card_tokens": projection_tokens(
                            " ".join(
                                [
                                    topic.title,
                                    topic.description,
                                    topic.problem,
                                    *topic.keywords,
                                ]
                            )
                        ),
                    }
                )
        result.sort(key=self._source_sort_key)
        return result

    def _cluster(self, sources: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not sources:
            return []
        clusters: list[list[dict[str, Any]]] = []
        cluster_sessions: list[set[str]] = []
        inverted: dict[str, set[int]] = defaultdict(set)
        for source in sources:
            candidates: set[int] = set()
            for token in source["title_tokens"]:
                candidates.update(inverted[token])
            compatible = [
                cluster_index
                for cluster_index in sorted(candidates)
                if source["topic"].session_id not in cluster_sessions[cluster_index]
                and all(
                    self._matches(existing, source)
                    for existing in clusters[cluster_index]
                )
            ]
            if compatible:
                cluster_index = max(
                    compatible,
                    key=lambda candidate: (
                        self._cluster_match_strength(clusters[candidate], source),
                        -candidate,
                    ),
                )
                clusters[cluster_index].append(source)
                cluster_sessions[cluster_index].add(source["topic"].session_id)
            else:
                cluster_index = len(clusters)
                clusters.append([source])
                cluster_sessions.append({source["topic"].session_id})
            for token in source["title_tokens"]:
                inverted[token].add(cluster_index)
        result = [
            sorted(group, key=self._source_sort_key)
            for group in clusters
            if len(group) > 1
        ]
        result.sort(key=lambda group: self._source_sort_key(group[0]))
        return result

    @staticmethod
    def _cluster_match_strength(
        cluster: list[dict[str, Any]], source: dict[str, Any]
    ) -> float:
        return min(
            jaccard(existing["card_tokens"], source["card_tokens"])
            for existing in cluster
        )

    def _matches(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_title = left["title_tokens"]
        right_title = right["title_tokens"]
        if not left_title or not right_title:
            return False
        title_score = jaccard(left_title, right_title)
        card_score = jaccard(left["card_tokens"], right["card_tokens"])
        if left_title == right_title:
            return card_score >= self.config.global_topics.title_similarity_threshold
        return (
            title_score >= self.config.global_topics.title_similarity_threshold
            and card_score >= self.config.global_topics.lexical_similarity_threshold
        )

    def _previous_groups(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        value = read_json(self.index_path)
        if not isinstance(value, dict) or not isinstance(value.get("topics"), list):
            raise ValueError(f"invalid global topic index: {self.index_path}")
        return list(value["topics"])

    def _build_projection_records(
        self,
        clusters: list[list[dict[str, Any]]],
        previous: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        used_previous: set[str] = set()
        records: list[dict[str, Any]] = []
        for cluster in clusters:
            topic_ids = [source["topic"].id for source in cluster]
            global_id = self._reuse_id(topic_ids, previous, used_previous)
            if not global_id:
                global_id = self._new_id(min(cluster, key=self._identity_anchor_key))
            used_previous.add(global_id)
            current = max(cluster, key=self._source_sort_key)
            records.append(
                {
                    "id": global_id,
                    "title": current["topic"].title,
                    "current_topic_id": current["topic"].id,
                    "source_count": len(cluster),
                    "first_seen_at": min(
                        source["session_started_at"] for source in cluster
                    ),
                    "last_seen_at": current["topic"].updated_at,
                    "current_updated_at": current["topic"].updated_at,
                    "topic_ids": topic_ids,
                    "_sources": cluster,
                }
            )
        records.sort(key=lambda item: item["id"])
        return records

    @staticmethod
    def _reuse_id(
        topic_ids: list[str],
        previous: list[dict[str, Any]],
        used_previous: set[str],
    ) -> str:
        current = set(topic_ids)
        matches: list[tuple[int, str]] = []
        for entry in previous:
            candidate_id = str(entry.get("id", ""))
            if not candidate_id or candidate_id in used_previous:
                continue
            overlap = len(current & {str(item) for item in entry.get("topic_ids", [])})
            if overlap:
                matches.append((overlap, candidate_id))
        return max(matches)[1] if matches else ""

    def _new_id(self, anchor: dict[str, Any]) -> str:
        topic: Topic = anchor["topic"]
        slug = "-".join(
            token
            for token in re.findall(r"[a-z0-9]+", topic.title.lower())
            if token not in STOP_WORDS
        )[:40] or "topic"
        digest = hashlib.sha256(topic.id.encode("utf-8")).hexdigest()[:10]
        return validate_id(f"global-{slug}-{digest}", "global topic id")

    def _write_projection_tree(self, records: list[dict[str, Any]]) -> None:
        stage = self.config.storage.root / f".global-topics.stage-{uuid.uuid4().hex}"
        previous = self.config.storage.root / ".global-topics.previous"
        stage.mkdir(parents=True, exist_ok=False)
        try:
            public_records: list[dict[str, Any]] = []
            for record in records:
                path = stage / record["id"]
                path.mkdir(parents=True, exist_ok=False)
                public = {key: value for key, value in record.items() if key != "_sources"}
                public_records.append(public)
                atomic_write_text(path / "current.md", self._render_current(record))
                atomic_write_text(path / "timeline.md", self._render_timeline(record))
                atomic_write_json(path / "sources.json", self._render_sources(record))
            atomic_write_json(
                stage / "index.json",
                {
                    "version": INDEX_VERSION,
                    "generated_at": utc_or_local_now(),
                    "topics": public_records,
                },
            )
            if previous.exists():
                shutil.rmtree(previous)
            self._mapping_mtime_ns = -1
            self._mapping_cache = {}
            if self.root.exists():
                os.replace(self.root, previous)
            try:
                os.replace(stage, self.root)
            except Exception:
                if previous.exists() and not self.root.exists():
                    os.replace(previous, self.root)
                raise
            if previous.exists():
                shutil.rmtree(previous)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _recover_interrupted_swap(self) -> None:
        previous = self.config.storage.root / ".global-topics.previous"
        if not self.root.exists() and previous.exists():
            os.replace(previous, self.root)
        elif self.root.exists() and previous.exists():
            shutil.rmtree(previous)
        for stage in self.config.storage.root.glob(".global-topics.stage-*"):
            if stage.is_dir():
                shutil.rmtree(stage)

    @staticmethod
    def _render_current(record: dict[str, Any]) -> str:
        source = max(record["_sources"], key=GlobalTopicStore._source_sort_key)
        topic: Topic = source["topic"]
        metadata = {
            "id": record["id"],
            "projection": "latest-session-snapshot",
            "title": topic.title,
            "source_topic_id": topic.id,
            "source_session_id": topic.session_id,
            "source_updated_at": topic.updated_at,
        }
        frontmatter = _yaml(metadata)
        return (
            f"---\n{frontmatter}\n---\n"
            "# Последний сессионный snapshot\n\n"
            "> Это последняя обновлённая сессионная версия, а не синтез всех "
            "исторических summaries.\n\n"
            f"{topic.summary.strip()}\n"
        )

    @staticmethod
    def _render_timeline(record: dict[str, Any]) -> str:
        lines = [f"# История: {record['title']}", ""]
        for source in record["_sources"]:
            topic: Topic = source["topic"]
            session_date = str(source["session_started_at"]) or "unknown"
            updated_date = topic.updated_at or "unknown"
            description = topic.description.strip() or topic.problem.strip()
            lines.append(
                f"- **updated {updated_date}** — {topic.title} (`{topic.id}`; "
                f"session started {session_date})"
                + (f": {description}" if description else "")
            )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _render_sources(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "global_topic_id": record["id"],
            "current_topic_id": record["current_topic_id"],
            "topic_ids": record["topic_ids"],
            "sources": [
                {
                    "topic_id": source["topic"].id,
                    "session_id": source["topic"].session_id,
                    "session_started_at": source["session_started_at"],
                    "topic_updated_at": source["topic"].updated_at,
                    "path": source["path"],
                }
                for source in record["_sources"]
            ],
        }

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"version": INDEX_VERSION, "topics": []}
        value = read_json(self.index_path)
        if not isinstance(value, dict) or not isinstance(value.get("topics"), list):
            raise ValueError(f"invalid global topic index: {self.index_path}")
        if value.get("version") != INDEX_VERSION:
            return {"version": INDEX_VERSION, "topics": []}
        return value

    def _portable_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.config.storage.root.resolve()))
        except ValueError:
            return str(path)

    @staticmethod
    def _source_sort_key(source: dict[str, Any]) -> tuple[str, str, str]:
        topic: Topic = source["topic"]
        return topic.updated_at, str(source["session_started_at"]), topic.id

    @staticmethod
    def _identity_anchor_key(source: dict[str, Any]) -> tuple[str, str, str]:
        topic: Topic = source["topic"]
        return str(source["session_started_at"]), topic.created_at, topic.id


def _yaml(value: dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write projection files") from exc
    return yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
