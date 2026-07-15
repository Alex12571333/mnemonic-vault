from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .catalog import Catalog
from .config import AppConfig
from .embeddings import Embedder, cosine_similarity, pack_vector, unpack_vector
from .models import Message, SearchHit, Topic
from .recorder import SessionRecorder
from .storage import estimate_tokens, read_messages, read_topic, validate_id


WORD = re.compile(r"[\w.+#/-]+", re.UNICODE)
PRECISION_PATTERNS = (
    "какая команда",
    "какую команду",
    "точное значение",
    "сколько токенов",
    "какой параметр",
    "какая ошибка",
    "какая версия",
    "command",
    "exact value",
    "parameter",
    "error message",
    "version",
)
PRECISION_QUERY = re.compile(
    r"\b(команд\w*|точн\w*|параметр\w*|ошиб\w*|верси\w*|адрес\w*|дат\w*|"
    r"токен\w*|сколько|command\w*|exact\w*|parameter\w*|error\w*|version\w*)\b",
    re.IGNORECASE,
)


def tokenize_query(query: str) -> list[str]:
    result: list[str] = []
    for token in WORD.findall(query.lower()):
        token = token.strip("/.-")
        if len(token) > 1 and token not in result:
            result.append(token)
    return result


def fts_query(query: str) -> str:
    tokens = tokenize_query(query)
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens[:24])


class Retriever:
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

    def search(self, query: str, max_topics: int | None = None) -> list[SearchHit]:
        if not query.strip():
            return []
        retrieval = self.config.retrieval
        lexical_rows = []
        expression = fts_query(query)
        if expression:
            lexical_rows = self.catalog.lexical_search(
                expression, retrieval.lexical_top_k
            )
        lexical_ids = [str(row["id"]) for row in lexical_rows]
        vector_ids = self._vector_search(query, retrieval.vector_top_k)

        ranks: dict[str, dict[str, int]] = {}
        for rank, topic_id in enumerate(lexical_ids, 1):
            ranks.setdefault(topic_id, {})["lexical"] = rank
        for rank, topic_id in enumerate(vector_ids, 1):
            ranks.setdefault(topic_id, {})["vector"] = rank
        if not ranks:
            return []

        raw_scores: dict[str, float] = {}
        for topic_id, channels in ranks.items():
            raw_scores[topic_id] = sum(
                1.0 / (retrieval.rrf_k + rank) for rank in channels.values()
            )
        maximum = max(raw_scores.values())
        ordered = sorted(raw_scores, key=raw_scores.get, reverse=True)
        hits: list[SearchHit] = []
        for topic_id in ordered:
            score = raw_scores[topic_id] / maximum if maximum else 0.0
            if score < retrieval.minimum_score:
                continue
            path = self.catalog.topic_path(topic_id)
            if path is None or not path.exists():
                continue
            topic = read_topic(path)
            topic.path = self._portable_path(path)
            channels = ranks[topic_id]
            hits.append(
                SearchHit(
                    topic=topic,
                    score=score,
                    lexical_rank=channels.get("lexical"),
                    vector_rank=channels.get("vector"),
                )
            )
            if len(hits) >= (max_topics or retrieval.final_top_k):
                break
        return hits

    def get_topic(self, topic_id: str) -> Topic:
        validate_id(topic_id, "topic id")
        path = self.catalog.topic_path(topic_id)
        if path is None or not path.exists():
            raise FileNotFoundError(topic_id)
        topic = read_topic(path)
        topic.path = self._portable_path(path)
        return topic

    def _portable_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.config.storage.root.resolve()))
        except ValueError:
            return str(path)

    def expand_topic(
        self,
        topic_id: str,
        query: str,
        max_fragments: int = 5,
        token_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        topic = self.get_topic(topic_id)
        messages: list[Message] = []
        for source in topic.source_ranges:
            messages.extend(
                self.recorder.read_turns(
                    source.session_id, source.from_message, source.to_message
                )
            )
        return rank_messages(
            messages,
            query,
            max_fragments=max_fragments,
            token_budget=token_budget or self.config.retrieval.source_budget_tokens,
        )

    def search_transcript(
        self,
        query: str,
        session_id: str | None = None,
        max_fragments: int = 10,
    ) -> list[dict[str, Any]]:
        sessions: Iterable[Path]
        if session_id:
            sessions = [self.recorder.locate(session_id)]
        else:
            sessions = (
                item.parent
                for item in self.config.storage.sessions_dir.glob("*/*/*/session.json")
            )
        candidates: list[Message] = []
        for path in sessions:
            for message in read_messages(path / "transcript.jsonl"):
                message.metadata["_session_id"] = path.name
                candidates.append(message)
        ranked = rank_messages(candidates, query, max_fragments=max_fragments)
        return ranked

    def _vector_search(self, query: str, limit: int) -> list[str]:
        if self.embedder is None:
            return []
        try:
            query_vector = self.embedder.embed([query])[0]
        except Exception:
            return []
        indexed = self.catalog.vector_search(
            self.embedder.model, pack_vector(query_vector), len(query_vector), limit
        )
        if len(indexed) >= limit:
            return indexed
        scored: list[tuple[float, str]] = []
        for row in self.catalog.list_embeddings():
            if row["model"] != self.embedder.model:
                continue
            if row["topic_id"] in indexed:
                continue
            if int(row["dimension"]) != len(query_vector):
                continue
            vector = unpack_vector(row["embedding"], int(row["dimension"]))
            score = cosine_similarity(query_vector, vector)
            if score > 0.0:
                scored.append((score, str(row["topic_id"])))
        scored.sort(reverse=True)
        return indexed + [topic_id for _, topic_id in scored[: max(0, limit - len(indexed))]]


class ContextBuilder:
    def __init__(self, config: AppConfig, retriever: Retriever):
        self.config = config
        self.retriever = retriever

    def build(
        self,
        query: str,
        max_topics: int | None = None,
        summary_budget_tokens: int | None = None,
        include_sources: str = "auto",
    ) -> dict[str, Any]:
        if include_sources not in {"auto", "always", "never"}:
            raise ValueError("include_sources must be auto, always, or never")
        config = self.config.retrieval
        max_topics = max_topics or config.final_top_k
        budget = summary_budget_tokens or config.summary_budget_tokens
        hits = self.retriever.search(query, max_topics=max_topics)
        used = 0
        summaries_opened = 0
        output: list[dict[str, Any]] = []
        auto_precision = bool(PRECISION_QUERY.search(query)) or any(
            pattern in query.lower() for pattern in PRECISION_PATTERNS
        )
        for hit in hits:
            item = hit.to_dict(include_summary=False)
            card_tokens = estimate_tokens(
                f"{hit.topic.title}\n{hit.topic.description}\n{hit.topic.problem}"
            )
            used += card_tokens
            summary_tokens = estimate_tokens(hit.topic.summary)
            if summaries_opened < config.auto_open_summaries and used + summary_tokens <= budget:
                item["summary"] = hit.topic.summary
                used += summary_tokens
                summaries_opened += 1
            should_expand = include_sources == "always" or (
                include_sources == "auto" and auto_precision
            )
            if should_expand and "summary" in item:
                fragments = self.retriever.expand_topic(
                    hit.topic.id,
                    query,
                    max_fragments=5,
                    token_budget=config.source_budget_tokens,
                )
                if fragments:
                    item["source_fragments"] = fragments
                    used += sum(estimate_tokens(value["text"]) for value in fragments)
            output.append(item)
        return {
            "topics": output,
            "used_tokens": used,
            "can_expand": bool(output),
        }


def rank_messages(
    messages: list[Message],
    query: str,
    max_fragments: int = 5,
    token_budget: int = 1_800,
) -> list[dict[str, Any]]:
    tokens = tokenize_query(query)
    if not messages:
        return []

    def score(message: Message) -> tuple[int, int]:
        text = message.text.lower()
        matches = sum(text.count(token) for token in tokens)
        exact = int(query.lower().strip() in text) if query.strip() else 0
        return exact * 100 + matches, message.id

    ranked = sorted(messages, key=score, reverse=True)
    if tokens:
        matched = [message for message in ranked if score(message)[0] > 0]
        if matched:
            ranked = matched
    result: list[dict[str, Any]] = []
    used = 0
    for message in ranked:
        cost = estimate_tokens(message.text)
        if result and used + cost > token_budget:
            continue
        text = message.text
        if not result and cost > token_budget:
            text = text[: max(800, token_budget * 4)] + "\n[fragment truncated]"
            cost = estimate_tokens(text)
        fragment = {
            "id": message.id,
            "role": message.role,
            "text": text,
            "created_at": message.created_at,
        }
        if message.metadata.get("_session_id"):
            fragment["session_id"] = message.metadata["_session_id"]
        result.append(fragment)
        used += cost
        if len(result) >= max_fragments:
            break
    result.sort(key=lambda item: item["id"])
    return result
