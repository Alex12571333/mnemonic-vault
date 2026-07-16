from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .catalog import Catalog
from .config import AppConfig
from .embeddings import Embedder, cosine_similarity, pack_vector, unpack_vector
from .global_topics import GlobalTopicStore
from .models import Message, SearchHit, Topic
from .recorder import SessionRecorder
from .session_aliases import SessionAliasStore
from .storage import estimate_tokens, read_topic, validate_id


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
HISTORICAL_QUERY = re.compile(
    r"(?:\b(?:19|20)\d{2}\b|раньше|тогда|до\s+перехода|первоначальн|"
    r"предыдущ|прежн|previously|before|back\s+then|originally|at\s+the\s+time)",
    re.IGNORECASE,
)
EXPLICIT_ISO_DATE = re.compile(
    r"\b(?P<year>(?:19|20)\d{2})(?:[-/.](?P<month>0?[1-9]|1[0-2])"
    r"(?:[-/.](?P<day>0?[1-9]|[12]\d|3[01]))?)?\b"
)
EXPLICIT_EUROPEAN_DATE = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])[./-]"
    r"(?P<month>0?[1-9]|1[0-2])[./-](?P<year>(?:19|20)\d{2})\b"
)
MONTH_NAMES = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "май": 5,
    "мая": 5,
    "мае": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
QUERY_STOP_WORDS = {
    "а", "был", "была", "были", "в", "во", "где", "год", "года", "году",
    "для", "и", "из", "использовали", "использовать", "как", "какая", "какие",
    "какой", "какую", "когда", "мы", "на", "наш", "наша", "наше", "о", "по",
    "применяли", "тогда", "у", "что", "это", "я", "a", "an", "at", "did",
    "do", "for", "how", "in", "is", "of", "on", "our", "the", "then", "used",
    "we", "what", "when", "which",
}


def tokenize_query(query: str) -> list[str]:
    result: list[str] = []
    for token in WORD.findall(query.lower()):
        token = token.strip("/.-")
        if len(token) > 1 and token not in result:
            result.append(token)
    return result


def relevance_query_tokens(query: str) -> list[str]:
    tokens = tokenize_query(query)
    filtered = [
        token
        for token in tokens
        if token not in QUERY_STOP_WORDS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
        and not EXPLICIT_ISO_DATE.fullmatch(token)
        and not EXPLICIT_EUROPEAN_DATE.fullmatch(token)
    ]
    return filtered or tokens


def temporal_scope(query: str) -> tuple[int, int | None, int | None] | None:
    """Extract an explicit calendar constraint without using a generative model."""
    match = EXPLICIT_EUROPEAN_DATE.search(query) or EXPLICIT_ISO_DATE.search(query)
    if not match:
        return None
    year = int(match.group("year"))
    month = int(match.group("month")) if match.groupdict().get("month") else None
    day = int(match.group("day")) if match.groupdict().get("day") else None
    lowered = query.lower()
    if month is None:
        window = lowered[
            max(0, match.start("year") - 32) : min(
                len(lowered), match.end("year") + 32
            )
        ]
        for stem, value in MONTH_NAMES.items():
            if re.search(rf"\b{re.escape(stem)}\w*\b", window):
                month = value
                break
    return year, month, day


def date_matches_scope(
    value: str, scope: tuple[int, int | None, int | None]
) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    year, month, day = scope
    return (
        parsed.year == year
        and (month is None or parsed.month == month)
        and (day is None or parsed.day == day)
    )


def temporal_prefix(scope: tuple[int, int | None, int | None]) -> str:
    year, month, day = scope
    if month is None:
        return f"{year:04d}-"
    if day is None:
        return f"{year:04d}-{month:02d}-"
    return f"{year:04d}-{month:02d}-{day:02d}"


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
        self.session_aliases = SessionAliasStore(
            config.storage.root / "session-aliases.json"
        )
        self.global_topics = GlobalTopicStore(config)

    def search(self, query: str, max_topics: int | None = None) -> list[SearchHit]:
        if not query.strip():
            return []
        retrieval = self.config.retrieval
        requested_date = temporal_scope(query)
        date_prefix = temporal_prefix(requested_date) if requested_date else None
        temporal_topic_ids = (
            self.catalog.topic_ids_for_session_started_prefix(date_prefix)
            if date_prefix
            else None
        )
        if temporal_topic_ids == []:
            return []
        lexical_rows = []
        expression = fts_query(query)
        if expression:
            lexical_rows = self.catalog.lexical_search(
                expression,
                retrieval.lexical_top_k,
                session_started_prefix=date_prefix,
            )
        query_tokens = set(relevance_query_tokens(query))
        lexical_relevance: dict[str, float] = {}
        for row in lexical_rows:
            searchable = " ".join(
                str(row[key]) for key in ("title", "description", "problem", "keywords", "summary")
            )
            topic_tokens = set(tokenize_query(searchable))
            lexical_relevance[str(row["id"])] = (
                len(query_tokens & topic_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
        vector_scores = self._vector_search(
            query, retrieval.vector_top_k, temporal_topic_ids
        )

        lexical_ids = [
            str(row["id"])
            for row in lexical_rows
            if lexical_relevance[str(row["id"])]
            >= retrieval.lexical_min_query_coverage
        ]
        vector_ids = [
            topic_id
            for topic_id, similarity in vector_scores.items()
            if similarity >= retrieval.vector_min_similarity
        ]

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
        ordered = sorted(raw_scores, key=raw_scores.get, reverse=True)
        candidates: list[SearchHit] = []
        for topic_id in ordered:
            score = max(
                lexical_relevance.get(topic_id, 0.0),
                vector_scores.get(topic_id, -1.0),
            )
            if score < retrieval.minimum_score:
                continue
            path = self.catalog.topic_path(topic_id)
            if path is None or not path.exists():
                continue
            topic = read_topic(path)
            topic.path = self._portable_path(path)
            self._attach_session_dates(topic)
            channels = ranks[topic_id]
            candidates.append(
                SearchHit(
                    topic=topic,
                    score=score,
                    lexical_rank=channels.get("lexical"),
                    vector_rank=channels.get("vector"),
                    lexical_relevance=lexical_relevance.get(topic_id, 0.0),
                    vector_similarity=vector_scores.get(topic_id),
                    rrf_score=raw_scores[topic_id],
                )
            )
        if requested_date:
            candidates = [
                candidate
                for candidate in candidates
                if date_matches_scope(
                    candidate.topic.session_started_at, requested_date
                )
            ]
        selected = self._diversify(
            candidates,
            max_topics or retrieval.final_top_k,
            preserve_history=bool(HISTORICAL_QUERY.search(query)),
        )
        projection_mapping = self.global_topics.topic_mapping()
        for hit in selected:
            hit.global_topic_id = projection_mapping.get(hit.topic.id)
        return selected

    def get_topic(self, topic_id: str) -> Topic:
        validate_id(topic_id, "topic id")
        path = self.catalog.topic_path(topic_id)
        if path is None or not path.exists():
            raise FileNotFoundError(topic_id)
        topic = read_topic(path)
        topic.path = self._portable_path(path)
        self._attach_session_dates(topic)
        return topic

    def _attach_session_dates(self, topic: Topic) -> None:
        row = self.catalog.get_session(topic.session_id)
        if row is not None:
            topic.session_started_at = str(row["started_at"] or "")
            topic.session_ended_at = str(row["ended_at"]) if row["ended_at"] else None

    def _diversify(
        self,
        candidates: list[SearchHit],
        limit: int,
        preserve_history: bool = False,
    ) -> list[SearchHit]:
        if preserve_history:
            return candidates[:limit]
        groups: list[SearchHit] = []
        for candidate in candidates:
            candidate_tokens = set(
                tokenize_query(f"{candidate.topic.title} {candidate.topic.problem}")
            )
            duplicate_index: int | None = None
            for index, existing in enumerate(groups):
                existing_tokens = set(
                    tokenize_query(f"{existing.topic.title} {existing.topic.problem}")
                )
                union = candidate_tokens | existing_tokens
                similarity = len(candidate_tokens & existing_tokens) / len(union) if union else 0.0
                if similarity >= 0.72:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                groups.append(candidate)
                continue
            existing = groups[duplicate_index]
            if candidate.topic.updated_at > existing.topic.updated_at:
                candidate.related_older_topic_ids = [
                    existing.topic.id,
                    *existing.related_older_topic_ids,
                ]
                groups[duplicate_index] = candidate
            else:
                existing.related_older_topic_ids.append(candidate.topic.id)
        return groups[:limit]

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
        expression = fts_query(query)
        if not expression:
            return []
        session_ids = self.session_aliases.members(session_id) if session_id else None
        rows = self.catalog.search_messages(
            expression,
            max(max_fragments * 8, 40),
            session_ids=session_ids,
        )
        candidates = [
            Message(
                id=int(row["message_id"]),
                role=str(row["role"]),
                text=str(row["content"]),
                created_at=str(row["created_at"]),
                metadata={"_session_id": str(row["session_id"])},
            )
            for row in rows
        ]
        ranked = rank_messages(candidates, query, max_fragments=max_fragments)
        return ranked

    def _vector_search(
        self,
        query: str,
        limit: int,
        topic_ids: list[str] | None = None,
    ) -> dict[str, float]:
        if self.embedder is None:
            return {}
        try:
            query_vector = self.embedder.embed([query])[0]
        except Exception:
            return {}
        indexed = (
            []
            if topic_ids is not None
            else self.catalog.vector_search(
                self.embedder.model,
                pack_vector(query_vector),
                len(query_vector),
                limit,
            )
        )
        scores = {
            str(item["topic_id"]): float(item["cosine_similarity"])
            for item in indexed
        }
        if len(scores) >= limit:
            return scores
        scored: list[tuple[float, str]] = []
        for row in self.catalog.list_embeddings(topic_ids):
            if row["model"] != self.embedder.model:
                continue
            if row["topic_id"] in scores:
                continue
            if int(row["dimension"]) != len(query_vector):
                continue
            vector = unpack_vector(row["embedding"], int(row["dimension"]))
            score = cosine_similarity(query_vector, vector)
            if score > 0.0:
                scored.append((score, str(row["topic_id"])))
        scored.sort(reverse=True)
        for score, topic_id in scored[: max(0, limit - len(scores))]:
            scores[topic_id] = score
        return scores


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
        total_context_budget_tokens: int | None = None,
    ) -> dict[str, Any]:
        if include_sources not in {"auto", "always", "never"}:
            raise ValueError("include_sources must be auto, always, or never")
        config = self.config.retrieval
        max_topics = max_topics or config.final_top_k
        total_budget = min(
            total_context_budget_tokens or config.total_context_budget_tokens,
            config.total_context_budget_tokens,
        )
        summary_budget = min(
            summary_budget_tokens or config.summary_budget_tokens,
            config.summary_budget_tokens,
            total_budget,
        )
        hits = self.retriever.search(query, max_topics=max_topics)
        used = 0
        card_used = 0
        summary_used = 0
        source_used = 0
        summaries_opened = 0
        output: list[dict[str, Any]] = []
        auto_precision = bool(PRECISION_QUERY.search(query)) or any(
            pattern in query.lower() for pattern in PRECISION_PATTERNS
        )
        for hit in hits:
            item = compact_hit_card(hit, config.card_budget_tokens - card_used)
            card_tokens = estimate_tokens(json.dumps(item, ensure_ascii=False))
            if (
                not item
                or card_used + card_tokens > config.card_budget_tokens
                or used + card_tokens > total_budget
            ):
                break
            used += card_tokens
            card_used += card_tokens
            remaining_summary = min(
                summary_budget - summary_used,
                total_budget - used,
            )
            summary, summary_tokens = bounded_json_text(
                "summary", hit.topic.summary, remaining_summary
            )
            if summaries_opened < config.auto_open_summaries and summary:
                item["summary"] = summary
                used += summary_tokens
                summary_used += summary_tokens
                summaries_opened += 1
            should_expand = include_sources == "always" or (
                include_sources == "auto" and auto_precision
            )
            if should_expand and "summary" in item:
                remaining_source = min(
                    config.source_budget_tokens - source_used,
                    total_budget - used,
                )
                fragments = self.retriever.expand_topic(
                    hit.topic.id,
                    query,
                    max_fragments=5,
                    token_budget=max(1, remaining_source - 24),
                ) if remaining_source > 0 else []
                if fragments:
                    fragment_tokens = estimate_tokens(
                        json.dumps({"source_fragments": fragments}, ensure_ascii=False)
                    )
                    while fragments and fragment_tokens > remaining_source:
                        fragments.pop()
                        fragment_tokens = estimate_tokens(
                            json.dumps({"source_fragments": fragments}, ensure_ascii=False)
                        )
                    if fragments:
                        item["source_fragments"] = fragments
                        used += fragment_tokens
                        source_used += fragment_tokens
            output.append(item)
        return {
            "topics": output,
            "used_tokens": used,
            "budget_tokens": total_budget,
            "budget_breakdown": {
                "cards": card_used,
                "summaries": summary_used,
                "sources": source_used,
            },
            "can_expand": bool(output),
        }


def compact_hit_card(hit: SearchHit, token_budget: int) -> dict[str, Any]:
    """Keep a useful topic card inside its sub-budget without hiding overflow."""
    if token_budget <= 0:
        return {}
    item = hit.to_dict(include_summary=False)
    item["keywords"] = list(item.get("keywords", []))
    if "related_older_topic_ids" in item:
        item["related_older_topic_ids"] = list(item["related_older_topic_ids"])

    def cost() -> int:
        return estimate_tokens(json.dumps(item, ensure_ascii=False))

    ranges = item.get("source_ranges", [])
    original_ranges = len(ranges) if isinstance(ranges, list) else 0
    while isinstance(ranges, list) and len(ranges) > 1 and cost() > token_budget:
        ranges.pop()
    if isinstance(ranges, list) and len(ranges) < original_ranges:
        item["source_ranges_omitted"] = original_ranges - len(ranges)
    keywords = item.get("keywords", [])
    while isinstance(keywords, list) and len(keywords) > 3 and cost() > token_budget:
        keywords.pop()
    older = item.get("related_older_topic_ids", [])
    while isinstance(older, list) and len(older) > 1 and cost() > token_budget:
        older.pop()
    for field in ("description", "problem", "title"):
        if cost() <= token_budget:
            break
        value = str(item.get(field, ""))
        overflow_chars = max(16, int((cost() - token_budget) * 2.5) + 16)
        keep = max(0, len(value) - overflow_chars)
        item[field] = value[:keep] + ("…" if keep else "")
    return item if cost() <= token_budget else {}


def bounded_json_text(field: str, text: str, token_budget: int) -> tuple[str, int]:
    """Truncate one JSON string field to a deterministic tokenizer-independent budget."""
    if token_budget <= 0:
        return "", 0

    def cost(value: str) -> int:
        return estimate_tokens(json.dumps({field: value}, ensure_ascii=False))

    if cost(text) <= token_budget:
        return text, cost(text)
    marker = "\n[truncated to context budget]"
    if cost(marker) > token_budget:
        return "", 0
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[:middle] + marker
        if cost(candidate) <= token_budget:
            low = middle
        else:
            high = middle - 1
    value = text[:low] + marker
    return value, cost(value)


def rank_messages(
    messages: list[Message],
    query: str,
    max_fragments: int = 5,
    token_budget: int = 1_800,
) -> list[dict[str, Any]]:
    tokens = tokenize_query(query)
    if not messages or not tokens:
        return []

    def score(message: Message) -> tuple[int, int]:
        text = message.text.lower()
        matches = sum(text.count(token) for token in tokens)
        exact = int(query.lower().strip() in text) if query.strip() else 0
        return exact * 100 + matches, message.id

    ranked = sorted(messages, key=score, reverse=True)
    matched = [message for message in ranked if score(message)[0] > 0]
    if not matched:
        return []
    ranked = matched
    result: list[dict[str, Any]] = []
    used = 0
    for message in ranked:
        cost = estimate_tokens(message.text)
        if result and used + cost > token_budget:
            continue
        text = message.text
        if not result and cost > token_budget:
            marker = "\n[fragment truncated]"
            marker_cost = estimate_tokens(marker)
            if token_budget <= marker_cost:
                return []
            allowed = max(1, int((token_budget - marker_cost) * 2.5))
            text = text[:allowed] + marker
            cost = estimate_tokens(text)
            while cost > token_budget and allowed > 1:
                allowed = max(1, allowed - 4)
                text = message.text[:allowed] + marker
                cost = estimate_tokens(text)
            if cost > token_budget:
                return []
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
