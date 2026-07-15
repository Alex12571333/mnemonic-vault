from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .retriever import Retriever


def evaluate_retrieval(
    retriever: Retriever, dataset: str | Path, top_k: int = 5
) -> dict[str, Any]:
    """Evaluate recall@k and negative rejection on a user-authored JSONL set."""
    cases: list[dict[str, Any]] = []
    positives = hits = negatives = rejected = 0
    with Path(dataset).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            raw = json.loads(line)
            query = str(raw.get("query", "")).strip()
            expected = {str(value) for value in raw.get("relevant_topic_ids", [])}
            if not query:
                raise ValueError(f"dataset line {line_number} has no query")
            returned = [hit.topic.id for hit in retriever.search(query, max_topics=top_k)]
            if expected:
                positives += 1
                passed = bool(expected & set(returned))
                hits += int(passed)
            else:
                negatives += 1
                passed = not returned
                rejected += int(passed)
            cases.append(
                {
                    "query": query,
                    "kind": str(raw.get("kind", "")),
                    "expected": sorted(expected),
                    "returned": returned,
                    "passed": passed,
                }
            )
    return {
        "top_k": top_k,
        "positive_cases": positives,
        "recall_at_k": hits / positives if positives else None,
        "negative_cases": negatives,
        "negative_rejection_rate": rejected / negatives if negatives else None,
        "cases": cases,
    }
