from __future__ import annotations

import json
import math
import os
import struct
import urllib.error
import urllib.request
from array import array
from pathlib import Path
from typing import Protocol

from .config import EmbeddingsConfig


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self.model = config.model

    @property
    def enabled(self) -> bool:
        return bool(self.config.base_url and self.config.model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            raise RuntimeError("embedding endpoint is not configured")
        payload = json.dumps(
            {"model": self.model, "input": texts}, ensure_ascii=False
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/embeddings",
            data=payload,
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
            raise RuntimeError(f"embedding endpoint returned {exc.code}: {detail}") from exc
        ordered = sorted(raw.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [[float(value) for value in item["embedding"]] for item in ordered]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding endpoint returned an unexpected vector count")
        return vectors


class FastEmbedEmbedder:
    """Lazy, in-process embeddings whose model cache lives inside the project."""

    def __init__(self, config: EmbeddingsConfig, cache_dir: str | Path):
        self.config = config
        self.model = config.model or "sentence-transformers/all-MiniLM-L6-v2"
        self.cache_dir = Path(cache_dir)
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._client is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is required for the local embedding provider"
                ) from exc
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._client = TextEmbedding(
                model_name=self.model,
                cache_dir=str(self.cache_dir),
            )
        return [
            [float(value) for value in vector]
            for vector in self._client.embed(texts)
        ]


def pack_vector(vector: list[float]) -> bytes:
    values = array("f", vector)
    if struct.pack("=I", 1) != struct.pack("<I", 1):
        values.byteswap()
    return values.tobytes()


def unpack_vector(value: bytes, dimension: int) -> list[float]:
    values = array("f")
    values.frombytes(value)
    if struct.pack("=I", 1) != struct.pack("<I", 1):
        values.byteswap()
    if len(values) != dimension:
        raise ValueError(f"invalid vector: expected {dimension}, got {len(values)}")
    return list(values)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot / (left_norm * right_norm)


def topic_embedding_text(
    title: str, description: str, problem: str, keywords: list[str], summary: str
) -> str:
    compact_summary = summary[:6000]
    return (
        f"Title: {title}\nDescription: {description}\nProblem solved: {problem}\n"
        f"Keywords: {', '.join(keywords)}\nSummary:\n{compact_summary}"
    )
