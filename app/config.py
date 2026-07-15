from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StorageConfig:
    root: Path = Path("data")
    sessions_dir: Path = Path("data/sessions")
    catalog_db: Path = Path("data/catalog.sqlite")


@dataclass(slots=True)
class SummarizationConfig:
    messages_per_batch: int = 20
    token_threshold: int = 12_000
    idle_minutes: int = 30
    max_input_tokens: int = 16_000
    max_output_tokens: int = 2_500


@dataclass(slots=True)
class RetrievalConfig:
    lexical_top_k: int = 30
    vector_top_k: int = 30
    final_top_k: int = 5
    auto_open_summaries: int = 2
    summary_budget_tokens: int = 1_800
    source_budget_tokens: int = 1_800
    minimum_score: float = 0.35
    rrf_k: int = 60


@dataclass(slots=True)
class EndpointConfig:
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 120.0


@dataclass(slots=True)
class MemoryLLMConfig(EndpointConfig):
    temperature: float = 0.0
    enable_thinking: bool = False


@dataclass(slots=True)
class EmbeddingsConfig(EndpointConfig):
    provider: str = "openai-compatible"
    dimension: int = 1_024


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    storage: StorageConfig = field(default_factory=StorageConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory_llm: MemoryLLMConfig = field(default_factory=MemoryLLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)

    @classmethod
    def load(cls, path: str | Path = "config/config.yaml") -> "AppConfig":
        path = Path(path).resolve()
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read config.yaml") from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        project_root = path.parent.parent
        return cls.from_mapping(raw, project_root)

    @classmethod
    def from_mapping(
        cls, raw: dict[str, Any], project_root: str | Path = "."
    ) -> "AppConfig":
        project_root = Path(project_root).resolve()
        storage_raw = raw.get("storage", {})

        def resolve(value: str, default: str) -> Path:
            result = Path(value or default)
            return result if result.is_absolute() else project_root / result

        storage = StorageConfig(
            root=resolve(storage_raw.get("root", "data"), "data"),
            sessions_dir=resolve(
                storage_raw.get("sessions_dir", "data/sessions"), "data/sessions"
            ),
            catalog_db=resolve(
                storage_raw.get("catalog_db", "data/catalog.sqlite"),
                "data/catalog.sqlite",
            ),
        )
        llm_raw = dict(raw.get("memory_llm", {}))
        embeddings_raw = dict(raw.get("embeddings", {}))
        environment_overrides = (
            (llm_raw, "base_url", "ETERNAL_MEMORY_LLM_BASE_URL"),
            (llm_raw, "model", "ETERNAL_MEMORY_LLM_MODEL"),
            (embeddings_raw, "base_url", "ETERNAL_MEMORY_EMBEDDINGS_BASE_URL"),
            (embeddings_raw, "model", "ETERNAL_MEMORY_EMBEDDINGS_MODEL"),
            (embeddings_raw, "provider", "ETERNAL_MEMORY_EMBEDDINGS_PROVIDER"),
        )
        for target, key, variable in environment_overrides:
            if os.environ.get(variable):
                target[key] = os.environ[variable]
        return cls(
            project_root=project_root,
            storage=storage,
            summarization=SummarizationConfig(**raw.get("summarization", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            memory_llm=MemoryLLMConfig(**llm_raw),
            embeddings=EmbeddingsConfig(**embeddings_raw),
        )
