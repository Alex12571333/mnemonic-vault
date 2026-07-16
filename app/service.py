from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog
from .config import AppConfig
from .embeddings import FastEmbedEmbedder, OpenAICompatibleEmbedder
from .explicit_memory import ExplicitMemoryStore
from .indexer import Indexer
from .recorder import SessionRecorder
from .retriever import ContextBuilder, Retriever
from .summarizer import (
    JobRunner,
    MemorySummarizer,
    OpenAICompatibleMemoryLLM,
)


@dataclass(slots=True)
class Services:
    config: AppConfig
    catalog: Catalog
    recorder: SessionRecorder
    indexer: Indexer
    retriever: Retriever
    context_builder: ContextBuilder
    summarizer: MemorySummarizer
    job_runner: JobRunner
    explicit_memory: ExplicitMemoryStore | None = None


def build_services(config: AppConfig) -> Services:
    config.storage.root.mkdir(parents=True, exist_ok=True)
    for name in ("attachments", "exports", "backups", "jobs", "spool"):
        (config.storage.root / name).mkdir(parents=True, exist_ok=True)
    (config.project_root / "logs").mkdir(parents=True, exist_ok=True)
    catalog = Catalog(config.storage.catalog_db)
    recorder = SessionRecorder(config, catalog)
    if config.embeddings.provider == "fastembed":
        embedder_client = FastEmbedEmbedder(
            config.embeddings, config.storage.root / "models"
        )
    else:
        embedder_client = OpenAICompatibleEmbedder(config.embeddings)
    embedder = embedder_client if embedder_client.enabled else None
    indexer = Indexer(config, catalog, embedder)
    explicit_memory = ExplicitMemoryStore(config, catalog, recorder, embedder)
    retriever = Retriever(config, catalog, recorder, embedder, explicit_memory)
    context_builder = ContextBuilder(config, retriever)
    llm = OpenAICompatibleMemoryLLM(
        config.memory_llm,
        config.summarization.max_output_tokens,
        config.summarization,
    )
    summarizer = MemorySummarizer(
        config, catalog, recorder, indexer, llm
    )
    job_runner = JobRunner(summarizer, recorder)
    return Services(
        config=config,
        catalog=catalog,
        recorder=recorder,
        indexer=indexer,
        retriever=retriever,
        context_builder=context_builder,
        summarizer=summarizer,
        job_runner=job_runner,
        explicit_memory=explicit_memory,
    )
