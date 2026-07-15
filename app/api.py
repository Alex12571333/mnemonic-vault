from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .config import AppConfig
from .models import Session
from .service import Services, build_services


class StartSessionRequest(BaseModel):
    agent: str = "unknown"
    session_id: str | None = None
    started_at: str | None = None


class AppendMessageRequest(BaseModel):
    role: str
    content: str
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EndSessionRequest(BaseModel):
    ended_at: str | None = None


class SearchRequest(BaseModel):
    query: str
    max_topics: int = Field(default=5, ge=1, le=50)
    summary_budget_tokens: int = Field(default=1800, ge=100, le=32000)
    include_sources: Literal["auto", "always", "never"] = "auto"


class ExpandRequest(BaseModel):
    query: str
    max_fragments: int = Field(default=5, ge=1, le=20)
    token_budget: int | None = Field(default=None, ge=100, le=32000)


def session_dict(session: Session) -> dict[str, Any]:
    return session.to_dict()


def create_app(
    config_path: str = "config/config.yaml",
    services: Services | None = None,
    start_worker: bool = True,
) -> FastAPI:
    services = services or build_services(AppConfig.load(config_path))
    stop_event = threading.Event()
    worker: threading.Thread | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal worker
        if start_worker:
            worker = threading.Thread(
                target=services.job_runner.run_forever,
                args=(stop_event,),
                name="memory-job-runner",
                daemon=True,
            )
            worker.start()
        yield
        stop_event.set()
        if worker:
            worker.join(timeout=5.0)

    app = FastAPI(
        title="Mnemonic Vault",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.services = services

    @app.exception_handler(FileNotFoundError)
    async def not_found(_: Request, exc: FileNotFoundError):
        return _json_error(404, f"not found: {exc}")

    @app.exception_handler(ValueError)
    async def invalid_input(_: Request, exc: ValueError):
        return _json_error(422, str(exc))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/sessions/start", status_code=201)
    def start_session(payload: StartSessionRequest) -> dict[str, Any]:
        try:
            session = services.recorder.start_session(
                payload.agent, payload.session_id, payload.started_at
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return session_dict(session)

    @app.post("/v1/sessions/{session_id}/messages", status_code=201)
    def append_message(
        session_id: str, payload: AppendMessageRequest
    ) -> dict[str, Any]:
        return services.recorder.append(
            session_id,
            payload.role,
            payload.content,
            payload.created_at,
            payload.metadata,
        ).to_dict()

    @app.post("/v1/sessions/{session_id}/end")
    def end_session(
        session_id: str, payload: EndSessionRequest | None = None
    ) -> dict[str, Any]:
        return session_dict(
            services.recorder.end_session(
                session_id, payload.ended_at if payload else None
            )
        )

    @app.get("/v1/sessions/{session_id}/turns")
    def read_turns(
        session_id: str,
        from_turn: int = Query(default=1, alias="from", ge=1),
        to_turn: int | None = Query(default=None, alias="to", ge=1),
    ) -> dict[str, Any]:
        if to_turn is not None and to_turn < from_turn:
            raise HTTPException(status_code=422, detail="to must be >= from")
        messages = services.recorder.read_turns(session_id, from_turn, to_turn)
        return {"session_id": session_id, "turns": [item.to_dict() for item in messages]}

    @app.post("/v1/memory/search")
    def search_memory(payload: SearchRequest) -> dict[str, Any]:
        return services.context_builder.build(
            payload.query,
            payload.max_topics,
            payload.summary_budget_tokens,
            payload.include_sources,
        )

    @app.get("/v1/memory/topics/{topic_id}")
    def open_topic(topic_id: str) -> dict[str, Any]:
        topic = services.retriever.get_topic(topic_id)
        return {**topic.card(), "summary": topic.summary}

    @app.post("/v1/memory/topics/{topic_id}/expand")
    def expand_topic(topic_id: str, payload: ExpandRequest) -> dict[str, Any]:
        return {
            "topic_id": topic_id,
            "fragments": services.retriever.expand_topic(
                topic_id,
                payload.query,
                payload.max_fragments,
                payload.token_budget,
            ),
        }

    @app.post("/v1/memory/search-transcript")
    def search_transcript(payload: ExpandRequest, session_id: str | None = None):
        return {
            "fragments": services.retriever.search_transcript(
                payload.query, session_id, payload.max_fragments
            )
        }

    return app


def _json_error(status_code: int, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})
