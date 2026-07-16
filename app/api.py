from __future__ import annotations

import threading
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .config import AppConfig
from .global_topics import GlobalTopicStore
from .models import Session
from .service import Services, build_services


class RequestBodyLimitMiddleware:
    """Reject request bodies by bytes read, including chunked requests."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await _json_error(413, "request body is too large")(
                        scope, receive, send
                    )
                    return
            except ValueError:
                await _json_error(400, "invalid content-length")(
                    scope, receive, send
                )
                return

        buffered = []
        received = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                await _json_error(413, "request body is too large")(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break

        async def replay_receive():
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)


class StartSessionRequest(BaseModel):
    agent: str = Field(default="unknown", min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    started_at: str | None = Field(default=None, max_length=64)


class AppendMessageRequest(BaseModel):
    role: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=1_000_000)
    created_at: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)
    external_event_id: str | None = Field(default=None, max_length=128)


class EndSessionRequest(BaseModel):
    ended_at: str | None = Field(default=None, max_length=64)


class MemoryScopeRequest(BaseModel):
    type: Literal["global", "agent", "project", "session"] = "global"
    id: str | None = Field(default=None, min_length=1, max_length=128)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=100_000)
    max_topics: int = Field(default=5, ge=1, le=50)
    summary_budget_tokens: int | None = Field(default=None, ge=100, le=32000)
    total_context_budget_tokens: int | None = Field(default=None, ge=100, le=32000)
    include_sources: Literal["auto", "always", "never"] = "auto"
    scope: MemoryScopeRequest | None = None
    context_scopes: list[MemoryScopeRequest] = Field(default_factory=list, max_length=8)
    scope_mode: Literal["boost", "strict"] = "boost"
    include_all_scopes: bool = False


class RememberRequest(BaseModel):
    verbatim: str = Field(min_length=1, max_length=100_000)
    normalized: str | None = Field(default=None, min_length=1, max_length=100_000)
    kind: Literal[
        "fact",
        "preference",
        "decision",
        "configuration",
        "identity",
        "constraint",
        "task",
        "correction",
    ] = "fact"
    scope: MemoryScopeRequest = Field(default_factory=MemoryScopeRequest)
    source_session_id: str | None = Field(default=None, max_length=128)
    source_message_id: int | None = Field(default=None, ge=1)
    idempotency_key: str | None = Field(default=None, max_length=128)
    supersedes: str | None = Field(default=None, max_length=128)
    created_at: str | None = Field(default=None, max_length=64)


class ExpandRequest(BaseModel):
    query: str = Field(min_length=1, max_length=100_000)
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
    global_topics = GlobalTopicStore(services.config)
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
        version="0.5.1",
        lifespan=lifespan,
    )
    app.state.services = services
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=services.config.api.max_request_bytes,
    )
    api_token = os.environ.get(services.config.api.bearer_token_env, "")

    @app.middleware("http")
    async def bearer_auth(request: Request, call_next):
        if api_token and request.url.path.startswith("/v1/"):
            authorization = request.headers.get("authorization", "")
            expected = f"Bearer {api_token}"
            if not secrets.compare_digest(authorization, expected):
                return _json_error(401, "missing or invalid bearer token")
        return await call_next(request)

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
        if len(payload.content) > services.config.api.max_message_chars:
            raise HTTPException(status_code=413, detail="message is too large")
        try:
            return services.recorder.append(
                session_id,
                payload.role,
                payload.content,
                payload.created_at,
                payload.metadata,
                payload.external_event_id,
            ).to_dict()
        except RuntimeError as exc:
            if "cannot append to" not in str(exc):
                raise
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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

    @app.get("/v1/sessions/{session_id}/aliases")
    def session_aliases(session_id: str) -> dict[str, Any]:
        alias = services.retriever.session_aliases.describe(session_id)
        if alias is None:
            raise HTTPException(status_code=404, detail="session alias not found")
        return alias

    @app.post("/v1/memory/search")
    def search_memory(payload: SearchRequest) -> dict[str, Any]:
        if len(payload.query) > services.config.api.max_query_chars:
            raise HTTPException(status_code=413, detail="query is too large")
        return services.context_builder.build(
            query=payload.query,
            max_topics=payload.max_topics,
            summary_budget_tokens=payload.summary_budget_tokens,
            include_sources=payload.include_sources,
            total_context_budget_tokens=payload.total_context_budget_tokens,
            scope_type=payload.scope.type if payload.scope else None,
            scope_id=payload.scope.id if payload.scope else None,
            context_scopes=[(scope.type, scope.id) for scope in payload.context_scopes],
            scope_mode=payload.scope_mode,
            include_all_scopes=payload.include_all_scopes,
        )

    @app.post("/v1/memory/remember", status_code=201)
    def remember(payload: RememberRequest) -> dict[str, Any]:
        if services.explicit_memory is None:
            raise HTTPException(status_code=503, detail="explicit memory is unavailable")
        return services.explicit_memory.remember(
            payload.verbatim,
            normalized=payload.normalized,
            kind=payload.kind,
            scope_type=payload.scope.type,
            scope_id=payload.scope.id,
            source_session_id=payload.source_session_id,
            source_message_id=payload.source_message_id,
            idempotency_key=payload.idempotency_key,
            supersedes=payload.supersedes,
            created_at=payload.created_at,
        )

    @app.get("/v1/memory/explicit/{memory_id}")
    def open_explicit_memory(memory_id: str) -> dict[str, Any]:
        if services.explicit_memory is None:
            raise HTTPException(status_code=503, detail="explicit memory is unavailable")
        return services.explicit_memory.get(memory_id).to_search_dict()

    @app.get("/v1/memory/global-topics")
    def list_global_topics(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        return {"topics": global_topics.list_cards(limit)}

    @app.get("/v1/memory/global-topics/{global_topic_id}")
    def open_global_topic(
        global_topic_id: str,
        max_timeline_entries: int = Query(default=50, ge=1, le=500),
        total_token_budget: int | None = Query(default=None, ge=300, le=32000),
    ) -> dict[str, Any]:
        return global_topics.get(
            global_topic_id, max_timeline_entries, total_token_budget
        )

    @app.get("/v1/memory/topics/{topic_id}")
    def open_topic(topic_id: str) -> dict[str, Any]:
        topic = services.retriever.get_topic(topic_id)
        return {**topic.card(), "summary": topic.summary}

    @app.post("/v1/memory/topics/{topic_id}/expand")
    def expand_topic(topic_id: str, payload: ExpandRequest) -> dict[str, Any]:
        if len(payload.query) > services.config.api.max_query_chars:
            raise HTTPException(status_code=413, detail="query is too large")
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
        if len(payload.query) > services.config.api.max_query_chars:
            raise HTTPException(status_code=413, detail="query is too large")
        return {
            "fragments": services.retriever.search_transcript(
                payload.query, session_id, payload.max_fragments
            )
        }

    return app


def _json_error(status_code: int, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})
