from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path

if (
    importlib.util.find_spec("agent_core") is None
    or importlib.util.find_spec("agent_sdk") is None
):
    raise unittest.SkipTest(
        "Corax integration tests require optional agent-core and agent-sdk packages"
    )

from agent_core import (
    CapabilityRequest,
    ExtensionRequest,
    MemoryProvider,
    MemoryQuery,
    MemoryRecord,
    PermissionLevel,
    ResultStatus,
    SideEffect,
)
from agent_sdk import ExtensionManifest, load_extension_instance

ROOT = Path(__file__).resolve().parents[1]
CORAX = ROOT / "integrations" / "corax"
sys.path.insert(0, str(CORAX))

from provider import MnemonicVaultProvider  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.remembered: list[dict] = []
        self.searched: list[dict] = []

    def remember(self, payload: dict) -> dict:
        self.remembered.append(payload)
        return {"stored": True, "memory_id": "m-1", "available_for_recall": True}

    def search(self, payload: dict) -> dict:
        self.searched.append(payload)
        return {"topics": [], "explicit_memories": [{"memory_id": "m-1"}]}

    def health(self) -> bool:
        return True


class FakeNativeLoop:
    def __init__(self) -> None:
        self.initialized: list[tuple[str, str]] = []
        self.synced: list[tuple[str, str, str, str]] = []
        self.tool_calls: list[tuple[str, dict]] = []
        self.stopped = False

    def initialize(self, session_id: str, **kwargs) -> None:
        self.initialized.append((session_id, kwargs["agent_context"]))

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return f"memory for {session_id}: {query}"

    def sync_turn(
        self,
        user: str,
        assistant: str,
        *,
        session_id: str = "",
        run_id: str = "",
    ) -> bool:
        self.synced.append((user, assistant, session_id, run_id))
        return True

    def shutdown(self) -> None:
        self.stopped = True

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "memory_search",
                "description": "Search memory.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "memory_remember",
                "description": "Remember an explicit user request.",
                "parameters": {
                    "type": "object",
                    "properties": {"verbatim": {"type": "string"}},
                    "required": ["verbatim"],
                },
            },
        ]

    def handle_tool_call(self, name: str, args: dict) -> str:
        self.tool_calls.append((name, args))
        return json.dumps({"tool": name, "args": args})


def test_manifest_loads_memory_contract() -> None:
    manifest = ExtensionManifest.load(CORAX)
    instance = load_extension_instance(
        manifest,
        CORAX,
        kwargs={"client": FakeClient()},
    )
    assert isinstance(instance, MemoryProvider)
    assert manifest.kind.value == "memory_provider"
    assert not manifest.agent_callable
    assert "agent.memoryloop/v1" in manifest.interfaces


def test_write_requires_explicit_user_request() -> None:
    client = FakeClient()
    provider = MnemonicVaultProvider(client=client)
    denied = asyncio.run(provider.remember(MemoryRecord(content="secret")))
    accepted = asyncio.run(
        provider.remember(
            MemoryRecord(
                content="Production is local",
                scope={"type": "project", "id": "corax"},
                metadata={"explicit_user_request": True},
                idempotency_key="turn-1",
            )
        )
    )
    assert denied.status is ResultStatus.POLICY_DENIED
    assert accepted.status is ResultStatus.SUCCESS
    assert client.remembered[0]["scope"]["id"] == "corax"


def test_recall_maps_scopes_and_limit() -> None:
    client = FakeClient()
    provider = MnemonicVaultProvider(client=client)
    result = asyncio.run(
        provider.recall(
            MemoryQuery(
                "production",
                scopes=({"type": "project", "id": "corax"},),
                limit=3,
            )
        )
    )
    assert result.status is ResultStatus.SUCCESS
    assert client.searched[0]["max_topics"] == 3
    assert client.searched[0]["scope"]["id"] == "corax"


def test_native_loop_recalls_and_losslessly_captures_turn() -> None:
    native = FakeNativeLoop()
    provider = MnemonicVaultProvider(client=FakeClient(), native_loop=native)

    recalled = asyncio.run(
        provider.handle(
            ExtensionRequest(
                operation="before_turn",
                payload={"text": "previous choice"},
                session_id="chat-1",
            )
        )
    )
    captured = asyncio.run(
        provider.handle(
            ExtensionRequest(
                operation="after_turn",
                payload={
                    "user_text": "same text",
                    "assistant_text": "complete answer",
                    "scope": {"channel": "console", "turn_id": "turn-2"},
                },
                session_id="chat-1",
            )
        )
    )
    asyncio.run(provider.stop())

    assert recalled.payload["context"] == "memory for chat-1: previous choice"
    assert captured.payload["captured"] is True
    assert native.synced == [
        ("same text", "complete answer", "chat-1", "turn-2")
    ]
    assert native.stopped is True


def test_native_loop_keeps_explicit_memory_compatible() -> None:
    client = FakeClient()
    provider = MnemonicVaultProvider(
        client=client,
        native_loop=FakeNativeLoop(),
    )
    result = asyncio.run(
        provider.handle(
            ExtensionRequest(
                operation="after_turn",
                payload={
                    "user_text": "Запомни: production на .14",
                    "assistant_text": "Запомнил.",
                    "explicit": True,
                    "scope": {"channel": "console", "turn_id": "turn-3"},
                },
                session_id="chat-1",
            )
        )
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.payload["stored"] is True
    assert result.payload["captured"] is True
    assert client.remembered[0]["kind"] == "fact"
    assert client.remembered[0]["scope"] == {"type": "global"}


def test_corax_exposes_native_memory_tools_through_policy_proxies() -> None:
    native = FakeNativeLoop()
    provider = MnemonicVaultProvider(client=FakeClient(), native_loop=native)
    proxies = {proxy.id: proxy for proxy in provider.tool_proxies()}

    search = proxies["memory_search"]
    result = asyncio.run(
        search.execute(
            CapabilityRequest(
                task_id="task-1",
                session_id="chat-1",
                input={"query": "что сохранено"},
            )
        )
    )

    assert result.status is ResultStatus.SUCCESS
    assert result.payload["trust"] == "untrusted_historical_reference"
    assert result.payload["result"]["tool"] == "memory_search"
    assert search.routing["always_available"] is True
    assert search.permission_level is PermissionLevel.SAFE
    assert search.side_effects == {SideEffect.NONE}
    assert proxies["memory_remember"].permission_level is PermissionLevel.CONFIRM
    assert proxies["memory_remember"].side_effects == {SideEffect.MEMORY_WRITE}
    assert native.tool_calls == [
        ("memory_search", {"query": "что сохранено"})
    ]


def test_correction_turn_is_captured_without_explicit_memory() -> None:
    client = FakeClient()
    native = FakeNativeLoop()
    provider = MnemonicVaultProvider(client=client, native_loop=native)
    result = asyncio.run(
        provider.handle(
            ExtensionRequest(
                operation="after_turn",
                payload={
                    "user_text": "Не Alex, а Bob",
                    "assistant_text": "Исправил.",
                    "explicit": True,
                    "retraction_mode": True,
                    "scope": {"turn_id": "turn-4"},
                },
                session_id="chat-1",
            )
        )
    )

    assert result.payload == {
        "stored": False,
        "captured": True,
        "reason": "correction turn captured",
    }
    assert native.synced == [
        ("Не Alex, а Bob", "Исправил.", "chat-1", "turn-4")
    ]
    assert client.remembered == []
