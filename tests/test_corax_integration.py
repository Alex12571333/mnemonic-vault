from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agent_core import MemoryProvider, MemoryQuery, MemoryRecord, ResultStatus
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
