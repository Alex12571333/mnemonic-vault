"""Corax ``MemoryProvider`` adapter for the Mnemonic Vault API."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_core import (
    CoreError,
    ErrorCode,
    HealthStatus,
    MemoryProvider,
    MemoryQuery,
    MemoryRecord,
    PermissionLevel,
    Result,
    RiskLevel,
)
from agent_sdk import memory_provider


class _Client(Protocol):
    def remember(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def search(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def health(self) -> bool: ...


class _HttpClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def remember(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/memory/remember", payload)

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/memory/search", payload)

    def health(self) -> bool:
        try:
            self._request("GET", "/health", None)
        except RuntimeError:
            return False
        return True

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            f"{self.base_url}{path}",
            data=(
                json.dumps(payload).encode("utf-8")
                if payload is not None
                else None
            ),
            method=method,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=8) as response:  # noqa: S310
                parsed = json.loads(response.read() or b"{}")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Mnemonic Vault HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Mnemonic Vault request failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Mnemonic Vault returned a non-object response")
        return parsed


@memory_provider(
    id="memory.mnemonic-vault",
    name="Mnemonic Vault",
    description="File-first long-term memory with bounded scoped recall.",
    version="0.6.0",
    tags=("memory", "file-first", "mnemonic-vault"),
    permission_level=PermissionLevel.CONFIRM,
    required_scopes=("memory.read", "memory.write", "network.outbound"),
    risk_level=RiskLevel.MEDIUM,
    side_effects=("network_request", "memory_write"),
    secrets=("MNEMONIC_VAULT_API_TOKEN",),
    config_schema={
        "type": "object",
        "properties": {
            "base_url": {"type": "string"},
        },
    },
    entrypoint="provider:MnemonicVaultProvider",
    min_core_version="0.2.0",
)
class MnemonicVaultProvider(MemoryProvider):
    """Map the Corax memory contract to Vault search and explicit remember."""

    def __init__(self, *, client: _Client | None = None) -> None:
        self._client = client or _HttpClient(
            os.getenv("MNEMONIC_VAULT_URL", "http://127.0.0.1:8765"),
            os.getenv("MNEMONIC_VAULT_API_TOKEN", ""),
        )

    async def remember(self, record: MemoryRecord) -> Result:
        if record.metadata.get("explicit_user_request") is not True:
            return Result.denied(
                "Mnemonic Vault accepts explicit user-directed memory only",
                session_id="",
                details={"required_metadata": "explicit_user_request=true"},
            )
        payload: dict[str, Any] = {
            "verbatim": record.content,
            "normalized": record.metadata.get("normalized", record.content),
            "kind": record.kind,
            "scope": record.scope or {"type": "global"},
        }
        for key in (
            "source_session_id",
            "source_message_id",
            "supersedes",
        ):
            if key in record.metadata:
                payload[key] = record.metadata[key]
        if record.idempotency_key:
            payload["idempotency_key"] = record.idempotency_key
        try:
            data = await asyncio.to_thread(self._client.remember, payload)
        except Exception as exc:  # noqa: BLE001
            return _failure(str(exc))
        return Result.ok(data, session_id="")

    async def recall(self, query: MemoryQuery) -> Result:
        payload: dict[str, Any] = {
            "query": query.text,
            "max_topics": query.limit,
            "summary_budget_tokens": int(
                query.metadata.get("summary_budget_tokens", 1500)
            ),
            "include_sources": str(query.metadata.get("include_sources", "auto")),
            "scope_mode": str(query.metadata.get("scope_mode", "boost")),
        }
        if query.scopes:
            payload["scope"] = query.scopes[0]
            payload["context_scopes"] = list(query.scopes)
        try:
            data = await asyncio.to_thread(self._client.search, payload)
        except Exception as exc:  # noqa: BLE001
            return _failure(str(exc))
        return Result.ok(data, session_id="")

    async def forget(self, memory_id: str, *, scope: dict | None = None) -> Result:
        return Result.fail(
            CoreError(
                ErrorCode.INVALID_INPUT,
                "Mnemonic Vault is append-only; supersede a memory instead of deleting it",
                {"memory_id": memory_id},
            ),
            session_id="",
        )

    async def health_check(self) -> HealthStatus:
        healthy = await asyncio.to_thread(self._client.health)
        return HealthStatus.HEALTHY if healthy else HealthStatus.DEGRADED


def _failure(message: str) -> Result:
    return Result.fail(
        CoreError(ErrorCode.CAPABILITY_FAILED, message),
        session_id="",
    )
