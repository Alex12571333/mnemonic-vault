from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .config import AppConfig
from .models import utc_or_local_now
from .storage import (
    atomic_write_json,
    read_json,
    read_messages,
    read_session,
    validate_id,
)


DEFAULT_AGENT_INSTANCES = {
    "openclaw": "openclaw-main",
    "hermes": "hermes-main",
}


def stable_session_id(
    external_session_id: str, agent: str, agent_instance_id: str
) -> str:
    """Match the stable session-ID algorithm used by both native adapters."""
    digest = hashlib.sha256(
        f"{agent_instance_id}:{external_session_id}".encode("utf-8")
    ).hexdigest()[:24]
    safe_agent = re.sub(r"[^a-z0-9_-]+", "-", agent, flags=re.IGNORECASE).lower()
    return f"session-{safe_agent}-{digest}"


class SessionAliasStore:
    """Portable, file-backed logical-session aliases.

    The manifest never merges or renumbers source turns. It only groups immutable
    session folders under the stable ID that future adapter writes use.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def aliases(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        value = read_json(self.path)
        raw_aliases = value.get("aliases", [])
        if not isinstance(raw_aliases, list):
            raise ValueError(f"invalid session alias manifest: {self.path}")
        result: list[dict[str, Any]] = []
        for raw in raw_aliases:
            if not isinstance(raw, dict):
                raise ValueError(f"invalid session alias entry: {raw!r}")
            canonical = validate_id(
                str(raw.get("canonical_session_id", "")), "canonical session id"
            )
            members = [
                validate_id(str(item), "session alias member")
                for item in raw.get("member_session_ids", [])
            ]
            result.append(
                {
                    "canonical_session_id": canonical,
                    "agent": str(raw.get("agent", "")),
                    "agent_instance_id": str(raw.get("agent_instance_id", "")),
                    "external_session_id": str(raw.get("external_session_id", "")),
                    "member_session_ids": list(dict.fromkeys(members)),
                }
            )
        return result

    def describe(self, session_id: str) -> dict[str, Any] | None:
        validate_id(session_id, "session id")
        for alias in self.aliases():
            if session_id == alias["canonical_session_id"] or session_id in alias[
                "member_session_ids"
            ]:
                return alias
        return None

    def members(self, session_id: str) -> list[str]:
        alias = self.describe(session_id)
        if alias is None:
            return [validate_id(session_id, "session id")]
        return list(
            dict.fromkeys(
                [alias["canonical_session_id"], *alias["member_session_ids"]]
            )
        )

    def write(self, aliases: list[dict[str, Any]]) -> None:
        atomic_write_json(
            self.path,
            {
                "version": 1,
                "generated_at": utc_or_local_now(),
                "aliases": aliases,
            },
        )


def migrate_session_ids(
    config: AppConfig,
    agent_instances: Mapping[str, str] | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Discover legacy process-scoped sessions and write reversible aliases."""
    configured_instances = {
        **DEFAULT_AGENT_INSTANCES,
        **{str(key): str(value) for key, value in (agent_instances or {}).items()},
    }
    grouped: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    skipped: list[dict[str, str]] = []
    scanned = 0

    for session_file in sorted(config.storage.sessions_dir.glob("*/*/*/session.json")):
        scanned += 1
        session = read_session(session_file.parent)
        external_ids: set[str] = set()
        instance_ids: set[str] = set()
        for message in read_messages(session_file.parent / "transcript.jsonl"):
            external = message.metadata.get("external_session_id")
            if isinstance(external, str) and external.strip():
                external_ids.add(external.strip())
            instance = message.metadata.get("agent_instance_id")
            if isinstance(instance, str) and instance.strip():
                instance_ids.add(instance.strip())

        if len(external_ids) != 1:
            reason = (
                "missing external_session_id"
                if not external_ids
                else "conflicting external_session_id"
            )
            skipped.append({"session_id": session.id, "reason": reason})
            continue
        if len(instance_ids) > 1:
            skipped.append(
                {"session_id": session.id, "reason": "conflicting agent_instance_id"}
            )
            continue
        instance_id = next(
            iter(instance_ids), configured_instances.get(session.agent, "")
        )
        if not instance_id:
            skipped.append(
                {
                    "session_id": session.id,
                    "reason": f"no stable instance configured for agent {session.agent}",
                }
            )
            continue
        external_id = next(iter(external_ids))
        grouped[(session.agent, instance_id, external_id)].append(
            (session.started_at, session.id)
        )

    aliases: list[dict[str, Any]] = []
    for (agent, instance_id, external_id), dated_members in sorted(grouped.items()):
        canonical = stable_session_id(external_id, agent, instance_id)
        members = [item[1] for item in sorted(dated_members)]
        if members == [canonical]:
            continue
        aliases.append(
            {
                "canonical_session_id": canonical,
                "agent": agent,
                "agent_instance_id": instance_id,
                "external_session_id": external_id,
                "member_session_ids": members,
            }
        )

    store = SessionAliasStore(config.storage.root / "session-aliases.json")
    if not dry_run:
        store.write(aliases)
    return {
        "scanned_sessions": scanned,
        "alias_groups": len(aliases),
        "aliased_sessions": sum(len(item["member_session_ids"]) for item in aliases),
        "skipped_sessions": len(skipped),
        "skipped": skipped,
        "dry_run": dry_run,
        "path": str(store.path),
        "aliases": aliases,
    }
