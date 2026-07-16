"""Append-only, crash-recoverable delivery spool for Hermes turns."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class DurableSpool:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.dead_letter_path = (
            self.path.with_name(f"{self.path.name[:-6]}.dead-letter.jsonl")
            if self.path.name.endswith(".jsonl")
            else self.path.with_name(f"{self.path.name}.dead-letter.jsonl")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        if created:
            _fsync_directory(self.path.parent)
        dead_letter_created = not self.dead_letter_path.exists()
        descriptor = os.open(
            self.dead_letter_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
        )
        os.close(descriptor)
        try:
            os.chmod(self.dead_letter_path, 0o600)
        except OSError:
            pass
        if dead_letter_created:
            _fsync_directory(self.path.parent)
        self._thread_lock = threading.Lock()
        self._ack_count = 0

    def append(self, event: dict[str, Any]) -> str:
        event_id = str(event.get("event_id") or uuid.uuid4().hex)
        record = {"record": "event", "event_id": event_id, **event}
        record["event_id"] = event_id
        self._append_record(record)
        return event_id

    def acknowledge(self, event_id: str) -> None:
        self._append_record({"record": "delivered", "event_id": event_id})
        self._ack_count += 1
        if self._ack_count >= 256:
            self.compact()

    def record_redirect(self, original_session_id: str, recovery_session_id: str) -> None:
        if self.redirect_for(original_session_id) == recovery_session_id:
            return
        self._append_record(
            {
                "record": "redirect",
                "original_session_id": original_session_id,
                "recovery_session_id": recovery_session_id,
            }
        )

    def redirect_for(self, session_id: str) -> str | None:
        with self._locked():
            _, redirects = self._scan_unlocked()
            return redirects.get(session_id)

    def redirects(self) -> dict[str, str]:
        with self._locked():
            _, redirects = self._scan_unlocked()
            return dict(redirects)

    def dead_letter(
        self, event: dict[str, Any], reason: str, status: int | None = None
    ) -> None:
        record = {
            "record": "dead-letter",
            "failed_at": _utc_now(),
            "reason": reason,
            **({"status": status} if status is not None else {}),
            "event": event,
        }
        self._append_to(self.dead_letter_path, record)
        self.acknowledge(str(event["event_id"]))

    def dead_letters(self) -> list[dict[str, Any]]:
        with self._locked():
            return [
                json.loads(line)
                for line in self.dead_letter_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def pending(self) -> list[dict[str, Any]]:
        with self._locked():
            pending, _ = self._scan_unlocked()
            return list(pending.values())

    def compact(self) -> None:
        with self._locked():
            pending, redirects = self._scan_unlocked()
            fd, temporary = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            temp_path = Path(temporary)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                    for original, recovery in redirects.items():
                        stream.write(
                            json.dumps(
                                {
                                    "record": "redirect",
                                    "original_session_id": original,
                                    "recovery_session_id": recovery,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    for event in pending.values():
                        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_path, self.path)
                _fsync_directory(self.path.parent)
                self._ack_count = 0
            finally:
                temp_path.unlink(missing_ok=True)

    def _append_record(self, record: dict[str, Any]) -> None:
        self._append_to(self.path, record)

    def _append_to(self, path: Path, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, ensure_ascii=False) + "\n"
        with self._locked():
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())

    def _scan_unlocked(
        self,
    ) -> tuple[OrderedDict[str, dict[str, Any]], OrderedDict[str, str]]:
        events: OrderedDict[str, dict[str, Any]] = OrderedDict()
        redirects: OrderedDict[str, str] = OrderedDict()
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                event_id = "corrupt-" + hashlib.sha256(
                    f"{index + 1}:{line}".encode()
                ).hexdigest()[:24]
                events[event_id] = {
                    "record": "event",
                    "event_id": event_id,
                    "kind": "corrupt",
                    "session_id": "",
                    "external_session_id": "",
                    "agent": "",
                    "metadata": {
                        "raw_record": line,
                        "line_number": index + 1,
                    },
                }
                continue
            if record.get("record") == "redirect":
                original = str(record.get("original_session_id", ""))
                recovery = str(record.get("recovery_session_id", ""))
                if original and recovery:
                    redirects[original] = recovery
                continue
            event_id = str(record.get("event_id", ""))
            if not event_id:
                continue
            if record.get("record") == "event":
                events[event_id] = record
            elif record.get("record") == "delivered":
                events.pop(event_id, None)
        return events, redirects

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            lock_path = self.path.with_suffix(self.path.suffix + ".lock")
            with lock_path.open("a+", encoding="utf-8") as stream:
                try:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                    yield
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    yield


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
