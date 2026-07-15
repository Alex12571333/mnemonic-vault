from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import Message, Session, SourceRange, Topic


SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def validate_id(value: str, kind: str = "id") -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except ImportError:
            yield


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_session(path: Path) -> Session:
    return Session.from_dict(read_json(path / "session.json"))


def write_session(path: Path, session: Session) -> None:
    atomic_write_json(path / "session.json", session.to_dict())


def append_message(path: Path, message: Message) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_messages(
    transcript: Path, from_message: int = 1, to_message: int | None = None
) -> list[Message]:
    result: list[Message] = []
    if not transcript.exists():
        return result
    with transcript.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {transcript}:{line_number}") from exc
            message_id = int(raw["id"])
            if message_id < from_message:
                continue
            if to_message is not None and message_id > to_message:
                break
            result.append(
                Message(
                    id=message_id,
                    role=str(raw["role"]),
                    text=str(raw.get("text", raw.get("content", ""))),
                    created_at=str(raw["created_at"]),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
    return result


def read_last_message(transcript: Path) -> Message | None:
    if not transcript.exists() or transcript.stat().st_size == 0:
        return None
    with transcript.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        buffer = b""
        while position > 0:
            size = min(4096, position)
            position -= size
            stream.seek(position)
            buffer = stream.read(size) + buffer
            lines = [line for line in buffer.splitlines() if line.strip()]
            if len(lines) >= 2 or position == 0:
                if not lines:
                    return None
                raw = json.loads(lines[-1].decode("utf-8"))
                return Message(
                    id=int(raw["id"]),
                    role=str(raw["role"]),
                    text=str(raw.get("text", raw.get("content", ""))),
                    created_at=str(raw["created_at"]),
                    metadata=dict(raw.get("metadata", {})),
                )
    return None


def merge_ranges(ranges: list[SourceRange]) -> list[SourceRange]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: (item.session_id, item.from_message, item.to_message))
    result = [ordered[0]]
    for current in ordered[1:]:
        previous = result[-1]
        if (
            previous.session_id == current.session_id
            and current.from_message <= previous.to_message + 1
        ):
            previous.to_message = max(previous.to_message, current.to_message)
        else:
            result.append(current)
    return result


def topic_to_markdown(topic: Topic) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write topic files") from exc
    metadata = {
        "id": topic.id,
        "session_id": topic.session_id,
        "title": topic.title,
        "description": topic.description,
        "problem": topic.problem,
        "status": topic.status,
        "keywords": topic.keywords,
        "source_ranges": [item.to_pair() for item in topic.source_ranges],
        "created_at": topic.created_at,
        "updated_at": topic.updated_at,
    }
    frontmatter = yaml.safe_dump(
        metadata, allow_unicode=True, sort_keys=False, default_flow_style=False
    ).strip()
    return f"---\n{frontmatter}\n---\n{topic.summary.strip()}\n"


def read_topic(path: Path) -> Topic:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        raise ValueError(f"topic has no YAML front matter: {path}")
    try:
        _, metadata_text, summary = content.split("---\n", 2)
    except ValueError as exc:
        raise ValueError(f"malformed topic file: {path}") from exc
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read topic files") from exc
    metadata = yaml.safe_load(metadata_text) or {}
    session_id = str(metadata["session_id"])
    ranges = [
        SourceRange(session_id=session_id, from_message=int(pair[0]), to_message=int(pair[1]))
        for pair in metadata.get("source_ranges", [])
    ]
    return Topic(
        id=str(metadata["id"]),
        session_id=session_id,
        title=str(metadata.get("title", "")),
        description=str(metadata.get("description", "")),
        problem=str(metadata.get("problem", "")),
        status=str(metadata.get("status", "active")),
        keywords=[str(item) for item in metadata.get("keywords", [])],
        source_ranges=ranges,
        created_at=str(metadata.get("created_at", "")),
        updated_at=str(metadata.get("updated_at", "")),
        summary=summary.strip(),
        path=str(path),
    )


def write_topic(path: Path, topic: Topic) -> None:
    atomic_write_text(path, topic_to_markdown(topic))


def estimate_tokens(text: str) -> int:
    # Conservative tokenizer-independent approximation for trigger/budget handling.
    return max(1, (len(text) + 3) // 4)
