from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.config import AppConfig
from app.models import utc_or_local_now
from app.service import build_services
from app.storage import atomic_write_json, estimate_tokens, read_messages, read_session, write_session


PROJECT_ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Portable file-first long-term memory")
    result.add_argument("--config", default=str(PROJECT_ROOT / "config/config.yaml"))
    result.add_argument("--verbose", action="store_true")
    commands = result.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the FastAPI service and job worker")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    rebuild = commands.add_parser("rebuild-index", help="recreate catalog.sqlite from files")
    rebuild.add_argument("--with-embeddings", action="store_true")

    process = commands.add_parser("process-jobs", help="process pending summary jobs")
    process.add_argument("--once", action="store_true")

    commands.add_parser("reembed-all", help="rebuild all topic embeddings")

    resummarize = commands.add_parser(
        "resummarize", help="recreate derived topic summaries for one session"
    )
    resummarize.add_argument("--session", required=True)
    resummarize.add_argument(
        "--enqueue-only", action="store_true", help="do not call the Memory LLM now"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "serve":
        try:
            import uvicorn
            from app.api import create_app
        except ImportError as exc:
            raise SystemExit("Install requirements.txt before running the API") from exc
        uvicorn.run(
            create_app(args.config),
            host=args.host,
            port=args.port,
            reload=False,
        )
        return 0

    config = AppConfig.load(args.config)
    services = build_services(config)
    if args.command == "rebuild-index":
        print(json.dumps(services.indexer.rebuild(args.with_embeddings), ensure_ascii=False))
        return 0
    if args.command == "reembed-all":
        print(json.dumps(services.indexer.reembed_all(), ensure_ascii=False))
        return 0
    if args.command == "process-jobs":
        services.job_runner.recover()
        count = 0
        while True:
            value = services.job_runner.run_once()
            if value is None:
                break
            count += 1
            print(json.dumps(value, ensure_ascii=False))
            if args.once:
                break
        print(json.dumps({"processed_jobs": count}, ensure_ascii=False))
        return 0
    if args.command == "resummarize":
        prepare_resummarize(services, args.session)
        if not args.enqueue_only:
            while services.job_runner.run_once() is not None:
                pass
        return 0
    return 2


def prepare_resummarize(services, session_id: str) -> None:
    path = services.recorder.locate(session_id)
    session = read_session(path)
    services.catalog.delete_session_jobs(session_id)
    services.catalog.delete_session_topics(session_id)
    for topic_path in (path / "topics").glob("*.md"):
        topic_path.unlink()
    session.processed_until_message = 0
    session.new_token_estimate = sum(
        estimate_tokens(message.text)
        for message in read_messages(path / "transcript.jsonl")
    )
    session.status = "finalizing" if session.ended_at else "active"
    write_session(path, session)
    atomic_write_json(
        path / "index.json",
        {"session_id": session.id, "overview": "", "topics": []},
    )
    services.catalog.upsert_session(session, path)
    if session.message_count:
        services.catalog.enqueue_job(
            session.id, 1, session.message_count, utc_or_local_now()
        )


if __name__ == "__main__":
    raise SystemExit(main())
