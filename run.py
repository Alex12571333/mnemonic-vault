from __future__ import annotations

import argparse
import json
import logging
import os
import ipaddress
from pathlib import Path

from app.config import AppConfig
from app.models import utc_or_local_now
from app.evaluation import evaluate_retrieval
from app.service import build_services
from app.session_aliases import migrate_session_ids
from app.storage import (
    atomic_write_json,
    estimate_tokens,
    exclusive_lock,
    read_messages,
    read_session,
    write_session,
)


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

    retry_failed = commands.add_parser(
        "retry-failed", help="return failed summary jobs to the pending queue"
    )
    retry_failed.add_argument("--session")

    evaluate = commands.add_parser(
        "evaluate-retrieval", help="measure recall@k and negative rejection on JSONL"
    )
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)

    resummarize = commands.add_parser(
        "resummarize", help="recreate derived topic summaries for one session"
    )
    resummarize.add_argument("--session", required=True)
    resummarize.add_argument(
        "--enqueue-only", action="store_true", help="do not call the Memory LLM now"
    )
    migrate = commands.add_parser(
        "migrate-session-ids",
        help="create portable aliases from legacy process-scoped session folders",
    )
    migrate.add_argument(
        "--agent-instance",
        action="append",
        default=[],
        metavar="AGENT=INSTANCE",
        help="stable installation identity override; may be repeated",
    )
    migrate.add_argument(
        "--dry-run", action="store_true", help="report aliases without writing them"
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
        config = AppConfig.load(args.config)
        validate_bind_security(args.host, config)
        uvicorn.run(
            create_app(args.config),
            host=args.host,
            port=args.port,
            reload=False,
        )
        return 0

    config = AppConfig.load(args.config)
    if args.command == "migrate-session-ids":
        instances: dict[str, str] = {}
        for raw in args.agent_instance:
            if "=" not in raw:
                raise SystemExit("--agent-instance must use AGENT=INSTANCE")
            agent, instance = (part.strip() for part in raw.split("=", 1))
            if not agent or not instance:
                raise SystemExit("--agent-instance must use non-empty AGENT=INSTANCE")
            instances[agent] = instance
        print(
            json.dumps(
                migrate_session_ids(config, instances, dry_run=args.dry_run),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    services = build_services(config)
    if args.command == "rebuild-index":
        print(json.dumps(services.indexer.rebuild(args.with_embeddings), ensure_ascii=False))
        return 0
    if args.command == "reembed-all":
        print(json.dumps(services.indexer.reembed_all(), ensure_ascii=False))
        return 0
    if args.command == "retry-failed":
        count = services.catalog.retry_failed_jobs(
            utc_or_local_now(), args.session
        )
        print(json.dumps({"retried_jobs": count}, ensure_ascii=False))
        return 0
    if args.command == "evaluate-retrieval":
        print(json.dumps(
            evaluate_retrieval(services.retriever, args.dataset, args.top_k),
            ensure_ascii=False,
            indent=2,
        ))
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


def validate_bind_security(host: str, config: AppConfig) -> None:
    """Refuse network exposure unless bearer authentication is configured."""
    loopback_names = {"localhost", "ip6-localhost"}
    is_loopback = host.lower() in loopback_names
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
    token = os.environ.get(config.api.bearer_token_env, "")
    if not is_loopback and not token:
        raise SystemExit(
            f"Refusing to bind {host} without {config.api.bearer_token_env}; "
            "set a bearer token or use 127.0.0.1"
        )


def prepare_resummarize(services, session_id: str) -> None:
    path = services.recorder.locate(session_id)
    with exclusive_lock(path / ".summary.lock"):
        with exclusive_lock(path / ".session.lock"):
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
