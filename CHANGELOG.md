# Changelog

## 0.3.0 — 2026-07-15

- Add fsync-backed OpenClaw and Hermes delivery spools with restart replay.
- Add API idempotency through `session_id + external_event_id`.
- Add rebuildable `messages_fts` and remove archive-wide transcript scans.
- Gate candidates by absolute lexical/vector relevance before RRF.
- Bound summarizer cards, selected existing summaries, and new-message input separately.
- Stage and validate complete LLM operation batches before atomic topic/catalog commit.
- Add summary/session locks, failed-job retry commands, one total context budget,
  topic/session dates, and near-duplicate recency diversification.
- Add bearer authentication, safe LAN binding, payload limits, a retrieval evaluation
  command, and Python 3.11–3.13 CI coverage.
- Switch the verified local embedding profile to multilingual MiniLM and calibrate the
  absolute cosine gate against real Russian positive, paraphrase, and negative queries.

## 0.2.0 — 2026-07-15

- Add a native OpenClaw memory-slot plugin with six tools, four lifecycle hooks,
  bounded automatic recall, automatic turn capture, and a bundled skill.
- Add a native Hermes Agent `MemoryProvider` with bounded prefetch, a
  non-blocking ordered writer, six tools, and session lifecycle support.
- Add the shared `mnemonic-vault-memory` skill and integration documentation.
- Add Python integration tests and a Node 22 OpenClaw plugin CI job.
- Verify both integrations end to end against OpenClaw 2026.7.1 and Hermes
  Agent 0.17.0 on the `.14` deployment.
