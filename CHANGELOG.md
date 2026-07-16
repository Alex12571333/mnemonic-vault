# Changelog

## 0.3.2 — 2026-07-16

- Add a reversible `migrate-session-ids` command and portable alias manifest for
  legacy process-scoped OpenClaw and Hermes session folders.
- Derive adapter event IDs from stable turn identity and enforce their idempotency
  globally, including re-emission after an agent restart or session recovery.
- Persist finalized-session recovery redirects in the durable spool and route later
  messages and `session_end` events through the complete recovery chain.
- Parse explicit years and dates deterministically, constrain lexical/vector
  candidates by `session_started_at` before top-k ranking, and keep historical versions.

## 0.3.1 — 2026-07-15

- Keep one logical OpenClaw or Hermes chat in one Vault session across agent restarts
  using a persistent `agent_instance_id` instead of process/time identity.
- Classify spool failures, quarantine permanent poison events in fsync-backed
  dead-letter files, stop on authentication errors, and recover finalized sessions.
- Apply absolute lexical/vector relevance gates when the summarizer chooses existing
  topic summaries, while retaining the compact topic-card overview.
- Preserve older near-duplicate topics for explicitly historical queries.
- Replace the unsafe characters/4 fallback with conservative characters/2.5 budgets
  and make oversized summarizer inputs obey the computed upper bound.
- Enforce the API body limit against bytes actually received, including chunked bodies.

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
