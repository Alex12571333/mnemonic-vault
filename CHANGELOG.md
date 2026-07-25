# Changelog

## Unreleased

- Add a typed Corax `memory_provider` adapter with explicit-write authorization.

## 0.5.1 — 2026-07-16

- Keep every agent, project, and session scope in one shared retrieval space;
  scopes are contextual ranking signals rather than access-control boundaries.
- Add default `scope_mode=boost`, optional explicit `scope_mode=strict`, multiple
  `context_scopes`, and `include_all_scopes` for unpenalized historical recall.
- Boost current project, session, agent, and global memories while strongly
  downranking—but never hiding—other sessions during automatic recall.
- Make unqualified `/remember` and tool calls global by default, and let OpenClaw
  and Hermes attach their stable agent, current session, and optional project IDs.

## 0.5.0 — 2026-07-16

- Add append-only `data/explicit-memory.jsonl` for direct user-authored memories,
  with SQLite materialization and immediate FTS recall independent of the Memory LLM.
- Add idempotent `memory_remember`, exact transcript-source validation, constrained
  kinds/scopes, append-only supersession history, and priority summary jobs.
- Include active explicit memories in bounded search context, boost matching scopes,
  and expose superseded facts only for deterministic historical queries.
- Rebuild explicit memory from files with `rebuild-index` and add its optional vectors
  to `reembed-all` without making embeddings part of the write commit path.
- Add the direct API/CLI remember path, the OpenClaw `/remember` LLM-bypass command,
  and `memory_remember` tools for OpenClaw and Hermes.

## 0.4.1 — 2026-07-16

- Choose the latest global-topic snapshot by `topic.updated_at` while retaining the
  original session-based identity anchor for stable global IDs.
- Build projections only from finalized sessions whose messages are fully processed.
- Replace union-find single-linkage clustering with deterministic complete-link
  assignment so similarity chains cannot merge unrelated topic endpoints.
- Label `current.md` as the latest session snapshot rather than a synthesized current
  state, and show both topic-update and session-start dates in the timeline.
- Enforce one `total_token_budget` across snapshot, timeline, sources, and metadata in
  the API and the OpenClaw/Hermes `memory_open_global_topic` tool.

## 0.4.0 — 2026-07-16

- Add an optional, rebuildable global-topic layer over immutable session summaries.
- Build conservative cross-session clusters without an LLM and only after the
  configured number of topic versions has accumulated.
- Generate atomic `current.md`, `timeline.md`, and `sources.json` projections while
  keeping every original topic Markdown file byte-for-byte unchanged.
- Expose global projection IDs in search cards and add bounded read APIs plus the
  `memory_open_global_topic` tool to OpenClaw and Hermes.
- Keep projection creation manual through `rebuild-global-topics`; the default
  threshold is 12 versions, so the new layer does not alter small archives.

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
