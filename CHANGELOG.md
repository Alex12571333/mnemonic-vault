# Changelog

## 0.2.0 — 2026-07-15

- Add a native OpenClaw memory-slot plugin with six tools, four lifecycle hooks,
  bounded automatic recall, automatic turn capture, and a bundled skill.
- Add a native Hermes Agent `MemoryProvider` with bounded prefetch, a
  non-blocking ordered writer, six tools, and session lifecycle support.
- Add the shared `mnemonic-vault-memory` skill and integration documentation.
- Add Python integration tests and a Node 22 OpenClaw plugin CI job.
- Verify both integrations end to end against OpenClaw 2026.7.1 and Hermes
  Agent 0.17.0 on the `.14` deployment.
