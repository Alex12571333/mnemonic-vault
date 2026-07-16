# Mnemonic Vault for OpenClaw

Native OpenClaw memory-slot plugin. It records user/assistant turns through
lifecycle hooks, injects bounded topic summaries before a turn, exposes
source-expansion and token-bounded latest-snapshot/timeline tools, and bundles the `mnemonic-vault-memory`
skill. Captured
events are `fsync`ed to a persistent spool before delivery and replayed with
stable event IDs after OpenClaw or Vault restarts.

The plugin exposes `memory_remember` for direct user requests and a guaranteed
LLM-bypass command:

```text
/remember project:mnemonic-vault Production runs on 192.168.0.14
```

Without an explicit scope selector, `/remember` uses `global`, so other agents can
recall it with normal semantic search. Use agent/project/session selectors only for
context-specific facts.

Requires OpenClaw 2026.7.1 or newer and a reachable Mnemonic Vault API.

```bash
npm ci
npm run check
openclaw plugins install -l .
openclaw plugins enable mnemonic-vault
openclaw config set plugins.slots.memory mnemonic-vault
openclaw config set plugins.entries.mnemonic-vault.config.baseUrl http://127.0.0.1:8765
openclaw config set plugins.entries.mnemonic-vault.hooks.allowConversationAccess true --strict-json
openclaw gateway restart
openclaw plugins inspect mnemonic-vault --runtime --json
```

The default spool is `~/mnemonic-vault/data/spool/openclaw.jsonl` for an
installed plugin. Override it with the plugin's `spoolPath` setting or set
`MNEMONIC_VAULT_PROJECT_ROOT`. For an authenticated LAN endpoint, export
`MNEMONIC_VAULT_API_TOKEN`; `apiTokenEnv` can point to another environment
variable name.

`agentInstanceId` defaults to `openclaw-main`. Keep it stable across restarts and use
a distinct value for each independent OpenClaw installation. Permanent delivery
failures are preserved in `openclaw.dead-letter.jsonl`; retryable failures remain in
the primary spool.

Recall searches the shared archive with `scope_mode=boost`. The plugin always sends
its stable agent ID and current Vault session as context, and also sends `projectId`
when configured. Other agents and projects remain visible. Use tool
`scope_mode=strict` only for an explicitly requested scope; use
`include_all_scopes=true` for broad historical search without session downranking.
