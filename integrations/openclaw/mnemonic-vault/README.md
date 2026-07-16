# Mnemonic Vault for OpenClaw

Native OpenClaw memory-slot plugin. It records user/assistant turns through
lifecycle hooks, injects bounded topic summaries before a turn, exposes
source-expansion and global-timeline tools, and bundles the `mnemonic-vault-memory`
skill. Captured
events are `fsync`ed to a persistent spool before delivery and replayed with
stable event IDs after OpenClaw or Vault restarts.

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
