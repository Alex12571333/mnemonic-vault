# Mnemonic Vault for OpenClaw

Native OpenClaw memory-slot plugin. It records user/assistant turns through lifecycle hooks, injects bounded topic summaries before a turn, exposes source-expansion tools, and bundles the `mnemonic-vault-memory` skill.

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
