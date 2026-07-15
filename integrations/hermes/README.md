# Mnemonic Vault for Hermes Agent

Native `MemoryProvider` for Hermes Agent 0.17.0 or newer. It captures completed
turns through an `fsync`-backed persistent spool, prefetches bounded memory
context, and exposes the complete Mnemonic Vault tool set. Pending events replay
in order after Hermes or Vault restarts; API retries use stable event IDs.

```bash
mkdir -p ~/.hermes/plugins ~/.hermes/skills
cp -a integrations/hermes/mnemonic_vault ~/.hermes/plugins/
cp -a skills/mnemonic-vault-memory ~/.hermes/skills/
```

Set the active provider in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mnemonic_vault
```

The default endpoint is `http://127.0.0.1:8765`. Override it when needed:

```bash
export MNEMONIC_VAULT_URL=http://127.0.0.1:8765
hermes memory status
```

The default spool is `~/mnemonic-vault/data/spool/hermes.jsonl` for an installed
plugin. Set `MNEMONIC_VAULT_PROJECT_ROOT` or `MNEMONIC_VAULT_SPOOL_DIR` when the
portable project lives elsewhere. For an authenticated LAN endpoint, export the
same `MNEMONIC_VAULT_API_TOKEN` used by the Vault service.

`MNEMONIC_VAULT_AGENT_INSTANCE_ID` defaults to `hermes-main`. Keep it stable across
restarts and assign different values to independent Hermes installations. Permanent
delivery failures are preserved in `hermes.dead-letter.jsonl`; retryable failures stay
in the primary spool.
