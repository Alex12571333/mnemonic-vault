# Mnemonic Vault for Hermes Agent

Native `MemoryProvider` for Hermes Agent 0.17.0 or newer. It captures completed
turns through a non-blocking worker queue, prefetches bounded memory context, and
exposes the complete Mnemonic Vault tool set.

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
