# Corax memory provider

This package implements `agent.memory/v1` and is loaded as a runtime-only
`memory_provider`. It is never exposed to the model as a tool.

Set `MNEMONIC_VAULT_URL` and `MNEMONIC_VAULT_API_TOKEN`, configure the
extension path as `../mnemonic-vault/integrations/corax`, and bind
`extensions.bindings.memory` to `memory.mnemonic-vault`.

Writes require `MemoryRecord.metadata["explicit_user_request"] = true`; this
preserves Mnemonic Vault's user-authorization rule. Deletion remains unavailable
because the archive is append-only; corrections use `supersedes`.
