# Native Corax memory provider

This package implements both `agent.memory/v1` and `agent.memoryloop/v1`.
Corax therefore uses the provider's native bounded recall and lossless turn
capture instead of its generic memory loop. Other memory providers continue to
use the generic loop. The provider also exposes the same eight memory tools as
OpenClaw and Hermes through Corax's normal routing, policy, tracing, and UI path.

Set `MNEMONIC_VAULT_URL` and `MNEMONIC_VAULT_API_TOKEN`, configure the
extension path as `../mnemonic-vault/integrations/corax`, and bind
`extensions.bindings.memory` to `memory.mnemonic-vault`.

Every completed user/assistant turn is first `fsync`ed to
`$CORAX_DATA_PATH/mnemonic-vault/spool/corax.jsonl` and delivered in order by a
background worker. Recall remains bounded and fail-soft. A stable Corax turn ID
makes retries idempotent while allowing identical text in different turns.

Explicit writes still require
`MemoryRecord.metadata["explicit_user_request"] = true`; `/memory remember …`
uses that path. Ordinary turns, including “remember/запомни” text, stay in the
immutable transcript for background processing. Deletion remains unavailable
because the archive is append-only; corrections use `supersedes`.
