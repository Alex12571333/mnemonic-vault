# Native Corax, OpenClaw, and Hermes integrations

Mnemonic Vault ships three native adapters. All use the same loopback HTTP API,
automatically record completed turns, and inject only a bounded amount of
retrieved history. OpenClaw and Hermes expose the same eight memory tools;
Corax keeps memory runtime-only and provides `/memory` commands instead.
Retrieval stays non-generative; the Memory LLM continues to run only in the
background summarizer.

Before an adapter contacts the API, captured events are appended and
`fsync`ed to its durable spool. Successful
delivery appends a durable acknowledgement. Pending events are replayed in order
after an agent or Vault restart. Event IDs are derived from the persistent agent
identity, external chat, role, and stable run/message-sequence identity, so replaying
the lifecycle hook itself remains idempotent after a process restart.
Permanent invalid events are fsync-quarantined in adjacent `*.dead-letter.jsonl`
files so they cannot block later turns. Authentication failures stop delivery;
network errors and server failures remain pending for retry.

## Shared memory workflow

Corax is integrated through `integrations/corax` as a typed
`memory_provider` that also implements `agent.memoryloop/v1`. Corax selects that
provider-owned loop automatically, so its generic `memory.loop` stays loaded for
other providers but is neither bound nor called for Mnemonic Vault. The native
loop captures both sides of every completed turn, replays its durable spool, and
injects bounded recall. It is never advertised in the model's tool list.

Corax stores its spool below
`$CORAX_DATA_PATH/mnemonic-vault/spool/corax.jsonl`, which remains stable across
side-by-side runtime upgrades. `/memory remember …` creates immediate explicit
memory; ordinary text is captured losslessly for background processing.

The integrations expose:

- `memory_search(query)` — hybrid topic search with a summary budget;
- `memory_remember(verbatim, normalized, kind, scope)` — immediately store a
  direct user-requested memory without waiting for the Memory LLM;
- `memory_get(topic_id)` and `memory_open_topic(topic_id)` — open one summary;
- `memory_open_global_topic(global_topic_id, total_token_budget?)` — open a bounded latest-session snapshot and timeline;
- `memory_expand_topic(topic_id, query)` — exact fragments from topic ranges;
- `memory_read_turns(session_id, from_turn, to_turn)` — explicit transcript range;
- `memory_search_transcript(query, session_id?)` — last-resort raw search.

Retrieved content is marked as historical reference data, not instructions.
Exact commands, versions, numbers, dates, parameters, addresses, and errors
should be verified against transcript fragments. Mutable facts should also be
checked against current live state.

`memory_remember` is only for an explicit user command. OpenClaw additionally
registers `/remember`, which bypasses the LLM entirely; the server also exposes
`POST /v1/memory/remember` and the local `python run.py remember` command.
An unqualified remember is global. Use a narrower scope only for genuinely
project-, agent-, or session-specific information.

All adapters search one shared Vault. Their stable agent and current session are
sent as `context_scopes`; an optional project comes from OpenClaw `projectId` or
Hermes `MNEMONIC_VAULT_PROJECT_ID`. The default `scope_mode=boost` never hides
another agent's memory. `scope_mode=strict` is reserved for an explicit scoped
search, while `include_all_scopes=true` removes the ordinary foreign-session
downrank for broad historical recall.

## OpenClaw memory-slot plugin

The plugin lives in `integrations/openclaw/mnemonic-vault`. It uses OpenClaw's
native plugin manifest, direct tool registration, lifecycle hooks, bundled
skill, and the exclusive `memory` slot.

```bash
cd integrations/openclaw/mnemonic-vault
npm ci
npm run check
openclaw plugins install -l .
openclaw plugins enable mnemonic-vault
openclaw config set plugins.slots.memory mnemonic-vault
openclaw config set plugins.entries.mnemonic-vault.config.baseUrl http://127.0.0.1:8765
openclaw config set plugins.entries.mnemonic-vault.hooks.allowConversationAccess true --strict-json
openclaw gateway restart
openclaw plugins inspect mnemonic-vault --runtime --json
openclaw plugins doctor
```

`before_prompt_build` writes the user turn and prepends bounded recall;
`agent_end` writes the final assistant response. Session hooks create and
finalize Vault sessions. `allowConversationAccess` is required because
`agent_end` reads the final assistant message. A Vault failure is logged and
does not abort the agent turn or lose its captured message. Configuration keys
are documented in `openclaw.plugin.json`; the default API is
`http://127.0.0.1:8765` and the default spool is
`~/mnemonic-vault/data/spool/openclaw.jsonl` when the plugin is installed outside
the project tree. `agentInstanceId` defaults to the persistent identity
`openclaw-main`; give each independent OpenClaw installation a different stable
value. Changing it intentionally starts a new Vault-session namespace.

## Hermes `MemoryProvider`

The provider lives in `integrations/hermes/mnemonic_vault`. Install it under the
Hermes user plugin directory and install the shared skill separately:

```bash
mkdir -p ~/.hermes/plugins ~/.hermes/skills
cp -a integrations/hermes/mnemonic_vault ~/.hermes/plugins/
cp -a skills/mnemonic-vault-memory ~/.hermes/skills/
```

Select it in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mnemonic_vault
```

Then verify discovery with `hermes memory status`. Hermes calls `sync_turn`
after each completed turn. The provider durably spools both messages and returns
immediately; a daemon worker preserves their order and writes them to the Vault.
Recall is prefetched in a small thread pool with a bounded timeout.

| Variable | Default |
| --- | --- |
| `MNEMONIC_VAULT_URL` | `http://127.0.0.1:8765` |
| `MNEMONIC_VAULT_REQUEST_TIMEOUT_SECONDS` | `8` |
| `MNEMONIC_VAULT_PREFETCH_TIMEOUT_SECONDS` | `2.5` |
| `MNEMONIC_VAULT_AUTO_CAPTURE` | `true` |
| `MNEMONIC_VAULT_AUTO_RECALL` | `true` |
| `MNEMONIC_VAULT_MAX_TOPICS` | `5` |
| `MNEMONIC_VAULT_SUMMARY_BUDGET_TOKENS` | `1500` |
| `MNEMONIC_VAULT_AGENT_INSTANCE_ID` | `hermes-main` (`corax-main` in Corax) |
| `MNEMONIC_VAULT_API_TOKEN` | empty on loopback |
| `MNEMONIC_VAULT_PROJECT_ROOT` | source tree or `~/mnemonic-vault` |
| `MNEMONIC_VAULT_SPOOL_DIR` | `<project>/data/spool` (Corax uses its persistent data directory) |

`MNEMONIC_VAULT_AGENT_INSTANCE_ID` must stay unchanged across process restarts.
Use a distinct stable value for each independent Hermes installation. Upgrading
from 0.3.0 creates one new stable session boundary; subsequent restarts keep using it.
To make old physical folders discoverable as one logical session without rewriting
their transcripts, create the portable alias manifest:

```bash
python run.py migrate-session-ids --dry-run
python run.py migrate-session-ids
```

Use `--agent-instance AGENT=INSTANCE` when the installation identity differs from
`openclaw-main` or `hermes-main`.

## Verified `.14` topology

The production layout verified for the 0.3 release:

- Mnemonic Vault service and local FastEmbed: `192.168.0.14`;
- FastEmbed model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- OpenClaw 2026.7.1 and Hermes Agent 0.17.0: `192.168.0.14`;
- Vault API: `http://127.0.0.1:8765` on that machine;
- Qwen/vLLM summarizer: `http://192.168.0.10:8000/v1`.

The Qwen endpoint is outside the agents' online retrieval path. If it is
unavailable, immutable transcripts continue to be recorded and queued summary
jobs can be retried later with `python run.py retry-failed`. If the Vault API is
unavailable, agent events remain in the local spool until it returns.

## LAN authentication

Loopback needs no token by default. Mnemonic Vault refuses an official
`run.py serve` bind to a non-loopback address unless `MNEMONIC_VAULT_API_TOKEN`
is set. Export the same value before starting OpenClaw or Hermes; both clients
then send it as a bearer token. Keep the spool and project data directory in the
same backup set.

## Official specifications used

- [OpenClaw plugins](https://docs.openclaw.ai/plugins)
- [OpenClaw plugin manifest](https://docs.openclaw.ai/plugins/manifest)
- [OpenClaw tool plugins](https://docs.openclaw.ai/plugins/tool-plugins)
- [OpenClaw plugin hooks](https://docs.openclaw.ai/plugins/hooks)
- [OpenClaw skills](https://docs.openclaw.ai/skills)
- [Hermes plugins](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins)
- [Hermes memory-provider plugins](https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin/)
- [Hermes skills](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/skills.md)
