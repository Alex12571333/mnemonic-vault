import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { Type } from "typebox";
import { buildJsonPluginConfigSchema, definePluginEntry, } from "openclaw/plugin-sdk/plugin-entry";
import { extractLastAssistant, formatMemoryContext, deterministicEventId, recoverySessionId, VaultClient, VaultHttpError, vaultSessionId, } from "./client.js";
import { DurableSpool } from "./spool.js";
const DEFAULTS = {
    baseUrl: "http://127.0.0.1:8765",
    agent: "openclaw",
    agentInstanceId: "openclaw-main",
    autoCapture: true,
    autoRecall: true,
    maxTopics: 5,
    summaryBudgetTokens: 1_500,
    includeSources: "auto",
    requestTimeoutMs: 8_000,
    spoolPath: "",
    spoolFlushMs: 2_000,
    apiTokenEnv: "MNEMONIC_VAULT_API_TOKEN",
};
const configSchema = buildJsonPluginConfigSchema({
    type: "object",
    additionalProperties: false,
    properties: {
        baseUrl: { type: "string" },
        agent: { type: "string" },
        agentInstanceId: { type: "string", minLength: 1 },
        autoCapture: { type: "boolean" },
        autoRecall: { type: "boolean" },
        maxTopics: { type: "integer", minimum: 1, maximum: 50 },
        summaryBudgetTokens: { type: "integer", minimum: 100, maximum: 32_000 },
        includeSources: { type: "string", enum: ["auto", "always", "never"] },
        requestTimeoutMs: { type: "integer", minimum: 250, maximum: 60_000 },
        spoolPath: { type: "string" },
        spoolFlushMs: { type: "integer", minimum: 250, maximum: 60_000 },
        apiTokenEnv: { type: "string" },
    },
});
function resolveConfig(value) {
    const raw = value && typeof value === "object" ? value : {};
    return {
        baseUrl: typeof raw.baseUrl === "string" ? raw.baseUrl : DEFAULTS.baseUrl,
        agent: typeof raw.agent === "string" ? raw.agent : DEFAULTS.agent,
        agentInstanceId: typeof raw.agentInstanceId === "string" && raw.agentInstanceId.trim()
            ? raw.agentInstanceId.trim()
            : DEFAULTS.agentInstanceId,
        autoCapture: raw.autoCapture ?? DEFAULTS.autoCapture,
        autoRecall: raw.autoRecall ?? DEFAULTS.autoRecall,
        maxTopics: finiteInt(raw.maxTopics, DEFAULTS.maxTopics),
        summaryBudgetTokens: finiteInt(raw.summaryBudgetTokens, DEFAULTS.summaryBudgetTokens),
        includeSources: raw.includeSources === "always" || raw.includeSources === "never"
            ? raw.includeSources
            : "auto",
        requestTimeoutMs: finiteInt(raw.requestTimeoutMs, DEFAULTS.requestTimeoutMs),
        spoolPath: typeof raw.spoolPath === "string" ? raw.spoolPath : DEFAULTS.spoolPath,
        spoolFlushMs: finiteInt(raw.spoolFlushMs, DEFAULTS.spoolFlushMs),
        apiTokenEnv: typeof raw.apiTokenEnv === "string" ? raw.apiTokenEnv : DEFAULTS.apiTokenEnv,
    };
}
function finiteInt(value, fallback) {
    return typeof value === "number" && Number.isFinite(value)
        ? Math.floor(value)
        : fallback;
}
function toolResult(value) {
    return {
        content: [{ type: "text", text: JSON.stringify(value, null, 2) }],
        details: value,
    };
}
function errorResult(error) {
    const message = error instanceof Error ? error.message : String(error);
    return toolResult({ error: "Mnemonic Vault request failed", detail: message });
}
class InvalidSpoolEventError extends Error {
}
export async function flushDurableSpool(spool, client, logger) {
    for (const event of spool.pending()) {
        try {
            if (!event.session_id || !event.external_session_id || !event.agent) {
                throw new InvalidSpoolEventError("missing session or agent identity");
            }
            const originalSessionId = event.session_id;
            let targetSessionId = spool.redirectFor(originalSessionId) ?? originalSessionId;
            await client.startSession(targetSessionId, event.agent, event.external_session_id);
            if (event.kind === "message") {
                if (!event.role || !event.content) {
                    throw new InvalidSpoolEventError("message role and content are required");
                }
                try {
                    await client.appendMessage(targetSessionId, event.role, event.content, targetSessionId === originalSessionId
                        ? (event.metadata ?? {})
                        : {
                            ...(event.metadata ?? {}),
                            recovered_from_session: originalSessionId,
                        }, event.event_id);
                }
                catch (error) {
                    if (!(error instanceof VaultHttpError) || error.status !== 409)
                        throw error;
                    const recoveryParentSession = targetSessionId;
                    const recoveryId = recoverySessionId(recoveryParentSession);
                    await client.startSession(recoveryId, event.agent, event.external_session_id);
                    // Persist the redirect before the recovered append. If the process
                    // stops after the append, a replay still targets the recovery chain.
                    spool.recordRedirect(originalSessionId, recoveryId);
                    targetSessionId = recoveryId;
                    await client.appendMessage(recoveryId, event.role, event.content, {
                        ...(event.metadata ?? {}),
                        recovered_from_session: originalSessionId,
                        recovery_parent_session: recoveryParentSession,
                    }, event.event_id);
                }
            }
            else if (event.kind === "end") {
                await client.endSession(targetSessionId);
            }
            else {
                throw new InvalidSpoolEventError(`unknown event kind: ${String(event.kind)}`);
            }
            spool.acknowledge(event.event_id);
        }
        catch (error) {
            if (error instanceof VaultHttpError && [401, 403].includes(error.status)) {
                logger.warn(`Mnemonic Vault spool delivery blocked by authentication/configuration error: ${String(error)}`);
                return "blocked";
            }
            const permanent = error instanceof InvalidSpoolEventError ||
                (error instanceof VaultHttpError &&
                    error.status >= 400 &&
                    error.status < 500 &&
                    ![408, 429].includes(error.status));
            if (permanent) {
                const status = error instanceof VaultHttpError ? error.status : undefined;
                spool.deadLetter(event, error instanceof Error ? error.message : String(error), status);
                logger.warn(`Mnemonic Vault moved a permanent spool failure to dead-letter: ${String(error)}`);
                continue;
            }
            logger.warn(`Mnemonic Vault spool delivery failed; will retry: ${String(error)}`);
            return "retry";
        }
    }
    return "drained";
}
export default definePluginEntry({
    id: "mnemonic-vault",
    name: "Mnemonic Vault",
    description: "File-first long-term memory with automatic capture and bounded recall.",
    configSchema,
    register(api) {
        const rawConfig = api.pluginConfig;
        const config = resolveConfig(rawConfig);
        const client = new VaultClient(config.baseUrl, config.requestTimeoutMs, process.env[config.apiTokenEnv] ?? "");
        const sessions = new Map();
        const captured = new Set();
        const sourceRoot = fileURLToPath(new URL("../../../../", import.meta.url));
        const projectRoot = process.env.MNEMONIC_VAULT_PROJECT_ROOT ??
            (existsSync(join(sourceRoot, "run.py")) ? sourceRoot : join(homedir(), "mnemonic-vault"));
        const spool = config.autoCapture
            ? new DurableSpool(config.spoolPath || join(projectRoot, "data", "spool", "openclaw.jsonl"))
            : undefined;
        const externalSession = (value) => value.sessionKey ?? value.sessionId ?? `run-${value.runId ?? "main"}`;
        const vaultSession = (externalId) => {
            const existing = sessions.get(externalId);
            if (existing)
                return existing;
            const vaultId = vaultSessionId(externalId, config.agent, config.agentInstanceId);
            sessions.set(externalId, vaultId);
            return vaultId;
        };
        let flushing;
        let deliveryBlocked = false;
        const flushSpool = async () => {
            if (!spool)
                return;
            const result = await flushDurableSpool(spool, client, api.logger);
            deliveryBlocked = result === "blocked";
        };
        const scheduleFlush = () => {
            if (!spool || flushing || deliveryBlocked)
                return;
            flushing = flushSpool().finally(() => {
                flushing = undefined;
            });
        };
        if (spool) {
            scheduleFlush();
            const timer = setInterval(scheduleFlush, config.spoolFlushMs);
            timer.unref();
        }
        const markCaptured = (key) => {
            if (captured.has(key))
                return false;
            captured.add(key);
            if (captured.size > 4_096) {
                const oldest = captured.values().next().value;
                if (oldest)
                    captured.delete(oldest);
            }
            return true;
        };
        api.registerTool({
            name: "memory_search",
            label: "Search memory",
            description: "Search Mnemonic Vault topics using hybrid lexical and vector retrieval.",
            parameters: Type.Object({
                query: Type.String(),
                max_topics: Type.Optional(Type.Integer({ minimum: 1, maximum: 50 })),
                summary_budget_tokens: Type.Optional(Type.Integer({ minimum: 100, maximum: 32_000 })),
                total_context_budget_tokens: Type.Optional(Type.Integer({ minimum: 100, maximum: 32_000 })),
                include_sources: Type.Optional(Type.Union([Type.Literal("auto"), Type.Literal("always"), Type.Literal("never")])),
                scope: Type.Optional(Type.Object({
                    type: Type.Union([
                        Type.Literal("global"),
                        Type.Literal("agent"),
                        Type.Literal("project"),
                        Type.Literal("session"),
                    ]),
                    id: Type.Optional(Type.String()),
                })),
            }),
            async execute(_id, rawParams) {
                const params = rawParams;
                try {
                    return toolResult(await client.search(params.query, {
                        maxTopics: params.max_topics,
                        summaryBudgetTokens: params.summary_budget_tokens,
                        totalContextBudgetTokens: params.total_context_budget_tokens,
                        includeSources: params.include_sources,
                        scope: params.scope,
                    }));
                }
                catch (error) {
                    return errorResult(error);
                }
            },
        });
        api.registerTool({
            name: "memory_remember",
            label: "Remember explicit memory",
            description: "Store an explicit user-requested memory immediately. Call only when the user directly asks to remember, save, not forget, or always keep something.",
            parameters: Type.Object({
                verbatim: Type.String(),
                normalized: Type.Optional(Type.String()),
                kind: Type.Optional(Type.Union([
                    Type.Literal("fact"),
                    Type.Literal("preference"),
                    Type.Literal("decision"),
                    Type.Literal("configuration"),
                    Type.Literal("identity"),
                    Type.Literal("constraint"),
                    Type.Literal("task"),
                    Type.Literal("correction"),
                ])),
                scope: Type.Object({
                    type: Type.Union([
                        Type.Literal("global"),
                        Type.Literal("agent"),
                        Type.Literal("project"),
                        Type.Literal("session"),
                    ]),
                    id: Type.Optional(Type.String()),
                }),
                source_session_id: Type.Optional(Type.String()),
                source_message_id: Type.Optional(Type.Integer({ minimum: 1 })),
                idempotency_key: Type.Optional(Type.String()),
                supersedes: Type.Optional(Type.String()),
            }),
            async execute(_id, rawParams) {
                const params = rawParams;
                try {
                    return toolResult(await client.remember(params.verbatim, {
                        normalized: params.normalized,
                        kind: params.kind,
                        scope: params.scope,
                        sourceSessionId: params.source_session_id,
                        sourceMessageId: params.source_message_id,
                        idempotencyKey: params.idempotency_key,
                        supersedes: params.supersedes,
                    }));
                }
                catch (error) {
                    return errorResult(error);
                }
            },
        });
        api.registerCommand({
            name: "remember",
            description: "Store explicit memory immediately without invoking the LLM.",
            acceptsArgs: true,
            exposeSenderIsOwner: true,
            handler: async (ctx) => {
                const raw = (ctx.args ?? "").trim();
                if (!raw) {
                    return {
                        text: "Usage: /remember [global|agent:<id>|project:<id>|session] <exact fact>",
                        isError: true,
                    };
                }
                const scoped = raw.match(/^(global|agent:[a-zA-Z0-9._-]+|project:[a-zA-Z0-9._-]+|session)\s+([\s\S]+)$/);
                const selector = scoped?.[1];
                const verbatim = (scoped?.[2] ?? raw).trim();
                const externalId = externalSession(ctx);
                let scope;
                if (selector === "global") {
                    scope = { type: "global" };
                }
                else if (selector === "session") {
                    scope = { type: "session", id: vaultSession(externalId) };
                }
                else if (selector?.startsWith("project:")) {
                    scope = { type: "project", id: selector.slice("project:".length) };
                }
                else if (selector?.startsWith("agent:")) {
                    scope = { type: "agent", id: selector.slice("agent:".length) };
                }
                else {
                    scope = { type: "agent", id: config.agentInstanceId };
                }
                const digest = createHash("sha256")
                    .update(`${config.agentInstanceId}\0${externalId}\0${ctx.commandBody}`)
                    .digest("hex")
                    .slice(0, 40);
                try {
                    const receipt = await client.remember(verbatim, {
                        normalized: verbatim,
                        scope,
                        idempotencyKey: `event-${digest}`,
                    });
                    return {
                        text: `Remembered (${String(receipt.memory_id ?? "stored")}): ${verbatim}`,
                    };
                }
                catch (error) {
                    return {
                        text: `Mnemonic Vault could not store memory: ${error instanceof Error ? error.message : String(error)}`,
                        isError: true,
                    };
                }
            },
        });
        const topicParameters = Type.Object({ topic_id: Type.String() });
        for (const name of ["memory_get", "memory_open_topic"]) {
            api.registerTool({
                name,
                label: "Open memory topic",
                description: "Open one Mnemonic Vault topic card and its complete thematic summary.",
                parameters: topicParameters,
                async execute(_id, rawParams) {
                    const params = rawParams;
                    try {
                        return toolResult(await client.openTopic(params.topic_id));
                    }
                    catch (error) {
                        return errorResult(error);
                    }
                },
            });
        }
        api.registerTool({
            name: "memory_open_global_topic",
            label: "Open global memory topic",
            description: "Open a bounded latest-session snapshot, timeline, and source-topic list.",
            parameters: Type.Object({
                global_topic_id: Type.String(),
                max_timeline_entries: Type.Optional(Type.Integer({ minimum: 1, maximum: 500 })),
                total_token_budget: Type.Optional(Type.Integer({ minimum: 300, maximum: 32_000 })),
            }),
            async execute(_id, rawParams) {
                const params = rawParams;
                try {
                    return toolResult(await client.openGlobalTopic(params.global_topic_id, params.max_timeline_entries, params.total_token_budget));
                }
                catch (error) {
                    return errorResult(error);
                }
            },
        });
        api.registerTool({
            name: "memory_expand_topic",
            label: "Expand memory topic",
            description: "Retrieve exact source fragments only inside a topic's transcript ranges.",
            parameters: Type.Object({
                topic_id: Type.String(),
                query: Type.String(),
                max_fragments: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })),
                token_budget: Type.Optional(Type.Integer({ minimum: 100, maximum: 32_000 })),
            }),
            async execute(_id, rawParams) {
                const params = rawParams;
                try {
                    return toolResult(await client.expandTopic(params.topic_id, params.query, params.max_fragments, params.token_budget));
                }
                catch (error) {
                    return errorResult(error);
                }
            },
        });
        api.registerTool({
            name: "memory_read_turns",
            label: "Read memory turns",
            description: "Read an inclusive range of immutable transcript turns from one session.",
            parameters: Type.Object({
                session_id: Type.String(),
                from_turn: Type.Optional(Type.Integer({ minimum: 1 })),
                to_turn: Type.Optional(Type.Integer({ minimum: 1 })),
            }),
            async execute(_id, rawParams) {
                const params = rawParams;
                try {
                    return toolResult(await client.readTurns(params.session_id, params.from_turn, params.to_turn));
                }
                catch (error) {
                    return errorResult(error);
                }
            },
        });
        api.registerTool({
            name: "memory_search_transcript",
            label: "Search memory transcript",
            description: "Search raw transcript turns when topic summaries do not contain enough detail.",
            parameters: Type.Object({
                query: Type.String(),
                session_id: Type.Optional(Type.String()),
                max_fragments: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })),
            }),
            async execute(_id, rawParams) {
                const params = rawParams;
                try {
                    return toolResult(await client.searchTranscript(params.query, params.session_id, params.max_fragments));
                }
                catch (error) {
                    return errorResult(error);
                }
            },
        });
        api.on("session_start", async (event, ctx) => {
            if (!config.autoCapture)
                return;
            vaultSession(externalSession({ ...event, ...ctx }));
            scheduleFlush();
        });
        api.on("before_prompt_build", async (event, ctx) => {
            const externalId = externalSession(ctx);
            if (config.autoCapture && event.prompt.trim()) {
                const eventId = deterministicEventId(config.agentInstanceId, externalId, "user", ctx.runId, event.messages.length, event.prompt);
                if (markCaptured(eventId)) {
                    const vaultId = vaultSession(externalId);
                    spool.append({
                        event_id: eventId,
                        kind: "message",
                        session_id: vaultId,
                        external_session_id: externalId,
                        agent: config.agent,
                        role: "user",
                        content: event.prompt,
                        metadata: {
                            source: "openclaw-plugin",
                            external_session_id: externalId,
                            agent_instance_id: config.agentInstanceId,
                            run_id: ctx.runId ?? "",
                        },
                    });
                    scheduleFlush();
                }
            }
            if (!config.autoRecall || !event.prompt.trim())
                return;
            try {
                const result = await client.search(event.prompt, {
                    maxTopics: config.maxTopics,
                    summaryBudgetTokens: config.summaryBudgetTokens,
                    includeSources: config.includeSources,
                });
                const context = formatMemoryContext(result);
                return context ? { prependContext: context } : undefined;
            }
            catch (error) {
                api.logger.warn(`Mnemonic Vault recall failed: ${String(error)}`);
                return undefined;
            }
        }, { timeoutMs: Math.min(config.requestTimeoutMs + 1_000, 60_000) });
        api.on("agent_end", async (event, ctx) => {
            if (!config.autoCapture || !event.success)
                return;
            const content = extractLastAssistant(event.messages);
            if (!content)
                return;
            const externalId = externalSession(ctx);
            const eventId = deterministicEventId(config.agentInstanceId, externalId, "assistant", event.runId ?? ctx.runId, event.messages.length, content);
            if (!markCaptured(eventId))
                return;
            const vaultId = vaultSession(externalId);
            spool.append({
                event_id: eventId,
                kind: "message",
                session_id: vaultId,
                external_session_id: externalId,
                agent: config.agent,
                role: "assistant",
                content,
                metadata: {
                    source: "openclaw-plugin",
                    external_session_id: externalId,
                    agent_instance_id: config.agentInstanceId,
                    run_id: event.runId ?? ctx.runId ?? "",
                },
            });
            scheduleFlush();
        });
        api.on("session_end", async (event, ctx) => {
            if (!config.autoCapture)
                return;
            const externalId = externalSession({ ...event, ...ctx });
            const vaultId = vaultSession(externalId);
            const eventId = deterministicEventId(config.agentInstanceId, externalId, "end", undefined, event.messageCount, "session_end");
            spool.append({
                event_id: eventId,
                kind: "end",
                session_id: vaultId,
                external_session_id: externalId,
                agent: config.agent,
            });
            sessions.delete(externalId);
            scheduleFlush();
        });
    },
});
