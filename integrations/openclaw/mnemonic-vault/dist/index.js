import { createHash } from "node:crypto";
import { Type } from "typebox";
import { buildJsonPluginConfigSchema, definePluginEntry, } from "openclaw/plugin-sdk/plugin-entry";
import { extractLastAssistant, formatMemoryContext, VaultClient, vaultSessionId, } from "./client.js";
const DEFAULTS = {
    baseUrl: "http://127.0.0.1:8765",
    agent: "openclaw",
    autoCapture: true,
    autoRecall: true,
    maxTopics: 5,
    summaryBudgetTokens: 1_800,
    includeSources: "auto",
    requestTimeoutMs: 8_000,
};
const configSchema = buildJsonPluginConfigSchema({
    type: "object",
    additionalProperties: false,
    properties: {
        baseUrl: { type: "string" },
        agent: { type: "string" },
        autoCapture: { type: "boolean" },
        autoRecall: { type: "boolean" },
        maxTopics: { type: "integer", minimum: 1, maximum: 50 },
        summaryBudgetTokens: { type: "integer", minimum: 100, maximum: 32_000 },
        includeSources: { type: "string", enum: ["auto", "always", "never"] },
        requestTimeoutMs: { type: "integer", minimum: 250, maximum: 60_000 },
    },
});
function resolveConfig(value) {
    const raw = value && typeof value === "object" ? value : {};
    return {
        baseUrl: typeof raw.baseUrl === "string" ? raw.baseUrl : DEFAULTS.baseUrl,
        agent: typeof raw.agent === "string" ? raw.agent : DEFAULTS.agent,
        autoCapture: raw.autoCapture ?? DEFAULTS.autoCapture,
        autoRecall: raw.autoRecall ?? DEFAULTS.autoRecall,
        maxTopics: finiteInt(raw.maxTopics, DEFAULTS.maxTopics),
        summaryBudgetTokens: finiteInt(raw.summaryBudgetTokens, DEFAULTS.summaryBudgetTokens),
        includeSources: raw.includeSources === "always" || raw.includeSources === "never"
            ? raw.includeSources
            : "auto",
        requestTimeoutMs: finiteInt(raw.requestTimeoutMs, DEFAULTS.requestTimeoutMs),
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
export default definePluginEntry({
    id: "mnemonic-vault",
    name: "Mnemonic Vault",
    description: "File-first long-term memory with automatic capture and bounded recall.",
    configSchema,
    register(api) {
        const rawConfig = api.pluginConfig;
        const config = resolveConfig(rawConfig);
        const client = new VaultClient(config.baseUrl, config.requestTimeoutMs);
        const instanceId = `${Date.now().toString(36)}-${process.pid.toString(36)}`;
        const sessions = new Map();
        const captured = new Set();
        const externalSession = (value) => value.sessionKey ?? value.sessionId ?? `run-${value.runId ?? "main"}`;
        const ensureSession = async (externalId) => {
            const existing = sessions.get(externalId);
            if (existing)
                return existing;
            const vaultId = vaultSessionId(externalId, config.agent, instanceId);
            const pending = client
                .startSession(vaultId, config.agent, externalId)
                .then(() => vaultId)
                .catch((error) => {
                sessions.delete(externalId);
                throw error;
            });
            sessions.set(externalId, pending);
            return pending;
        };
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
        const captureKey = (role, externalId, runId, text) => `${role}:${externalId}:${runId ?? createHash("sha256").update(text).digest("hex").slice(0, 20)}`;
        api.registerTool({
            name: "memory_search",
            label: "Search memory",
            description: "Search Mnemonic Vault topics using hybrid lexical and vector retrieval.",
            parameters: Type.Object({
                query: Type.String(),
                max_topics: Type.Optional(Type.Integer({ minimum: 1, maximum: 50 })),
                summary_budget_tokens: Type.Optional(Type.Integer({ minimum: 100, maximum: 32_000 })),
                include_sources: Type.Optional(Type.Union([Type.Literal("auto"), Type.Literal("always"), Type.Literal("never")])),
            }),
            async execute(_id, rawParams) {
                const params = rawParams;
                try {
                    return toolResult(await client.search(params.query, {
                        maxTopics: params.max_topics,
                        summaryBudgetTokens: params.summary_budget_tokens,
                        includeSources: params.include_sources,
                    }));
                }
                catch (error) {
                    return errorResult(error);
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
            try {
                await ensureSession(externalSession({ ...event, ...ctx }));
            }
            catch (error) {
                api.logger.warn(`Mnemonic Vault session start failed: ${String(error)}`);
            }
        });
        api.on("before_prompt_build", async (event, ctx) => {
            const externalId = externalSession(ctx);
            if (config.autoCapture && event.prompt.trim()) {
                const key = captureKey("user", externalId, ctx.runId, event.prompt);
                if (markCaptured(key)) {
                    try {
                        const vaultId = await ensureSession(externalId);
                        await client.appendMessage(vaultId, "user", event.prompt, {
                            source: "openclaw-plugin",
                            external_session_id: externalId,
                            run_id: ctx.runId ?? "",
                        });
                    }
                    catch (error) {
                        api.logger.warn(`Mnemonic Vault user capture failed: ${String(error)}`);
                    }
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
            const key = captureKey("assistant", externalId, event.runId ?? ctx.runId, content);
            if (!markCaptured(key))
                return;
            try {
                const vaultId = await ensureSession(externalId);
                await client.appendMessage(vaultId, "assistant", content, {
                    source: "openclaw-plugin",
                    external_session_id: externalId,
                    run_id: event.runId ?? ctx.runId ?? "",
                });
            }
            catch (error) {
                api.logger.warn(`Mnemonic Vault assistant capture failed: ${String(error)}`);
            }
        });
        api.on("session_end", async (event, ctx) => {
            if (!config.autoCapture)
                return;
            const externalId = externalSession({ ...event, ...ctx });
            const pending = sessions.get(externalId);
            if (!pending)
                return;
            try {
                await client.endSession(await pending);
                sessions.delete(externalId);
            }
            catch (error) {
                api.logger.warn(`Mnemonic Vault session end failed: ${String(error)}`);
            }
        });
    },
});
