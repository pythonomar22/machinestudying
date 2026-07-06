

# OpenClaw Cheatsheet

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Entry Point                          │
│                   src/index.ts                               │
│                     library.ts                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Plugins      │  │   Channels      │  │     Gateway     │
│  Runtime      │  │    System       │  │   Server        │
└───────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 🔑 Core Exports & Entry Points

### Root Library (`src/library.ts`)
| Export | Description |
|--------|-------------|
| `loadConfig()` | Load/openclaw config from files |
| `saveSessionStore()` | Save session data to storage |
| `deriveSessionKey()` | Create deterministic session keys |
| `resolveSessionKey()` | Resolve session keys |
| `runExec()` | Execute shell commands |
| `runCommandWithTimeout()` | Exec with timeout wrapper |
| `monitorWebChannel()` | Monitor web channel activity |
| `getReplyFromConfig()` | Get auto-reply rules |

### Plugin Runtime (`src/plugins/runtime/index.ts`)
```typescript
createPluginRuntime(options?: {
  subagent?: PluginRuntime["subagent"];
  allowGatewaySubagentBinding?: boolean;
})
```

**Runtime Structure:**
- `config` - Configuration access
- `agent` - Agent-related methods
- `subagent` - Subagent spawning/wrapping
- `system` - System-level operations
- `media` - Media understanding operations
- `channel` - Channel plugin access
- `events` - Event system
- `logging` - Runtime logging
- `tasks/taskFlow` - Task execution flows
- `webSearch/search` - Web search operations
- `state.resolveStateDir()` - State directory resolution
- `version` - Runtime version string

**Lazy-loaded runtimes:** `tts`, `stt`, `mediaUnderstanding`, `modelAuth`, `imageGeneration`, `videoGeneration`, `musicGeneration`

---

## 🔌 Channel System

### Plugin Registry (`src/channels/plugins/registry.ts`)
```typescript
getChannelPlugin(id: string): Promise<ChannelPlugin | undefined>
listChannelPlugins(): ChannelPlugin[]
normalizeChannelId(channelId: string): string
```

### Channel Types (`src/channels/plugins/types.core.ts`)

**ChannelCapabilities:**
```typescript
{
  chatTypes: Array<ChatType | "thread">;
  polls?: boolean;
  reactions?: boolean;
  edit?: boolean;
  unsend?: boolean;
  reply?: boolean;
  effects?: boolean;
  groupManagement?: boolean;
  threads?: boolean;
  media?: boolean;
  nativeCommands?: boolean;
  blockStreaming?: boolean;
}
```

**ChannelMessageActionName:**
```typescript
"send", "replyToThread", "deleteMessage", 
"editMessage", "unsend", "poll", "reaction",
"search", "webFetch", "callOutbound", 
"speak", "translate", "share", "focus",
"block", "ignore"
```

### Channel Messaging Adapter
```typescript
type ChannelMessagingAdapter = {
  normalizeTarget?: (raw: string) => string | undefined;
  defaultMarkdownTableMode?: MarkdownTableMode;
  deriveLegacySessionChatType?: (sessionKey: string) => "direct" | "group" | "channel";
  isLegacyGroupSessionKey?: (key: string) => boolean;
};
```

---

## 🤖 Agent Harness

### Harness Interface (`src/agents/harness/types.ts`)
```typescript
interface AgentHarness {
  id: string;
  label: string;
  pluginId?: string;
  supports(ctx: AgentHarnessSupportContext): AgentHarnessSupport;
  runAttempt(params: EmbeddedRunAttemptParams): Promise<EmbeddedRunAttemptResult>;
  compact?(params: CompactEmbeddedPiSessionParams): Promise<AgentHarnessCompactResult | undefined>;
  reset?(params: AgentHarnessResetParams): Promise<void> | void;
  dispose?(): Promise<void> | void;
}
```

### Supported Runtimes
```typescript
type EmbeddedAgentRuntime = "embedded-ran" | "openai-codex-native" | "anthropic-native" | "other";
```

---

## 📦 Config System

### Config Loading (`src/config/config.ts`)
```typescript
loadConfig()              // Full config + defaults
readBestEffortConfig()    // Try config file without errors
parseConfigJson5(str)     // Parse JSON5 string
mutateConfigFile(patch)   // Atomically update config
```

### Runtime Config Access
```typescript
getRuntimeConfig()        // Current runtime state
getRuntimeConfigSnapshot() // Last snapshot for diffing
```

### Session Paths (`src/config/sessions/paths.ts`)
```typescript
resolveStorePath(config: OpenClawConfig, name?: string)
// Returns path like ~/.openclaw/state/session-store.json
```

### Secrets System (`src/secrets/`)
```typescript
getActiveSecretsRuntimeSnapshot()          // All active secrets
clearSecretsRuntimeSnapshot()              // Clear cached secrets
getRequiredSharedGatewaySessionGeneration() // Auth generation check
```

---

## 🌐 Gateway Server

### Gateway API (`src/gateway/server.impl.ts`)
```typescript
export async function startGatewayServer(
  port = 18789,
  opts: GatewayServerOptions = {},
): Promise<{
  close: (opts?: { reason?: string }) => Promise<void>;
}>;
```

### Gateway Options
```typescript
interface GatewayServerOptions {
  bind?: "loopback" | "lan" | "tailnet" | "auto";
  host?: string;
  controlUiEnabled?: boolean;
  openAiChatCompletionsEnabled?: boolean;
  openResponsesEnabled?: boolean;
  auth?: GatewayAuthConfig;
  tailscale?: GatewayTailscaleConfig;
}
```

### Gateway Methods List
All available endpoints in `src/gateway/server-methods-list.js`:
- `/v1/chat/completions` - LLM streaming endpoint
- POST `/v1/connect` - WebSocket connection
- GET `/health` - Health check
- `/v1/tools/capabilities` - Tool discovery
- `/v1/chat/transcript-inject` - Inject messages
- `/v1/exec-approvals/*` - Command approvals

---

## 🔄 Provider Attribution

### Endpoint Resolution (`src/agents/provider-attribution.ts`)
```typescript
resolveProviderEndpoint(baseUrl: string): ProviderEndpointResolution;

// Common mappings:
api.openai.com       → openai-public
api.anthropic.com    → anthropic-public
llm.chutes.ai        → chutes-native
api.deepseek.com     → deepseek-native
*.githubcopilot.com  → github-copilot-native
generativelanguage.googleapis.com → google-generative-ai
aiplatform.googleapis.com → google-vertex
```

### Provider Normalization (`src/agents/provider-id.ts`)
```typescript
normalizeProviderId("bedrock")        → "amazon-bedrock"
normalizeProviderId("z.ai")           → "zai"
normalizeProviderId("kimi")           → "kimi"
normalizeProviderId("opencode-zen")   → "opencode"
```

### Request Overrides (`src/agents/provider-request-config.ts`)
```typescript
type ProviderRequestTransportOverrides = {
  headers?: Record<string, string>;
  auth?: ProviderRequestAuthOverride;  // bearer/header mode
  proxy?: ProviderRequestProxyOverride;
  tls?: ProviderRequestTlsOverride;
  allowPrivateNetwork?: boolean;
};
```

---

## 🔐 Approval System

### Approval Errors (`src/infra/approval-errors.ts`)
```typescript
ApprovalDeniedError
ApprovalExpiredError
ApprovalNotFoundError
```

### Approval Handler (`src/infra/approval-handler-runtime.ts`)
```typescript
class ApprovalHandlerRuntime {
  async handleRequest(request: ApprovalRequest);
  getPendingApprovals(accountId?: string): Promise<ApprovalRequest[]>;
  acknowledgeApproval(approvalId: string): Promise<void>;
  denyApproval(approvalId: string): Promise<void>;
}
```

### Execution Approvals (`src/infra/exec-approvals.ts`)
```typescript
execAllowAlways()          // Override all approvals
execAllowList(patterns)    // Whitelist commands by pattern
```

---

## ⚠️ Error Handling

### Error Utilities (`src/infra/errors.ts`)
```typescript
formatErrorMessage(err: unknown): string           // Human-readable with redaction
formatUncaughtError(err: unknown): string          // Stack trace with redaction
detectErrorKind(err): "refusal"|"timeout"|"rate_limit"|"context_length"|"unknown"
hasErrnoCode(err, code): boolean                   // Check errno codes
```

### Error Kinds Detection
```typescript
"refusal"         → content_filter, refusal_policy
"timeout"         → timeout, etimedout
"rate_limit"      → 429, rate limit, too many requests
"context_length"  → token limit, context_window
```

---

## 🛠️ CLI Tools

### Bash Tools (`src/agents/bash-tools.ts`)
```typescript
execForeground(args, options={pty:true, timeoutMs: 30000})
execShellCommand(command, opts)
execBackground(command, callbackOnComplete)
```

### Process Supervision (`src/process/supervisor/`)
Process registry with PTY support, background process tracking, and foreground execution with terminal interaction.

---

## 🎯 Context Engine

### Types (`src/context-engine/types.ts`)
```typescript
type ContextEngine = {
  compile(context: ContextEngineContext): ContextEngineCompileResult;
  rewriteTranscript(transcript: TranscriptData): RewriteResult;
  ingest(message: Message, context: IngestContext): void;
};
```

### Built-in Engines (`src/hooks/bundled/`)
- `bootstrap-extra-files` - Initial onboarding prompts
- `session-memory` - Persistent memory across sessions
- `command-logger` - Audit logging

---

## 📝 Manifest Schema

### Plugin Manifest (`src/plugins/manifest-types.ts`)
```typescript
type PluginManifest = {
  id: string;
  version: string;
  entryPoint: string;
  metadata: PluginMetadata;
  description?: string;
  setupWizard?: PluginSetupWizard[];
  providerEndpoints?: ProviderEndpointEntry[];
  gatewayMethods?: string[];
};
```

---

## 🔍 Key File Locations

| Purpose | Location |
|---------|----------|
| Main entry point | `src/index.ts` |
| Library exports | `src/library.ts` |
| Plugin runtime | `src/plugins/runtime/index.ts` |
| Channel plugins | `src/channels/plugins/` |
| Config system | `src/config/` |
| Gateway server | `src/gateway/server.impl.ts` |
| Provider attribution | `src/agents/provider-attribution.ts` |
| Error utils | `src/infra/errors.ts` |
| Bash tools | `src/agents/bash-tools.ts` |
| Auth profiles | `src/agents/auth-profiles/` |
| Sessions | `src/config/sessions/` |
| Secrets | `src/secrets/` |

---

## 🧪 Testing Patterns

### Lazy Imports Pattern
Many modules use lazy loading to avoid cold-start overhead:
```typescript
const loadSomething = createLazyRuntimeModule(() => import("./something.js"));
const method = createLazyRuntimeMethod(loadSomething, (runtime) => runtime.method);
```

### Test Helpers
- `__resetModelCatalogCacheForTest()` - Clear model cache
- `setGatewaySubagentRuntime(subagent)` - Inject test subagent
- `clearGatewaySubagentRuntime()` - Clean up between tests
- `minimalTestGateway` env var for reduced test startup

---

## 🚀 Quick Reference Commands

```bash
# Start gateway
openclaw gateway --port 18789

# Setup wizard
openclaw onboard

# Config management
openclaw config set models.providers.amazon-bedrock.models.nova-micro.maxTokens 100000
openclaw config get | # show full config

# Models management
openclaw models list
openclaw models add --provider anthropic
openclaw models delete my-model

# Channels
openclaw channels list
openclaw channels enable slack
```

---

*Generated after 50 iterations of study using grep, glob, and read_file tools.*