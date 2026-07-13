# OpenClaw Cheatsheet

## Repository Structure

```
src/
├── acp/                 # Approval classifier - routes messages to agents/channels
├── agents/              # Agent spawning, auth profiles, tool execution
├── channels/            # Channel integration contracts (slack, discord, etc.)
│   └── plugins/        # Channel plugin type definitions
├── config/              # Configuration loading, validation, schema
├── gateway/             # HTTP API server - exposes services via RPC
├── infra/               # Cross-cutting concerns (secrets, dedupe, approval handling)
├── plugin-sdk/          # Public plugin/host contract definitions
├── plugins/             # Provider plugins (bedrock, anthropic, etc.)
├── routing/             # Session keys, account lookup, peer resolution
├── sessions/            # Message history, delivery, state
├── tools/               # Tool cataloging, input parsing, authorization
└── utils/               # Shared utilities
extensions/              # Bundled plugins (bluebubbles, open-prose, etc.)
    └── <extension>/     # Each follows: api.ts, openclaw.plugin.json, src/**
```

## Core Architecture Flow

```
User Request → Gateway/Hook → ACP Approval Classifier → Routing (Session Key) 
→ Agent Runtime → Tool Execution (if needed) → Channel Plugin → Network/Display
```

### Message Flow Components

**ACP (Approval Classifier)**: `src/acp/approval-classifier.ts`
- Classifies tool calls into approval classes: readonly_scoped, readonly_search, mutating, exec_capable, control_plane, interactive, other
- Keys by tool name pattern: EXEC_CAPABLE_TOOL_IDS, SAFE_SEARCH_TOOL_IDS, CONTROL_PLANE_TOOL_IDS
- Determines auto-approval eligibility based on scope and trust

**Routing**: `src/routing/session-key.ts`, `src/routing/account-id.ts`
- Builds session keys: `agent:<id>:<channel>:<peerKind>:<peerId>`
- Handles identity linking for scoped DM security
- Manages thread conversation binding

**Agents**: `src/agents/auth-profiles/*.ts`
- OAuth manager handles token refresh queues, locking paths, effective identity resolution
- Spawn logic manages multi-agent routing via session keys
- Tool execution via agent-step with input validation

**Channels**: `src/channels/plugins/types*.ts`
- ChannelPlugin type defines capabilities: setup, config, messaging, gateway, commands, etc.
- Adapter pattern separates capability implementations
- Support for custom gateway methods and external integrations

## Plugin System

### Plugin Types

**Provider Plugin** (`src/plugin-sdk/provider-entry.ts`):
```typescript
defineSingleProviderPluginEntry({
  id: "anthropic",
  label: "Anthropic",
  docsPath: "./docs/anthropic/",
  auth: [{ methodId: "apiKey", envVar: "ANTHROPIC_API_KEY" }],
  catalog: { buildProvider: /* returns ModelCatalog */ },
  augmentModelCatalog: () => [],
  prepareModels: (models) => models,
})
```

**Channel Plugin** (`src/plugin-sdk/core.ts`):
```typescript
defineChannelPluginEntry({
  id: "slack",
  name: "Slack",
  description: "Integration...",
  plugin: {
    setup: { accountId: /* resolve from input */ },
    config: { listAccountIds, resolveAccount, ... },
    pairing: { text: { idLabel, message, notify: fn } },
    security: { dm: { policy, allowFrom, ... } },
    messaging: { sendText, sendMedia, ... },
  },
})
```

**Non-channel Plugin** (`src/plugin-sdk/plugin-entry.ts`):
```typescript
definePluginEntry({
  id: "tool-provider",
  name: "Custom Provider",
  register(api) {
    api.registerProvider({ id, label, catalog, auth });
  },
});
```

### Plugin Boundaries

- **Extensions**: Should import from `openclaw/plugin-sdk/*`, not core internals (`src/**`)
- **Metadata**: Keep accurate in `openclaw.plugin.json` and package exports
- **Exports**: Define narrow subpaths over broad barrels
- **Registration**: Use manifest ownership + targeted activation, not eager global seeding

## Channel Capabilities

Required/Optional adapters define channel functionality:

| Capability | Type File | Purpose |
|------------|-----------|---------|
| Setup | `types.adapters.ts` | Account configuration during onboarding |
| Config | `types.adapters.ts` | List accounts, resolve, describe states |
| Pairing | `pairing.types.ts` | QR/text-based channel verification |
| Security | `types.adapters.ts` | DM policies, allow-from lists, audit findings |
| Threading | `types.core.ts` | Reply modes, conversation binding |
| Messaging | `types.core.ts` | sendText, sendMedia, sendPoll |
| Gateway | `types.adapters.ts` | Custom gateway methods, channelRuntime access |
| Commands | `types.adapters.ts` | Native command discovery/enforcement |
| Doctor | `types.adapters.ts` | Config repair, warning collection |
| Auth | `types.adapters.ts` | Login/logout flows |
| Secrets | `types.adapters.ts` | Secret target registry entries |

## ACP Approval Classification

File: `src/acp/approval-classifier.ts`

```typescript
classifyAcpToolApproval({
  toolCall: { title, _meta, rawInput },
  cwd: "/path/to/cwd"
})
// Returns: { toolName?, approvalClass, autoApprove }
```

Classes:
- `readonly_scoped`: Safe read ops scoped to cwd (auto-approve if trusted tool)
- `readonly_search`: Search/web-search tools (trusted)
- `mutating`: Data mutations (requires explicit approval)
- `exec_capable`: Shell/process execution (never auto-approve)
- `control_plane`: Session management tools
- `interactive`: Human-in-the-loop required
- `other`: Default fallback
- `unknown`: Unrecognized tool

Key Sets:
```typescript
SAFE_SEARCH_TOOL_IDS = new Set(["search", "web_search", "memory_search"])
EXEC_CAPABLE_TOOL_IDS = new Set(["exec", "spawn", "shell", "bash", "process"])
CONTROL_PLANE_TOOL_IDS = new Set(["sessions_spawn", "sessions_send", "session_status"])
```

## Tool Execution Flow

File: `src/agents/tools/common.ts`

Helper functions for tool parameters:
- `readStringParam(params, key, options)` - Validates and returns string params
- `readNumberParam(params, key, options)` - Returns parsed number or throws if required
- `readStringArrayParam(params, key, options)` - Returns trimmed array
- `createActionGate(actions)` - Gatekeeper for conditional tool availability

Error Classes:
- `ToolInputError` (400) - Bad request/input format
- `ToolAuthorizationError` (403) - Access denied

## Configuration System

Key Files:
- `src/config/schema.shared.ts` - Shared schema validators
- `src/config/runtime-schema.ts` - Runtime overrides
- `src/config/channel-configured.ts` - Per-channel configured state
- `src/config/io.ts` - Config write/read operations

Important Functions:
- `buildPluginConfigSchema()` - Create plugin-specific schemas
- `loadSecretFileSync()` - Secure file-based secrets with path enforcement
- `applyAccountNameToChannelSection()` - Migrate legacy config formats

## Sessions & Delivery

Key Files:
- `src/sessions/session-key-utils.ts` - Parse/serialize session identifiers
- `src/sessions/transcript-events.ts` - Event logging for transcripts
- `src/infra/dedupe.ts` - Deduplication cache with TTL

Session Key Format:
```
agent:<normalizedAgentId>:<channelOrMain>[:<accountId>:<peerKind>:<peerId>]
```

Thread conversations:
```typescript
resolveThreadSessionKeys({
  baseSessionKey: "agent:main",
  threadId: "thread-123"
})
// Returns: { sessionKey: "agent:main:thread:thread-123", parentSessionKey: "agent:main" }
```

## Gateway Server

File: `src/gateway/server-methods/types.js`

- Exposes plugin-owned Gateway behavior via HTTP/RPC
- Lightweight artifact resolvers prefer static descriptors over full runtime loads
- Maintain alignment between descriptor and implementation

Guardrail: Don't load bundled plugin runtime from Gateway just to answer static questions.

## Infrastructure Utilities

Cross-cutting concerns shared across modules:

| Module | File | Purpose |
|--------|------|---------|
| Secrets | `src/infra/secret-file.ts` | Secure secret file reading (mode 0o600, path enforcement) |
| Dedupe | `src/infra/dedupe.ts` | TTL+maxSize deduplication cache |
| Abort Signal | `src/infra/abort-signal.ts` | Cancel pattern propagation |
| Time Format | `src/infra/format-time/*.ts` | DateTime formatting/offsetless parsing |
| Fetch | `src/infra/fetch.ts` | HTTP client wrapper |
| Backoff | `src/infra/backoff.ts` | Retry with exponential backoff |

## Auth Profiles System

File: `src/agents/auth-profiles/oauth-manager.ts`

Manages credential lifecycles:
- Token refresh queues with cooldown detection
- Effective identity resolution when multiple profiles exist
- Fallback mechanisms for token expiration
- Mirror refresh strategies across agents

Key Types:
- `Profile` - Stored auth profile with credentials, scopes, lastUsed
- `CredentialState` - Valid, expiring, expired statuses
- `Order` - Priority ordering of profiles

## Guardrails & Best Practices (from AGENTS.md)

### General Development

1. **Performance**: Benchmark before/after changes. Slow test files often signal architecture problems.
2. **Lazy Loading**: Expensive bootstrap work behind dependency injection or narrow helpers
3. **Boundary Respect**: Plugins shouldn't reach into arbitrary host internals
4. **Backwards Compatibility**: Additive changes default; breaking changes require major version bump
5. **Test Coverage**: Preserve exact production composition in helpers; don't remove behavior proofs just because old tests were slow

### Specific Areas

**Plugin SDK**:
- Prefer narrow, purpose-built subpaths over broad convenience barrels
- Keep facades acyclic (no back-edge re-exports)
- If setup requires runtime, make explicit in plugin's declared setup/runtime surface

**Agents**:
- Treat channel/plugin lookups inside hot paths as suspect
- Use lightweight typed artifacts before falling back to full runtime
- Avoid broad `importOriginal()` partial mocks in hot agent tests

**Gateway**:
- For plugin-owned behavior, prefer lightweight public artifact resolver
- Run `pnpm build` when changing lazy-loading or bundled plugin artifacts
- Keep schedulers disabled in manual-RPC tests unless specifically testing scheduling

**Config**:
- Use schema validators for all config sections
- Enforce private file modes (0o700 dirs, 0o600 files) for secrets
- Path resolution must stay within designated root directories

**Channels**:
- Maintain capability contracts in adapter type files
- External channel plugins should check for optional `channelRuntime` before use
- Align descriptor and full-plugin export behavior

## Quick Reference Patterns

### Creating a New Provider Plugin
```typescript
import { defineSingleProviderPluginEntry } from "openclaw/plugin-sdk";

export default defineSingleProviderPluginEntry({
  id: "my-provider",
  name: "My Provider",
  description: "...",
  provider: {
    id: "my-provider",
    label: "My Provider",
    docsPath: "./docs/my-provider/",
    auth: [
      {
        methodId: "apiKey",
        envVar: "MY_PROVIDER_API_KEY",
        label: "API Key",
      },
    ],
    catalog: {
      buildProvider: ({ ctx }) => {
        // Return model catalog with id, label, docsPath, aliases, etc.
      },
    },
    augmentModelCatalog: (catalog) => catalog,
    prepareModels: (models) => models,
  },
});
```

### Creating a New Channel Plugin
```typescript
import { defineChannelPluginEntry } from "openclaw/plugin-sdk/core";

export const plugin = defineChannelPluginEntry({
  id: "custom-channel",
  name: "Custom Channel",
  description: "...",
  plugin: {
    id: "custom-channel",
    meta: { displayName: "Custom" },
    setup: {
      applyAccountConfig: (params) => { /* ... */ },
    },
    config: {
      listAccountIds: (cfg) => [...],
      resolveAccount: (cfg, accountId) => {},
    },
    pairing: {
      text: {
        idLabel: "Pairing ID",
        message: "Please scan QR...",
        notify: async (ctx) => { /* ... */ },
      },
    },
    security: {
      dm: {
        policy: (account) => "...",
        allowFrom: () => ["example.com"],
      },
    },
    messaging: {
      sendText: async (ctx) => { /* ... */ },
    },
  },
});
```

### Building Session Keys
```typescript
import { buildGroupHistoryKey, buildAgentPeerSessionKey } from "openclaw/routing";

const groupKey = buildGroupHistoryKey({
  channel: "slack",
  accountId: "user-123",
  peerKind: "group",
  peerId: "room-abc",
});

const peerSessionKey = buildAgentPeerSessionKey({
  agentId: "main",
  mainKey: undefined,
  channel: "slack",
  accountId: "user-123",
  peerKind: "direct",
  peerId: "alice",
});
```

## Testing Guidelines

- Use suite-level servers and authenticated contexts when possible
- Reset runtime state explicitly instead of restarting per case
- Benchmark affected Gateway test files with `pnpm test <file>`
- Record seconds and RSS for agent performance changes
- Run `pnpm build` when touching SDK seams or bundled plugin artifacts

---

*Generated from systematic study of OpenClaw repository architecture and documentation.*

---

# Corrections from self-quizzing (verified against the source; trust these over the summary above)

## Repo map
- `src/agents/`: subagent-announce.format.e2e.test.ts, openai-ws-stream.test.ts, attempt.ts
- `src/gateway/`: server.sessions.gateway-server-sessions-a.test.ts, gateway-models.profiles.live.test.ts, chat.ts
- `src/infra/`: host-env-security.test.ts, update-runner.test.ts, restart-stale-pids.test.ts
- `src/auto-reply/`: highlight.min.js, dispatch-from-config.test.ts, session.test.ts
- `src/config/`: schema.base.generated.ts, bundled-channel-config-metadata.generated.ts, schema.help.ts
- `src/commands/`: doctor-config-flow.test.ts, onboard-channels.e2e.test.ts, status.test.ts
- `src/plugins/`: loader.test.ts, loader.ts, install.test.ts
- `extensions/matrix/`: handler.test.ts, sdk.test.ts, handler.ts
- `extensions/discord/`: thread-bindings.lifecycle.test.ts, native-command.ts, provider.ts
- `src/cli/`: capability-cli.ts, config-cli.test.ts, update-cli.test.ts
- `extensions/telegram/`: bot-message-dispatch.test.ts, bot.create-telegram-bot.test.ts, bot.test.ts
- `extensions/feishu/`: bot.test.ts, docx.ts, monitor.comment.ts
- `extensions/browser/`: pw-tools-core.interactions.ts, pw-tools-core.interactions.navigation-guard.test.ts, pw-session.ts
- `extensions/qa-lab/`: ui-render.ts, server.test.ts, server.ts
- `extensions/memory-core/`: qmd-manager.test.ts, qmd-manager.ts, cli.runtime.ts
- `extensions/msteams/`: channel.ts, message-handler.ts, messenger.test.ts
- `src/plugin-sdk/`: channel-config-helpers.ts, core.ts, channel-config-helpers.test.ts
- `extensions/slack/`: interactions.test.ts, slash.test.ts, media.test.ts
- `src/channels/`: setup-wizard-helpers.test.ts, setup-wizard-helpers.ts, bundled.shape-guard.test.ts
- `src/cron/`: timer.ts, delivery-dispatch.double-announce.test.ts, timer.regression.test.ts

## src/agents
- **You believe that when calling `acpSpawn()` with `thread=true` but without specifying the `mode` parameter, the function will utilize the **Default** (or **Standard**) spawn mode.** The function will actually use the **"session"** spawn mode (which resolves to the runtime mode **"persistent"**) because `thread=true` triggers a specific fallback logic in `resolveSpawnMode` rather than using a generic default. When `threadRequested` is true and no explicit mode is provided, the logic defaults to "session".
  > `src/agents/acp-spawn.ts:334`: `// Thread-bound spawns should default to persistent sessions.`

## src/gateway
- **You believe that without provided input context, you cannot identify the specific implementation details of the channel manager's exponential backoff policy, including its file location and configuration values for `initialMs`, `maxMs`, `factor`, and `jitter`.** The exponential backoff policy is explicitly defined in **`src/gateway/server-channels.ts`** at lines 22-27 as the `CHANNEL_RESTART_POLICY` constant, with `initialMs: 5_000`, `maxMs: 5 * 60_000` (300,000ms), `factor: 2`, and `jitter: 0.1`. This policy is actively used on line 460 via `computeBackoff()` when channels fail to start.
  > `src/gateway/server-channels.ts:22`: `const CHANNEL_RESTART_POLICY: BackoffPolicy = {
  initialMs: 5_000,
  maxMs: 5 * 60_000,
  factor: 2,
  jitter: 0.1,`

## src/infra
- **you believe the default expiration timeout value in milliseconds for execution approvals when no override is provided is 86,400,000** the actual default expiration timeout value is 120000 milliseconds when no override is provided
  > `src/agents/pi-tools.before-tool-call.ts:254`: `timeoutMs: approval.timeoutMs ?? 120_000,`

## src/auto-reply
- **You believe the primary entry point function and its required parameters for the auto-reply module remain unknown or indeterminate.** The primary entry point is `dispatchInboundMessage` exported from `src/auto-reply/dispatch.ts`, which requires `ctx`, `cfg`, and `dispatcher` arguments.
  > `src/auto-reply/dispatch.ts:20`: `export async function dispatchInboundMessage(params: {`

## src/config
- **you believe there is no documentation or implementation details regarding `loadConfig()` or `getRuntimeConfig()` functions specifically covering their caching behavior differences or recommended usage patterns for long-lived runtimes.** Both `loadConfig()` and `getRuntimeConfig()` exhibit identical caching behavior - they both utilize a process-level snapshot cache where the first successful load becomes the process-wide snapshot. Neither function performs a fresh config file read on subsequent calls. For long-lived runtimes, either function should be avoided on hot code paths.
  > `src/config/io.ts:1801`: `// First successful load becomes the process snapshot. Long-lived runtimes`

## src/commands
- **You believe the error output will display a generic parameter validation message like "Invalid value for option '--section'" without specifying the allowable section identifiers, and that execution halts purely through argument parsing middleware.** When an invalid value is passed, execution stops immediately at line 32 in `src/commands/configure.commands.ts` via `runtime.exit(1)`, terminating the process before the wizard runs. The exact error output explicitly lists valid options: "Invalid --section: ... Expected one of: workspace, model, web, gateway, daemon, channels, plugins, skills, health."
  > `src/commands/configure.commands.ts:30`: `Invalid --section: ${invalid.join(", ")}. Expected one of: ${CONFIGURE_WIZARD_SECTIONS.join(", ")}.`

## src/plugins
- **You believed there is no specific information detailing the exact conditions under which a plugin receives the activation cause 'blocked-by-denylist'.** A plugin receives the activation cause 'blocked-by-denylist' when the plugin's ID is included in the deny array of the configuration parameters, specifically when `params.config.deny.includes(params.id)` evaluates to true.
  > `src/plugins/config-state.ts:276`: `if (params.config.deny.includes(params.id)) {`

## extensions/matrix
- **You believe there is no specific information regarding the security risks associated with setting `autoJoin='always'` on a Matrix account in the provided documentation.** When `autoJoin` is set to 'always' on a Matrix account, any invited room will be joined before message policy applies.
  > `extensions/matrix/src/channel.ts:209`: `- Matrix invites: autoJoin="always" joins any invited room before message policy applies. Set ${autoJoinPath}="allowlist" + ${autoJoinAllowlistPath} (or ${autoJoinPath}="off") to restrict joins.`

## extensions/discord
- **You believe that the REQUIRED_DISCORD_PERMISSIONS constant definition and its specified minimum requirements for channel access are not documented in the correction notes.** The constant is defined in extensions/discord/src/channel.ts at line 135, specifying "ViewChannel" and "SendMessages" as the required permissions for channel access.
  > `extensions/discord/src/channel.ts:135`: `const REQUIRED_DISCORD_PERMISSIONS = ["ViewChannel", "SendMessages"] as const;`

## src/cli
- **You believe that there is no information available regarding the environment variable that determines whether subcommands are eagerly registered.** Subcommands are eagerly registered when the environment variable `OPENCLAW_DISABLE_LAZY_SUBCOMMANDS` is set to a truthy value, which is checked within the function `shouldEagerRegisterSubcommands` in the file `src/cli/command-registration-policy.ts`.
  > `src/cli/command-registration-policy.ts:24`: `return isTruthyEnvValue(env.OPENCLAW_DISABLE_LAZY_SUBCOMMANDS);`

## extensions/telegram
- **You believe the specific behavior of handling a duplicate Telegram bot token is not explicitly documented and may result in throwing an error or blocking the connection setup entirely.** The system marks the conflicting account as **not configured** and sets `isConfigured` to `false` if `findTelegramTokenOwnerAccountId` detects that another `accountId` already owns the token, providing a clear unconfigured reason message.
  > `extensions/telegram/src/shared.ts:176`: `return !findTelegramTokenOwnerAccountId({ cfg, accountId: account.accountId });`

## extensions/feishu
- **You believe that marking a `ResolvedFeishuAccount` as 'enabled' depends on multiple validation steps like valid credentials, populated fields, and error-checking phases.** A `ResolvedFeishuAccount` is marked 'enabled' **only** when BOTH the channel-level config (`channels.feishu.enabled`) and account-specific config (`channels.feishu.accounts[accountId].enabled`) are NOT explicitly set to `false`. No additional validations are required.
  > `extensions/feishu/src/accounts.ts:266`: `const enabled = baseEnabled && accountEnabled;`
