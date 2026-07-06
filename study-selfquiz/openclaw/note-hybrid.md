# OpenClaw Developer Cheatsheet

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Plugin SDK API Surface](#plugin-sdk-api-surface)
3. [Extension Development Patterns](#extension-development-patterns)
4. [Channel/Provider/Tool Infrastructure](#channelprovider-tool-infrastructure)
5. [Configuration System](#configuration-system)
6. [Gateway & Runtime](#gateway--runtime)
7. [Memory & Context Engine](#memory--context-engine)
8. [Hooks & Extensibility](#hooks--extensibility)
9. [Common Utilities](#common-utilities)
10. [Performance & Testing](#performance--testing)

---

## Architecture Overview

### Repository Structure
```
root/
├── src/                 # Core application logic
│   ├── agents/          # Agent spawning, policy, auth-profiles
│   ├── channels/        # Channel infrastructure, plugins
│   ├── plugins/         # Plugin loader, registry, installation
│   ├── config/          # Configuration loading, sessions
│   ├── context-engine/  # Memory, compaction, state
│   ├── gateway/         # API server, agent-list, protocols
│   └── shared/          # Common utilities
├── extensions/          # Bundled plugin directory (same boundary as 3rd-party)
│   ├── <plugin-id>/     # Each extension package
│   │   ├── index.ts     # Plugin entry point
│   │   ├── api.ts       # Public export barrel
│   │   └── package.json # OpenCLaw manifest in openclaw block
└── src/plugin-sdk/      # SDK definition files for all extensions
```

### Key Boundary Principles
- **Extensions should NOT import from** `src/**`, `src/channels/**`, `src/plugin-sdk-internal/**`, or other extensions' `src/**`
- **Extensions SHOULD import from**: `openclaw/plugin-sdk/*` (core), their own `api.ts`/`runtime-api.ts` exports
- **No relative imports escaping current extension package root**
- **Plugin availability comes from manifest ownership + targeted activation, NOT eager global registry seeding**

---

## Plugin SDK API Surface

### Core Entry Helpers

#### Non-Channel Plugins
```typescript
import { definePluginEntry } from 'openclaw/plugin-sdk/core';

const myPlugin = definePluginEntry({
  id: 'my-plugin',
  name: 'My Plugin',
  description: 'Does something useful',
  register: async (api) => {
    // Register tools, commands, services here
    api.registerTool('foo', (ctx) => ...);
    api.registerCommand('bar', (ctx) => ...);
  },
});
```

#### Channel Plugins
```typescript
import { defineChannelPluginEntry } from 'openclaw/plugin-sdk/core';
import type { ChannelPlugin } from 'openclaw/plugin-sdk/channel-contract';

const myChannel = defineChannelPluginEntry({
  id: 'my-channel',
  name: 'My Channel',
  description: 'Connects to X service',
  plugin: {
    setup: { /* ... */ },
    capabilities: {},
    commands: [],
  },
});
```

### Plugin Definition Types
```typescript
export type OpenClawPluginDefinition = {
  kind?: string;           // 'provider' | 'tool' | 'command' | 'service'
  configSchema?: any;      // Zod/via buildPluginConfigSchema
  reload?: () => void;     // Hot reload function
  nodeHostCommands?: any;  // Node host command definitions
  securityAuditCollectors?: any[];
  register: (api: OpenClawPluginApi) => Promise<void> | void;
};
```

### Provider Context Types (All Contexts Follow Pattern)
```typescript
export type ProviderCatalogContext = { cfg, modelId, accountId, region, providerName };
export type ProviderBuildMissingAuthMessageContext = { modelId, providerName, ... };
export type ProviderResolveDynamicModelContext = { resolvedModel, providerName, accountId };
export type ProviderPrepareExtraParamsContext = { modelId, providerName, params };
export type ProviderReplayPolicy = (params) => PolicyResult | undefined;
```

---

## Extension Development Patterns

### Standard Extension Package Structure
```typescript
// extensions/my-extension/index.ts
export const plugin = {
  meta: { /* metadata */ },
  setup: { /* setup flow */ },
  capabilities: [], // Array of capability IDs
  commands: [],     // Command definitions
  doctor: async ({ cfg }) => { /* diagnostics */ },
};

// extensions/my-extension/api.ts
export { plugin };
export type { MyPluginCapabilities, MyPluginCommands } from './types';

// extensions/my-extension/runtime-api.ts
export { registerTool, runExec } from 'openclaw/plugin-sdk/runtime';
```

### Config Schema Pattern
```typescript
import { buildPluginConfigSchema } from 'openclaw/plugin-sdk/core';

export const configSchema = buildPluginConfigSchema({
  $schema: 'http://json-schema.org/draft-07/schema#',
  type: 'object',
  properties: {
    apiKey: { type: 'string', secret: true },
    endpoint: { type: 'string', default: 'https://api.example.com' },
  },
  required: ['apiKey'],
});
```

### Channel Security Options
```typescript
import { createChatChannelPlugin } from 'openclaw/plugin-sdk/core';

const security = {
  dm: {
    channelKey: 'my-channel-key',
    resolvePolicy: (account) => account.policy ?? 'blocked',
    resolveAllowFrom: () => ['trusted-user@example.com'],
    approveHint: '/accept-invite',
  },
};

const myChannel = createChatChannelPlugin({
  id: 'my-channel',
  setup,
  security,
});
```

---

## Configuration System

### Load and Save Config
```typescript
import { loadConfig } from 'openclaw/config';
import { saveSessionStore } from 'openclaw/config/sessions/store';

const cfg = await loadConfig();
await saveSessionStore(sessionId, storeData);
```

### Session Keys & Resolution
```typescript
import { deriveSessionKey, resolveSessionKey } from 'openclaw/config/sessions/session-key';
import { resolveDefaultAgentId } from 'openclaw/agents/agent-scope';

const defaultAgentId = resolveDefaultAgentId(cfg);
const sessionKey = await deriveSessionKey(cfg, sessionId);
```

### Config Paths Resolution
```typescript
import { resolveStateDir, resolveStorePath } from 'openclaw/config/paths';
import path from 'path';

const stateDir = resolveStateDir(); // ~/.local/share/openclaw-state/
const storePath = resolveStorePath(agentId, sessionId);
```

### Config Writing
```typescript
import { setAccountEnabledInConfigSection } from 'openclaw/channels/plugins/config-helpers';
import { updateGroupMembers } from 'openclaw/channels/plugins/group-policy-warnings';

setAccountEnabledInConfigSection(channelId, accountId, enabled);
await writeConfigSection(cfg, channelId, newConfig);
```

---

## Gateway & Runtime

### List Available Agents
```typescript
import { listGatewayAgentsBasic } from 'openclaw/gateway/agent-list';

const { defaultId, mainKey, scope, agents } = listGatewayAgentsBasic(cfg);
// returns: [{ id: 'agent1', name?: string }, ...]
```

### Channel Health Monitoring
```typescript
import { monitorWebChannel } from 'openclaw/plugins/runtime/runtime-web-channel-plugin';

await monitorWebChannel(channelId, callback);
```

### Model Override via Hooks
```typescript
import { HookRunnerRegistry } from 'openclaw/plugins/hook-types';

async function handleBeforeModelResolve(event, ctx) {
  // Check conditions, then return override
  return {
    modelOverride: 'gpt-4-turbo',
    providerOverride: 'anthropic',
  };
}
```

---

## Memory & Context Engine

### Context Engine Factory
```typescript
import { getContextEngineFactory, registerContextEngine } from 'openclaw/context-engine';

// In plugin
registerContextEngine('lancedb-memory', () => {
  return { ingest, compact, query, delete, maintain };
});
```

### Legacy vs New Context Engines
```typescript
import { LegacyContextEngine } from 'openclaw/context-engine/legacy';
import { delegateCompactionToRuntime } from 'openclaw/context-engine/delegate';

// For simple cases, use delegate pattern
delegateCompactionToRuntime(sessionId, transcript);
```

---

## Hooks & Extensibility

### Available Hook Phases
```typescript
// Priority-based modifying hooks (first result wins)
before_model_resolve       // Override provider/model
before_prompt_build        // Inject context/system prompt

// Claiming hooks (first handled:true wins)
before_agent_reply        // Short-circuit LLM with synthetic reply
inbound_claim             // Handle incoming messages
subagent_spawning         // Customize subagent delivery

// Fire-and-forget parallel hooks
agent_end                 // Post-analysis of completed conversations
llm_input                 // Observe exact input payload
llm_output                // Observe output
```

### Hook Runner Example
```typescript
import { createHookRunner } from 'openclaw/plugins/hooks';

const runner = createHookRunner(registry);

await runner.runModifyingHook(
  'before_prompt_build',
  event,
  ctx,
  { mergeResults: (acc, next) => acc?.systemPrompt || next.systemPrompt }
);
```

---

## Common Utilities

### String Coercion & Normalization
```typescript
import { normalizeLowercaseStringOrEmpty, normalizeHyphenSlug } from 'openclaw/shared/string-coerce';

const normalized = normalizeLowercaseStringOrEmpty(input);
const slug = normalizeHyphenSlug(name);
```

### Number Coercion
```typescript
import { parseStrictPositiveInteger } from 'openclaw/infra/parse-finite-number';

const safeNumber = parseStrictPositiveInteger("123"); // 123
const invalid = parseStrictPositiveInteger("abc"); // throws
```

### Date/Time Formatting
```typescript
import { formatZonedTimestamp } from 'openclaw/infra/format-time/format-datetime';

const formatted = formatZonedTimestamp(date); // "2024-01-15T10:30:00+00:00"
```

### Secret Handling
```typescript
import { loadSecretFileSync, readSecretFileSync, tryReadSecretFileSync } from 'openclaw/infra/secret-file';

const secret = await readSecretFileSync('/path/to/.env');
```

### Error Handling
```typescript
import { formatErrorMessage } from 'openclaw/infra/errors';

try {
  await riskyOperation();
} catch (error) {
  console.error(formatErrorMessage(error)); // User-friendly message
}
```

---

## Performance & Testing

### Critical Performance Guidelines

1. **Avoid Cold Loading Full Runtime in Tests**
   ```typescript
   // ❌ Don't do this in tests
   import fullChannelRuntime; // slow bootstrap

   // ✅ Do this instead
   import lightweightArtifact; // pure helper, no runtime
   ```

2. **Use Dependency Injection for Lazy Loads**
   ```typescript
   // Keep expensive work behind DI boundaries
   function getChannelPluginSafe() {
     if (!loaded) loaded = lazyLoadFunctionality();
     return loaded;
   }
   ```

3. **Benchmark Before/After Performance Edits**
   ```bash
   /usr/bin/time -l pnpm test specific.test.ts
   ```

4. **Avoid broad `importOriginal()` partial mocks in hot paths**

5. **Keep Routing/Delivery-Context Normalization Deterministic**

6. **Treat slow test files as architecture signals**

### Testing Strategies
```typescript
// Use explicit mock factories over reset-only mocks
// One-time imports where possible
// Reset only state the test mutates
```

### Build Commands
```bash
pnpm build              # Full rebuild
pnpm test               # Run tests
pnpm test --watch       # Watch mode
pnpm lint               # Linting checks
```

---

## Quick Reference: Essential Imports

```typescript
// From plugin-sdk/core
import { definePluginEntry, defineChannelPluginEntry, buildPluginConfigSchema } from 'openclaw/plugin-sdk/core';
import type { OpenClawPluginApi, OpenClawConfig, ChannelPlugin } from 'openclaw/plugin-sdk/core';

// From agents
import { deriveSessionKey, resolveSessionKey } from 'openclaw/config/sessions/session-key';
import { resolveDefaultAgentId } from 'openclaw/agents/agent-scope';

// From channels
import { listGatewayAgentsBasic } from 'openclaw/gateway/agent-list';
import { setAccountEnabledInConfigSection } from 'openclaw/channels/plugins/config-helpers';

// From hooks
import { createHookRunner } from 'openclaw/plugins/hooks';
import type { PluginHookBeforeModelResolveEvent, PluginHookContext } from 'openclaw/plugins/hook-types';

// From utils
import { normalizeLowercaseStringOrEmpty } from 'openclaw/shared/string-coerce';
import { parseStrictPositiveInteger } from 'openclaw/infra/parse-finite-number';
```

---

## Manifest Requirements (package.json)

```json
{
  "name": "my-openclaw-extension",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "build": "tsc",
    "test": "vitest"
  },
  "devDependencies": {
    "vitest": "^1.0.0",
    "@vitest/ui": "^1.0.0"
  },
  "openclaw": {
    "id": "my-extension",
    "kind": "provider",
    "entryPoint": "./index.ts"
  }
}
```

---

*Generated from comprehensive OpenClaw repository exploration.*

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
- **You believe the functions for loading and saving the authentication profile store are named generically like "loadProfile" and "saveProfile" rather than identifying the specific exports defined in the source file.** The three functions are "loadAuthProfileStore()" returning "AuthProfileStore", "saveAuthProfileStore()" returning "void", and "updateAuthProfileStoreWithLock()" returning "Promise<AuthProfileStore | null>".
  > `src/agents/auth-profiles/store.ts:151`: `export function loadAuthProfileStore(): AuthProfileStore {`

## src/gateway
- **You believe that the `GatewayClient.selectConnectAuth()` method prioritizes JWT/Token-based Authentication from an `AuthenticationSourceList` followed by TLS Certificate-based Authentication.** The method actually prioritizes `explicitGatewayToken` over `resolvedDeviceToken` using the nullish coalescing operator to build the `authToken`, before potentially using `explicitBootstrapToken`.
  > `src/gateway/client.ts:709`: `const authToken = explicitGatewayToken ?? resolvedDeviceToken;`

## src/infra
- **You believe the exact file path for the default socket path configuration for exec approvals cannot be localized to a specific repository file, and that tilde expansion relies on standard shell processing or generic library functions like `os.path.expanduser`.** The configuration is explicitly declared in `src/infra/exec-approvals.ts` as the constant `DEFAULT_SOCKET`, and tilde expansion is resolved using a custom `expandHomePrefix` function that prioritizes specific environment variables (`OPENCLAW_HOME`, `HOME`, `USERPROFILE`) over system defaults.
  > `src/infra/exec-approvals.ts:173`: `const DEFAULT_SOCKET = "~/.openclaw/exec-approvals.sock"`

## src/auto-reply
- **You believe a user message is filtered out during heartbeat processing if the heartbeat verification indicates that the connection is stale, expired, or compromised, such as when the message arrives after the configured timeout period or fails required security/integrity checks for the heartbeat session.** A message is filtered out specifically when `params.trigger` is not equal to "heartbeat" OR the cleaned message body does not contain the expected event text verified via `hasEventToken`.
  > `extensions/memory-core/src/dreaming-phases.ts:1679`: `if (params.trigger !== "heartbeat" || !hasEventToken) {`

## src/config
- **You believe there is no specific information about `writeConfigFile()` behavior regarding environment variable reference templates when API keys are present, and that templates would typically be preserved as-is** Actual values are persisted directly to the config file, replacing any previously stored environment variable reference templates for those specific paths; only unchanged paths get their env var reference templates restored from the snapshot via merge patch logic
  > `src/config/io.write-prepare.ts:310`: `if (!isPathChanged(path, changedPaths)) {`

## src/commands
- **You believe that the 'conflicts' array returned by applyAgentBindings() contains objects documenting the conflicting route key, both agent IDs involved (existing and new), and conflict metadata including the operation being attempted.** The 'conflicts' array contains objects with exactly two fields: `binding` (the incoming/new AgentRouteBinding that caused the conflict) and `existingAgentId` (a string containing only the agentId that was previously assigned to that route key before the attempt to change it).
  > `src/commands/agents.bindings.ts:78`: `conflicts: Array<{ binding: AgentRouteBinding; existingAgentId: string }>`

## src/plugins
- **You believe registered agent harnesses are accessed through `src/plugin-sdk/core.ts`, compaction providers are registered in `extensions/memory-core/qmd-manager.ts`, and all components follow a generic plugin lifecycle contract with file-system scanning discovery.** Registered agent harnesses are accessed via `src/agents/harness/registry.ts` using dedicated registry modules with global symbol-based singleton patterns. Compaction providers, memory embedding providers, and conversation binding handlers each have their own dedicated registry files following the same symbol-based singleton API pattern for process-wide storage with paired getter/register functions.
  > `src/plugins/compaction-provider.ts:50`: `const COMPACTION_PROVIDER_REGISTRY_STATE = Symbol.for("openclaw.compactionProviderRegistryState")`

## extensions/matrix
- **You believe that `content` is a valid parameter name for specifying media content with a fallback priority after `file`, and that the validation logic is primarily located in `handler.ts`.** The valid media specification parameters are `file` (highest priority), `url` (secondary fallback), `filename`, `mimetype`, and `imageInfo`. When both `file` and `url` are provided, `file` takes precedence and `url` is ignored; the code evaluates this condition directly in `media.ts`.
  > `extensions/matrix/src/matrix/send/media.ts:87`: `if (!params.file && params.url) {`

## extensions/discord
- **You believe that the provided documentation lacks information regarding `runtime-api.ts` and its exports for Discord moderation operations, asserting that `runtime.moderation-shared.ts` and `runtime.moderation.ts` are not documented within the study notes.** The documentation confirms that `runtime-api.ts` re-exports core moderation functionality from `runtime.moderation-shared.ts` (shared logic/types) and `runtime.moderation.ts` (guild execution), enabling external access to actions like banning, kicking, and timeout handling.
  > `extensions/discord/runtime-api.ts:3`: `export * from "./src/actions/runtime.moderation-shared.js";`

## src/cli
- **You believe the provided study notes from the OpenClaw repository contain no information about the function `applyCliExecutionStartupPresentation` or the logical conditions determining when it should NOT emit the CLI banner.** The function determines it should NOT emit the CLI banner if any of the following conditions are met: `params.startupPolicy.hideBanner` is truthy, `params.showBanner` equals false, or `params.version` is falsy. If so, the function returns immediately before emitting the banner.
  > `src/cli/command-execution-startup.ts:39`: `if (params.startupPolicy.hideBanner || params.showBanner === false || !params.version) {`

## extensions/telegram
- **You believe the specific parameters required for Telegram channel topics and the distinction between regular chats and channel topics are not available in the provided study notes.** When sending a message to a Telegram channel topic, you must include the `messageThreadId` parameter to ensure the message is directed to the correct thread rather than the default chat.
  > `extensions/telegram/src/send.ts:98`: `messageThreadId?: number;`

## extensions/feishu
- **you believe that `threadId` is the appropriate identifier for replying to a thread and that the `withReplyDispatcher` method is the correct mechanism to invoke the Feishu action API without needing a specific action string or guaranteed mandatory fields like `messageId`.** you must explicitly set the action string to `"thread-reply"` and ensure a mandatory `messageId` parameter is included in the `params`; omitting `messageId` triggers a runtime error validating thread reply requirements.
  > `extensions/feishu/src/channel.ts:664`: `if (ctx.action === "thread-reply" && !replyToMessageId) {
              throw new Error("Feishu thread-reply requires messageId.");
}`
