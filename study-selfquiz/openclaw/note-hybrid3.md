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
