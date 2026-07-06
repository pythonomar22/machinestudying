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