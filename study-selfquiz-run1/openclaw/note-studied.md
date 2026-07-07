# OpenClaw — studied reference (grounded in prior study)

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
The `src/agents` module contains core execution and spawning logic for CLI agents and ACP sessions across three key files. In `cli-runner.ts`, the `runCliAgent()` function handles session management via `runPreparedCliAgent()`. When catching a `FailoverError`, it checks `err.reason === "session_expired"` alongside `retryableSessionId` and `params.sessionKey` availability. If all conditions are met, it executes `await executePreparedCliRun(context, undefined)` to restart with a fresh session ID.

In `acp-spawn.ts`, `spawnAcpDirect()` performs initial policy validation through `isAcpEnabledByPolicy(cfg)`, returning error code `"acp_disabled"` if disabled. For `params.mode="session"` operations, it requires `params.thread === true`; failure results in `"thread_required"` error. The `resolveRuntimeCwdForAcpSpawn()` function resolves working directories with graceful error handling: if `fs.access()` fails, it uses `isMissingPathError()` to catch the exception and returns `undefined` rather than throwing.

In `auth-profiles/store.ts`, three core store functions manage authentication profiles. `loadAuthProfileStore()` returns `AuthProfileStore`, loading from disk with legacy and external overlays. `saveAuthProfileStore(store, agentDir?)` returns `void`, persisting secrets JSON and state. `updateAuthProfileStoreWithLock(params)` returns `Promise<AuthProfileStore | null>`, providing atomic load-update-save operations under file locking. These functions enable secure profile lifecycle management with thread-safety guarantees. Developers should validate these error conditions before calling respective functions and respect their return types for proper error handling workflows.

## src/gateway
The `GatewayClient.selectConnectAuth()` method resolves authentication sources in a fixed priority sequence. Explicit gateway tokens provided via `opts.token` hold highest precedence (Line 709). If absent, the method falls back to a resolved device token retrieved from `opts.deviceToken` or stored state (Lines 696-701). Next, bootstrap tokens (`opts.bootstrapToken`) serve as a conditional backup at Line 711, utilized only if both gateway and device tokens are missing. Stored credentials appear after these, followed by password options (`opts.password`) processed separately at Line 686. The resulting signature token chain at Line 717 confirms this hierarchy: `signatureToken: authToken ?? authBootstrapToken ?? undefined`.

Security protocols strictly govern connection establishment. If `GatewayClient.start()` encounters a plaintext `ws://` connection to a public hostname (non-loopback), it does not throw an exception. Instead, it constructs a specific error message—"SECURITY ERROR: Cannot connect to [host] over plaintext ws://"—and invokes the optional `onConnectError` callback (Line 245) before returning early (Line 246). This prevents credential interception attacks. To bypass this check, the environment variable `OPENCLAW_ALLOW_INSECURE_PRIVATE_WS=1` must be set. When configuring `authorizeGatewayConnect`, the serialization of Tailscale authentication attempts relies on the concurrent provision of a `rateLimiter` flag AND the `allowTailscaleHeaderAuth` parameter set to enabled (Lines 561-620). This mechanism also verifies `auth.allowTailscale` and excludes local direct connections (`!localDirect`).

Rate limiting logic distinguishes between missing and invalid credentials to manage counter balance. A `token_missing` response avoids burning rate-limit slots and skips `recordFailure` calls entirely (Line 463), protecting clients during initialization. Conversely, a `token_mismatch` response records a failure via `params.limiter?.recordFailure(params.ip, params.rateLimitScope)` (Line 466), ensuring repeated attempts contribute to quota exhaustion. Finally, the reconnect loop management in `GatewayClient.pause` reacts to specific error codes. General failures like `AUTH_RATE_LIMITED`, `PAIRING_REQUIRED`, or identity mismatches trigger an immediate pause (Line 328). Specifically for `AUTH_TOKEN_MISMATCH`, untrusted endpoints (non-loopback without TLS fingerprints) pause immediately. Trusted endpoints (loopback or valid TLS fingerprint) delay pausing until consuming one retry token.

## src/infra
To configure execution approval policies, set `tools.exec.ask` in your configuration file. Valid enum values are `"off"`, `"on-miss"`, and `"always"` (defined as type `ExecAsk`). The `normalizeExecAsk(value?: string | null)` function (line 52) processes inputs; invalid strings strip to null, causing fallback to `DEFAULT_ASK: ExecAsk = "off"` (line 170) following sanitization in `sanitizeExecApprovalPolicy`.

Socket path defaults define `~/.openclaw/exec-approvals.sock` (line 173). Resolution relies on `expandHomePrefix()` (lines 82-100), expanding tildes by checking `OPENCLAW_HOME`, then `HOME`, then `USERPROFILE` environment variables before falling back to `os.homedir()`.

Routing distinguishes approval semantics by `approvalId`. Adapters (matrix, discord, telegram) utilize `.startsWith("plugin:")` checks (lines 129, 1355). An ID prefixed with "plugin:" triggers plugin authorization logic; all other IDs route to standard exec handling.

Deduplication prevents redundant delivery within request cycles. `approval-native-runtime.ts` uses a `deliveredKeys` Set (line 82) indexed by `dedupeKey`. If a key exists, execution skips delivery, invoking `onDuplicateSkipped`. Additional session-level checks in `exec-approval-session-target.ts` validate recent deliveries via `isDuplicateDelivery()` (lines 39-54), ensuring idempotency against duplicate sends and warning logs.

Standard error formatting functions handle these exceptions systematically. Failure reporting structures `FileNotFoundError` events comprehensively. Error formatters collect `err.message` and `err.name` (line 71), traversing nested causes via `err.cause` (line 74). Execution results utilize `ExecProcessOutcome` types (lines 156-158), distinguishing `status` ("completed"|"failed"), `exitCode`, `durationMs`, and `aggregated` content including stderr/stdout captures. These fields categorize runtime errors via `failureKind` and `reason` strings.

## src/auto-reply
Module: Auto-Reply

1. **Heartbeat Filtering**: Within `extensions/memory-core/src/dreaming-phases.ts`, `runPhaseIfTriggered()` filters incoming messages during heartbeat phases. A message returns `undefined` if either condition is met: `params.trigger !== "heartbeat"` OR `!hasEventToken`. The `hasEventToken` function checks if `params.cleanedBody.trim().split(/\s+/).includes(params.eventText)`. Thus, processing advances only if `trigger` is exactly `"heartbeat"` AND the cleaned body contains the specific event text string.

2. **Command Argument Parsing**: Use `src/auto-reply/commands-registry.ts`'s `parseCommandArgs(command, rawArgsString?)` for `ChatCommandDefinition` objects. If `rawArgsString` is falsy, it returns `undefined`. If `command.argsParsing === "none"`, the result is `{ raw: trimmedString }`. Standard execution parses positional arguments using `parsePositionalArgs()`, returning `{ raw: trimmedString, values: { /* populated parameters */ } }`. Parameters with `captureRemaining: true` retain all subsequent tokens. Reverse conversion utilizes `serializeCommandArgs()`.

3. **Reply Dispatcher Setup**: To send replies using a pre-initialized dispatcher, call `core.channel.reply.withReplyDispatcher()` (lines 1117+ in `extensions/feishu/src/bot.ts`). Pass an object containing your `dispatcher` instance, a `run` callback for actual sending logic, and an optional `onSettled` callback for post-transmission state updates.

4. **Pending Queue Management**: Outbound traffic in `src/auto-reply/reply/reply-dispatcher.ts` uses a `pending` integer. Each outbound attempt increments `pending` (+1). The `.finally()` handler decrements `pending` (-1) after delivery completes (lines 146, 172). When `pending` hits 0 (line 180), `unregister()` fires to stop global tracking, followed by `options.onIdle?.()` (lines 182-183).

5. **Human-Delay Timing**: Delays inject natural pauses between block replies (`kind === "block"`). Delays apply only after the first block (`sentFirstBlock` flag, lines 149-151). Default timing draws a random ms between 800ms and 2500ms (constants in `config/schema.base.generated.ts`, lines 26, 4815). Disable via `humanDelay.mode: "off"` or specify `minMs`/`maxMs` via `"custom"` mode.

## src/config
This module governs configuration lifecycle operations, including writing, loading, and validating schema-compliant settings. When invoking `writeConfigFile(cfg)` with a new configuration object containing actual API keys (like `gateway.auth.token`), the actual values are persisted directly, overriding any environment variable reference templates. Logic resides in `src/config/io.write-prepare.ts`; templates (e.g., `${VAR}`) remain only for paths NOT tracked in `changedPaths` during the merge patch operation. Explicitly provided values replace prior snapshots completely.

`loadConfig()` handles missing files and invalid JSON structures gracefully by returning an empty object `{}`. Specifically, if `deps.fs.existsSync(configPath)` evaluates to false, or if `typeof effectiveConfigRaw !== "object"` at the root level (including null, strings, arrays), no error is thrown; instead, the caller receives an empty default state. This prevents crashes during initial setup or corrupted file reads.

Binding schemas enforce strict type discrimination between routing and authentication. Routes use `type: "route"` while authentication profiles require `type: "acp"`. Attempting to set `bindings[0].type = "route"` while including `authProfileId` triggers a schema validation error, typically stating "Unrecognized key: \"authProfileId\"" due to `additionalProperties: false` on the route branch. Consumers validate `binding.type === "acp"` before processing authorization.

Gateway network modes impose specific binding restrictions. When `gateway.tailscale.mode` is "serve" or "funnel", `gateway.bind` accepts "loopback" or "custom". The "custom" variant mandates `gateway.customBindHost="127.0.0.1"`. Suspicious modifications (e.g., size drops) create backups using the pattern `{configPath}.clobbered.{formatted_timestamp}`. Timestamps are processed via `formatConfigArtifactTimestamp`, replacing colons and periods with hyphens (e.g., `2026-01-01T00:00:00.000Z` becomes `2026-01-01T00-00-00-000Z`) to facilitate forensic restoration via unique identifiers.

## src/commands
The `src/commands` module governs agent configuration updates and backup lifecycle operations. Within `src/commands/agents.bindings.ts`, `applyAgentBindings()` processes `AgentRouteBinding` configurations. If an incoming binding matches an existing route key with a conflicting `agentId`, the returned `conflicts` array contains objects typed explicitly as `{ binding: AgentRouteBinding; existingAgentId: string }` (lines 78, 93, 103). Conversely, non-route bindings are identified by `!isRouteBinding(binding)` where the type is "acp" (line 10). These bindings are extracted at line 81 and appended directly to the final output at line 147, bypassing duplicate detection and upgrade logic defined for route bindings, which may result in unexpected state corruption.

For account resolution, `resolveBindingAccountId()` evaluates sources sequentially. First, it checks `params.explicitAccountId` (line 239). Second, it attempts to invoke `plugin?.setup?.resolveBindingAccountId?.()` (line 248). Third, if `plugin?.meta.forceAccountBinding` is true, it uses the result from `resolveDefaultAccountId()` (line 252).

In `parseBindingSpecs()`, input validation is strict. A spec like `telegram:test-account1` produces a valid binding. However, `discord:` triggers an error at lines 304-307 stating "Invalid binding ... (empty account id)" because the account section cannot be undefined but also lacks content. Similarly, `unknown` generates an error at lines 299-301 indicating "Unknown channel".

Backup integrity checks utilize `openclaw backup create --verify`. Verification execution depends on `opts.verify && !opts.dryRun` (line 24 in backup.ts). Developers must omit `--dry-run`; including both `--verify` and `--dry-run` disables actual verification checks according to line 26 in register.backup.ts. Always verify that `--verify` is executed without concurrent dry-run flags to ensure archive validation runs correctly.

## src/plugins
In `src/plugins/loader.ts`, `loadOpenClawPlugins()` enforces a strict coupling: setting `options.activate` to `false` mandates that `options.cache` is explicitly `false`. Violating this combination throws Error: "loadOpenClawPlugins: activate:false requires cache:false to prevent command registry divergence". This behavior prevents caching registries whose commands were never globally registered, protecting the integrity of the snapshot (non-activating) loads.

Developers access registered agent harnesses via `src/agents/harness/registry.ts`, utilizing functions such as `registerAgentHarness(harness: AgentHarness)` or `getRegisteredAgentHarness(id: string)`. Compaction providers handle registration through `src/plugins/compaction-provider.ts` using `registerCompactionProvider()`. Memory embedding providers manage availability through `src/plugins/memory-embedding-providers.ts` via `registerMemoryEmbeddingProvider()`.

Runtime registration mode determination follows a hierarchy in `src/plugins/loader.ts` (lines 1726-1744). An enabled plugin assumes "setup-runtime" mode if not validating and `shouldLoadChannelPluginInSetupRuntime` returns true. This check confirms a valid `setupSource`, channel definitions, and considers `preferSetupRuntimeForChannelPlugins` or startup defer states. If these fail, assignment defaults to "full". Conversely, plugins entering "setup-only" mode satisfy conditions involving `includeSetupOnlyChannelPlugins` and limited manifest configuration like `manifestRecord.channels.length > 0`.

Duplicate pluginId resolution uses `resolveCandidateDuplicateRank()` to order discovery candidates. Priority scores range from 0 to 4: "config" (0), "global" explicit (1), "bundled" (2), "workspace" (3), and others (4). Candidates sort ascending; the first occurrence processes normally into `seenIds`. Subsequent candidates sharing the same pluginId remain disabled, emitting errors containing "overridden by [existingOrigin] plugin".

Diagnostic warnings occur when bundle-format plugins declare capabilities unwired into OpenClaw. The diagnostic level emits "warn" specifically for unsupported categories such as "apps" and "rules". These checks evaluate inline capability values or files within the bundle root directory structure to identify missing wiring support.

## extensions/matrix
The Matrix extension implements message handling and account management through specific code modules. When calling a `send` action to post media, required parameters are `msgtype`, `body`, and `size`. Media content prioritization is defined in `extensions/matrix/src/matrix/send/media.ts` (lines 87-91): `file` (EncryptedFile) takes highest priority for E2EE, followed by `url` (MXC upload string). Optional parameters include `filename`, `mimetype`, `imageInfo`, and `durationMs`.

Account initialization occurs via the `startAccount` gateway in `extensions/matrix/src/channel.ts` (lines 439-486). Its workflow comprises: 1. Setting the account status context with accountId and baseUrl; 2. Enforcing a Promise-based lock mechanism to serialize dynamic imports and prevent race conditions between concurrent startups; 3. Lazily importing the monitor module (`./matrix/monitor/index.js`) and invoking `monitorMatrixProvider` with runtime configuration.

Security requirements mandate explicit allowlisting. If `channels.matrix.groupPolicy` is omitted or set to "open" while `channels.matrix.groups` defines rooms, the warning displays: "- Matrix rooms: groupPolicy=\"open\" allows any room to trigger". Remediation requires setting `channels.matrix.groupPolicy="allowlist"` and adding permitted rooms to `channels.matrix.groups`. Additionally, `channels.matrix.groupAllowFrom` can be configured to restrict which users may initiate actions.

Message action availability relies on gating flags defined in `extensions/matrix/src/actions.ts`. Actions available immediately by default include `poll` and `poll-vote`. Access to `send`, `read`, `edit`, and `delete` requires the `messages` gate to be enabled (lines 68-72), verified by `params.gate("messages")`. For profile updates via `set-profile`, developers can use `avatarUrl` or `avatar_url` (camel/snake case) for remote MXC:// or HTTP(s) URLs. For local uploads, `avatarPath` or `avatar_path` accepts filesystem paths, automatically uploading the file and assigning the resulting MXC URI (lines 35, 48).

## extensions/discord
The Discord extension orchestrates slash commands, native skills, and moderation via specific runtime modules. Configuration resolution primarily utilizes `resolveDiscordSlashCommandConfig()` in `extensions/discord/src/monitor/commands.ts`. Default ephemeral responses are enabled (`true`), determined by `raw?.ephemeral !== false` logic confirmed in `commands.ts` line 7 and verified in `commands.test.ts` lines 7-10. Access group policies default to `true` when `cfg.commands?.useAccessGroups !== false`, a check repeated in `provider.ts` (672), `native-command.ts` (422), and `security-audit.ts` (177).

Moderation operations export core functionality from `runtime-api.ts`, pulling shared definitions from `./src/actions/runtime.moderation-shared.js`. This module exposes `DiscordModerationAction` types (string union of "timeout", "kick", "ban") along with utility guards like `isDiscordModerationAction()` and `requiredGuildPermissionForModerationAction()`. This function maps actions to Discord PermissionFlagsBits. Additionally, `readDiscordModerationCommand()` extracts and validates moderation parameters. Execution logic resides in `runtime.moderation.ts`, providing `discordModerationActionRuntime` object methods including `banMemberDiscord`.

Native command enabling requires two separate flags: `commands.native` and `commands.nativeSkills`. Both accept `boolean` or `"auto"` strings and can be set at the global `channels.discord.commands` level or nested under `channels.discord.accounts.<id>.commands`. Initialization validation occurs in `provider.ts` at lines 658-662 using `resolveNativeCommandsEnabled()`.

During startup, the Application ID fetch follows config initialization but precedes command compilation in `monitorDiscordProvider`. Phases `fetch-application-id:start` (line 709) and `fetch-application-id:done` (line 723) track progress. If unresolved, `Error("Failed to resolve Discord application id")` is thrown at line 721. Finally, a limit of `maxDiscordCommands = 100` (line 731) governs spec counts. If both native modes are active and specs exceed 100, `skillCommands` are emptied to preserve `/skill` commands, triggering the "removing per-skill commands" log. Single native mode exceeds 100 without automatic adjustment, issuing only a potential failure warning.

## src/cli
The CLI execution startup presentation is governed by `applyCliExecutionStartupPresentation` in `src/cli/command-execution-startup.ts`. Lines 39-40 demonstrate that the function suppresses the CLI banner immediately if `params.startupPolicy.hideBanner` is truthy, `params.showBanner === false`, or `!params.version`. Execution executes an immediate `return` statement before attempting to call `emitCliBanner()` when any of these conditions are satisfied.

Subcommand registration modes are determined by `shouldEagerRegisterSubcommands()` in `src/cli/command-registration-policy.ts`. The environment variable `OPENCLAW_DISABLE_LAZY_SUBCOMMANDS` controls this toggle directly. When set to any truthy value, subcommands are registered eagerly via `registerSubCliCommands()` (line 245 of `register.subclis-core.ts`), passed to `registerCommandGroups()` to determine the registration mode using `PluginCliRegistrationMode = "eager" | "lazy"`. Otherwise, subcommands register lazily.

Invocation resolution within `src/cli/argv-invocation.ts` handles path extraction in `resolveCliArgvInvocation`. The `commandPath` is derived by invoking `getCommandPathWithRootOptions(argv, 2)`, processing the argv array starting from index 2. The primary command identifier is extracted separately via `getPrimaryCommand(argv)` imported at line 3.

Plugin loading is managed through the `loadPlugins` property in `CliCommandPathPolicy` defined in `src/cli/command-catalog.ts`. Three modes exist: "always" (unconditional loading regardless of output mode), "text-only" (loading skipped during JSON output), and "never" (zero plugin initialization). This policy dictates available commands during execution and impacts performance versus feature parity.

Configuration validation guards are bypassed explicitly for security-admin commands. Paths including "backup", "doctor", "secrets", "completion", and "config" (subcommands "validate" and "schema") utilize `policy: { bypassConfigGuard: true }` in `src/cli/command-catalog.ts` (lines 93-113). For config subcommands, the policy uses `exact: true` alongside the guard bypass. These entries may also include `hideBanner: true` settings alongside the guard bypass mechanism.

## extensions/telegram
The `extensions/telegram` module handles bot lifecycle, threading, and configuration validation. When sending messages to Telegram channel topics, developers must include the optional `messageThreadId` parameter to direct output to the correct thread rather than the default chat (src/send.ts:97-98). This integer parameter specifies the unique thread ID. The system distinguishes posting to a regular chat versus a channel topic forum; regular chats do not require thread parameters, whereas channels auto-determine scope type as "forum" vs "dm" (src/send.ts:56). Errors are handled by validating thread existence with retries if not found.

Configuration conflicts occur when multiple gateway accounts share a token. The `isConfigured` function in `shared.ts` (lines 160-177) detects duplicates via `findTelegramTokenOwnerAccountId`. If a duplicate is found, the account status becomes "not configured" (lines 189-199), generating an error message: "Duplicate Telegram bot token: account "[accountId]" shares a token with..." (shared.ts:61).

Conversation hierarchy relies on consistent `parentConversationId` assignment. For commands, `native-command.ts` (line 888) sets `threadParentId` through `resolveDiscordThreadParentInfo()`. Similarly, inbound messages in `message-handler.preflight.ts` (line 624) set `earlyThreadParentId` using the same resolution logic (lines 589-604), ensuring unified context regardless of message origin, whether user input or automated updates.

Thread binding states are managed by the `createTelegramThreadBindingManager` function (thread-bindings.ts:410). This manager initializes three critical parameters: `idleTimeoutMs` (default 24 hours per thread-bindings.ts:24) to expire inactive threads, `maxAgeMs` (default 0, disabling absolute expiration per thread-bindings.ts:25), and `persist` (default true) to save state across application restarts.

Target resolution distinguishes between usernames and IDs. Resolving a DM via `@username` fails silently if the bot token is missing (channel.ts:526), returning `resolved: false` with the note "Telegram bot token is required to resolve @username targets." Conversely, group membership data exists in persistent config (channels.telegram.groups), allowing static entries in allowlists to bypass runtime token checks (channel.ts:545). When resolving groups, the system calls `lookupTelegramChatId` (channel.ts:545), though the initial failure mode for missing tokens applies to users specifically.

## extensions/feishu
The Feishu extension integrates with the `@larksuiteoapi/node-sdk` library through `extensions/feishu/src/chat.ts` and `src/docx.ts`. Chat retrieval utilizes `client.im.chat.get` and `client.contact.user.get`, while document conversion uses `client.docx.document.convert()`. Both rely on imported `Lark.Client` instances (src/chat.ts:1, src/docx.ts:1).

Message dispatch enforces strict structural rules. For thread replies, specify `action: "thread-reply"` with a mandatory `messageId`; omitting this throws `Feishu thread-reply requires messageId.` (src/channel.ts:664). The `to` parameter accepts `chat_id` (formats like `oc_group_1`, `ou_user`), `user_id`, or email. Resolved IDs are obtained via `resolveFeishuMessageId(ctx.params)` if `action === "thread-reply"` (src/channel.ts:662). Sending `card` and `mediaUrl` concurrently raises `Feishu ${ctx.action} does not support card with media.` (src/channel.ts:674), separating interactive card logic from native media delivery routes.

Markdown-to-document conversion has table restrictions. Creating block types 31 (Table) or 32 (TableCell) via `documentBlockChildren.create` triggers error 1770029 (src/docx.ts:93). Standard insertion skips these in `cleanBlocksForInsert()`. Developers must use `insertBlocksWithDescendant`, leveraging `documentBlockDescendant.create()` instead. Tables remain readable via `list_blocks`, but individual cell updates are preferred over children creation.

Feature gating controls interaction capabilities. Reactions ('react', 'reactions') depend on boolean flags in `channels.feishu.actions.reactions` or `accounts.<account_id>.config.actions.reactions` (src/config-schema.ts:8). If disabled, tools exclude these actions from discovery schemas. Direct invocation results in `Unsupported Feishu action: react` exceptions (src/channel.ts:253). Core channel logic resides in `src/channel.ts`, managing send entries and validation flows at lines 657-674.

## extensions/browser
## Browser Extension Module Reference Summary

**BrowserToolSchema Actions**: The Browser API supports 16 valid actions: `status, start, stop, profiles, tabs, open, focus, close, snapshot, screenshot, navigate, console, pdf, upload, dialog, act`. While TypeScript allows unknown strings at compile time, runtime validation via `stringEnum()` will reject unrecognized actions (e.g., `'runCommand'`) with a validation error before handler execution.

**Profile Name Validation**: Enforced at `profiles-service.ts:92` BEFORE existence checks. Regex pattern `/^[a-z0-9][a-z0-9-]*$/` requires names to start with a lowercase letter or digit, followed by lowercase letters, digits, or hyphens only. Max length: 64 characters. Invalid names throw `BrowserValidationError` with message "use lowercase letters, numbers, and hyphens only".

**Profile Creation Parameters**: Use `driver: "existing-session"` to attach to a running Chrome instance. When `userDataDir` is provided, it MUST accompany `driver="existing-session"` or throw "driver=existing-session is required when userDataDir is provided" (line 115). If omitted with existing-session, the system uses Chrome's default user data path without error (conditional spread at line 150).

**CDP Port Management**: Standard profiles allocate from default range 18800-18899 (100 ports total). On exhaustion, `allocateCdpPort()` returns `null`, triggering `BrowserResourceExhaustedError` ("no available CDP ports in range"). Custom ranges override defaults when specified.

All validations prioritize format checking before resource availability checks to fail-fast on invalid input.

## extensions/qa-lab
The `qa-lab` extension governs test execution and credential management through rigid CLI structures and typed scenarios. Executing `qa suite` requires the `--runner` flag, accepting strictly `host` or `multipass` as valid options; invalid selections trigger an error at `cli.runtime.ts:345`. The system defaults the runner kind to `host` within `cli.ts:227`. When developing custom QA runners, developers must register commands via `registerQaLabCli`. This process prevents naming collisions through `assertNoQaSubcommandCollision` (line 611), which iterates existing subcommands using `qa.commands.some()`. If a duplicate name is detected, the system throws an Error stating: "QA runner command \"{commandName}\" conflicts with an existing qa subcommand", halting registration immediately. This validation executes before the lane object is registered, ensuring the CLI tree remains clean.

Managing credentials involves security-conscious filtering. Listing credentials normally hides secrets, but applying `--show-secrets` to `qa credentials list` (defined at `cli.ts:470`) modifies the output stream significantly. While a standard table appears first, the `cli.runtime.ts:648` logic injects a Payloads section when `opts.showSecrets` is true. Each entry displays the `credentialId` followed by a JSON-stringified representation of `credential.payload ?? null` (`cli.runtime.ts:671`). To add credentials, users invoke `qa credentials add` which requires two mandatory arguments defined in `runQaCredentialsAddCommand` (`cli.runtime.ts:543-550`): a string `--kind` (e.g., "telegram") and a path `--payload-file`.

Test scenarios define conditional logic within `scenario-catalog.ts` using the `qaFlowIfShapeBase` structure (lines 127-132). This shape accommodates expressions for branching into `then` or `else` arrays. Recovery logic operates inside try/catch blocks defined at `scenario-catalog.ts:154-160`, which accept three distinct action types: `call` for executing API tools (`schema.ts:96-100`), `set` for modifying state variables (`schema.ts:102-105`), and `assert` for validating expressions (`schema.ts:107-115`). These actions are valid within any flow array—actions, catch, or finally—allowing robust error handling and state restoration throughout the execution lifecycle. All schema validations employ Zod types to prevent runtime mismatches.

## extensions/memory-core
The `extensions/memory-core` module implements critical synchronization and retrieval logic to ensure stability during process bootstrapping and query execution, relying on specific file-level implementations for core behaviors.

Within `MemoryIndexManager.search()`, integrity checks prevent failure on empty indices. At line 306 of `manager.ts`, if `hasIndexedContent()` evaluates to false, the system executes `await this.sync({ reason: "search", force: true })`. This forced synchronous bootstrap guarantees that initial lookups following a restart return data instead of failing closed (manager.ts:305-307). Additionally, session initialization relies on `warmSession()` in `manager.ts` (lines 273-287). When an agent starts a session with `onSessionStart` enabled, this method verifies `this.settings.sync.onSessionStart` and triggers `void this.sync({ reason: "session-start" })` to load the necessary index state before interaction.

Retrieval tools exhibit distinct behaviors based on configuration. The `memory_get` tool handles the `corpus='wiki'` flag by completely bypassing the builtin memory backend, querying only registered compiled-wiki supplements (tools.ts:344). If no supplement corresponds to the file path, the response object includes `{ path: <relPath>, text: "", disabled: true, error: "wiki corpus result not found" }` (tools.ts:357), ensuring developers are explicitly notified of unregistered sources. For `memory_search`, result formatting depends on the backend mode. When using QMD mode, the system enforces character injection limits via `clampResultsByInjectedChars()` against `resolved.qmd?.limits.maxInjectedChars` (tools.ts:253), processing results sequentially while tracking the remaining character budget and stopping entirely once exhausted. Conversely, the builtin mode returns decorated results without applying any budget enforcement or truncation logic.

Finally, scoring thresholds interact dynamically with hybrid search weights. If `MemorySearchTool` minScore exceeds `hybrid.textWeight` (default 0.3), the system automatically relaxes the threshold using `Math.min(minScore, hybrid.textWeight)` (manager.ts:424). This ensures keyword-only lexical matches are not inadvertently filtered out when `minScore` values sit above the weight baseline. Strict filtering occurs subsequently for entries where `entry.score >= minScore` (manager.ts:446). Developers must consider this automatic relaxation strategy to maintain recall while tuning search sensitivity parameters.

## extensions/msteams
Microsoft Teams extension authentication relies on `channels.msteams.authType`, supporting two explicit modes: `"secret"` and `"federated"`. `MSTeamsSecretCredentials` (used for `"secret"`) mandates `appId`, `appPassword`, and `tenantId` at `cfg.channels.msteams.appPassword`, `cfg.channels.msteams.appId`, and `cfg.channels.msteams.tenantId`. `MSTeamsFederatedCredentials` (used for `"federated"`) requires `appId` and `tenantId`, supplemented by either certificate details (`certificatePath`, `certificateThumbprint`) or managed identity flags (`useManagedIdentity`, `managedIdentityClientId`). Configuration validation confirms only `"secret"` or `"federated"` are accepted.

Token resolution is handled by `resolveDelegatedAccessToken()` in `extensions/msteams/src/token.ts`. At line 177, it verifies expiration. If tokens are expired but `refreshToken` exists, it invokes `refreshMSTeamsDelegatedTokens()` with the stored `refreshToken` (line 187). Upon success, it executes `saveDelegatedTokens(refreshed)` (line 190) and returns the new access token.

Outbound routing is managed by `resolveMSTeamsOutboundSessionRoute()` in `extensions/msteams/src/session-route.ts` (line 9). Accepting `ChannelOutboundSessionRouteParams`, it returns a `ChannelOutboundSessionRoute`. Peer identification distinguishes users via `lower.startsWith("user:")` and channels via `/@@thread\\.tacv2/i` regex. Addresses are constructed with "msteams" prefixes using `buildChannelOutboundSessionRoute()`.

Messaging operations define four core async functions with `MSTeams` suffixes. `sendMessageMSTeams` (line 99), `editMessageMSTeams` (line 564), and `deleteMessageMSTeams` (line 607) are exported in `extensions/msteams/src/send.ts`. `reactMessageMSTeams` is exported in `extensions/msteams/src/send.reactions.ts` (line 68).

Security policies depend on `channels.msteams.groupPolicy` in `extensions/msteams/src/channel.ts` (line 87). Setting this to `"open"` generates a warning stating any member can trigger mention-gated content. Mitigation requires setting `channels.msteams.groupPolicy="allowlist"` plus `channels.msteams.groupAllowFrom` to restrict sender permissions.

---

# Verified corrections (trust these over the summaries)

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
