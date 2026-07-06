# OpenClaw — corrections from studying (your beliefs vs. this repository)

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
- **You believe the error reason values that trigger the retry are `'SESSION_EXPIRED'` or `'INVALID_SESSION'` (using uppercase letters with underscores).** The actual behavior requires `err.reason` to match `"session_expired"` (strictly lowercase), and additionally validates that both `retryableSessionId` and `params.sessionKey` exist.
  > `src/agents/cli-runner.ts:99`: `if (err.reason === "session_expired" && retryableSessionId && params.sessionKey) {`
- **You believe the functions for loading and saving the authentication profile store are named generically like "loadProfile" and "saveProfile" rather than identifying the specific exports defined in the source file.** The three functions are "loadAuthProfileStore()" returning "AuthProfileStore", "saveAuthProfileStore()" returning "void", and "updateAuthProfileStoreWithLock()" returning "Promise<AuthProfileStore | null>".
  > `src/agents/auth-profiles/store.ts:151`: `export function loadAuthProfileStore(): AuthProfileStore {`
- **You believe that when calling `spawnAcpDirect()` with `params.mode="session"`, the `thread` parameter must be set to `NULL` and the function returns error code `ACPE_E_INVALID_THREAD_MODE`.** The `params.thread` parameter must be set to `true` to bind the ACP session to a thread, and the error code returned if unmet is `thread_required`.
  > `src/agents/acp-spawn.ts:1049`: `errorCode: "thread_required",`
- **You believe the function throws an exception or attempts to create the directory if the resolved path does not exist.** It catches the access error and returns `undefined` to allow the caller to handle the absence of the specified directory.
  > `src/agents/acp-spawn.ts:504`: `return undefined;`
- **You believe that a retry attempt without a session ID cannot occur because the system is designed to require valid session identifiers (`retryableSessionId` and `params.sessionKey`) before allowing any retry logic to execute.** The function actually executes a retry attempt without a session ID specifically when the error reason is `"session_expired"`, calling `executePreparedCliRun()` with `undefined` to ensure a new CLI session is created rather than continuing with expired context.
  > `src/agents/cli-runner.ts:105`: `const output = await executePreparedCliRun(context, undefined);`

## src/gateway
- **You believe that the `GatewayClient.selectConnectAuth()` method prioritizes JWT/Token-based Authentication from an `AuthenticationSourceList` followed by TLS Certificate-based Authentication.** The method actually prioritizes `explicitGatewayToken` over `resolvedDeviceToken` using the nullish coalescing operator to build the `authToken`, before potentially using `explicitBootstrapToken`.
  > `src/gateway/client.ts:709`: `const authToken = explicitGatewayToken ?? resolvedDeviceToken;`
- **You believe that GatewayClient.start() throws a SecurityException when connecting via ws:// to a public hostname.** GatewayClient.start() does not throw an exception directly; instead, it creates an Error object and calls the optional onConnectError callback.
  > `src/gateway/client.ts:235`: `const error = new Error(
    `SECURITY ERROR: Cannot connect to "${displayHost}" over plaintext ws://. ` +
      "Both credentials and chat data would be exposed to network interception.")`
- **you believe the rate limiter distinguishes between 'token_missing' and 'token_mismatch' by evaluating the presence and validity of the token before invoking the final authorization checks.** The distinction depends on whether a failed authentication attempt is recorded in the rate limiter; 'token_missing' returns immediately without calling recordFailure() to avoid burning rate-limit slots, whereas 'token_mismatch' explicitly calls recordFailure() before returning to count against the limit.
  > `src/gateway/auth.ts:460`: `// Don't burn rate-limit slots for missing credentials — the client
// simply hasn't provided a token yet (e.g. bare browser open).
// Only actual *wrong* credentials should count as failures.`

## src/infra
- **You believe the exact file path for the default socket path configuration for exec approvals cannot be localized to a specific repository file, and that tilde expansion relies on standard shell processing or generic library functions like `os.path.expanduser`.** The configuration is explicitly declared in `src/infra/exec-approvals.ts` as the constant `DEFAULT_SOCKET`, and tilde expansion is resolved using a custom `expandHomePrefix` function that prioritizes specific environment variables (`OPENCLAW_HOME`, `HOME`, `USERPROFILE`) over system defaults.
  > `src/infra/exec-approvals.ts:173`: `const DEFAULT_SOCKET = "~/.openclaw/exec-approvals.sock"`
- **you believe the system determines whether to use exec or plugin semantics based on the registration configuration or metadata definition of the custom approval handler within the system's registry.** The system determines whether an incoming approval uses exec or plugin semantics by examining the approvalId field and checking if it starts with the prefix "plugin:".
  > `extensions/matrix/src/exec-approvals.ts:129`: `return request.id.startsWith("plugin:") ? "plugin" : "exec";`
- **You believe that standard error formatting functions provide detailed system metadata including timestamps, execution locations, user session IDs, and allowlist violation checks.** Standard error formatting functions only collect the error message, error name, optional errno code, and cause chain, ensuring sensitive content is redacted before returning or logging.
  > `src/infra/errors.ts:71`: `formatted = err.message || err.name || "Error";`

## src/auto-reply
- **You believe a user message is filtered out during heartbeat processing if the heartbeat verification indicates that the connection is stale, expired, or compromised, such as when the message arrives after the configured timeout period or fails required security/integrity checks for the heartbeat session.** A message is filtered out specifically when `params.trigger` is not equal to "heartbeat" OR the cleaned message body does not contain the expected event text verified via `hasEventToken`.
  > `extensions/memory-core/src/dreaming-phases.ts:1679`: `if (params.trigger !== "heartbeat" || !hasEventToken) {`
- **You believe you typically need to manually identify command prefixes, tokenize payloads, and map tokens to schemas to parse arguments from a chat command string.** You should invoke the `parseCommandArgs()` function located in `src/auto-reply/commands-registry.ts` instead of writing custom parsing logic.
  > `src/auto-reply/commands-registry.ts:192`: `export function parseCommandArgs(
  command: ChatCommandDefinition,
  raw?: string,
): CommandArgs | undefined {`
- **You believe the `dispatch` method should be called on the pre-created ReplyDispatcher instance.** To dispatch an inbound message, you should call `withReplyDispatcher()` on the channel reply object (such as `core.channel.reply`), passing the pre-created dispatcher instance within the configuration object arguments rather than calling a method directly on the dispatcher.
  > `extensions/feishu/src/bot.ts:1117`: `await core.channel.reply.withReplyDispatcher({`
- **You believe the dispatcher allows the system to continue accepting new outbound or inbound requests without restriction once the pending count drops to zero.** In reality, when pending reaches zero, the dispatcher unregisters from global tracking to prevent new inbound messages from being routed and invokes the idle callback; outbound replies will not be processed until pending is incremented again.
  > `src/auto-reply/reply/reply-dispatcher.ts:182`: `unregister();`
- **You believe there is no specific information documenting when human-like delays occur in block reply sequences or their default range values.** Human-like delays occur only for block replies after the first block in a multi-block reply sequence, with a configured default range of 800ms minimum to 2500ms maximum.
  > `src/auto-reply/reply/reply-dispatcher.ts:149`: `// Determine if we should add human-like delay (only for block replies after the first).`

## src/config
- **You believe there is no specific information about `writeConfigFile()` behavior regarding environment variable reference templates when API keys are present, and that templates would typically be preserved as-is** Actual values are persisted directly to the config file, replacing any previously stored environment variable reference templates for those specific paths; only unchanged paths get their env var reference templates restored from the snapshot via merge patch logic
  > `src/config/io.write-prepare.ts:310`: `if (!isPathChanged(path, changedPaths)) {`
- **You believe backup files are stored in the `~/.openclaw/backups/` directory with the naming convention `<timestamp>_<original_filename>`.** Backup files follow the pattern `{configPath}.clobbered.{formatted_timestamp}`, where timestamps replace colons and dots with hyphens using `formatConfigArtifactTimestamp`.
  > `src/config/io.ts:511`: `const targetPath = `${params.configPath}.clobbered.${formatConfigArtifactTimestamp(params.observedAt)}`;`
- **You believe there is no specific information about authentication binding configuration validation errors related to setting `bindings[0].type = "route"` in the provided study notes.** You must set `bindings[0].type` to `"acp"` for authentication bindings because the `"route"` type uses `additionalProperties: false` and excludes `authProfileId`, resulting in a schema validation error if included.
  > `src/config/schema.base.generated.ts:17648`: `additionalProperties: false,`
- **You believe that setting `bindings[0].type = "route"` fails validation because the schema configuration excludes the `authProfileId` property via `additionalProperties: false`.** Authentication bindings require `type="acp"` to function because the architecture separates transient routing (`type="route"`) from persistent ACP harness bindings (`type="acp"`), which are necessary for managing authentication profiles.
  > `src/config/schema.base.generated.ts:17793`: `Top-level binding rules for routing and persistent ACP conversation ownership. Use type=route for normal routing and type=acp for persistent ACP harness bindings.`

## src/commands
- **You believe that the 'conflicts' array returned by applyAgentBindings() contains objects documenting the conflicting route key, both agent IDs involved (existing and new), and conflict metadata including the operation being attempted.** The 'conflicts' array contains objects with exactly two fields: `binding` (the incoming/new AgentRouteBinding that caused the conflict) and `existingAgentId` (a string containing only the agentId that was previously assigned to that route key before the attempt to change it).
  > `src/commands/agents.bindings.ts:78`: `conflicts: Array<{ binding: AgentRouteBinding; existingAgentId: string }>`
- **You believe the behavior of non-route bindings in applyAgentBindings() is not documented and would require inspecting source code to determine if they cause unexpected issues.** Non-route bindings are actually extracted and directly appended to the output configuration without undergoing duplicate detection, conflict analysis, or upgrade logic. They bypass all safety checks applied to route bindings, which can lead to state corruption from duplicate ACP configurations.
  > `src/commands/agents.bindings.ts:147`: `bindings: [...existingRoutes, ...added, ...nonRouteBindings]`
- **You believe the function prioritizes runtime context/store and configuration defaults for account resolution.** You must first check the explicit parameter, then the plugin callback, and finally the conditional default bind.
  > `src/commands/agents.bindings.ts:248`: `if (pluginAccountId?.trim()) {`
- **You believe there is no information available regarding backup archive verification options or dry-run behavior in the provided context.** Verification is enabled using the `--verify` flag on the `openclaw backup create` command, and it will execute properly only if the `--dry-run` option is NOT included in the arguments.
  > `src/commands/backup.ts:24`: `if (opts.verify && !opts.dryRun) {`

## src/plugins
- **You believe registered agent harnesses are accessed through `src/plugin-sdk/core.ts`, compaction providers are registered in `extensions/memory-core/qmd-manager.ts`, and all components follow a generic plugin lifecycle contract with file-system scanning discovery.** Registered agent harnesses are accessed via `src/agents/harness/registry.ts` using dedicated registry modules with global symbol-based singleton patterns. Compaction providers, memory embedding providers, and conversation binding handlers each have their own dedicated registry files following the same symbol-based singleton API pattern for process-wide storage with paired getter/register functions.
  > `src/plugins/compaction-provider.ts:50`: `const COMPACTION_PROVIDER_REGISTRY_STATE = Symbol.for("openclaw.compactionProviderRegistryState")`
- **you believe the diagnostic level is 'warning' and that typically 5-8 major capability domains trigger warnings for unimplemented bundle-format plugins.** the diagnostic level emitted is 'warn', and exactly 2 capability categories ('apps' and 'rules') trigger warnings.
  > `src/plugins/loader.ts:1780`: `level: "warn"`

## extensions/matrix
- **You believe that `content` is a valid parameter name for specifying media content with a fallback priority after `file`, and that the validation logic is primarily located in `handler.ts`.** The valid media specification parameters are `file` (highest priority), `url` (secondary fallback), `filename`, `mimetype`, and `imageInfo`. When both `file` and `url` are provided, `file` takes precedence and `url` is ignored; the code evaluates this condition directly in `media.ts`.
  > `extensions/matrix/src/matrix/send/media.ts:87`: `if (!params.file && params.url) {`
- **You believe there is no information available about the 'startAccount' gateway method implementation for the Matrix plugin in the provided notes.** The 'startAccount' gateway method is implemented in `extensions/matrix/src/channel.ts` within the `gateway` export structure (lines 439-486), handling context setup, startup serialization, and module loading with provider invocation.
  > `extensions/matrix/src/channel.ts:440`: `startAccount: async (ctx) => {`
- **You believe the security warning is described generically as "a security warning regarding unrestricted group access" without knowing the exact text displayed by the system.** The system displays an exact warning message stating: "- Matrix rooms: groupPolicy="open" allows any room to trigger (mention-gated). Set channels.matrix.groupPolicy="allowlist" + channels.matrix.groups (and optionally channels.matrix.groupAllowFrom) to restrict rooms." To fix this, you must set channels.matrix.groupPolicy="allowlist", configure channels.matrix.groups with your allowed rooms list, and optionally set channels.matrix.groupAllowFrom for additional user/group restrictions.
  > `extensions/matrix/src/channel.directory.test.ts:251`: `'- Matrix rooms: groupPolicy="open" allows any room to trigger (mention-gated). Set channels.matrix.groupPolicy="allowlist" + channels.matrix.groups (and optionally channels.matrix.groupAllowFrom) to restrict rooms.'`
- **You believe the provided note does not contain specific details regarding which Matrix message actions are available by default versus those requiring the 'messages' gate flag.** The `extensions/matrix/src/actions.ts` file explicitly initializes `poll` and `poll-vote` without gating, while `send`, `read`, `edit`, and `delete` are added conditionally only when `params.gate("messages")` is enabled.
  > `extensions/matrix/src/actions.ts:67`: `const actions = new Set<ChannelMessageActionName>(["poll", "poll-vote"]);`

## extensions/discord
- **You believe that the provided documentation lacks information regarding `runtime-api.ts` and its exports for Discord moderation operations, asserting that `runtime.moderation-shared.ts` and `runtime.moderation.ts` are not documented within the study notes.** The documentation confirms that `runtime-api.ts` re-exports core moderation functionality from `runtime.moderation-shared.ts` (shared logic/types) and `runtime.moderation.ts` (guild execution), enabling external access to actions like banning, kicking, and timeout handling.
  > `extensions/discord/runtime-api.ts:3`: `export * from "./src/actions/runtime.moderation-shared.js";`
- **You believe that the provided repository map and study notes do not contain specific information regarding the Discord extension startup phase timing, the error condition if application ID verification fails, or the logging phases surrounding this operation.** The system actually fetches and verifies the Discord application ID immediately after configuration initialization and allowlist resolution but before command specification compilation; if the ID resolves as falsy, it throws a specific error, while logging markers denote the start and completion of the fetch operation.
  > `extensions/discord/src/monitor/provider.ts:721`: `throw new Error("Failed to resolve Discord application id")`
- **You believe that the provided study notes do not document the specific configuration conflict conditions between native skills and standard commands nor the relationship between maxDiscordCommands limits and skill command prioritization.** When both nativeEnabled and nativeSkillsEnabled are true and the total number of command specs exceeds the maxDiscordCommands limit (100), the system clears all per-skill commands to prioritize native commands (/skill).
  > `extensions/discord/src/monitor/provider.ts:755`: `discord: ${initialCommandCount} commands exceeds limit; removing per-skill commands and keeping /skill.`

## src/cli
- **You believe the provided study notes from the OpenClaw repository contain no information about the function `applyCliExecutionStartupPresentation` or the logical conditions determining when it should NOT emit the CLI banner.** The function determines it should NOT emit the CLI banner if any of the following conditions are met: `params.startupPolicy.hideBanner` is truthy, `params.showBanner` equals false, or `params.version` is falsy. If so, the function returns immediately before emitting the banner.
  > `src/cli/command-execution-startup.ts:39`: `if (params.startupPolicy.hideBanner || params.showBanner === false || !params.version) {`
- **You believe that the implementation details of `resolveCliArgvInvocation`, including how it derives `commandPath` from the raw argv array or what specific helper function extracts the primary command identifier, are not documented in the repository study notes.** The `commandPath` is derived by calling `getCommandPathWithRootOptions(argv, 2)`, which processes the argv array starting from index 2 (skipping the node executable and application binary). The primary command identifier is extracted by calling the helper function `getPrimaryCommand(argv)`.
  > `src/cli/argv-invocation.ts:19`: `commandPath: getCommandPathWithRootOptions(argv, 2),`
- **You believe there is no information available about the 'loadPlugins' property within a CliCommandPathPolicy or its three possible values, and that you would need to consult source code or configuration files to find this documentation.** The three possible values for the 'loadPlugins' property are "always", "text-only", and "never". Always loads plugins regardless of output mode; text-only loads plugins only when not in JSON mode; never skips plugin loading entirely during command execution to minimize startup overhead.
  > `src/cli/command-catalog.ts:1`: `export type CliCommandPluginLoadPolicy = "never" | "always" | "text-only";`
- **You believe there is no specific information documented regarding which command paths in the cliCommandCatalog are configured to bypass the default ConfigGuard check.** Six specific command paths ("backup", "doctor", "completion", "secrets", "config validate", "config schema") are configured to bypass the check via `{ bypassConfigGuard: true }` in the cliCommandCatalog, and the ensureCliCommandBootstrap function respects this by checking params.skipConfigGuard to conditionally skip loading the ConfigGuard module.
  > `src/cli/command-catalog.ts:93`: `{ commandPath: ["backup"], policy: { bypassConfigGuard: true } }`

## extensions/telegram
- **You believe the specific parameters required for Telegram channel topics and the distinction between regular chats and channel topics are not available in the provided study notes.** When sending a message to a Telegram channel topic, you must include the `messageThreadId` parameter to ensure the message is directed to the correct thread rather than the default chat.
  > `extensions/telegram/src/send.ts:98`: `messageThreadId?: number;`
- **You believe the provided notes lack detailed content descriptions covering the error condition for duplicate Telegram tokens, lifecycle handler location, or specific actions taken.** The system marks the duplicate account as `not configured` because it cannot acquire the token assigned to another account; detection occurs in `extensions/telegram/src/shared.ts` (specifically `isConfigured`), where `findTelegramTokenOwnerAccountId` identifies the conflict.
  > `extensions/telegram/src/shared.ts:176`: `return !findTelegramTokenOwnerAccountId({ cfg, accountId: account.accountId });`
- **You believe there is no specific information available regarding the Telegram thread binding manager initialization within the channel plugin in the provided study notes.** The Telegram thread binding manager is initialized within `extensions/telegram/src/thread-bindings.ts` in the `createTelegramThreadBindingManager` function. The three main configuration parameters are `idleTimeoutMs`, `maxAgeMs`, and `persist`.
  > `extensions/telegram/src/thread-bindings.ts:410`: `export function createTelegramThreadBindingManager(`
- **you believe this specific behavior regarding Telegram username resolution failure with an incorrect bot token is not documented in the provided repository notes or correction materials** the target resolver explicitly returns `resolved: false` with the note "Telegram bot token is required to resolve @username targets." when attempting to resolve a direct-message target using only a Telegram @username input with an incorrectly configured bot token
  > `extensions/telegram/src/channel.ts:528`: `return params.inputs.map((input) => ({
  input,
  resolved: false as const,
  note: "Telegram bot token is required to resolve @username targets.",
))`

## extensions/feishu
- **you believe that `threadId` is the appropriate identifier for replying to a thread and that the `withReplyDispatcher` method is the correct mechanism to invoke the Feishu action API without needing a specific action string or guaranteed mandatory fields like `messageId`.** you must explicitly set the action string to `"thread-reply"` and ensure a mandatory `messageId` parameter is included in the `params`; omitting `messageId` triggers a runtime error validating thread reply requirements.
  > `extensions/feishu/src/channel.ts:664`: `if (ctx.action === "thread-reply" && !replyToMessageId) {
              throw new Error("Feishu thread-reply requires messageId.");
}`
- **You believe the core client function for fetching Feishu chat information is located in `extensions/feishu/bot.ts`.** The core function is actually located in `extensions/feishu/src/chat.ts` and wraps SDK methods such as `client.im.chat.get` to retrieve chat data.
  > `extensions/feishu/src/chat.ts:17`: `const res = await client.im.chat.get({ path: { chat_id: chatId } });`
- **You believe there is no information available about converting markdown containing block_type=31 (Table) or block_type=32 (TableCell) into documentBlockChildren.create payloads, suggesting this functionality may not exist or is undocumented.** When attempting to convert markdown with table blocks (31/32) directly to documentBlockChildren.create, it returns error 1770029 because Tables cannot be created via this endpoint. Use the Descendant API (documentBlockDescendant.create) via insertBlocksWithDescendant instead to properly handle table conversions.
  > `extensions/feishu/skills/feishu-doc/references/block-types.md:78`: `**Important:** Table blocks CANNOT be created via the `documentBlockChildren.create` API (error 1770029).`
