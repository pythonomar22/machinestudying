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
- **You believe that the specific error message shown when attempting to spawn an ACP session with `sandbox='require'` from a requester already running in sandboxed mode is "Error: ACP session creation failed; requester is already in sandboxed mode."** The request actually fails immediately with the message: "Sandboxed sessions cannot spawn ACP sessions because runtime=\"acp\" runs on the host. Use runtime=\"subagent\" from sandboxed sessions." This happens because the code validates the requester's sandbox status before processing the `sandbox` parameter.
  > `src/agents/acp-spawn.ts:175`: `return 'Sandboxed sessions cannot spawn ACP sessions because runtime="acp" runs on the host. Use runtime="subagent" from sandboxed sessions.';`
- **You believe that when calling `acpSpawn()` with `thread=true` but without specifying the `mode` parameter, the function will utilize the **Default** (or **Standard**) spawn mode.** The function will actually use the **"session"** spawn mode (which resolves to the runtime mode **"persistent"**) because `thread=true` triggers a specific fallback logic in `resolveSpawnMode` rather than using a generic default. When `threadRequested` is true and no explicit mode is provided, the logic defaults to "session".
  > `src/agents/acp-spawn.ts:334`: `// Thread-bound spawns should default to persistent sessions.`

## src/gateway
- **You believe that without provided input context, you cannot identify the specific implementation details of the channel manager's exponential backoff policy, including its file location and configuration values for `initialMs`, `maxMs`, `factor`, and `jitter`.** The exponential backoff policy is explicitly defined in **`src/gateway/server-channels.ts`** at lines 22-27 as the `CHANNEL_RESTART_POLICY` constant, with `initialMs: 5_000`, `maxMs: 5 * 60_000` (300,000ms), `factor: 2`, and `jitter: 0.1`. This policy is actively used on line 460 via `computeBackoff()` when channels fail to start.
  > `src/gateway/server-channels.ts:22`: `const CHANNEL_RESTART_POLICY: BackoffPolicy = {
  initialMs: 5_000,
  maxMs: 5 * 60_000,
  factor: 2,
  jitter: 0.1,`
- **You believe the error type thrown is `InvalidGatewayCodeError`.** The client actually throws a `GatewayClientRequestError` exception, which includes `gatewayCode`, `message`, and `details` properties populated from the server response.
  > `src/gateway/client.ts:82`: `class GatewayClientRequestError extends Error {`

## src/infra
- **you believe the default expiration timeout value in milliseconds for execution approvals when no override is provided is 86,400,000** the actual default expiration timeout value is 120000 milliseconds when no override is provided
  > `src/agents/pi-tools.before-tool-call.ts:254`: `timeoutMs: approval.timeoutMs ?? 120_000,`
- **You believe the mandatory fields for an ExecApprovalRequest include generic fields like `requestId`, `userId`/`approverId`, `action`/`targetResource`, and `timestamp`.** The actual mandatory fields when constructing an ExecApprovalRequest object are: `id` (string), `request` (ExecApprovalRequestPayload with at minimum a `command` field), `createdAtMs` (Unix timestamp in milliseconds), and `expiresAtMs` (Unix timestamp in milliseconds). These four fields are enforced by the runtime execution type in exec-approvals.ts, despite gateway protocol schema showing them as optional.
  > `src/infra/exec-approvals.ts:108`: `id: string;`
- **You believe that requiresExecApproval can return false even when ask is set to 'always' if specific parameters like dry_run, force, or role are configured.** requiresExecApproval unconditionally returns true whenever ask is 'always', overriding any other parameter logic or conditions.
  > `src/infra/exec-approvals.ts:780`: `return true;`
- **You believe that the `writeExecApprovalsRaw` method primarily throws errors due to general system-level operational constraints such as insufficient disk space, read-only permission restrictions, or data serialization failures during the save process.** In fact, the implementation specifically validates path safety by throwing errors when directory components contain symlinks or if the destination file is a symlink, and relies on atomic file operations (exclusive temp creation followed by rename) rather than direct write attempts.
  > `src/infra/exec-approvals.ts:236`: `throw new Error(`Refusing to use unsafe exec approvals directory: ${dir}`)`

## src/auto-reply
- **You believe the primary entry point function and its required parameters for the auto-reply module remain unknown or indeterminate.** The primary entry point is `dispatchInboundMessage` exported from `src/auto-reply/dispatch.ts`, which requires `ctx`, `cfg`, and `dispatcher` arguments.
  > `src/auto-reply/dispatch.ts:20`: `export async function dispatchInboundMessage(params: {`
- **You believe individual command handlers are implemented within dedicated `handlers` or `commands` sub-directories using Python files with naming patterns like `echo_command.py` or `forget_handler.py`.** Command handlers are actually located in the `extensions/whatsapp/src/auto-reply/monitor/` directory using TypeScript files with a `[feature-name].ts` naming convention (e.g., `echo.ts`, `commands.ts`) that directly indicates the functionality being handled.
  > `extensions/whatsapp/src/auto-reply/monitor/echo.ts:1`: `export type EchoTracker = {`
- **You believe the separation between message intake and reply generation is managed through generic modular file organization without a specific intermediate processing pipeline layer.** The architecture implements a three-layer separation using `on-message.ts` for intake, `process-message.ts` for the pipeline, and `deliver-reply.ts` for generation.
  > `extensions/whatsapp/src/auto-reply/monitor/on-message.ts:178`: `await processForRoute(msg, route, groupHistoryKey);`
- **you believe the safeguard is likely implemented using locks or semaphores in `src/auto-reply/dispatch.ts` or platform-specific monitors.** The system silently skips the operation if `startedReplyLifecycle` is already true, ensuring no new lifecycle starts while one is active, implemented specifically in `src/auto-reply/reply/dispatch-acp-delivery.ts`.
  > `src/auto-reply/reply/dispatch-acp-delivery.ts:214`: `if (state.startedReplyLifecycle) {`

## src/config
- **you believe there is no documentation or implementation details regarding `loadConfig()` or `getRuntimeConfig()` functions specifically covering their caching behavior differences or recommended usage patterns for long-lived runtimes.** Both `loadConfig()` and `getRuntimeConfig()` exhibit identical caching behavior - they both utilize a process-level snapshot cache where the first successful load becomes the process-wide snapshot. Neither function performs a fresh config file read on subsequent calls. For long-lived runtimes, either function should be avoided on hot code paths.
  > `src/config/io.ts:1801`: `// First successful load becomes the process snapshot. Long-lived runtimes`

## src/commands
- **You believe the error output will display a generic parameter validation message like "Invalid value for option '--section'" without specifying the allowable section identifiers, and that execution halts purely through argument parsing middleware.** When an invalid value is passed, execution stops immediately at line 32 in `src/commands/configure.commands.ts` via `runtime.exit(1)`, terminating the process before the wizard runs. The exact error output explicitly lists valid options: "Invalid --section: ... Expected one of: workspace, model, web, gateway, daemon, channels, plugins, skills, health."
  > `src/commands/configure.commands.ts:30`: `Invalid --section: ${invalid.join(", ")}. Expected one of: ${CONFIGURE_WIZARD_SECTIONS.join(", ")}.`
- **You believe exported CLI command functions typically follow a signature pattern of `export async function commandName(params: { ctx: Context; args?: Args; [option]: any })` where common parameters like `ctx` and `args` are passed as part of a single parameters object.** In this repository, exported CLI command registrations follow a Commander.js pattern using a wrapper function like `registerXCommands` that accepts a `Command` instance and a parent options generator, with optional parameters defined explicitly via `.option()` methods instead of function arguments.
  > `extensions/browser/src/cli/browser-cli-manage.ts:134`: `export function registerBrowserManageCommands(
  browser: Command,
  parentOpts: (cmd: Command) => BrowserParentOpts
)`

## src/plugins
- **You believed there is no specific information detailing the exact conditions under which a plugin receives the activation cause 'blocked-by-denylist'.** A plugin receives the activation cause 'blocked-by-denylist' when the plugin's ID is included in the deny array of the configuration parameters, specifically when `params.config.deny.includes(params.id)` evaluates to true.
  > `src/plugins/config-state.ts:276`: `if (params.config.deny.includes(params.id)) {`
- **You believe there is no information regarding plugin discovery cache expiration timing or how to disable the plugin discovery caching mechanism.** The plugin discovery cache expires after 1 second (1000ms) by default, and users can disable it by passing `cache: false` or setting the `NO_PLUGIN_MANIFEST_CACHING=true` environment variable.
  > `src/plugins/manifest-registry.ts:136`: `const DEFAULT_MANIFEST_CACHE_MS = 1000;`
- **You believe setting the environment variable `NO_PLUGIN_MANIFEST_CACHING=true` disables the plugin discovery cache.** To completely disable the plugin discovery caching mechanism, you must set the environment variable `OPENCLAW_DISABLE_PLUGIN_DISCOVERY_CACHE=1`.
  > `src/plugins/discovery.ts:86`: `const disabled = env.OPENCLAW_DISABLE_PLUGIN_DISCOVERY_CACHE?.trim();`
- **You believe that a plugin manifest lacking an 'id' field is immediately excluded from the plugin registry entirely.** If a manifest lacks an 'id', the system attempts to use the directory name (`entry.name`) as the plugin ID; it is only silently skipped if both the manifest 'id' and directory name are falsy.
  > `extensions/qa-lab/src/bundled-plugin-staging.ts:127`: `const pluginId = typeof manifest.id === "string" ? manifest.id.trim() : entry.name;`

## extensions/matrix
- **You believe you must manually verify encryption configuration, inspect SDK contexts, or route messages through specific handlers to enable E2EE when sending Matrix messages.** Execute the command `openclaw message send --channel matrix`, which automatically starts one-off Matrix send clients to ensure messages are sent through encrypted channels rather than plain events.
  > `extensions/matrix/CHANGELOG.md:99`: `- Matrix/CLI send: start one-off Matrix send clients before outbound delivery so `openclaw message send --channel matrix` restores E2EE in encrypted rooms instead of sending plain events.`
- **You believe there is no specific information regarding the security risks associated with setting `autoJoin='always'` on a Matrix account in the provided documentation.** When `autoJoin` is set to 'always' on a Matrix account, any invited room will be joined before message policy applies.
  > `extensions/matrix/src/channel.ts:209`: `- Matrix invites: autoJoin="always" joins any invited room before message policy applies. Set ${autoJoinPath}="allowlist" + ${autoJoinAllowlistPath} (or ${autoJoinPath}="off") to restrict joins.`
- **You believe the gateway methods for Matrix device verification are registered in `extensions/matrix/handler.ts`.** The gateway methods for Matrix device verification are registered in `extensions/matrix/src/plugin-entry.runtime.ts` via exported handler functions such as `handleVerifyRecoveryKey`, `handleVerificationBootstrap`, and `handleVerificationStatus`.
  > `extensions/matrix/src/plugin-entry.runtime.ts:25`: `export async function handleVerifyRecoveryKey({`

## extensions/discord
- **You believe that the REQUIRED_DISCORD_PERMISSIONS constant definition and its specified minimum requirements for channel access are not documented in the correction notes.** The constant is defined in extensions/discord/src/channel.ts at line 135, specifying "ViewChannel" and "SendMessages" as the required permissions for channel access.
  > `extensions/discord/src/channel.ts:135`: `const REQUIRED_DISCORD_PERMISSIONS = ["ViewChannel", "SendMessages"] as const;`
- **You believe there is no specific documentation or code reference included that addresses the error condition "Discord reactions are disabled" during Discord messaging action execution.** The error is thrown when the reaction action gate check fails during a Discord messaging action execution, specifically in the "react" action case when `isActionEnabled("reactions")` returns `false`.
  > `extensions/discord/src/actions/runtime.messaging.ts:137`: `throw new Error("Discord reactions are disabled.");`
- **You believe that Discord direct message target IDs must use the `dm:` prefix followed by the numeric user ID.** You must use the `user:` prefix followed by the numeric ID (e.g., `user:123456789012345678`) or the mention format `<@123456789012345678>`. Using the `dm:` prefix is incorrect and may cause routing errors.
  > `extensions/discord/src/target-parsing.ts:33`: `"Discord DMs require a user id (use user:<id> or a <@id> mention)"`
- **You believe that both the `user:<numeric_id>` format AND the mention format `<@numeric_id>` are valid identifier formats for specifying a direct message to a user with their numeric ID in Discord. You also believe there's no issue with various alternative prefixes beyond these two options.** When specifying a direct message to a user with their numeric ID in Discord, you must use either the "user:{numeric_id}" format OR the "discord:{numeric_id}" format - both require a colon prefix with one of those two keywords. The mention format `<@id>` is NOT a valid identifier format for programmatic DM targeting in OpenClaw.
  > `extensions/discord/src/channel.ts:818`: `normalizeAllowEntry: createPairingPrefixStripper(/^(discord|user):/i)`

## src/cli
- **You believe that there is no information available regarding the environment variable that determines whether subcommands are eagerly registered.** Subcommands are eagerly registered when the environment variable `OPENCLAW_DISABLE_LAZY_SUBCOMMANDS` is set to a truthy value, which is checked within the function `shouldEagerRegisterSubcommands` in the file `src/cli/command-registration-policy.ts`.
  > `src/cli/command-registration-policy.ts:24`: `return isTruthyEnvValue(env.OPENCLAW_DISABLE_LAZY_SUBCOMMANDS);`
- **You believe there is no specific information detailing what Error type is thrown by a path parser when encountering invalid path segments during command execution.** The path parser actually throws a standard JavaScript `Error` object. The `parsePath` function specifically fails if a closing bracket `]` is missing or if brackets contain no content between them.
  > `src/cli/config-cli.ts:144`: `throw new Error(`Invalid path (missing "]"): ${raw}`);`
- **You believe the `--validate` option allows validation-only testing and returns a structure with `current`, `proposed`, `differences`, and `applied` fields.** Use the `--dry-run` option to enable validation-only testing, which returns a `ConfigSetDryRunResult` object with `ok`, `operations`, `configPath`, `inputModes`, `checks`, `refsChecked`, `skippedExecRefs`, and `errors`.
  > `src/cli/config-cli.ts:1364`: `--dry-run`

## extensions/telegram
- **you believe there is no specific documentation regarding the priority or conflict resolution between configuring both webhookUrl and poll-based operation mode simultaneously.** when both are configured simultaneously, webhook mode is used by default if webhookUrl exists in account.config, overriding the polling fallback, so no manual preference setting is needed beyond configuring the URL.
  > `extensions/telegram/src/channel.ts:864`: `mode: runtime?.mode ?? (account.config.webhookUrl ? "webhook" : "polling")`
- **You believe the specific behavior of handling a duplicate Telegram bot token is not explicitly documented and may result in throwing an error or blocking the connection setup entirely.** The system marks the conflicting account as **not configured** and sets `isConfigured` to `false` if `findTelegramTokenOwnerAccountId` detects that another `accountId` already owns the token, providing a clear unconfigured reason message.
  > `extensions/telegram/src/shared.ts:176`: `return !findTelegramTokenOwnerAccountId({ cfg, accountId: account.accountId });`
- **You believe that the reply-to functionality fails silently when **`startedReplyLifecycle` is already true** in `dispatch-acp-delivery.ts:214`, causing operations to be skipped silently.** The reply-to functionality actually fails silently when either the **`requireKnownShortId` option is omitted or set to `false`** in the call to `resolveBlueBubblesMessageId`, OR when the **`replyToId` parameter is empty/null/whitespace**. Without requiring known short IDs, invalid IDs pass through unchanged and cause silent failure rather than throwing errors.
  > `extensions/bluebubbles/src/monitor-reply-cache.ts:103`: `if (opts?.requireKnownShortId) {
        throw new Error(
          `BlueBubbles short message id "${trimmed}" is no longer available. Use MessageSidFull.`,
        );
    }`

## extensions/feishu
- **You believe that marking a `ResolvedFeishuAccount` as 'enabled' depends on multiple validation steps like valid credentials, populated fields, and error-checking phases.** A `ResolvedFeishuAccount` is marked 'enabled' **only** when BOTH the channel-level config (`channels.feishu.enabled`) and account-specific config (`channels.feishu.accounts[accountId].enabled`) are NOT explicitly set to `false`. No additional validations are required.
  > `extensions/feishu/src/accounts.ts:266`: `const enabled = baseEnabled && accountEnabled;`
- **You believe that there is no information regarding error codes that trigger a fallback to direct messaging when replying to a message fails with a 'withdrawn' code in `send.ts`.** The system does have two specific error codes (230011 and 231003) defined in the WITHDRAWN_REPLY_ERROR_CODES set that trigger fallback to direct messaging when reply attempts fail with withdrawn status.
  > `extensions/feishu/src/send.ts:17`: `const WITHDRAWN_REPLY_ERROR_CODES = new Set([230011, 231003]);`
- **you believe that there is no information about a `resolveBroadcastAgents` function definition in the provided note.** the `resolveBroadcastAgents` function is defined in `extensions/feishu/src/bot.ts` at line 72.
  > `extensions/feishu/src/bot.ts:72`: `export function resolveBroadcastAgents(cfg: ClawdbotConfig, peerId: string): string[] | null {`
- **You believe that there is no explicit documentation detailing the specific mechanism for handling bot self-references in Feishu message parsing to ensure slash commands remain parseable.** The system handles bot self-references by replacing mentions with empty strings in `normalizeMentions()` when the `open_id` matches `botStripId`, and strips all `<at[^>]*>` tags and standalone `@` mentions in `normalizeFeishuCommandProbeBody()` before command probe execution.
  > `extensions/feishu/src/bot-content.ts:277`: `.replace(/<at\b[^>]*>[^<]*<\/at>/giu, " ")`

## extensions/browser
- **You believe that specific OpenClawPluginApi method signatures for tool and service provider registration are not explicitly documented and that implementation details must be inferred from other extensions.** You must utilize `api.registerTool()`, `api.registerCli()`, `api.registerGatewayMethod()`, and `api.registerService()` directly from the OpenClawPluginApi interface within your plugin registration function.
  > `extensions/browser/plugin-registration.ts:29`: `api.registerTool(((ctx: OpenClawPluginToolContext) =>`
- **You believe there is no information available regarding Chrome stale target error detection circumstances or special handling requirements for the 'user' profile compared to other profiles in the study notes.** The system detects Chrome stale target errors when CDP sessions are invalidated due to browser process restarts, tab closures without proper cleanup, WebSocket disconnections, or network interruptions. Special handling for the 'user' profile is necessary because it often uses Chrome MCP transport instead of direct CDP, relies on ephemeral HTTP-based endpoints that cannot maintain persistent connections, requires different lifecycle management during stops, and demands that users re-establish CDP sessions by running action=tabs to retrieve fresh targetIds when stale targets occur.
  > `extensions/browser/src/browser-tool.actions.ts:397`: ``Chrome tab not found (stale targetId?). Run action=tabs profile="${profile}" and use one of the returned targetIds.``
- **You believe there is no specific documentation or defined set of browser action kind values considered safe to retry without a targetId when encountering transient errors.** The safe browser action kind values are "hover", "scrollIntoView", and "wait". This determination is made by the `canRetryChromeActWithoutTargetId` function in `extensions/browser/src/browser-tool.actions.ts`, which validates that the action is a read-only operation using selectors. These retries are triggered only when exactly one tab is attached (`tabs.length === 1`).
  > `extensions/browser/src/browser-tool.actions.ts:154`: `return kind === "hover" || kind === "scrollIntoView" || kind === "wait";`
- **You believe the documentation does not contain specific details about browser act parameter formats or the distinction between legacy flattened versus structured request object approaches.** The system determines the format by checking if `params.request` exists as an object; if present, structured format is used, otherwise legacy flattened format is applied with keys such as `doubleClick`, `modifiers`, and `text`.
  > `extensions/browser/src/browser-tool.ts:161`: `if (requestParam && typeof requestParam === "object") {`

## extensions/qa-lab
- **You believe there is no explicit documentation detailing the specific behavior of QaBusState when an inbound message is added to a conversation that doesn't already exist in its map.** Actually, the `ensureConversation()` method is automatically called to check if the conversation ID exists in the `conversations` Map and creates a new entry with the same properties if it is missing, followed by message registration and event emission.
  > `extensions/qa-lab/src/bus-state.ts:102`: `conversations.set(created.id, created);`
- **you believe the retry logic for transient startup races during gateway child initialization is implemented in standard gateway files such as src/gateway/startup.ts.** The retry logic is implemented in extensions/qa-lab/src/gateway-child.ts, utilizing `isRetryableGatewayStartupError()` to detect port conflicts and binding new ports up to 5 times.
  > `extensions/qa-lab/src/gateway-child.ts:689`: `[qa-lab] gateway child startup attempt ${attempt}/${QA_GATEWAY_CHILD_STARTUP_MAX_ATTEMPTS} hit a transient startup race on port ${gatewayPort}; retrying with a new port\n`
- **You believe there is no documented way to configure base URL and output directory when starting a QA Lab server programmatically, including uncertainty about parameter names and the default port value.** The QA Lab server can be configured programmatically via `startQaLabServer()` function. Pass optional parameters to configure: `host` (bind address), `port` (listening port, default 0 for dynamic assignment), `repoRoot` (output directory, defaults to current working directory), and `advertiseHost`/`advertisePort` for external URL overrides. Use `process.cwd()` as default output location if not specified.
  > `extensions/qa-lab/src/lab-server.ts:169`: `const repoRoot = path.resolve(params?.repoRoot ?? process.cwd());`

## extensions/memory-core
- **you believe there is no documentation or implementation details regarding hybrid search functionality or the `selectScoredResults` method within the provided repository notes.** The method calculates a `relaxedMinScore` as `Math.min(minScore, hybrid.textWeight)` and falls back to using this relaxed threshold to return keyword-only matches if the initial strict filter produces no results.
  > `extensions/memory-core/src/memory/manager.ts:424`: `const relaxedMinScore = Math.min(minScore, hybrid.textWeight);`
- **you believe there is **no information** available regarding a `search()` function, file indexing operations, or pre-query index lookup behavior within the provided note materials.** `search()` performs a synchronous bootstrap sync via `await this.sync({ reason: "search", force: true })` if content isn't indexed to prevent "fail closed" empty results upon restart.
  > `extensions/memory-core/src/memory/manager.ts:306`: `await this.sync({ reason: "search", force: true });`
- **You believe there is no documentation about MemoryIndexManager's FTS-only fallback mechanisms for conversational queries in the memory-core section, and that the memory-core entries remain unexpanded without implementation details.** MemoryIndexManager implements a multi-stage fallback: it first searches with the cleaned query as-is, and if no results are returned, it extracts meaningful keywords using extractKeywords() while filtering stop words and invalid terms, then searches each extracted keyword individually to broaden recall before merging all results with deduplication keeping highest scores.
  > `extensions/memory-core/src/memory/manager.ts:355`: `// Fallback: broaden recall for conversational queries when the
// exact AND query is too strict to return any results.`

## extensions/msteams
- **you believe the provided note content does not contain details about file consent mechanisms or media attachment behavior in personal chat contexts.** For personal (1:1) chats, the system uses direct base64 embedding only for small images (<4MB); otherwise, large files (≥4MB) and non-image files trigger the FileConsentCard flow.
  > `extensions/msteams/src/send.ts:96`: `- Personal (1:1) chats: small images (<4MB) use base64, large files and non-images use FileConsentCard`
- **You believe there is no specific documentation or code reference included that addresses the delivery mechanism for non-image files during a proactive message to a personal chat account.** When a user attempts to send a non-image file during a proactive message to a personal chat account, the system triggers a FileConsentCard flow that requires user consent before delivery based on the conversation type being "personal" AND either having a large size or non-image content type.
  > `extensions/msteams/src/file-consent-helpers.ts:5`: `FileConsentCard is required for:
- Personal (1:1) chats with large files (>=4MB)
- Personal chats with non-image files (PDFs, documents, etc.)`
- **You believe that there is no specific information documenting where the maximum outbound media size limit (100MB) is defined for MSTeams in the codebase and that it requires searching multiple directories without a clear source.** The maximum outbound media size limit is defined as a constant named MAX_DOCUMENT_BYTES set to 100 megabytes in the src/media/constants.ts file at line 4.
  > `src/media/constants.ts:4`: `export const MAX_DOCUMENT_BYTES = 100 * 1024 * 1024; // 100MB`
- **You believe there is no information about a delegated login function or its decision logic between automatic browser opening and manual URL entry in the repository documentation.** The delegated login function decides between automatic browser opening and manual URL entry based on the execution environment's remoteness status determined by the `isRemoteEnvironment()` function. When `isRemote` is false (local environment), the browser is automatically opened via `openUrl()`.
  > `src/plugins/provider-oauth-flow.ts:36`: `await params.openUrl(url);`
