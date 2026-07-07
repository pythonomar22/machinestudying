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

## src/plugins
- **You believed there is no specific information detailing the exact conditions under which a plugin receives the activation cause 'blocked-by-denylist'.** A plugin receives the activation cause 'blocked-by-denylist' when the plugin's ID is included in the deny array of the configuration parameters, specifically when `params.config.deny.includes(params.id)` evaluates to true.
  > `src/plugins/config-state.ts:276`: `if (params.config.deny.includes(params.id)) {`
- **You believe there is no information regarding plugin discovery cache expiration timing or how to disable the plugin discovery caching mechanism.** The plugin discovery cache expires after 1 second (1000ms) by default, and users can disable it by passing `cache: false` or setting the `NO_PLUGIN_MANIFEST_CACHING=true` environment variable.
  > `src/plugins/manifest-registry.ts:136`: `const DEFAULT_MANIFEST_CACHE_MS = 1000;`

## extensions/matrix
- **You believe you must manually verify encryption configuration, inspect SDK contexts, or route messages through specific handlers to enable E2EE when sending Matrix messages.** Execute the command `openclaw message send --channel matrix`, which automatically starts one-off Matrix send clients to ensure messages are sent through encrypted channels rather than plain events.
  > `extensions/matrix/CHANGELOG.md:99`: `- Matrix/CLI send: start one-off Matrix send clients before outbound delivery so `openclaw message send --channel matrix` restores E2EE in encrypted rooms instead of sending plain events.`
- **You believe there is no specific information regarding the security risks associated with setting `autoJoin='always'` on a Matrix account in the provided documentation.** When `autoJoin` is set to 'always' on a Matrix account, any invited room will be joined before message policy applies.
  > `extensions/matrix/src/channel.ts:209`: `- Matrix invites: autoJoin="always" joins any invited room before message policy applies. Set ${autoJoinPath}="allowlist" + ${autoJoinAllowlistPath} (or ${autoJoinPath}="off") to restrict joins.`
- **You believe the gateway methods for Matrix device verification are registered in `extensions/matrix/handler.ts`.** The gateway methods for Matrix device verification are registered in `extensions/matrix/src/plugin-entry.runtime.ts` via exported handler functions such as `handleVerifyRecoveryKey`, `handleVerificationBootstrap`, and `handleVerificationStatus`.
  > `extensions/matrix/src/plugin-entry.runtime.ts:25`: `export async function handleVerifyRecoveryKey({`
