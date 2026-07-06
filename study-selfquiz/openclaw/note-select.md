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
