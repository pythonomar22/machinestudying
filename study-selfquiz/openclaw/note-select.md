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
