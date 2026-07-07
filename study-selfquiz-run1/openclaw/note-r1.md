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
