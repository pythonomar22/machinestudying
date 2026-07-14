# 012 — SmallDSPy paired baseline and forced-50 cheatsheet screen

## Objective and status

Measure two fast iteration signals on the user-selected SmallDSPy slice:

1. no-note baseline lenient expertise (WAUC); and
2. newly studied forced-50 cheatsheet lenient expertise (WAUC).

Generation and local grading are separate phases. Episodes, grades, reports,
and the paired comparison are immutable JSON artifacts. The primary two-number
summary is the local-Qwen pure weighted-rubric lenient WAUC for each arm; the
four budget-level means, token costs, compile diagnostics, uncertainty, and
paired deltas remain mandatory context.

Status: both final `h` generation populations and the forced-50 study artifact
are complete and immutable, but no valid two-arm score has yet been observed.
Replacement `b` and `c` runs exposed harness accounting defects and were
stopped without reuse. Both `d` generation grids later completed, but the first
isolated grading smoke exposed a deeper vLLM structured-output configuration
defect that also contaminated generation: JSON schemas were enforced inside
Qwen's hidden reasoning channel, where a schema-complete response could
terminate with final `content=null`. Because the same path powered DSPy's JSON
format repair, the defect contaminated all 11 baseline and 18 treatment adapter
failures; it does not prove what their counterfactual answers would have been.
All `d` artifacts are preserved as invalidated diagnostics.

The corrected `h` generation was not regenerated after outcomes were seen. Its
first full local-grading pass was stopped after a new judge-only defect became
clear: the local judge could spend its entire 8,192-token allowance in hidden
reasoning and return `content=null`, while the schema also asked the model to
repeat deterministic weighted arithmetic. That old grading namespace contains
50 stored grade files, four terminal failed-judge audits, and six interrupted
intent records; it is diagnostic and permanently unreportable. The later `i`
and `j` protocols each exposed a distinct constrained-verdict defect and are
also terminal; their full chronology is recorded below. The preregistered `k`
protocol was executed as its declared last attempt and stopped before benchmark
grading when all three replicas failed its synthetic candidate-entailment
check. That failure remains immutable and `k` has no score. The user's earlier
instruction to continue engineering the local judge until it is sound
authorized the separately disclosed `l` screen. `l` then failed its expanded
qualification before benchmark contact. Protocol `m` changes only the local
judge's thinking setting and completion ceiling, as documented in Results.
Neither `l` nor `m` complies with, rescues, or replaces `k`: both are post-hoc
local engineering experiments. Every correction followed observed failures,
so any completed `m` numbers remain adaptive, exploratory, and non-claim-ready.

## What this dataset is—and is not

`data/smalldspy.jsonl` contains five records with SHA-256
`b152153a9ec159dc99f89d9a1ca085a88d04be818b348e58cebf620513b2c75d`:

| question ID | full DSPy row | topic |
|---|---:|---|
| `dspy_2de37073e8e4` | 11 | `react_agents_and_tools` |
| `dspy_7329144ef1e9` | 12 | `react_agents_and_tools` |
| `dspy_a5b116f00083` | 13 | `react_agents_and_tools` |
| `dspy_e175093485fd` | 14 | `react_agents_and_tools` |
| `dspy_0b4b420ebeb4` | 15 | `react_agents_and_tools` |

The parsed objects are exact copies of those public `data/dspy.jsonl` rows.
Together they contain 26 weighted claims and 25 evidence spans. Every excerpt
was re-matched byte-for-byte to the pinned Git blobs, all rubric weights sum to
100, and all five reference programs execute successfully against pinned DSPy.

This is **not an independent held-out test**. The questions and rubrics are
public, the slice was chosen adaptively after inspecting the benchmark and
source layout, all five questions share one topic, and the pre-experiment-012
repository inventory already contained 1,080 rollout artifacts and 963 grade
artifacts for these IDs. Later diagnostic namespaces add still more reuse; the
historical counts are not presented as the current filesystem inventory.
The screen can debug plumbing and compare variants on this narrow slice; it
cannot establish general DSPy performance, out-of-sample improvement, or a
publishable studying result. Repeated use makes it a development set.

The exact upstream rows also retain three inherited audit limitations rather
than silently changing the evaluation after selection:

- `dspy_e175093485fd` asks for an exactly one-step router, while its gold demo
  exposes a caller-controlled `route_budget` whose default is one;
- `dspy_7329144ef1e9` claim `c6` does not cite the row's strongest existing
  `DummyLM` evidence span; and
- `dspy_a5b116f00083` claim `c2` cites adapter mechanics but not the strongest
  available example of per-call `dspy.context(adapter=...)` scoping.

These are dataset-quality caveats. Editing them here would create a corrected
derivative and break exact-subset fidelity, so this run preserves and discloses
them.

## Studying and evaluation corpus

`corpora/smalldspy` is a reproducible sparse checkout of DSPy commit
`9cdb0aac28b2a04b064e40697ccd301872cf6a43`. Exactly 59 allowed Python files
(12,931 lines, 497,627 characters) are exposed through repository tools under:

- `dspy/adapters`
- `dspy/predict`
- `dspy/primitives`
- `tests/predict`

Setup creates or verifies those exact sparse roots. The validator accepts
`skip-worktree` only outside the configured roots, requires all in-scope files
to be normally indexed, and rechecks each exposed file's pinned blob and mode.
Neither study nor evaluation tools can read benchmark, run, grade, or experiment
files. Both arms use this same scoped corpus in every tool-enabled evaluation
budget; `direct` is intentionally tool-free in both arms.

## Frozen paired design

The design is an adaptive exploratory screen:

| field | frozen value |
|---|---|
| generator and local judge | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| harness | author-confirmed `dspy.ReAct`; parse-only Chat-to-JSON format repair with independent provider-attempt audit; structured JSON constrained only after Qwen reasoning ends |
| answer generation mode | paper sampling parameters with Qwen thinking enabled |
| current local-judge mode (`m`) | temperature 0, seed 0, maximum 4,096 output tokens, and `chat_template_kwargs.enable_thinking=true`; Qwen may reason privately before an answer-centered, xgrammar-constrained final JSON object; the JSON-escaped candidate-last payload contains every exact keyed binary claim label and `needs_regrade`, with no rationale or total |
| local-judge topology | three homogeneous TP=2 replicas on six L40S GPUs; the immutable generation-manifest `episode.server_slot` routes each cell to the same slot in both arms |
| local-judge lifecycle | one authenticated `m` launcher lifecycle for the 20-case qualification, both arm grades, both reports, and the paired comparison; its high-entropy non-secret launch ID is grading-spec-bound, so interruption/relaunch makes the namespace terminal rather than resumable |
| lenient scoring | binary `0/1` claims following Jacob's correction; the harness alone computes the pure weighted sum, following Jacob's clarification that lenient is just the weights summed together |
| arms | no note; exact forced-50 SmallDSPy note prepended |
| answer-time corpus access | `grep`, `glob`, and pinned `read_file` in both arms |
| budgets | `direct`, `k5`, `k20`, `k20f` |
| rollouts | 3 per question and budget; 60 episodes per arm |
| master seed | `44001` |
| paired seed group | `smalldspy-local-cheatsheet-screen-20260713` |
| judge evidence | dataset excerpts, identically in both arms |
| bootstrap | 10,000 paired replicates, seed `45001` |
| primary summary | pure-sum lenient WAUC for each arm |

The published Appendix A.5 still shows the older `0/0.5/1` claim scale, asks
the judge to echo `question_score`, and requests concise rationales. In
`docs/jacob.md`, the author explicitly corrected the claim scale to `0/1`
because partial credit increased variance and later answered that “lenient is
just weights summed together.” The implementation therefore computes the
weighted total deterministically from binary labels. Removing the redundant
model total cannot mechanically add credit. Removing local rationales after
the `j` failure is more consequential: rationales never entered the arithmetic,
but changing elicitation can change Qwen's labels and weakens auditability.
That divergence is compensated only partially by a fixed manual post-freeze
audit, not treated as paper fidelity. The paper used GPT-5.4 with whole evidence
files, whereas this deliberately cheaper screen uses the same pinned 9B Qwen
family as answerer and judge with dataset excerpts. Consequently this is a
local proxy, not a byte-exact paper replication or an external validation of
the rubric decisions.

The full baseline grid runs before the full treatment grid. This fixed ordering
is operationally simple but not counterbalanced, so time-dependent server or
hardware drift could be confounded with the arm contrast even though both runs
share one server lifecycle and every substantive recorded environment field.
This is another reason to treat the comparison as diagnostic only.

The treatment note is generated once by the existing forced-50 study protocol
over only the scoped corpus, without benchmark access. It is not the historical
full-DSPy `cheatsheets/dspy.md`. The immutable construction is
`studies/cheatsheet/smalldspy-cheatsheet-s50-20260714h/smalldspy/`; evaluation
must supply both its content-addressed note and manifest. Study tokens are
reported separately and excluded from the evaluation token axis, matching the
paper's estimand.

The generator also serves as the local judge, so grading errors may be
correlated with generation behavior. Five questions provide very low precision,
and bootstrap intervals are descriptive on this adaptively selected population,
not a remedy for selection bias. Console output must retain the repository's
`EXPLORATORY LOCAL-QWEN PROXY: NOT CLAIM-READY` warning; report JSON must encode
`claim_ready=false` and the local-proxy provenance, and comparison JSON must
retain its diagnostic banner.

## Infrastructure preflight

On 2026-07-13, allocation `16142825` exposed seven idle L40S GPUs. The 262k
context requires TP=2 on this GPU class, so both generation and final local
grading use six GPUs as three homogeneous TP=2 servers; no scientifically
different TP=1 seventh server is introduced. The seventh GPU remains idle
*inside Omar's seven-GPU parent allocation* and is therefore not advertised as
free to another Slurm user. The final grader now has an explicit ordered
multi-endpoint contract: endpoint count must equal the immutable run-manifest
server count, every cell routes strictly through its stored slot with no
fallback or modulo remapping, grades and failure audits retain their complete
topology, reports distinguish recorded routes from actual answered-cell
contacts, and comparison requires the paired arms' slot maps to match. The
authenticated per-launch ID is also part of the substantive local-runtime
digest. Thus a later otherwise identical relaunch cannot be spliced into an
interrupted arm or compared against the other arm; it makes the active grading
namespace terminal.

The `j` grader used one TP=2 replica. The frozen `k` grader instead used three
TP=2 replicas because the user requested full useful GPU utilization and the
paired slot block is now implemented and tested. The earlier `f` diagnostic
already showed temperature-zero replica divergence, so neither `k` nor `l` is
comparable to `j` and replica is not treated as exchangeable. `l` retains the
same three-replica topology but uses a fresh launch, ports, IDs, policy, and
source commit; it never reuses `k` responses. Across the same one-launch
lifecycle, each paired cell uses the same manifest slot in baseline and
treatment. The only planned `l` chat-completion calls before shutdown are 60
synthetic qualification calls (20 frozen cases exactly once per endpoint) and
119 benchmark judge calls (60
answered baseline cells and 59 answered treatment cells; the treatment's one
generation `no_answer` contacts no judge). Three post-qualification model-list
health requests do not generate text and are recorded separately.

The first full-corpus launcher diagnostics exposed two pre-provider failures:
PyTorch 2.11 returns a typed CUDA UUID, and Hugging Face's offline repository-ID
loader rejected an inference-complete snapshot because `.gitattributes`,
`LICENSE`, and `README.md` were absent. The launcher now validates typed UUIDs,
byte-attests the exact local snapshot, passes that snapshot path directly to
vLLM, and preserves `Qwen/Qwen3.5-9B` as the served response identity. The fresh
`vllm-16142825-smalldspy-infra-a` preflight loaded all three servers, completed
warmup, passed authenticated `/models` readiness, re-attested the cache, and
cleanly released all GPU processes. No SmallDSPy question or study prompt was
sent during that infrastructure preflight.

Offline preflight passed 331 non-sandbox tests and 11 sandbox-environment tests,
compiled the Python tree in both pinned environments, syntax-checked every
shell launcher, reproduced the sparse checkout from a fresh local clone, passed
`git diff --check`, and confirmed `AGENTS.md` is byte-identical to `CLAUDE.md`.

The isolated run `smalldspy-generation-smoke-20260713` produced two successful,
ungraded plumbing episodes for one question. The direct episode used one model
call and 3,954 generated tokens; `k5` used four calls, three ReAct iterations,
two tool iterations, one caught `finish`, and 4,861 generated tokens. Its
manifest exposed the editable-import validator defect. The attempted study ID
`smalldspy-cheatsheet-s50-20260713` created only a zero-byte process lock: no
intent, episode, note, provider request, or study response exists. The fix now
requires the exact pinned origin `corpora/dspy/dspy/__init__.py` and its exact
hash; substituted editable or site-package origins remain invalid. The used
smoke/study IDs, `generation-a` prefix, and ports `34100`-`34102` are retired.

The replacement smoke `smalldspy-generation-smoke-20260713b` completed both
plumbing cells. The full baseline `smalldspy-local-base-20260713b` then exposed
four completed provider-response parse failures (`content=None` with only
`reasoning_content`) among the first 19 terminal cells. Pinned DSPy discarded
the partial ReAct trajectory and the harness labeled these `error`, contrary to
the repository policy that exhausted model format repair is a model non-answer.
Generation was deliberately stopped; all 54 written attempt intents, 15 final
episodes, four failed attempts, and the log remain preserved. The correction
counts provider attempts independently, allows Chat-to-JSON fallback only after
a typed parse failure, preserves escaped trajectories, and accepts a parse
non-answer only when every attempted provider call has a complete usage ledger.
It never promotes `reasoning_content` into an action or answer. This is an
adaptive implementation correction, so the screen remains non-confirmatory.
All `b` IDs, prefix `generation-b`, and ports
`34300`-`34302` are retired.

The `c` smoke passed 2/2 and the full baseline produced 45 valid finals before
the forced budget exposed three typed action-format failures at 7, 4, and 0 of
20 recorded iterations. Treating those completed model-response failures as
`forced_short` made the one-shot screen terminal even though the same failure
in an unforced budget was correctly an ITT zero. The final policy therefore
records typed forced-action parse exhaustion as `no_answer`, retains the exact
partial trajectory and token ledger, and records `forced_budget_complete=false`.
Full-budget forced answers and extraction failures remain explicitly marked
complete; context, transport, tool, and program failures remain nonfinal. The
`c` namespace and all its 60 attempt intents, 45 finals, three failed attempts,
and interrupted cells remain preserved and are never spliced into `d`. All `c`
IDs, prefix `generation-c`, and ports `34500`-`34502` are retired.

The `d` generation smoke passed, its baseline and treatment grids each wrote
all 60 terminal episodes, and its forced-50 study completed. The baseline had
48 `ok` episodes and 12 `no_answer` episodes: 11 DSPy adapter parse failures
and one parsed empty answer. It used 351 LM calls and 456,283 generated tokens.
The forced-50 study used 51 LM calls across 50 turns, 898,618 prompt tokens,
and 53,490 generated tokens; its note hash was
`a6be8b2162f49a042d3769743540951250735d22b12894773389d918a447618e`.
The treatment had 42 `ok` episodes and 18 `no_answer` episodes, all DSPy
adapter parse failures, and used 303 LM calls and 358,763 generated tokens. A
two-attempt isolated judge smoke then returned HTTP 200 twice with complete
model identities and usage ledgers—4,173 prompt and 418 completion tokens per
attempt—but final `message.content=null` both times, so the grader correctly
wrote no grade and preserved its failed-attempt audit.

The shared cause was the vLLM launch option
`structured-outputs-config={"enable_in_reasoning":true}`. The pinned Qwen chat
template begins in a reasoning channel, and vLLM's Qwen reasoning parser does
not enter final content until that channel closes. Applying the JSON grammar
inside reasoning allowed a schema-complete sequence to terminate before final
content began. This affected the judge's mandatory verdict schema and DSPy's
schema-based answer-format repair. It therefore cannot be interpreted as an
honest model non-answer, and the asymmetric failures cannot be scored as
intention-to-treat zeroes. Hidden reasoning is never promoted into a final
answer or verdict. The launcher now explicitly sets
`enable_in_reasoning=false`, preserving reasoning while applying schemas only
to final content, and a script-contract regression test rejects the retired
setting. Every `d` ID, its note, prefixes `generation-d`/`grading-d`, and ports
`34700`-`34800` are retired. The artifacts remain available for diagnosis but
are excluded from all scores and comparisons.

The first corrected `e` topology loaded all three TP=2 servers and recorded
`enable_in_reasoning=False` in every engine configuration. Contemporaneous
launcher output records that a strict-schema probe used temperature zero, seed
zero, and an 8,192-token ceiling, and that its endpoint-zero SDK response had
final `content=null`. The durable server logs establish three HTTP 200
responses and a ceiling-length endpoint-zero request, but the probe process
raised before harvesting the other two futures or serializing any response.
Its zero-byte `.structured-smoke.jsonl.tmp` cannot independently establish the
request fields, response identities, usage, or content. Those details are
therefore disclosed as contemporaneous observations rather than upgraded into
artifact-backed claims.

The precommitted all-endpoints gate therefore failed, the launcher terminated
all six GPU workers, and no benchmark, study, run, grade, or report artifact
was created. All `e` IDs, prefix `generation-e`, ports `34900`-`34902`, and its
partial probe file are preserved and retired. The unused planned `grading-e`
prefix and port `35000` are also retired.

The precommitted `f` gate then produced a complete 5,290-byte audit with SHA-256
`f4bd29fa361e2e2423cbfcac722c6455e680e1556cc930bc3fc63c094a0053a9`
and request hash
`75ec9cfcb8c5ed25b89a9fb48f09ea41785d0a3fbacf3615fe6bfa76795b109a`.
All response identities, fingerprints, finish reasons, content hashes/status,
and usage were written without hidden reasoning. Slots zero and one returned
the same valid 28-byte final JSON on their first attempts after 1,243 and 890
completion tokens, respectively. Slot two returned complete HTTP responses
with `finish_reason=length`, 8,192 completion tokens, and final `content=null`
on both bounded attempts. The all-slots gate failed, all six workers exited,
and no SmallDSPy artifact was created. All `f` run/study/grade IDs,
`generation-f`/`grading-f` prefixes, and ports `35100`-`35200` are retired.

This result shows that an all-slots content gate samples model behavior rather
than only configuration: temperature zero and seed zero did not make replicas
identical. Relaunching topologies until every model response happened to pass
would be optional selection on favorable outputs.

The first `g` launcher reached deterministic authenticated readiness on three
TP=2 servers and all three logs recorded `enable_in_reasoning=False`. A local
preflight then compared `HEAD` with an incorrectly transcribed full commit hash
and terminated the shell before invoking `studybench.react`. No output-bearing
model request, SmallDSPy prompt, run manifest, episode, study, grade, or report
was created. The six workers exited with no GPU memory retained. This was a
pre-outcome operator error, not a model-output qualification failure, but its
logs, ports `35300`-`35302`, and all planned `g` IDs remain preserved and
retired rather than reused.

The final `h` topology is therefore qualified only by deterministic
authenticated identity, exact model/environment/cache attestation, and all
engine logs recording
`enable_in_reasoning=False`; it receives no model-output qualification probe.
There is exactly one outcome-bearing generation topology. Each versioned
grading protocol receives at most one topology and is never relaunched to seek
more favorable judge output. Complete model responses that fail to expose a
usable answer remain intention-to-treat non-answers; exhausted judge attempts
produce no grade, never a fabricated zero or a topology retry.

That `h` generation topology completed the preregistered sequence. The baseline
manifest has SHA-256
`53acd1a4f5fca258a190fc01f8330806c166c64ed5aad0e95b8d33ac3149bacd`;
all 60 episodes are `ok`, with 465 model calls, 3,263,217 prompt tokens,
543,322 generated tokens, and 3,806,539 total tokens. The forced-50 study then
completed exactly 50 turns plus extraction. Its manifest has SHA-256
`3444965f71f5a37d63e889a481b6d58a25f534b85224eae5ddfc9900842638e3`;
the 4,553-byte note has SHA-256
`8e3733d16c669642fa9bf96dfd2eb1de89366b75c6c7f42a5f7c9a7ba6eb33cb`;
and study usage is 1,158,099 prompt, 23,557 generated, and 1,181,656 total
tokens across 51 model calls. The treatment manifest has SHA-256
`06da43b3014a22637d51c13be4357ead6faf344dbd0b81cda17fea260a9b18c3`.
It contains 59 `ok` episodes and one protocol-valid intention-to-treat
`no_answer`, with 500 model calls, 4,445,998 prompt tokens, 561,544 generated
tokens, and 5,007,542 total tokens. The non-answer is
`k20f/r1/dspy_2de37073e8e4` on canonical server slot 2 after 13 model calls and
130,315 total tokens. Every arm has exactly 20 episodes per slot and paired
cells share a slot. Both manifests bind generator commit
`1d94e6777c6d50655a557d6c4ae8e904662bde28`, source-tree SHA-256
`26f3434f843068e2fbcad636810e2bfaf808894be79fff797a845c2fd825546d`,
and source-record SHA-256
`3ca811f440d95b405b74b09c67df9458f5cac371c1b9c3b61f5dfad20050e4ef`.
Deep population, note-binding, usage-ledger, environment, seed, slot, and
manifest validation passed before the generation servers were shut down.

The first `h` grading topology used one TP=2 local judge. Its isolated smoke
returned a schema-valid grade, after which the full baseline pass was started.
The pass was manually stopped as soon as the failure pattern became
diagnostically clear. The preserved old namespace
`qwen-local-base-20260714h` contains 50 grade files, four terminal failed-judge
audits, and all 60 pre-contact intent records; six intents were in flight when
the process was terminated. No treatment judge request was made. Three cells
(`direct/r1/dspy_a5b116f00083`, `k5/r0/dspy_e175093485fd`, and
`k5/r1/dspy_a5b116f00083`) each returned two complete 8,192-token responses
whose final `content` was null. At `k5/r1/dspy_2de37073e8e4`, both responses
gave the same atomic labels but supplied redundant totals of 2 and 40, while
the harness recomputed 32. The stored rationale for one positive claim was
also substantively questionable, so neither the labels nor 32 are salvaged.
The partial namespace is terminal, cannot produce a report, and is never mixed
with the corrected pass.

The first corrected (`i`) judge contract has grade schema 7,
judge-attempt-intent schema 2, and failed-judge-audit schema 5. It removes
model-generated `question_score`
from the prompt and response schema, retains only atomic binary labels and
concise rationales, and records the harness-computed weighted total as
`question_score`. For the pinned Qwen template,
`chat_template_kwargs.enable_thinking=false` renders an empty closed thinking
block before generation, so strict JSON is generated in final content rather
than hidden reasoning. The exact request policy and options are hashed into
every grade specification and repeated in every grade, intent, failure audit,
report, and comparison contract.

The `i` baseline completed all 60 cells with no failed-judge audit and produced
the immutable diagnostic report
`reports/smalldspy-local-base-20260714h/qwen-local-base-20260714i/smalldspy/report-c1ead5f076ef15e02c11868f4632fb500408267f5bf5d4033c25600f2f6561b5.json`.
Its lenient WAUC was 27.105522198059212, but it is not a final arm result because
the matched treatment did not form a complete population. The `i` treatment
wrote 59 grades and one terminal failed-judge audit. At
`k20/r2/dspy_7329144ef1e9`, the array-shaped response schema required six
allowed claim IDs but did not enforce their uniqueness. Qwen returned IDs
`c1,c2,c2,c2,c2,c2`; the validator correctly rejected the verdict. The second
nominal retry repeated the exact temperature-zero request and returned the
exact same 1,743-byte content (SHA-256
`dd3f1c5918c570c3955c6532e2985ee3cabf592ed4eb940c1f23349072f146fe`).
Each duplicate response used 6,984 prompt and 511 completion tokens, so the
second request added 7,495 tokens without new information. No grade was written
for that cell, no treatment report exists, and the `i` arms are never compared.

The fresh `j` contract changes the verdict to an object whose properties are
the exact rubric claim IDs. Every ID is required, additional properties are
forbidden, and each value contains only a binary score and nonblank rationale.
The strict parser still rejects duplicate JSON object keys, and the harness
canonicalizes the keyed verdict back to rubric order before computing scores.
This makes duplicate or missing claim IDs structurally impossible under the
pinned constrained decoder. The policy also permits exactly one judge request
per answered cell and fails closed instead of repeating an identical
temperature-zero request. The contract has grade schema 8,
judge-attempt-intent schema 3, failed-judge-audit schema 6, report schema 11,
and local screen-comparison schema 3. The attempt policy and maximum request
count are explicitly bound into the grading specification, grades, pre-contact
intents, failure audits, reports, and paired-screen contract. Before any `j`
benchmark request, the exact real six-claim schema compiled successfully under
the pinned xgrammar runtime; this was an offline grammar check and produced no
benchmark judgment. Because this repair follows inspection of `i`, `j` remains
adaptive, exploratory, and non-claim-ready even if complete.

The `j` baseline completed all 60 cells with 60 pre-contact intents and no
failed-judge audit. The matched treatment did not complete: it contains 54
grade artifacts, of which 53 are accepted judge verdicts and one is the
pre-existing generation `no_answer`, plus six terminal failed-judge audits
among the 59 answered cells. Thus the answered-cell judge failure rate was
6/59 (10.17%) in treatment versus 0/60 in baseline. Every failure was one HTTP
200 response from the pinned model and fingerprint with complete usage, and
every one consumed the full 8,192 completion-token allowance. The six cells
were `direct/r0/dspy_e175093485fd`, `direct/r1/dspy_7329144ef1e9`,
`k5/r1/dspy_0b4b420ebeb4`, `k5/r1/dspy_2de37073e8e4`,
`k5/r2/dspy_2de37073e8e4`, and `k20/r2/dspy_7329144ef1e9`.

Manual inspection showed a narrower mechanism than an overlong substantive
rationale. In each failed response, Qwen tried to mention a literal double
quote inside a rationale, closed the JSON string prematurely, and then emitted
only grammar-legal spaces, tabs, or newlines until the token ceiling. Across
the failures, 119,950 of 124,830 response bytes (96.1%) were trailing JSON
whitespace; their aggregate judge cost was 34,855 prompt plus 49,152 completion
tokens. The old parser message called the incomplete ASCII JSON “not valid
UTF-8 JSON”; that message was inaccurate and is corrected prospectively. The
53 accepted treatment verdicts all had exact keyed IDs, binary scores, and
nonblank rationales, as did the baseline verdicts. This supports the keyed-ID
repair but does not authorize salvaging the 13 partial labels visible in failed
prefixes, selectively retrying six cells, or reporting the completed `j`
baseline alone. Both `j` namespaces are terminal diagnostics and no `j` report
or comparison will be made.

The frozen `k` protocol removed every free-generation field from the *local
proxy only*. Its raw verdict was exactly a keyed map from every required rubric
ID directly to integer `0` or `1`, plus the bounded `needs_regrade` boolean.
The external GPT-5.4 contract remains rationale-bearing and Appendix-A.5-like.
Local accepted grade artifacts store only canonical claim IDs and scores; no
placeholder or fabricated rationale is introduced. `needs_regrade=true`, any
non-`stop` provider finish reason, any malformed identity or usage, or any
schema/parse failure remains terminal and produces no grade. The local output
ceiling is 256 tokens. The server is pinned to xgrammar with
`disable_any_whitespace=true` and `enable_in_reasoning=false`; the former
disables *arbitrary* whitespace but, with the pinned xgrammar defaults, still
uses fixed `", "` and `": "` JSON separators. Raw compact serialization was
never required by the production strict parser.

`k` was executed from clean, pushed commit
`0257db7bcd2347502d986ed390afe7ca31f1a6c2` in one six-GPU launch. Its immutable
calibration audit is
`logs/vllm-16142825-smalldspy-grading-k.calibration.json`, SHA-256
`067f21065ac807ef8500884d25ce9083dbb7a354de1c4cbc6801803847d1b58f`.
Each endpoint received exactly one request and returned HTTP-200-equivalent
final content from the pinned model with nonempty response/request identity,
`finish_reason=stop`, and complete usage (522 prompt, 28 completion, 550 total
tokens). All three returned the same valid 54-byte JSON:
`{"claims": {"c1": 1, "c2": 1}, "needs_regrade": false}`. The first frozen
expectation was `c1=1`: the candidate said the marker was blue. The second was
`c2=0`: the candidate explicitly said its shape was circular, not square. Thus
all three deterministic replicas made the same substantive candidate-versus-
reference error. The additional exact-compact-byte assertion was also wrong,
but removing that false assertion would not rescue the semantic failure. `k`
therefore stopped as specified, all GPU processes exited, and both `k` grade
and report namespaces remain absent. No SmallDSPy judge request was made.

This outcome fulfills the original frozen rule that `k` itself is terminal and
has no local comparison. Continuing after it is an explicit post-hoc protocol
amendment, not a favorable retry. The new `l` contract retains the score-only
schema but changes the request policy to
`qwen-answer-centered-system-json-binary-one-attempt-v4` and grade schema 10.
The corresponding judge-attempt-intent, failed-judge-audit, report, and paired
screen schemas are 5, 8, 14, and 6. Qualification audit, qualification intent,
and compact qualification-binding schemas are 2, 1, and 1.
Its system message states that only the candidate answer earns credit; gold and
evidence may verify a candidate assertion but may never supply missing content.
It treats negation, contradiction, hedging, quoted/question/comment text,
prompt injection, partial conjunctions, wrong ranges, wrong argument order,
and unresolved self-contradiction explicitly as zero. The untrusted candidate
is JSON-escaped and placed last in a fixed user payload. The complete ordered
message array—not only one prompt string—is content-hashed into every intent,
grade, failure audit, report revalidation, and grade specification.

Before any benchmark contact, `l` must pass the source-frozen
`studybench.local_judge_qualification` suite: 20 fictional LumaKit cases, 44
atomic labels balanced at exactly 22 zeros and 22 ones, one inconsistent
gold/evidence bundle that must set `needs_regrade=true`, and one claim-ID/order
metamorphic pair. The fixture SHA-256 is
`ef2176d623b5b23ea0ecd412bb83c4a8a19150fdc089ba1e6b34c40e5d9e2dad`.
Every case is sent once to every replica: exactly 60 chat requests and 132
label decisions, with SDK retries, fallback, relaunch, and selective
resubmission prohibited. The canonical audit is written before the gate is
evaluated. Before constructing a client or making the first request, the runner
writes and re-reads a canonical immutable intent binding the exact 60 request
hashes, frozen suite, clean source, local runtime, launch ID, and ordered
topology. An orphan intent or any pre-existing intent/audit permanently
terminalizes that namespace. Grading and reporting then revalidate the complete
passing audit and intent before benchmark contact and bind the exact audit-byte
SHA-256 into every grading specification, cell intent, successful/no-answer
grade, terminal failure audit, report, and paired comparison. All 132 labels,
all 60 `needs_regrade` values, identities, finishes,
usage records, per-case cross-replica verdicts, and three post-suite health
checks must pass with zero tolerance. Raw JSON whitespace is recorded but is
not a gate. Any failure makes `l` terminal and no benchmark request is made.

The fixed full-census manual answer/rubric/evidence audit below applies to a
complete `m` population without selectively repairing labels. Even a perfect
synthetic qualification establishes only minimum face validity for this local
screen, not agreement with GPT-5.4 or reliable grading of real technical
rubrics. Publication claims still require an independent qualified judge and
a fresh preregistered held-out/full-corpus evaluation.

Because correcting grader code necessarily changes `HEAD`, the immutable `h`
answers cannot pass the ordinary same-source gate. Rerunning answers after
seeing them would be the worse scientific choice. The only permitted exception
therefore requires the exact full historical generator commit on both the
grading and reporting command lines, reconstructs every scoped file directly
from Git and checks it against each run manifest, separately binds the current
clean grader source and runtime, and hard-codes `claim_ready=false` and
`paper_comparison_allowed=false`. It is unavailable to smoke, external,
confirmatory, or paper-comparison paths. The paired comparison requires both
arms to have byte-identical generator source records, grader source records,
request policies, runtimes, and checker contracts. The default current-source
gate remains unchanged.

The `f` divergence also makes server replica an observed blocking variable.
The implementation now assigns every expected episode a canonical server slot
from the immutable full-grid ordinal, stores the complete path-to-slot map and
policy in each arm's manifest, binds the slot into attempt intents and final or
failed episodes, and routes by that stored slot even after an interruption.
Normalized loopback endpoints are ordered by ascending numeric port, and that
endpoint-order policy is manifest-bound, so a permuted `--base-urls` argument
cannot silently remap a stored slot on resume.
Comparison rejects assignment drift or a per-cell slot mismatch. In each arm,
each of the three slots receives exactly five cells per budget and 20 of 60
cells overall; paired baseline and treatment cells use the same slot within the
same six-GPU server lifecycle.

## Completed blinded pre-grade generation census

Before `l` qualification or grading, an isolated reviewer read all 120 frozen
generation episodes without opening any grade, report, or outcome. The census
confirmed 60 cells per arm (`5 questions × 4 budgets × 3 rollouts`), exact
question/prompt/manifest/note/environment/source hashes, balanced server slots,
120 matching generation-attempt intents, and complete token ledgers. Base has
60 nonempty `ok` records; treatment has 59 and the one preregistered ITT
`no_answer` at `k20f/r1/dspy_2de37073e8e4`. All 119 nonempty answers are unique
both bytewise and after case/whitespace normalization. There was no off-topic
answer, cross-question or game leakage, prompt injection, copied study-note
text, template contamination, or NUL/replacement character.

The full read found four answer-level presentation/completeness defects that
remain immutable and eligible for ordinary grading:

| Arm/cell | Finding |
|---|---|
| base `k5/r1/dspy_a5b116f00083` | hard-truncated 862-character answer stops inside an open Python fence and an unfinished `dspy.Predict` string |
| base `k20/r0/dspy_0b4b420ebeb4` | otherwise complete answer has one extra trailing code fence |
| cheatsheet `k20/r0/dspy_2de37073e8e4` | otherwise complete answer omits its closing code fence |
| cheatsheet `k20/r0/dspy_7329144ef1e9` | otherwise complete answer splits `mock_search_rides` across a newline inside the function identifier |

Here `status=ok` means that the preregistered generation harness durably
captured a nonempty answer without an infrastructure failure; it does not claim
that the answer is complete or correct. The truncated answer is therefore an
answer defect to score as generated, not grounds for a retry, status rewrite,
or exclusion.

The reviewer also inspected all 780 recorded ReAct turns: 286 `finish`, 130
`glob`, 146 `grep`, and 218 `read_file`. Of 494 non-finish calls, 490 replayed
byte-for-byte against the pinned corpus. One treatment
`k5/r1/dspy_a5b116f00083` call passed unsupported line arguments to `grep` and
honestly stored the validation traceback. Three grep observations were
semantically consistent but not byte-replayable because the original tool hit
its time budget and emitted a truncation marker (treatment
`k5/r0/dspy_a5b116f00083` turns 0 and 2, and treatment
`k20/r2/dspy_7329144ef1e9` turn 1). The remaining recorded lookup errors, empty
searches, invalid finish arguments, and forced-budget catches were internally
consistent; no fabricated or mismatched tool observation was found.

The treatment note SHA-256 is
`8e3733d16c669642fa9bf96dfd2eb1de89366b75c6c7f42a5f7c9a7ba6eb33cb`;
its study manifest, study episode, and study intent hashes are respectively
`3444965f71f5a37d63e889a481b6d58a25f534b85224eae5ddfc9900842638e3`,
`3966de9aa726df6571113eedfe028c2c748438f199960099306114a03547b513`, and
`cc0010d860b628dca477ed0cba7947db5b194b590ca37376507b795be66b9646`.
The note is byte-identical across its study, run-input, and provenance copies,
contains none of the five target scenarios, and is bound to treatment only.
All 80 frozen research-source files reproduce exactly from generator commit
`1d94e6777c6d50655a557d6c4ae8e904662bde28`; the pinned SmallDSPy corpus is
clean at `9cdb0aac28b2a04b064e40697ccd301872cf6a43`.

## Frozen post-score integrity and manual-audit protocol

This protocol was fixed before `k` calibration and is carried forward unchanged
for any complete `m` population. It is an audit of immutable outputs, not a
second adaptive grader and not a way to repair a preferred result.

First, a machine census covers all 120 generation cells and all 120 terminal
grade artifacts. It rechecks canonical encoding, schema versions, source and
prompt hashes, question/budget/rollout identity, generation status, the one
treatment `no_answer`, manifest slot, full stored judge topology, substantive
launch ID, request policy/options, response model, accepted response identity,
`finish_reason=stop`, complete token arithmetic, exact keyed binary labels, and
deterministic weighted-score recomputation. For the 119 answered cells there
must be exactly 119 pre-contact intents, 119 accepted one-request judge audits,
and zero failed-judge audits; the no-answer cell must have neither a judge
intent nor a provider contact. Any mismatch makes `m` incomplete and prevents
reporting the two scores.

Second, the human/agent-assisted review is a full census rather than a sample:

1. Inspect all 120 candidate answers for relevance, source-grounded factual
   correctness, material omissions, and whether the stored `ok`/`no_answer`
   status is truthful. Inspect every atomic rubric claim for each of the 119
   answered cells (all judge labels, not only positive labels).
2. Build rows only after grade files are immutable: exactly 120 answer-review
   rows plus 619 answered-cell claim-review rows (739 first-pass rows total).
   Compute every JSON digest with `studybench.integrity.sha256_json`, which
   hashes `canonical_json_bytes` including its canonical trailing newline.
   Define `cell_binding_sha256` from exactly
   `{"audit_schema":1,"arm":arm,"qid":qid,"budget":budget,"rollout":rollout,"episode_sha256":episode_sha256,"grade_sha256":grade_sha256}`,
   where `arm` is exactly `base` or `cheatsheet`, `qid` is the dataset question
   ID, `budget` is the manifest budget string, `rollout` is the integer rollout,
   and both digests are the 64-character lowercase content hashes of the exact
   immutable files. Define `row_id` with `sha256_json` over exactly
   `{"audit_schema":1,"cell_binding_sha256":cell_binding_sha256,"unit":"answer","claim_id":null}`
   for an answer row, or the same object with `unit="claim"` and the exact
   claim-ID string. Sort by SHA-256 of the literal UTF-8 bytes `46001:` followed by
   the 64 lowercase ASCII `row_id` bytes; duplicate IDs or order keys are
   fatal. During the first pass, hide arm, budget, rollout, run/grade path,
   server slot, Qwen label, weighted total, and aggregate result. An answer row
   shows the question, candidate answer, gold answer, complete rubric, and all
   dataset evidence (with the same pinned corpus read access available for
   source verification). A claim row shows the question, candidate answer,
   gold answer, exactly one atomic claim with its type/weight, and that claim's
   cited evidence.
3. For each answer record `answer_ok`, `answer_incorrect`, or `uncertain`, plus
   a concise reason and any corpus/evidence issue. For each claim independently
   record `0`, `1`, or `uncertain`, confidence (`high`, `medium`, or `low`), an
   ambiguity flag, and a rubric/evidence-defect flag with a source-grounded
   note. Do not infer the hidden judge label from neighboring repeated rows.
4. Reveal Qwen labels only after the entire first pass is durably written.
   Send every reviewer-Qwen disagreement, every `uncertain`, and every flagged
   rubric/evidence defect to a second reviewer in a new deterministic order:
   sort by SHA-256 of the literal UTF-8 bytes `46002:` followed by the 64
   lowercase ASCII `row_id` bytes; duplicate IDs or order keys remain fatal.
   The second reviewer remains blind to arm, Qwen label, aggregate result, and
   first-review label. Preserve both judgments; adjudication never overwrites
   either one.
5. Publish overall and per-arm determinate agreement, the `0/1` confusion
   counts, uncertainty/defect counts, claim-weighted disagreement, and a table
   of every disagreement with the relevant answer/claim/evidence and both
   reviewer decisions. Also disclose any answer-level factual defect even when
   its atomic labels happen to agree.

Before a packet is opened, its audit manifest records each reviewer agent ID,
the exposed model/version string (or explicit `unavailable`), review-prompt
SHA-256, packet SHA-256, allowed tool/corpus access, and independence role. The
first and second passes use fresh isolated agent contexts with no conversation
history and instructions to read only their blinded packets and cited pinned
corpus evidence—not grades, reports, experiment outcomes, or sibling reviewer
files. The main operator, who can see grading logs, constructs and verifies the
packets but does not supply first-pass labels. If this isolation cannot be
maintained, the deviation is disclosed and the review is not described as
blinded.

The audit cannot alter a grade, trigger a retry, drop a cell, tune the prompt,
or substitute reviewer labels into the two requested local-Qwen numbers. There is
no post-hoc agreement threshold that upgrades this public adaptive screen into
a research claim. Systematic, asymmetric, or high-weight disagreement weakens
or invalidates interpretation and is reported as such; it does not get repaired
away. The local-Qwen scores and the reviewer sensitivity summary remain visibly
separate.

## Execution namespaces and phase commands

The outcome-bearing source tree must be committed, pushed, and clean. A failed
smoke or full namespace is never deleted or reused; use a fresh suffix/ID.

Generation runs inside one six-GPU Slurm step and one authenticated server
lifecycle. It first runs a one-question baseline smoke and the full baseline
grid, then the forced-50 construction, a one-question note-bearing smoke, and
the full treatment grid. Each smoke/construction gate is inspected before the
next dependent phase. The two 60-episode evaluation grids remain sequential:

Before any SmallDSPy prompt, require the deterministic checks implemented by
`serve_and_wait.sh` and independently inspect all three engine configurations
for `enable_in_reasoning=False`. Do not issue another output-bearing topology
qualification request. This rule is frozen after the diagnostic `f` result and
before any outcome prompt.

```bash
srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=60 --cpu-bind=none --gres=gpu:l40s:6 \
  env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 SB_PORT_BASE=35500 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smalldspy-generation-h \
  bash --noprofile --norc

# Inside that step, after setup and authenticated readiness:
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/setup_common.sh
verify_env_file
sync_dspy_environment
source scripts/serve_and_wait.sh
PY=.venv-dspy/bin/python

"$PY" -m studybench.react --task smalldspy \
  --run-id smalldspy-generation-smoke-20260714h --seed 43999 \
  --seed-group smalldspy-generation-smoke-20260714h \
  --budgets direct,k5 --rollouts 1 --limit 1 --smoke \
  --base-urls "$BASE_URLS" --concurrency 2

"$PY" -m studybench.react --task smalldspy \
  --run-id smalldspy-local-base-20260714h --seed 44001 \
  --seed-group smalldspy-local-cheatsheet-screen-20260713 \
  --budgets direct,k5,k20,k20f --rollouts 3 --exploratory \
  --base-urls "$BASE_URLS" --concurrency 12

STUDY_ID=smalldspy-cheatsheet-s50-20260714h
"$PY" -m studybench.react --task smalldspy --study \
  --study-id "$STUDY_ID" --seed 43001 --base-urls "$BASE_URLS"
MANIFEST="studies/cheatsheet/$STUDY_ID/smalldspy/manifest.json"
NOTE=$("$PY" -c 'import json,sys; from pathlib import Path; m=Path(sys.argv[1]); p=Path(json.loads(m.read_text(encoding="utf-8"))["note_path"]); assert not p.is_absolute() and len(p.parts)==1; print(m.parent/p)' "$MANIFEST")
test -f "$MANIFEST" -a ! -L "$MANIFEST" -a -f "$NOTE" -a ! -L "$NOTE"

"$PY" -m studybench.react --task smalldspy \
  --run-id smalldspy-treatment-smoke-20260714h --seed 43999 \
  --seed-group smalldspy-treatment-smoke-20260714h \
  --budgets direct,k5 --rollouts 1 --limit 1 --smoke \
  --note "$NOTE" --note-manifest "$MANIFEST" \
  --base-urls "$BASE_URLS" --concurrency 2

"$PY" -m studybench.react --task smalldspy \
  --run-id smalldspy-local-cheatsheet-20260714h --seed 44001 \
  --seed-group smalldspy-local-cheatsheet-screen-20260713 \
  --budgets direct,k5,k20,k20f --rollouts 3 --exploratory \
  --note "$NOTE" --note-manifest "$MANIFEST" \
  --base-urls "$BASE_URLS" --concurrency 12
```

The following `k` recipe is preserved as executed history and must never be
rerun. After the generation step exited and all six generation-server processes
were confirmed gone, a separate six-GPU Slurm step started three fresh TP=2
local judges. An offline template test and one predeclared, non-benchmark
strict-JSON calibration per ordered endpoint must first verify that the exact
no-thinking option reaches all three replicas, produces final content, and
distinguishes one trivially satisfied atomic claim from one trivially
contradicted claim. Exactly three requests are issued concurrently, with SDK
retries disabled and no endpoint fallback, remapping, relaunch, or selective
resubmission. All responses or errors are harvested, all clients are closed,
and one canonical immutable calibration audit is written before the all-slots
assertion was evaluated. This was only a minimum transport/semantic sanity check;
it cannot validate real-rubric judgment quality, score or select a benchmark
answer, or establish agreement with GPT-5.4. Any slot failure made `k` terminal
and no benchmark grading request is made.

There was no historical benchmark smoke: historical source separation is
deliberately restricted to complete exploratory populations. The unreached
`k` success branch would have uniformly graded both complete frozen populations
exactly once before either report, then compared their content-addressed
reports with the frozen description `forced-50 scoped SmallDSPy cheatsheet
versus no note`:

```bash
srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=60 --cpu-bind=none --gres=gpu:l40s:6 \
  env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 SB_PORT_BASE=35900 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smalldspy-grading-k \
  bash --noprofile --norc

# Inside that step; the launcher reaches authenticated readiness only after the
# clean-source and fresh-namespace checks below:
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export GRADER_MODEL=local OPENAI_API_KEY= SAKANA_API_KEY=
source scripts/setup_common.sh
verify_env_file
sync_main_environment
PY=.venv/bin/python
VLLM_PY=.venv-vllm/bin/python
CALIBRATION_AUDIT=logs/vllm-16142825-smalldspy-grading-k.calibration.json
GEN_COMMIT=1d94e6777c6d50655a557d6c4ae8e904662bde28

# Freeze the exact outcome-bearing source and every final namespace before the
# first calibration model call. Untracked historical artifacts are outside the
# source record, but no tracked research file may be dirty.
"$PY" - <<'PY'
from studybench.provenance import source_record
record = source_record()
assert record["dirty"] is False, record
print(f"clean grader source commit: {record['git_commit']}")
PY
for path in \
  "$CALIBRATION_AUDIT" \
  grades/smalldspy-local-base-20260714h/qwen-local-base-20260714k \
  grades/smalldspy-local-cheatsheet-20260714h/qwen-local-cheatsheet-20260714k \
  reports/smalldspy-local-base-20260714h/qwen-local-base-20260714k \
  reports/smalldspy-local-cheatsheet-20260714h/qwen-local-cheatsheet-20260714k; do
  test ! -e "$path" -a ! -L "$path"
done
if compgen -G 'logs/vllm-16142825-smalldspy-grading-k*' >/dev/null; then
  echo 'final k log namespace already exists' >&2
  exit 1
fi

source scripts/serve_and_wait.sh
JUDGE_BASE_URLS=$BASE_URLS

# Verify the pinned template closes an empty thinking block under the exact
# local-judge option. Then run one predeclared synthetic atomic calibration
# containing no StudyBench benchmark question, answer, gold, rubric, or evidence.
"$VLLM_PY" - <<'PY'
from pathlib import Path
from transformers import AutoTokenizer

snapshot = Path("/matx/u/omarah/hf/hub/models--Qwen--Qwen3.5-9B/snapshots") \
    / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": "synthetic transport check"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
assert rendered.endswith("<think>\n\n</think>\n\n")
PY

JUDGE_BASE_URLS="$JUDGE_BASE_URLS" \
CALIBRATION_AUDIT="$CALIBRATION_AUDIT" "$PY" - <<'PY'
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from types import SimpleNamespace
from studybench.grade import (
    LOCAL_GRADER_MODEL,
    LOCAL_GRADER_MODEL_REVISION,
    LOCAL_GRADER_REQUEST_OPTIONS,
    LOCAL_GRADER_REQUEST_POLICY,
    LOCAL_GRADER_RATIONALE_POLICY,
    LOCAL_GRADER_VERDICT_CONTRACT,
    build_prompt,
    judge_schema,
    parse_json,
    validate_verdict,
)
from studybench.integrity import (
    canonical_json_bytes,
    read_artifact_bytes,
    sha256_json,
    write_immutable_json,
)
from studybench.provenance import validate_local_server_urls

# This predeclared non-benchmark calibration has one clearly satisfied and one
# clearly contradicted atomic claim. It tests only a minimum semantic invariant,
# not real-rubric validity or judge agreement with GPT-5.4.
row = {
    "id": "synthetic_atomic_calibration",
    "topic": "synthetic",
    "question": "What color and shape is the marker?",
    "gold_answer": "The marker is blue and square.",
    "rubric": [
        {
            "claim_id": "c1", "claim_type": "core", "weight": 50,
            "statement": "Says that the marker is blue.",
            "span_ids": ["s1"],
        },
        {
            "claim_id": "c2", "claim_type": "core", "weight": 50,
            "statement": "Says that the marker is square.",
            "span_ids": ["s1"],
        },
    ],
    "evidence": [{
        "span_id": "s1", "path": "synthetic.txt",
        "start_line": 1, "end_line": 1,
        "excerpt": "The marker is blue and square.",
    }],
}
prompt = build_prompt(
    SimpleNamespace(display="Synthetic"),
    row,
    'The marker is blue; config={"shape":"circular"}, not "square".',
    whole_files=False,
    judge_model=LOCAL_GRADER_MODEL,
)
schema = judge_schema(row, LOCAL_GRADER_MODEL)
urls = validate_local_server_urls(os.environ["JUDGE_BASE_URLS"], expected_count=3)
request = {
    "model": LOCAL_GRADER_MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "response_format": schema,
    **LOCAL_GRADER_REQUEST_OPTIONS,
}

async def call(slot, url):
    client = None
    try:
        client = AsyncOpenAI(
            api_key=os.environ["SB_VLLM_API_KEY"],
            base_url=url,
            max_retries=0,
        )
        response = await client.chat.completions.create(
            **request,
        )
        request_id = (
            getattr(response, "_request_id", None)
            or getattr(response, "request_id", None)
        )
        content = response.choices[0].message.content
        usage = response.usage.model_dump(mode="json")
        errors = []
        scores = None
        try:
            value = parse_json(content, label="synthetic judge verdict")
            _, scores = validate_verdict(row, value, LOCAL_GRADER_MODEL)
            if content != json.dumps(
                value, ensure_ascii=False, separators=(",", ":")
            ):
                errors.append("response JSON is not the exact compact serialization")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if response.model != LOCAL_GRADER_MODEL:
            errors.append("response model mismatch")
        if response.choices[0].finish_reason != "stop":
            errors.append("finish reason is not stop")
        if not isinstance(response.id, str) or not response.id:
            errors.append("response id missing")
        if not isinstance(request_id, str) or not request_id:
            errors.append("request id missing")
        if (
            type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
            or type(usage.get("total_tokens")) is not int
            or usage["prompt_tokens"] + usage["completion_tokens"]
            != usage["total_tokens"]
            or not 0 < usage["completion_tokens"] < 256
        ):
            errors.append("usage is missing, inconsistent, or reaches the ceiling")
        if scores != {"c1": 1, "c2": 0}:
            errors.append("synthetic atomic scores differ from the frozen expectation")
        encoded = content.encode("utf-8") if isinstance(content, str) else None
        return {
            "slot": slot,
            "url": url,
            "response_id": response.id,
            "request_id": request_id,
            "response_model": response.model,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": response.choices[0].finish_reason,
            "content": content,
            "content_bytes": len(encoded) if encoded is not None else None,
            "content_sha256": sha256(encoded).hexdigest()
            if encoded is not None else None,
            "usage": usage,
            "scores": scores,
            "errors": errors,
            "passed": not errors,
        }
    except Exception as exc:
        return {
            "slot": slot,
            "url": url,
            "response_id": None,
            "request_id": None,
            "response_model": None,
            "system_fingerprint": None,
            "finish_reason": None,
            "content": None,
            "content_bytes": None,
            "content_sha256": None,
            "usage": None,
            "scores": None,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "passed": False,
        }
    finally:
        if client is not None:
            await client.close()

async def main():
    raw = await asyncio.gather(
        *(call(slot, url) for slot, url in enumerate(urls)),
        return_exceptions=True,
    )
    responses = []
    for slot, (url, result) in enumerate(zip(urls, raw, strict=True)):
        if isinstance(result, BaseException):
            result = {
                "slot": slot, "url": url, "response_id": None,
                "request_id": None, "response_model": None,
                "system_fingerprint": None, "finish_reason": None,
                "content": None, "content_bytes": None,
                "content_sha256": None, "usage": None, "scores": None,
                "errors": [f"{type(result).__name__}: {result}"],
                "passed": False,
            }
        responses.append(result)
    artifact = {
        "calibration_schema_version": 1,
        "claim_ready": False,
        "purpose": "non-benchmark-local-judge-transport-and-atomic-sanity",
        "judge_model": LOCAL_GRADER_MODEL,
        "judge_model_revision": LOCAL_GRADER_MODEL_REVISION,
        "judge_request_policy": LOCAL_GRADER_REQUEST_POLICY,
        "judge_verdict_contract": LOCAL_GRADER_VERDICT_CONTRACT,
        "judge_rationale_policy": LOCAL_GRADER_RATIONALE_POLICY,
        "server_launch_id": os.environ["SB_SERVER_LAUNCH_ID"],
        "ordered_urls": urls,
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "request_sha256": sha256_json(request),
        "request_options": LOCAL_GRADER_REQUEST_OPTIONS,
        "response_format": schema,
        "expected_scores": {"c1": 1, "c2": 0},
        "responses": responses,
        "all_passed": all(result["passed"] for result in responses),
    }
    path = Path(os.environ["CALIBRATION_AUDIT"])
    write_immutable_json(path, artifact)
    data = read_artifact_bytes(path)
    assert data == canonical_json_bytes(artifact)
    print(f"calibration audit: {path} sha256={sha256(data).hexdigest()}")
    if not artifact["all_passed"]:
        raise SystemExit("all-endpoint synthetic calibration failed; k is terminal")

asyncio.run(main())
PY

for arm in base cheatsheet; do
  "$PY" -m studybench.grade --task smalldspy \
    --run-id "smalldspy-local-$arm-20260714h" \
    --grade-id "qwen-local-$arm-20260714k" \
    --judge-base-url "$JUDGE_BASE_URLS" --excerpt-evidence --concurrency 24 \
    --historical-exploratory-source-commit "$GEN_COMMIT"
done

for arm in base cheatsheet; do
  "$PY" -m studybench.report --tasks smalldspy \
    --run-id "smalldspy-local-$arm-20260714h" --grader local \
    --grade-id "qwen-local-$arm-20260714k" \
    --judge-base-url "$JUDGE_BASE_URLS" --excerpt-evidence \
    --historical-exploratory-source-commit "$GEN_COMMIT" \
    --ci 10000 --ci-seed 45001
done

mapfile -t BASE_REPORTS < <(find \
  reports/smalldspy-local-base-20260714h/qwen-local-base-20260714k/smalldspy \
  -maxdepth 1 -type f -name 'report-*.json' | LC_ALL=C sort)
mapfile -t CHEAT_REPORTS < <(find \
  reports/smalldspy-local-cheatsheet-20260714h/qwen-local-cheatsheet-20260714k/smalldspy \
  -maxdepth 1 -type f -name 'report-*.json' | LC_ALL=C sort)
test "${#BASE_REPORTS[@]}" -eq 1 -a "${#CHEAT_REPORTS[@]}" -eq 1
"$PY" -m studybench.screen_compare \
  --control-report "${BASE_REPORTS[0]}" \
  --treatment-report "${CHEAT_REPORTS[0]}" \
  --intervention-description \
    'forced-50 scoped SmallDSPy cheatsheet versus no note' \
  --bootstrap-replicates 10000 --bootstrap-seed 45001
```

`k` exited at its calibration command, so none of the grading/report/comparison
commands in that historical block ran. The following is the separately frozen
post-hoc `l` recipe. It uses a fresh source commit, launch, port range, log
prefix, qualification artifact, grade IDs, and report paths. The qualification
module itself validates the exact 20-case/44-label fixture and clean source,
collects all 60 request/response records, performs one authenticated model-list
health check per endpoint, and then writes the canonical terminal audit before
evaluating and returning zero-tolerance success. Both grades still complete
before either report:

```bash
srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=60 --cpu-bind=none --gres=gpu:l40s:6 \
  env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 SB_PORT_BASE=36100 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smalldspy-grading-l \
  bash --noprofile --norc

# Inside that step:
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export GRADER_MODEL=local OPENAI_API_KEY= SAKANA_API_KEY=
source scripts/setup_common.sh
verify_env_file
sync_main_environment
PY=.venv/bin/python
VLLM_PY=.venv-vllm/bin/python
GEN_COMMIT=1d94e6777c6d50655a557d6c4ae8e904662bde28

"$PY" - <<'PY'
from studybench.provenance import source_record
record = source_record()
assert record["dirty"] is False, record
print(f"clean grader source commit: {record['git_commit']}")
PY
for path in \
  grades/smalldspy-local-base-20260714h/qwen-local-base-20260714l \
  grades/smalldspy-local-cheatsheet-20260714h/qwen-local-cheatsheet-20260714l \
  reports/smalldspy-local-base-20260714h/qwen-local-base-20260714l \
  reports/smalldspy-local-cheatsheet-20260714h/qwen-local-cheatsheet-20260714l; do
  test ! -e "$path" -a ! -L "$path"
done
if compgen -G 'logs/vllm-16142825-smalldspy-grading-l*' >/dev/null; then
  echo 'final l log namespace already exists' >&2
  exit 1
fi

source scripts/serve_and_wait.sh
JUDGE_BASE_URLS=$BASE_URLS
QUALIFICATION_AUDIT="logs/local-judge-qualification-${SB_SERVER_LAUNCH_ID}.json"
for path in "$QUALIFICATION_AUDIT" "${QUALIFICATION_AUDIT}.intent.json"; do
  test ! -e "$path" -a ! -L "$path"
done
for slot in 0 1 2; do
  log="logs/vllm-16142825-smalldspy-grading-l-$slot.log"
  rg -q "structured_outputs_config.*backend='xgrammar'.*disable_any_whitespace=True.*enable_in_reasoning=False" "$log"
done

"$VLLM_PY" - <<'PY'
from pathlib import Path
from transformers import AutoTokenizer

snapshot = Path("/matx/u/omarah/hf/hub/models--Qwen--Qwen3.5-9B/snapshots") \
    / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
rendered = tokenizer.apply_chat_template(
    [{"role": "system", "content": "synthetic system"},
     {"role": "user", "content": "synthetic user"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
assert rendered.endswith("<think>\n\n</think>\n\n")
PY

"$PY" -m studybench.local_judge_qualification \
  --judge-base-url "$JUDGE_BASE_URLS" \
  --output "$QUALIFICATION_AUDIT"

for arm in base cheatsheet; do
  "$PY" -m studybench.grade --task smalldspy \
    --run-id "smalldspy-local-$arm-20260714h" \
    --grade-id "qwen-local-$arm-20260714l" \
    --judge-base-url "$JUDGE_BASE_URLS" --excerpt-evidence --concurrency 24 \
    --qualification-audit "$QUALIFICATION_AUDIT" \
    --historical-exploratory-source-commit "$GEN_COMMIT"
done

for arm in base cheatsheet; do
  "$PY" -m studybench.report --tasks smalldspy \
    --run-id "smalldspy-local-$arm-20260714h" --grader local \
    --grade-id "qwen-local-$arm-20260714l" \
    --judge-base-url "$JUDGE_BASE_URLS" --excerpt-evidence \
    --qualification-audit "$QUALIFICATION_AUDIT" \
    --historical-exploratory-source-commit "$GEN_COMMIT" \
    --ci 10000 --ci-seed 45001
done

mapfile -t BASE_REPORTS < <(find \
  reports/smalldspy-local-base-20260714h/qwen-local-base-20260714l/smalldspy \
  -maxdepth 1 -type f -name 'report-*.json' | LC_ALL=C sort)
mapfile -t CHEAT_REPORTS < <(find \
  reports/smalldspy-local-cheatsheet-20260714h/qwen-local-cheatsheet-20260714l/smalldspy \
  -maxdepth 1 -type f -name 'report-*.json' | LC_ALL=C sort)
test "${#BASE_REPORTS[@]}" -eq 1 -a "${#CHEAT_REPORTS[@]}" -eq 1

QUALIFICATION_AUDIT="$QUALIFICATION_AUDIT" \
BASE_REPORT="${BASE_REPORTS[0]}" CHEAT_REPORT="${CHEAT_REPORTS[0]}" \
"$PY" - <<'PY'
import os
from pathlib import Path
from studybench.integrity import read_artifact_bytes, sha256_bytes, strict_json_loads

qualification_bytes = read_artifact_bytes(Path(os.environ["QUALIFICATION_AUDIT"]))
qualification_sha256 = sha256_bytes(qualification_bytes)
qualification = strict_json_loads(
    qualification_bytes,
    label="local judge qualification",
)
assert qualification["all_passed"] is True
assert len(qualification["responses"]) == 60
assert all(item["passed"] for item in qualification["responses"])
assert all(item["passed"] for item in qualification["post_qualification_health"])
for variable in ("BASE_REPORT", "CHEAT_REPORT"):
    report = strict_json_loads(
        read_artifact_bytes(Path(os.environ[variable])), label=variable
    )
    config = report["grading_manifest"]["config"]
    assert config["judge_request_policy"] == qualification["judge_request_policy"]
    assert config["local_judge_qualification_sha256"] == qualification_sha256
    assert config["local_judge_qualification"]["audit"]["sha256"] == qualification_sha256
    assert config["local_judge_qualification"]["binding"]["audit_sha256"] == qualification_sha256
    assert config["local_judge_runtime_sha256"] == qualification["local_judge_runtime_sha256"]
    assert config["local_judge_runtime"]["server_launch_id"] == qualification["server_launch_id"]
    assert config["judge_validation_urls"] == qualification["ordered_urls"]
    assert config["generation_source_validation"]["grader_source"] == qualification["source"]
PY

"$PY" -m studybench.screen_compare \
  --control-report "${BASE_REPORTS[0]}" \
  --treatment-report "${CHEAT_REPORTS[0]}" \
  --intervention-description \
    'forced-50 scoped SmallDSPy cheatsheet versus no note' \
  --bootstrap-replicates 10000 --bootstrap-seed 45001
```

The `l` content-addressed report paths are resolved after reporting and supplied to
`studybench.screen_compare` with 10,000 paired bootstrap replicates and seed
`45001`. The comparison is an additional diagnostic artifact, not one of the
two requested primary numbers.

## Results

`k` is terminal with no benchmark grades for the reasons above. `l` is also
terminal with no benchmark grades. Its pushed clean launch completed all 60
qualification requests, but only 43 request verdicts matched every frozen
expectation and replica consensus failed for C15. The 17 failed request
verdicts were complete, schema-valid, stop-terminated Qwen outputs: three
replicas each missed C03, C09, C13, C18, and C20, while two replicas missed
C15. This rules out transport, truncation, parsing, and structured-output
failure; the no-thinking elicitation was semantically inadequate. The complete
immutable audit is
`logs/local-judge-qualification-38a7c81c25a19e6269fbb9b6daaafa27e0da0aac36451ddd96eb9f85a715058e.json`,
SHA-256
`cc3615daedbbbae96802e93369d7b21a97feaea91da374f51f5f8c4c9ce924c6`.

Protocol `m` is a disclosed post-hoc correction, not a retry of `l`. It keeps
the full candidate-last input, exact score-only schema, temperature zero, seed
zero, one request, no fallback, and the same zero-tolerance qualification. Its
only elicitation change is to permit Qwen's native private reasoning before
the final constrained JSON and raise the completion ceiling from 256 to 4,096
tokens so that reasoning cannot crowd out that JSON. It uses a fresh launch,
canonical qualification audit, grade IDs, reports, and comparison namespace.
If `m` qualification fails, no benchmark call is permitted.

The immutable `h` generation populations are
the only answer populations eligible for this screen. The `d` generation grids
and failed judge smoke are invalidated diagnostics; `e` failed its
pre-benchmark structured-output gate; and `f` failed its fully audited
output-bearing gate. The old partial `h` grading and the incomplete matched `i`
grading are also terminal diagnostic evidence, not arm results. The standalone
`i` baseline report is retained and disclosed above, but its WAUC is not paired
with a treatment estimate. No valid full-grid arm comparison has yet been
observed. Only uniform `m` grades over all 120 frozen cells, after
the all-replica qualification passes, may populate this section; such numbers
remain explicitly post-hoc and non-claim-ready.
