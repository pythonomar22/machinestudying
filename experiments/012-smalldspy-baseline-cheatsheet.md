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
intent records; it is diagnostic and permanently unreportable. The next and only
score-bearing grading pass uniformly regrades both frozen `h` populations in
fresh `i` namespaces with an atomic-only, non-thinking local-judge protocol.
This is an adaptive correction made after inspecting failures, so any resulting
numbers remain exploratory and non-claim-ready.

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
source layout, all five questions share one topic, and repository history
already contains 1,080 rollout artifacts and 963 grade artifacts for these IDs.
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
| corrected local-judge mode | temperature 0, seed 0, maximum 8,192 output tokens, and `chat_template_kwargs.enable_thinking=false`; strict JSON contains only atomic claim labels, rationales, and `needs_regrade` |
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

The published Appendix A.5 still shows the older `0/0.5/1` claim scale and asks
the judge to echo `question_score`. In `docs/jacob.md`, the author explicitly
corrected the claim scale to `0/1` because partial credit increased variance and
later answered that “lenient is just weights summed together.” The implementation
therefore treats each atomic label as the judge's only substantive decision and
computes the weighted total deterministically. Removing the redundant model
total cannot add credit; it removes an arithmetic failure mode. The paper used
GPT-5.4 with whole evidence files, whereas this deliberately cheaper screen uses
the same pinned 9B Qwen family as answerer and judge with dataset excerpts.
Consequently this is a local proxy, not a byte-exact paper replication or an
external validation of the rubric decisions.

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
context requires TP=2 on this GPU class, so generation uses six GPUs as three
identical TP=2 servers; no scientifically different TP=1 seventh server is
introduced. Local grading uses one TP=2 server on two GPUs. The grader currently
has a one-endpoint provenance contract; adding three judge replicas after the
protocol failure would require a preregistered, paired per-cell assignment and
new report/compare validation. For only 119 actual judge calls under the
no-thinking protocol, that late change is not worth the additional confound or
implementation risk. The other five GPUs remain free during grading rather
than being claimed without scientifically valid work.

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

After the generation step exits and all six server processes are confirmed
gone, a separate two-GPU Slurm step starts one fresh corrected local judge. An
offline template test and one predeclared, non-benchmark strict-JSON calibration
must first verify that the exact no-thinking option reaches vLLM, produces final
content, and distinguishes one trivially satisfied atomic claim from one
trivially contradicted claim. This is only a minimum transport/semantic sanity
check; it cannot validate real-rubric judgment quality, score or select a
benchmark answer, or establish agreement with GPT-5.4. If it fails, no benchmark
grading request is made. There is no historical benchmark smoke:
historical source separation is deliberately restricted to complete
exploratory populations. After the synthetic gate, uniformly grade and report
both complete frozen populations exactly once, then compare their
content-addressed reports with the already frozen description `forced-50 scoped
SmallDSPy cheatsheet versus no note`:

```bash
srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=20 --cpu-bind=none --gres=gpu:l40s:2 \
  env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 SB_PORT_BASE=35700 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smalldspy-grading-j \
  bash --noprofile --norc

# Inside that step, after setup and authenticated readiness:
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export GRADER_MODEL=local OPENAI_API_KEY= SAKANA_API_KEY=
source scripts/setup_common.sh
verify_env_file
sync_main_environment
source scripts/serve_and_wait.sh
PY=.venv/bin/python
VLLM_PY=.venv-vllm/bin/python
JUDGE_BASE_URL=$BASE_URLS
GEN_COMMIT=1d94e6777c6d50655a557d6c4ae8e904662bde28

# Verify the pinned template closes an empty thinking block under the exact
# local-judge option. Then run one predeclared synthetic atomic calibration
# containing no StudyBench question, answer, gold, rubric, or evidence.
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

JUDGE_BASE_URL="$JUDGE_BASE_URL" "$PY" - <<'PY'
import asyncio
import json
import os
from openai import AsyncOpenAI
from types import SimpleNamespace
from studybench.grade import (
    LOCAL_GRADER_MODEL,
    LOCAL_GRADER_REQUEST_OPTIONS,
    build_prompt,
    judge_schema,
    validate_verdict,
)

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
            "claim": "Says that the marker is blue.",
            "evidence_span_ids": ["s1"],
        },
        {
            "claim_id": "c2", "claim_type": "core", "weight": 50,
            "claim": "Says that the marker is square.",
            "evidence_span_ids": ["s1"],
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
    "The marker is blue and circular.",
    whole_files=False,
)

async def main():
    client = AsyncOpenAI(
        api_key=os.environ["SB_VLLM_API_KEY"],
        base_url=os.environ["JUDGE_BASE_URL"],
        max_retries=0,
    )
    try:
        response = await client.chat.completions.create(
            model=LOCAL_GRADER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=judge_schema(row),
            **LOCAL_GRADER_REQUEST_OPTIONS,
        )
    finally:
        await client.close()
    assert response.model == LOCAL_GRADER_MODEL
    content = response.choices[0].message.content
    value = json.loads(content)
    _, scores = validate_verdict(row, value)
    assert scores == {"c1": 1, "c2": 0}
    print("synthetic no-thinking atomic calibration: PASS")

asyncio.run(main())
PY

for arm in base cheatsheet; do
  "$PY" -m studybench.grade --task smalldspy \
    --run-id "smalldspy-local-$arm-20260714h" \
    --grade-id "qwen-local-$arm-20260714j" \
    --judge-base-url "$JUDGE_BASE_URL" --excerpt-evidence --concurrency 8 \
    --historical-exploratory-source-commit "$GEN_COMMIT"
  "$PY" -m studybench.report --tasks smalldspy \
    --run-id "smalldspy-local-$arm-20260714h" --grader local \
    --grade-id "qwen-local-$arm-20260714j" \
    --judge-base-url "$JUDGE_BASE_URL" --excerpt-evidence \
    --historical-exploratory-source-commit "$GEN_COMMIT" \
    --ci 10000 --ci-seed 45001
done

mapfile -t BASE_REPORTS < <(find \
  reports/smalldspy-local-base-20260714h/qwen-local-base-20260714j/smalldspy \
  -maxdepth 1 -type f -name 'report-*.json' | LC_ALL=C sort)
mapfile -t CHEAT_REPORTS < <(find \
  reports/smalldspy-local-cheatsheet-20260714h/qwen-local-cheatsheet-20260714j/smalldspy \
  -maxdepth 1 -type f -name 'report-*.json' | LC_ALL=C sort)
test "${#BASE_REPORTS[@]}" -eq 1 -a "${#CHEAT_REPORTS[@]}" -eq 1
"$PY" -m studybench.screen_compare \
  --control-report "${BASE_REPORTS[0]}" \
  --treatment-report "${CHEAT_REPORTS[0]}" \
  --intervention-description \
    'forced-50 scoped SmallDSPy cheatsheet versus no note' \
  --bootstrap-replicates 10000 --bootstrap-seed 45001
```

The content-addressed report paths are resolved after reporting and supplied to
`studybench.screen_compare` with 10,000 paired bootstrap replicates and seed
`45001`. The comparison is an additional diagnostic artifact, not one of the
two requested primary numbers.

## Results

Pending corrected `j` grading. The immutable `h` generation populations are
the only answer populations eligible for this screen. The `d` generation grids
and failed judge smoke are invalidated diagnostics; `e` failed its
pre-benchmark structured-output gate; and `f` failed its fully audited
output-bearing gate. The old partial `h` grading and the incomplete matched `i`
grading are also terminal diagnostic evidence, not arm results. The standalone
`i` baseline report is retained and disclosed above, but its WAUC is not paired
with a treatment estimate. No valid full-grid arm comparison has yet been
observed. Only uniform fresh-schema `j` grades over all 120 frozen cells may
populate this section.
