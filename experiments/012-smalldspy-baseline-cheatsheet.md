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

Status: implementation and infrastructure preflight complete, but no valid
two-arm score has yet been observed. Replacement `b` and `c` runs exposed
harness accounting defects and were stopped without reuse. Both `d` generation
grids later completed, but the first isolated grading smoke exposed a deeper
vLLM structured-output configuration defect that also contaminated generation:
JSON schemas were enforced inside Qwen's hidden reasoning channel, where a
schema-complete response could terminate with final `content=null`. Because the
same path powered DSPy's JSON format repair, the defect contaminated all 11
baseline and 18 treatment adapter failures; it does not prove what their
counterfactual answers would have been. All `d` artifacts are preserved as
invalidated diagnostics. No `d` population will be graded or scored further;
the one failed grading smoke remains diagnostic. The corrected protocol is
frozen in fresh `h` namespaces below; no prior episode, study note, or judge
response is reused.

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
| arms | no note; exact forced-50 SmallDSPy note prepended |
| answer-time corpus access | `grep`, `glob`, and pinned `read_file` in both arms |
| budgets | `direct`, `k5`, `k20`, `k20f` |
| rollouts | 3 per question and budget; 60 episodes per arm |
| master seed | `44001` |
| paired seed group | `smalldspy-local-cheatsheet-screen-20260713` |
| judge evidence | dataset excerpts, identically in both arms |
| bootstrap | 10,000 paired replicates, seed `45001` |
| primary summary | pure-sum lenient WAUC for each arm |

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
introduced. Local grading uses one TP=2 server on two GPUs.

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
There is exactly one outcome topology and one later grading topology. Complete
model responses that fail to expose a usable answer remain intention-to-treat
non-answers; exhausted judge attempts produce no grade, never a fabricated
zero or a topology retry.

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
gone, a separate two-GPU Slurm step starts one fresh local judge. Grade one
smoke first, inspect it, then grade and report both complete populations before
the judge exits. Finally compare the two content-addressed reports with the
already frozen description `forced-50 scoped SmallDSPy cheatsheet versus no
note`:

```bash
srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=20 --cpu-bind=none --gres=gpu:l40s:2 \
  env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 SB_PORT_BASE=35600 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smalldspy-grading-h \
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
JUDGE_BASE_URL=$BASE_URLS

"$PY" -m studybench.grade --task smalldspy \
  --run-id smalldspy-generation-smoke-20260714h \
  --grade-id qwen-local-smoke-20260714h \
  --judge-base-url "$JUDGE_BASE_URL" --excerpt-evidence \
  --concurrency 1 --local-smoke

for arm in base cheatsheet; do
  "$PY" -m studybench.grade --task smalldspy \
    --run-id "smalldspy-local-$arm-20260714h" \
    --grade-id "qwen-local-$arm-20260714h" \
    --judge-base-url "$JUDGE_BASE_URL" --excerpt-evidence --concurrency 8
  "$PY" -m studybench.report --tasks smalldspy \
    --run-id "smalldspy-local-$arm-20260714h" --grader local \
    --grade-id "qwen-local-$arm-20260714h" \
    --judge-base-url "$JUDGE_BASE_URL" --excerpt-evidence \
    --ci 10000 --ci-seed 45001
done

mapfile -t BASE_REPORTS < <(find \
  reports/smalldspy-local-base-20260714h/qwen-local-base-20260714h/smalldspy \
  -maxdepth 1 -type f -name 'report-*.json' | LC_ALL=C sort)
mapfile -t CHEAT_REPORTS < <(find \
  reports/smalldspy-local-cheatsheet-20260714h/qwen-local-cheatsheet-20260714h/smalldspy \
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

Pending. The `d` generation grids and failed judge smoke are invalidated
diagnostics, and `e` failed its pre-benchmark structured-output gate. Neither
`e` nor the fully audited failed `f` gate is a result. No valid full-grid grade,
WAUC, or arm comparison has yet been observed; only fresh `h` artifacts may
populate this section.
