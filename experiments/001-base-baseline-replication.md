# 001 — Replicating the Machine Studying base baselines (Study-DSPy, Study-OpenClaw)

**Goal.** Reproduce Table 1's `Qwen3.5-9B (base)` rows: lenient accuracy and mean
generated tokens at four inference budgets, and the expertise WAUC (paper: DSPy 6.49,
OpenClaw 7.64), using the exact grading procedure the first author described to us
(docs/jacob.md).

## Setup (what is pinned by the paper/dataset/author)

- **Questions**: `jacobli/studybench` HF dataset (30 DSPy + 20 OpenClaw), vendored in
  `data/*.jsonl`. Rubric weights sum to 100; claims are core/supporting.
- **Corpora**: `corpora/dspy` @ `9cdb0aac...`, `corpora/openclaw` @ `da228660...`
  (Table 2). Test-time code roots inferred from the dataset's evidence paths and the
  "no docs at answer time" rule: DSPy → `dspy/`, `tests/`; OpenClaw → `src/`,
  `extensions/`.
- **Model**: `Qwen/Qwen3.5-9B`, thinking mode, sampling per §B: temp 1.0, top-p 0.95,
  top-k 20, min-p 0, presence 1.5, repetition 1.0, 32768 max tokens per ReAct turn.
  Served with vLLM (`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`).
- **Budgets**: direct (no tools), ≤5 voluntary iters, ≤20 voluntary iters, forced 20
  ("no early exit"). 3 rollouts per budget, scores averaged (paper §5).
- **Grading** (per the author, docs/jacob.md — followed to the letter):
  1. sandbox: run the generated code, syntax/compile gate;
  2. GPT-5.4 judge with the paper's A.5 prompt, **claims scored 0/1 only** (his
     correction: the 0.5 level was removed);
  - **strict** = automatic zero if the compile gate fails or any core claim ≠ 1;
  - **lenient** (what Table 1 reports) = weighted claim sum, no automatic zeros.
- **Expertise**: Appendix C WAUC. Our implementation reproduces the paper's published
  values from its own table (6.49 / ~7.66-vs-7.64 rounding) and the worked example
  (10.8), so the formula is verified.

## Replication inferences (not specified in the paper; our educated guesses)

1. **ReAct implementation**: native tool-calling loop (vLLM qwen3_coder parser) with
   a bare-bones system prompt naming the repo and tools. One iteration = one
   assistant turn that makes tool calls. Forced mode uses `tool_choice="required"`
   for 20 iterations, then a final no-tools answer turn; voluntary mode answers when
   the model responds without tool calls (cap reached → final no-tools turn).
2. **Tool semantics/caps** (unspecified): grep = case-sensitive regex, ≤50 matching
   lines; glob ≤200 paths; read_file ≤500 lines per call (pageable); observations
   ≤25k chars. Sized to keep forced-20 episodes inside the 262k context.
3. **Sandbox**: DSPy answers (Python, self-contained DummyLM programs by
   construction) are executed against a venv with the pinned DSPy checkout
   installed; pass = exit 0. OpenClaw answers (TS that imports repo internals —
   cannot run standalone) get a tree-sitter syntax parse (node on this cluster is
   v12, too old for modern tsc). Compile gate takes the *largest* fenced code block
   of the task's language; no code block = gate fails.
4. **Judge**: `gpt-5.4` (author confirmed the model), default reasoning effort,
   structured output; we recompute the weighted sum from claim scores ourselves
   rather than trusting the judge's arithmetic. Evidence given to the judge =
   dataset's span excerpts (the public release ships excerpts, not whole files; its
   README states excerpts are the only code context the judge needs).
5. **Token axis** = sum of `completion_tokens` over every request in an episode
   (thinking tokens included), averaged over all episodes of a budget.
6. **Seeds**: per-episode deterministic seed (crc32 of task/qid/budget/rollout)
   passed to vLLM.

## Harness findings from smoke tests (2026-07-06)

Smoke rounds on 1 GPU (2 questions × direct/k5/k20f × both tasks) surfaced three
behavioral issues, each fixed and re-smoked:

1. **vLLM 0.24 returns thinking under `reasoning`**, not `reasoning_content` — we
   recorded nothing until fixed.
2. **The model cannot tell when its tool budget is gone.** At the cap the request
   drops the `tools` param, but the model keeps writing tool calls — first as parsed
   calls, then as plain `<tool_call>` XML that became the "answer" (51-token
   answers). Fixes: an explicit user notice when the budget exhausts, plus
   tool-call-shaped text is never accepted as an answer without nudges.
3. **Interleaved thinking is the model's intended agentic mode.** The Qwen3.5 chat
   template renders prior turns' `<think>` blocks inside the current tool loop and
   renders *empty* think blocks when reasoning isn't passed back — visibly
   conditioning the model to stop thinking (~60-100 tokens/turn in tool loops vs
   ~2k thinking in direct mode). vLLM 0.24's chat parser explicitly supports
   assistant `reasoning` passback "for interleaved thinking". Round 2 tokens without
   passback: k20f ≈ 2.5-5.2k vs the paper's 34.6k mean — passback (round 3) is the
   mechanism that plausibly closes this gap.

## Smoke grading (2026-07-06, 12 episodes, n=2 questions — NOT representative)

End-to-end pipeline verified: sandbox + GPT-5.4 judge + report. Qualitative patterns
match the paper: direct answers hallucinate APIs (`GeoTeleprompter`,
`dsp.utils.dotdict`) → compile-fail → lenient 0; lenient rises with budget (DSPy
0 → 38.5 → 55.5; OpenClaw 0 → 7 → 26); strict ≈ 0 everywhere (paper: "most answers
would have otherwise received zero"); one direct answer wrote Python for a
TypeScript question (correctly gated). Judge rationales are code-grounded and cite
rubric mechanisms. n=2 lenient levels are far above the paper's 30-question means at
k5/k20f — expected variance at this sample size; the full run decides.

## RESULTS — primary run (2026-07-06; 600 episodes, 8×H100, ~24 min; grading GPT-5.4)

All 600 episodes status=ok, zero invariant violations (forced=20 exact, caps
respected, no tool-call-shaped answers, no empty answers). Grading clean: zero
judge claim-id mismatches, zero needs_regrade, zero errors.

| | direct | k5 | k20 | k20f | WAUC |
|---|---|---|---|---|---|
| **DSPy ours (lenient)** | 4.9 | 21.3 | 35.9 | 40.4 | 26.40 |
| **DSPy paper** | 3.3 | 8.6 | 9.6 | 29.4 | 6.49 |
| **DSPy ours tok(k)** | 2.9 | 4.2 | 5.9 | 5.9 | |
| **DSPy paper tok(k)** | 4.1 | 7.9 | 8.6 | 34.6 | |
| **OpenClaw ours (lenient)** | 5.8 | 12.1 | 27.3 | 28.6 | 20.20 |
| **OpenClaw paper** | 2.3 | 6.9 | 15.8 | 17.6 | 7.64 |
| **OpenClaw ours tok(k)** | 2.8 | 4.2 | 5.1 | 5.0 | |
| **OpenClaw paper tok(k)** | 4.1 | 4.6 | 9.7 | 24.3 | |

**What replicates (direction/shape):** monotone lenient accuracy in budget; tools
help a lot; DSPy k20f is the best cell; OpenClaw k20≈k20f plateau (theirs
15.8→17.6, ours 27.3→28.6); strict ≈ 0 at low budgets (paper: "most answers would
otherwise receive zero"); direct-mode answers hallucinate stale APIs and
compile-fail (DSPy direct compile rate 5.6%).

**What does not (levels):** lenient ~1.5-3.7x higher across cells; WAUC 3-4x
(compounded by our lower token means: cheaper budgets carry more WAUC weight).
Biggest structural difference: paper's voluntary k20 barely beats k5 (9.6 vs 8.6,
tokens 8.6k≈7.9k → their agent stops early), while ours keeps searching (mean 16.9
iters, 35.9 lenient). Our harness produces a genuinely stronger agent — exactly the
effect the first author warned about ("you could improve 'performance' by using
more complicated harness", docs/jacob.md).

**Judge-severity checks:**
- Rationale spot-reads: judge is strict and mechanism-specific (denies a 50-weight
  core claim for using `retrieve_module(query,k=..)` instead of `dspy.Retrieve`
  backed by the configured rm).
- Lenient definition readings (OpenClaw): rubric-only 5.8/12.1/27.3/28.6 (~2x
  high); +core-conjunctive 0/1.6/9.7/9.6 (~2x low); +compile-gate 1.0/10.0/22.1/21.2.
  The paper's numbers sit between the rubric-only and core-conjunctive readings.
- Whole-evidence-files judge (A.5-faithful) vs dataset excerpts: OpenClaw direct
  5.8→4.7, k20 27.3→27.0 — judge evidence context is NOT the inflation source.

**Full sweep results (all 600 episodes per variant; lenient means, tokens in k):**

| DSPy | direct | k5 | k20 | k20f | WAUC |
|---|---|---|---|---|---|
| paper Table 1 | 3.3 (4.1) | 8.6 (7.9) | 9.6 (8.6) | 29.4 (34.6) | 6.49 |
| ours, rubric-sum | 4.9 (2.9) | 21.3 (4.2) | 35.9 (5.9) | 40.4 (5.9) | 26.40 |
| ours, **core-conjunctive** | 0.0 (2.9) | 2.9 (4.2) | 10.3 (5.9) | 17.8 (5.9) | **9.68** |
| judge whole-files (rubric-sum) | 5.1 | 23.5 | 34.4 | 40.6 | 27.04 |
| harness no-think-history (rubric-sum) | 4.9 (2.8) | 23.2 (3.7) | 34.2 (6.2) | 37.1 (6.7) | 26.37 |

| OpenClaw | direct | k5 | k20 | k20f | WAUC |
|---|---|---|---|---|---|
| paper Table 1 | 2.3 (4.1) | 6.9 (4.6) | 15.8 (9.7) | 17.6 (24.3) | 7.64 |
| ours, rubric-sum | 5.8 (2.8) | 12.1 (4.2) | 27.3 (5.1) | 28.6 (5.0) | 20.20 |
| ours, **core-conjunctive** | 0.0 (2.8) | 1.6 (4.2) | 9.7 (5.1) | 9.6 (5.0) | **5.98** |
| judge whole-files (rubric-sum) | 4.7 | 12.9 | 27.0 | 28.7 | 20.05 |
| judge xhigh (rubric-sum) | 4.0 | 10.2 | 26.1 | 24.2 | 17.95 |
| harness no-think-history (rubric-sum) | 6.6 (2.7) | 11.5 (3.0) | 28.5 (4.8) | 29.1 (5.2) | 22.53 |

**Interpretation.**
1. **The lenient definition dominates everything else.** The raw weighted rubric sum
   runs 2-4x above the paper; applying the author's core-conjunctive rule (any core
   claim ≠ 1 → question scores 0; docs/jacob.md, stated precisely while explaining a
   3x-inflated baseline replication) lands both tasks in the paper's range: WAUC
   9.68 vs 6.49 (DSPy), 5.98 vs 7.64 (OpenClaw) — over-shooting one task,
   under-shooting the other, i.e. no systematic bias. `report.py` now reports
   core-conjunctive lenient as the Table 1 comparison, with the raw rubric sum
   alongside. The two readings bracket the paper everywhere.
2. **Judge-side knobs are second-order.** Whole evidence files vs excerpts: no
   effect (±1pt). Judge reasoning effort xhigh: −10-30% relative. Claim-id schema
   enforcement: zero mismatches in 1800+ judge calls.
3. **Harness thinking passback is also second-order for scores** (rubric-sum
   28.5 vs 27.3 at OpenClaw k20) though it changes voluntary persistence
   (DSPy k20 16.9 → 12.9 iters) and token counts.
4. **Residual structural gap: token axis.** The paper's k20f means (34.6k / 24.3k)
   imply ~1.2-1.6k generated tokens per forced turn sustained for 20 turns; every
   harness variant we ran produces 5-7k per forced episode (the model thinks briefly
   once oriented). Their voluntary-k20 token counts (≈ k5's) also show their agent
   stopping early where ours keeps searching. Both point to a differently-shaped
   ReAct loop (plausibly text-based rather than native tool calling, and no
   interleaved thinking), which the paper does not specify.

**Error bars (2026-07-06; two-stage cluster bootstrap, 10k replicates, questions
then rollouts, `report.py --ci`):**

| WAUC (lenient) | point | 95% CI | paper | inside CI? |
|---|---|---|---|---|
| DSPy, core-conjunctive | 9.68 | [4.4, 16.5] | 6.49 | **yes** |
| DSPy, rubric-sum | 26.40 | [19.9, 34.5] | 6.49 | no |
| OpenClaw, core-conjunctive | 5.98 | [0.9, 15.5] | 7.64 | **yes** |
| OpenClaw, rubric-sum | 20.20 | [14.4, 30.0] | 7.64 | no |

So under the core-conjunctive reading we are **statistically consistent with the
paper on both tasks**, and the rubric-sum reading is excluded on both — the
strongest evidence yet that Table 1's lenient applies the core-claim gate
(pending Jacob's confirmation, DM sent 2026-07-06).

Why the CIs are wide: the signal is carried by a handful of questions — under
core-conjunctive grading only 3/7/10 DSPy questions (k5/k20/k20f) and 1/3/3
OpenClaw questions ever score above zero across all rollouts; direct is 0/30 and
0/20. Design implications for the study-procedure experiments:
1. **Compare paired by question** (per-question score differences vs the base),
   which removes the dominant between-question variance — marginal CIs (±6-7 WAUC)
   would swamp cheatsheet-sized effects (paper: +3.2 DSPy, +0.5 OpenClaw).
2. More rollouts are cheap (~8 GPU-min per extra rollout wave per task) if paired
   CIs are still too wide.
3. OpenClaw effects of the paper's size are likely undetectable at this n even
   paired; DSPy is the primary battleground.

**Open questions for the first author** (to confirm before we treat the baseline as
fully pinned): (a) does Table 1 "lenient" apply the core-conjunctive zero (skipping
only the deterministic compile/hallucinated-API zeros), or is it the pure weighted
sum? (b) was the ReAct loop native tool-calling or text-format, and was prior-turn
thinking kept in context? (c) judge reasoning effort.

## AUTHOR CORRECTIONS (2026-07-06, docs/jacob.md lines 116-140) — supersede the above

Jacob answered (a) and (b):
1. **"lenient is just weights summed together."** Table 1 lenient = the pure
   weighted claim sum. Our core-conjunctive reconciliation above is the WRONG
   reading — the correct comparison is our rubric-sum row, which is **3-4x above
   the paper and excluded by our own CIs** (26.4 vs 6.49 DSPy; 20.2 vs 7.64
   OpenClaw). The core-conjunctive rule from the earlier DMs evidently belongs to
   *strict* grading only. The real discrepancy is therefore open again, and since
   the judge knobs we swept move scores ≤30%, the harness must carry it → see (2).
2. **"I was using dspy.ReAct ... (and yes model emits reasoning every turn)."**
   The paper's harness is DSPy's own text-field ReAct (signature fields
   next_thought/next_tool_name/next_tool_args, a `finish` tool, trajectory dict
   re-fed each turn, final extract step) — not a native tool-calling loop. This
   coherently explains BOTH anomalies we flagged: fresh ~1.5k-token thinking every
   turn with no reasoning carry-over (their k20f 34.6k vs our 6k) and voluntary
   early stopping via the `finish` tool (their k20 ≈ k5 tokens).

Consequence: our harness is materially stronger than the paper's (Jacob's own
warning — "you could improve 'performance' by using more complicated harness").
A faithful replication of Table 1's absolute numbers requires a dspy.ReAct-based
harness (the pinned library is already in corpora/dspy). Faithfulness audit +
consolidated Jacob questions: see the three-agent audit in this session and
experiments/003 (planned).

## Status log

- 2026-07-04: dataset vendored; corpora cloned at pinned commits; model downloaded to
  HF cache; expertise formula verified against paper numbers; tools benchmarked
  (in-memory grep ~2-90ms after 23s preload of the 11.3k-file OpenClaw tree).
- 2026-07-05 (pre-flight): all 294 dataset evidence spans verified byte-exact against
  the pinned checkouts; all rubric weights sum to 100; all evidence paths fall under
  the configured code roots. Multi-agent adversarial review of the pipeline found and
  we fixed:
  - `report.py` crash: expertise loop mutated the dict it iterated (blocker);
  - population mismatch: report now takes tokens/status from the grade files
    themselves (grades embed `gen_tokens`), and `grade.py` re-grades any episode whose
    run file is newer than its grade (stale grades after a retry);
  - expertise: `next_w` now capped at 1.0 (two sub-3k budgets used to yield a
    negative segment weight); docstring corrected — OpenClaw base from the paper's own
    rounded Table 1 inputs gives 7.66, not the published 7.64 (token rounding).
  - rollout: forced-mode guard (an early answer with `tool_choice=required` not
    honored is nudged, then marked `forced_short`, not `ok`); resume now also skips
    `no_answer` (a genuine model outcome), retrying only infra failures; both async
    loops now isolate per-episode exceptions instead of cancelling the whole pass.
  - sandbox: env scrubbed (no API keys reach generated code; HOME/TMPDIR = temp dir);
    fence regex tolerates info strings (```python3 ...); tsx blocks parsed with the
    tsx grammar; judge structured-output schema is now per-question (claim_id enum,
    exact claim count).
  - cluster scripts: job-derived ports (shared nodes; fixed 8100 collides), SB_NGPU
    passed explicitly, `.venv-vllm` sentinel checks `bin/vllm` (not the dir), vllm
    pinned to 0.24.0, crash detection in the health wait, `set -euo pipefail` +
    process-group cleanup, `logs/slurm/.gitkeep` tracked (sbatch --output needs the
    dir to exist).
- 2026-07-05 (vLLM/parser verification, static): Qwen3.5-9B's chat template emits the
  Qwen3-Coder XML tool format → `--tool-call-parser qwen3_coder` is correct;
  `--reasoning-parser qwen3` and `--language-model-only` exist in vllm 0.24.0.
  `tool_choice="required"` is implemented via an xgrammar structural tag for
  `qwen_3_coder` (forces ≥1 native-format tool call; parsing falls back to the auto
  path). Whether thinking survives the grammar in forced mode → smoke test.
- 2026-07-05: smoke test job 22923 (1 GPU): dspy+openclaw × {direct,k5,k20f} × 1
  rollout × 2 questions = 12 episodes.
- 2026-07-05 (review round 2, all 28 agents): fixed the confirmed findings:
  - **tools.py grep blocker**: `re` → `regex` with a hard match timeout — a
    catastrophic-backtracking pattern used to hold the GIL and freeze every
    concurrent episode; now it returns partial matches at the deadline.
  - rollout: `parallel_tool_calls=False` (one action per ReAct iteration — the
    faithful reading of the paper's "tool iterations"; also bounds context growth);
    rollout exits non-zero when any episode ends in error/forced_short so the batch
    job fails visibly; read_file now errors on out-of-range instead of returning an
    empty success.
  - sandbox: the compile gate prefers explicitly-tagged code blocks (an untagged
    output/config blob can no longer displace the program).
  - grade: judge retried once if it returns duplicate/missing claim ids.
  - sbatch/serve: job-unique vLLM log names (a stale Traceback from a previous run
    used to falsely abort startup), port stride 8 per job id (concurrent jobs with
    adjacent ids no longer overlap ports), unconditional `uv sync`.
  - Verified refutations worth recording: vLLM 0.24.0 clamps `max_tokens`
    server-side to the remaining context (no context-overflow failures); the
    forced-mode xgrammar structural tag constrains only post-`</think>` tokens, so
    thinking survives `tool_choice="required"`; slurm ≥23.02 creates missing
    --output dirs.
  - NOTE: smoke episodes were generated with pre-round-2 rollout semantics —
    runs/ and grades/ will be wiped before the full run.
- 2026-07-05 (sandbox validated against golds): all 50/50 gold answers pass the
  compile gate (paper A.1 stage 5: references must pass the checker). Required
  installing `optuna` into `.venv-dspy` (two golds run MIPROv2); done in
  `setup_grading.sh`. This also exercised the env-scrubbed sandbox
  (no API keys, HOME=tempdir) — no gold needs network or user env.
- 2026-07-05 (vLLM startup on H100): Qwen3.5-9B GDN linear-attention prefill and the
  flashinfer top-k/top-p sampler are JIT-compiled at server start → need `ninja` and
  `nvcc` on PATH inside the batch job. Fixed by pip-installing ninja into .venv-vllm
  and prepending `.venv-vllm/bin` + `/usr/local/cuda-12.8/bin` to PATH in
  serve_vllm.sh (smoke jobs 22923/22949 died on this; 22961 has the fix). Crash
  detector now keys on "EngineCore failed to start" — GDN warmup prints non-fatal
  WARNING tracebacks that must not abort the job.
