# 006 — Faithful replication redo on dspy.ReAct (2026-07-06)

**Goal.** Re-run the base grid on the paper's actual harness, now that the author
answered every blocking question (docs/jacob.md, round 3):
- forced-20 / study-50: *"Just catch the finish and return something like you
  gotta keep searching type of logic, no need to remove that specific turn"*;
- direct budget: *"Dspy predict"*;
- tools: *"Grep glob, and read file (lines, Capped at 200lines)"*;
- (round 2) harness = dspy.ReAct, lenient = pure weighted claim sum, model
  re-thinks every turn.

Implementation: `studybench/react.py` (+ `scripts/react.sbatch`), runs land in
`runs/react/`, graded with fugu (user decision: no more gpt-5.4 spend) into
`grades/react/fugu/`. dspy version = the pinned benchmark checkout
(corpora/dspy @ 9cdb0aac), run in `.venv-dspy` (Python 3.12).

## The two harnesses, precisely

| dimension | native (harness-v1, `rollout.py`) | dspy.ReAct (`react.py`) |
|---|---|---|
| loop | OpenAI chat loop: `tools=` param, vLLM qwen3_coder parser, tool-role messages | text-field loop: each step one `dspy.Predict` call emitting `next_thought` / `next_tool_name` / `next_tool_args`, parsed by ChatAdapter |
| state | one growing message history per episode | **stateless per step** — the trajectory dict is re-serialized into every prompt |
| thinking | interleaved: prior turns' `<think>` passed back → model thinks ~50-150 tok/turn after turn 0 | **fresh `<think>` every step** (nothing carries) → ~0.5-1.5k tok/step |
| voluntary stop | model answers without a tool call | dedicated **`finish` tool** in the action space |
| forced mode | `tool_choice="required"` (xgrammar) + cap-notice + nudges | catch `finish` → observation "you gotta keep searching", turn stays, loop runs all N |
| direct | chat request, no tools, our system prompt | bare `dspy.Predict("question -> answer")` — no system prompt of ours |
| answer channel | last assistant message | separate **ChainOfThought extract** call over the whole trajectory |
| system prompt | ours (names repo + roots) | dspy's auto-generated ReAct instructions (tool list only, **no repo-layout disclosure**) |
| read_file cap | 500 lines (invented) | **200 lines (author-confirmed)** |
| parse failures | parser-level; nudge machinery | ChatAdapter → JSONAdapter fallback retries (dspy-native) |
| iteration count | one assistant turn = one iteration; parallel calls disabled | one react step = one iteration; finish-catches occupy slots (logged as `catches`) |
| context overflow | vLLM clamps max_tokens server-side | dspy trajectory truncation |
| invented machinery | cap-notice, nudges, tool-shaped-answer rejection, forced_short, seeds | none — dspy defaults throughout |

**Measured behavioral gap (same model, same server, same questions):**

| signature | native | react | paper |
|---|---|---|---|
| DSPy k20 voluntary iters (mean) | 16.9 | **3.4** | early stop implied (k20 tok ≈ k5) |
| DSPy k20 tokens | 5.9k | **7.0k** | 8.6k |
| DSPy k5 tokens | 4.2k | **6.0k** | 7.9k |
| OpenClaw k20 tokens | 5.1k | **9.5k** | 9.7k |
| DSPy k20f tokens | 5.9k | **12-24k** (heavy tails, e.g. 8 finish-catches → 23k) | 34.6k |
| per-LM-call gen tokens (k20f) | ~250 | **~550-1100** | ~1.6k implied |

The native agent is the stronger, cheaper searcher (never presses finish, carries
its reasoning); the react agent re-derives its plan every step and stops early —
which is exactly the paper's regime. Jacob's own read, verbatim: *"it's possible
that native tool calling helps with the performance (but I wasn't expecting 2x
improvement)"* and *"it wouldn't matter much after all, the reason for including
this lenient grading was to demonstrate a prettier inf scaling curve. And a
qualifying MS algorithm will need to shift that frontier."*

## Implementation notes & bugs caught

1. **Sampling passthrough verified at the wire**: pointed `dspy.LM` at a local
   capture server; the payload carried temperature=1.0, top_p=0.95, top_k=20,
   min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, max_tokens=32768 —
   all of §B, byte-exact. (litellm forwards extra_body verbatim.)
2. **`glob.translate` is Python 3.13-only** — every glob call in the 3.12 react
   venv raised AttributeError (caught by reading a smoke transcript: the model
   said "the glob command failed"). Fixed with `_glob_to_regex`, equivalence-
   tested against the 3.13 reference over both corpora × 12 pattern shapes
   (0 mismatches); tainted smoke episodes wiped before the full run.
3. dspy LM caching **disabled** (`cache=False`) — otherwise 3 rollouts would be
   byte-identical.
4. Token accounting = sum of `usage.completion_tokens` over every LM call in
   `lm.history` (react steps + extract + adapter retries), thinking included.
5. Episode JSON schema kept identical to native so grade.py/report.py work
   unchanged (`--variant react`); extra fields: `finish_catches`, `n_lm_calls`.

## Run log

- Smoke (job 25128, 12 episodes): all ok; signatures confirmed (fresh thinking,
  finish-catches firing, Predict direct = 1 call).
- Full grid (25156/25157, 8×H100): **killed at 08:26 by cluster-wide a3
  maintenance drain** (all 7 nodes; the documented 10-11-day reboot cycle).
  510/600 episodes were already on disk; resubmitted jobs (25196/25197) resumed
  automatically post-maintenance and re-ran only the remainder. An L40S runbook
  was prepared as a fallback (scripts/setup.sh; SB_TP for 48GB cards) and stood
  down when GCP returned. Provenance: all react episodes are H100.
- OpenClaw complete (240/240) → fugu grading fired; DSPy finishing its k20f tail.

## Remaining known divergences from the paper (accepted, small)

- dspy version: theirs unknown; ours = the pinned benchmark commit.
- grep/glob output caps and the 25k-char observation cap: ours (only the
  200-line read cap is author-confirmed).
- Judge: fugu instead of gpt-5.4 (user decision; measured calibration on
  identical answers: ±7% WAUC).
- Judge effort/wording details, hallucinated-API deterministic check (strict
  only), study instruction, prepend placement — still open from the
  experiments/003 register (P1s), none blocking the lenient comparison.

## Insights carried forward from the native-harness phase (kept for the studying program)

1. **Harness strength interacts with studying gains.** On the strong native agent
   the cheatsheet's paired WAUC effect vanished (+0.84 n.s., experiments/002):
   direct/k5 gains were offset by k20f giveback, because the strong agent's search
   already recovers what the note knows. Studying methods get evaluated on the
   faithful (react) harness where headroom exists at every budget — and "does
   studying still help stronger agents?" is itself a future research question. The
   raw data for that arm is retained: runs/base + runs/cheatsheet with
   grades/{base,cheatsheet}/.
2. **Interleaved thinking is the single biggest agent-strength lever we found**
   (persistence + per-turn depth, ~0 score change from context alone) — a candidate
   *studying target* later (procedures that improve per-step thinking rather than
   notes).
3. **Judge economics**: fugu ≈ gpt-5.4 within ±7% WAUC under our 0/1 + strict-schema
   pipeline; judge evidence/effort are second-order. Offline grading means judge
   sweeps are free-ish — never let a judge question block a rollout.
4. **Power**: paired-by-question deltas + the two-stage bootstrap are mandatory;
   at 3 rollouts the native paired CI half-width was ±5.6 WAUC. Recompute the
   noise floor on the react harness before pre-registering the self-quizzing
   success criterion (experiments/005 §4).
5. **Design assets that survive the harness switch**: the self-quizzing design
   v1.1 (experiments/005) — two-phase independent VERIFY, mechanical quote-check
   gate, dev-exam audit protocol, RETEST slots — now to be instantiated on the
   react harness (ATTEMPT = Predict closed-book, VERIFY = react episodes).
6. Cleanup record (this date): deleted cheatsheets/* (native study products;
   in git history), runs/no-think-history + grades (ablation, conclusion recorded),
   grades/base judge-ablation dirs (regenerable), studybench/study.py +
   scripts/study.sbatch (native study loop; react.py --study replaces them).

## Results

(to be filled when fugu grades land — comparison targets: paper Table 1 base,
DSPy 3.3/8.6/9.6/29.4 lenient @ 4.1/7.9/8.6/34.6k tok, WAUC 6.49;
OpenClaw 2.3/6.9/15.8/17.6 @ 4.1/4.6/9.7/24.3k, WAUC 7.64.)
