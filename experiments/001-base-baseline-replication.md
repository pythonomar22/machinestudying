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
