# 009 — Independent full-tree audit (2026-07-13)

**Auditor.** A separate session from the one that produced the 008 hardening
pass. Every file in the repository outside `corpora/` and the raw `runs/` JSON
was read in full: all experiment logs and docs and every then-present
`studybench/`, `scripts/`, and test file. Historical claims were
re-derived from the raw run/grade artifacts with independent scripts (no
studybench code on the recompute path). The offline test suite, byte-level
compile checks, shell syntax checks, and full dataset validation were executed.
No workflow was launched, no GPU/API/model call was made, and the auditor made
no repository edit other than this report. A final adversarial follow-up then
found four additional hardening gaps; those fixes and their verification are
recorded in §3.

**Audited state.** The working tree was being actively modified by another
session *while this audit ran* (continuous writes from ~03:20 to 03:58 PDT;
the uncommitted hardening diff grew from +9,803 to +10,337 lines during the
audit). All verification below was performed at the quiescent final state:

- HEAD commit `48328be60e34111df718e556d258a044ecc677eb` (clean tree required
  before any research run; the hardening diff is **not yet committed**).
- Files that changed after the first read (react.py, rollout.py, dataset.py,
  serve_vllm.sh, integrity.py, provenance.py, selfquiz.py, tools.py, grade.py,
  and several tests) were re-read at the final state; every mid-audit concern I
  had noted was already closed by those later edits (details in §3).

## 1. Bottom line

**The codebase is in a good state to move on to a new studying method — after
one commit and with three prerequisites understood.** Specifically:

1. The existing experimental record is honest. Every number I could recompute
   from the raw artifacts matches the experiment logs, including all the
   negative results. Nothing is overclaimed; the strongest supported
   conclusions are correctly stated as negative/parity findings with CIs.
2. The 008 audit's defect register is accurate. Every mechanically checkable
   finding (39 stale grades, hybrid3's non-fresh DSPy curriculum, dev-holdout
   leakage counts, token-accounting corrections, overwritten cheatsheets)
   reproduced exactly.
3. The uncommitted hardening code is correct as far as offline verification can
   establish: 213/213 tests pass (165 main env + 48 DSPy env), all Python
   compiles, all shell scripts parse, and both benchmark bundles validate
   byte-exact against the pinned corpus commits. After the four follow-up fixes
   listed in §3, no concrete correctness or integrity blocker remains.
4. The pipeline is now *deliberately* incapable of producing casual benchmark
   numbers. That is the right integrity posture, but it changes the research
   workflow in ways the team must consciously accept before starting the next
   method (§4).

## 2. Verification of the existing experimental record

All recomputation was done directly from `runs/` and `grades/` JSON with
independent scripts; the expertise formula was re-derived from Appendix C and
checked against the paper's worked example (10.8), DSPy base (6.49), and
OpenClaw base (7.66-vs-7.64 rounding) before use.

### 2.1 Population counts and statuses — all match the 008 ledger exactly

Every arm's episode count, per-status breakdown, and grade count matches
experiments/008 §3, including: react 594 ok + 6 no_answer with 600 grades;
react-cheatsheet 1,185 ok + 15 no_answer with 1,200 grades; hybrid2 DSPy
385/720 grades (fugu quota exhaustion) with OpenClaw 480/480; hybrid3
1,182 ok / 9 no_answer / 9 error with **zero** grades. No `ok` episode in any
react-family arm has an empty answer.

### 2.2 Headline numbers — recomputed from raw grades

| Claim (experiments/006–007) | Recomputed | Verdict |
|---|---|---|
| react base WAUC 12.31 (dspy) / 8.45 (openclaw) | 12.31 / 8.45 | exact |
| selfquiz r1 13.63 / 10.17; r2 11.20 / 10.64; r4 11.76 / 9.36 | identical | exact |
| native base gpt-5.4 26.40 / 20.20; fugu 27.99 / 18.16 | identical | exact |
| native cheatsheet fugu 28.45 / 17.91 | 28.49 / 17.96 | ±0.05 (post-log retries) |
| select cells 4.4/19.8/22.3/28.7; usage k20f 34.3 (best ever); hybrid direct 9.2 | identical | exact |
| cheatsheet−base paired (3 rollouts) +2.88 / +2.36 | +2.87 / +2.14 | matches; CI incl. 0 either way |
| hybrid−cheatsheet final 6v6 −0.61 / −0.84 | −0.71 / −0.73 | matches (parity) |
| hybrid2−cheatsheet openclaw −0.08 [−2.63,+2.70] | +0.10 [−3.17,+3.25] | matches (null) |

The sub-±0.25 drifts on three cells trace to the documented retry sweep (job
26575) and the 39 stale grades — the artifacts changed after the log lines were
written, exactly as 008 discloses. No qualitative conclusion moves: every
"n.s." stays n.s., the parity verdicts stay parity, and the pre-registered
success criterion remains failed.

### 2.3 The 008 defect register — spot-verified findings all reproduce

- **39 stale grades**, decomposed 3 (react) + 2 (react-cheatsheet) + 4 (r1) +
  8 (r2) + 7 (r4) + 5 (select) + 10 (hybrid2) — exact match to 008 §5.7.
- **Hybrid3 DSPy is not fresh** (I8): all 86/86 question records in the
  "fresh" `study-selfquiz/dspy` pipeline are normalized-exact repeats of the
  archived `study-selfquiz-run1` curriculum; OpenClaw overlap is 0/92. The
  vLLM-determinism root cause stands.
- **Dev-holdout leakage** (I1/I3): retests of dev-origin items exist in both
  archived and fresh pipelines; retest-derived entry counts are exactly 008's
  7/46, 4/57, 6/45, 5/50 (archived/fresh × dspy/openclaw).
- **Token accounting corrections** (008 §5.1): round-summary sums are exactly
  920,709 / 1,832,447 (archived dspy/openclaw) and 935,111 / 1,747,223 (fresh),
  contradicting experiments/007's table exactly as 008 states.
- **Root cheatsheets were overwritten** (I10): `cheatsheets/*.episode.json` now
  carry gen_tokens 39,467 / 21,044 — the hybrid3 study products, not the
  evaluated originals (64,363 / 25,183, recoverable only from git history).
- Dataset bundles: SHA-256 of `data/dspy.jsonl` and `data/openclaw.jsonl`
  match the constants pinned in `dataset.py`; 30 + 20 records; all 294
  evidence excerpts byte-exact against the pinned corpus commits (verified by
  executing the strict loader). `.env` was never committed and is gitignored.

**Assessment.** The experiment logs are a faithful record. The honest summary
in README/008 — *no tested self-quizzing or static-note variant beat the cheap
forced-50 cheatsheet with a 95% paired CI excluding zero; the pre-registered
criterion failed; the DSPy>OpenClaw ordering was falsified; hybrid ties the
cheatsheet* — is fully supported by the raw artifacts. The historical
provenance defects (mutable notes, no note hashes in episodes, fugu judge,
excerpt evidence, adaptive reuse of 50 public questions) are real, are
documented, and correctly demote everything from that era to
exploratory/diagnostic status. None of them threatens the negative
conclusions; they only preclude positive claims — which none of the logs make.

## 3. Verification of the code (uncommitted hardening pass)

At the final quiescent state, the 12 main-environment test modules report
**165 tests OK**; `.venv-dspy` over test_selfquiz_integrity and
test_react_integrity reports **48 tests OK**. All 16 Python package modules
compile, and `bash -n` is clean for all ten shell/sbatch files;
`load_questions()` fully validates both bundles against the pinned checkouts.

I read every line of the final source. The architecture is a coherent
fail-closed chain: pinned corpora verified blob-by-blob against the exact Git
tree → suffix-allowlisted in-memory tools → immutable run manifests binding
questions, seeds, prompts, note bytes, environment snapshots, and the
preregistration → per-episode provider ledgers with no invented usage →
grading that revalidates everything (including an independent checker rerun
and byte-reparse of the accepted judge verdict) before contacting a judge →
strict reports that recompute themselves before writing content-addressed JSON
→ comparisons that rebuild both arms from disk and match every non-intervention
manifest leaf. The selfquiz rebuild structurally eliminates the historical
leakage classes (dev/retest records cannot distill; the dev exam is cumulative
with immutable blind references; anchors and quotes are validated exactly;
seeds are deterministic; freshness is gated against every stored curriculum;
`claim_ready` is permanently false without a pre-registered blinded human
audit). The statistics are correct: the expertise integral matches Appendix C,
and the paired bootstrap shares question and rollout draws across arms.

Concerns noted during the read that the concurrent session closed before
quiescence (verified in the final state):

- react.py fabricated `prompt_tokens or 0` for missing provider usage → now
  `_dspy_usage_record` fails the episode instead of inventing zeros.
- No producer-side validation before persisting episodes → rollout.py now
  exports `_validate_final_episode` / `_reject_invalid_final_episode`, used by
  both runners, so an invalid `ok` artifact can never become durable.
- serve_vllm.sh placed the server API key in the child environment/topology
  command line → the key now flows via stdin into a bootstrap that sets
  `VLLM_API_KEY` and execs vllm (the 0600 topology file still holds it, which
  is necessary for runner authentication).
- `git update-index --assume-unchanged` could hide corpus drift from the
  porcelain check → `validate_corpus_snapshot` now rejects hidden index flags
  (and every code read still verifies the exact blob hash regardless).
- Main research-source cleanliness could still be hidden with index flags, and
  the test tree was absent from the freeze → `source_record()` now compares
  scoped HEAD, index, stable live bytes, and executable modes; rejects all
  hidden index state; and freezes the recursive test tree, including ignored
  fixtures.
- Extra human-audit protocol clauses looked pre-registered but were not
  machine-enforced → the protocol now has an exact closed five-field schema;
  unsupported assignment/escalation/adjudication promises are rejected at
  registration, promotion, evaluation, and grading.
- Model-cache hashing used ordinary path reads and left a hash/load race → the
  shared attestor now uses stable `openat`/`O_NOFOLLOW` traversal, validates
  Hugging Face leaf links and resolved blobs, and regenerates the complete
  canonical inventory immediately before launch and after authenticated
  readiness. This brackets accidental or persistent drift, while honestly not
  claiming protection from an adversarial same-user write-and-restore between
  checks.

**Remaining issues found (none blocking, none affecting correctness):**

1. **Docs nits.** CLAUDE.md contains a stray token ("minimalist hi"); the
   README repository map lists `studies/`, `reports/`, `comparisons/`,
   `preregistrations/` which do not exist until first use (intended, but worth
   knowing); experiments/007's token table remains uncorrected in place (008
   holds the correction — by design, logs are append-only).

## 4. What the hardened pipeline means for the next studying method

These are deliberate design decisions, verified in code, that change the
day-to-day workflow. They should be consciously accepted (or consciously
revisited) before starting the new method — not discovered mid-run.

1. **There is no benchmark-grading path for exploratory runs.** `grade.py`
   only accepts claim-ready, `purpose: "confirmatory"` manifests bound to a
   committed preregistration, and refuses entirely while the deterministic
   checker is unpinned. Exploratory generation runs are possible
   (`--exploratory`) but can never be scored against StudyBench by the new
   code; `report.py --legacy-partial` only reads *pre-existing* legacy grades
   under a loud diagnostic banner. Method iteration therefore happens on the
   selfquiz-internal dev exams, and the 50 public questions are touched only
   through preregistered two-arm confirmations. This is the strongest possible
   anti-overfitting stance — and the biggest workflow change from the
   iteration-heavy 007 era.
2. **Grading has hard external prerequisites**: a content-pinned Apptainer
   executable + Python SIF (DSPy), a content-pinned real TypeScript compiler
   (OpenClaw), and the selected judge key (`OPENAI_API_KEY` pinned to
   api.openai.com, or `SAKANA_API_KEY`). None exists yet; building/pinning them
   is on the critical path for the first confirmatory grade.
3. **The frozen-source chain couples docs to runs.** The source inventory
   includes `experiments/*.md`, `docs/*.md`, README, CLAUDE.md, AGENTS.md.
   Editing any of them between generation and grading/reporting/comparison
   makes the downstream stage refuse. Operationally: finish the whole
   generate→grade→report→compare chain, then write up.
4. **The preregistration two-commit rule is strict**: the execution HEAD must
   be the single-parent direct child of `source_commit`, adding only
   `preregistrations/*.json`. Any intervening commit invalidates the contract.
5. **Hybrid2's DSPy grades (385/720) can never be completed.** The legacy
   grader is gone and the new one cannot touch manifest-less runs. Treat
   hybrid2 as formally abandoned (008 already recommends this) and do not wait
   on the fugu quota reset for it. Hybrid3 must not be graded as a fresh DSPy
   confirmation regardless — its curriculum is not fresh.
6. **Selfquiz is strict to the point of brittleness (by design).** A chapter
   that yields fewer than exactly M valid, non-near-duplicate questions fails
   the round, and deterministic seeds mean a rerun reproduces the failure —
   the study ID is burned. The freshness gate compares against *every* stored
   curriculum (including both archived pipelines), so repeated fresh studies
   get progressively harder. Budget study IDs and GPU time accordingly, and
   snapshot the human-audit protocol at round 1 or promotion is permanently
   impossible for that study.

## 5. Pre-flight checklist before the new method

1. **Commit the hardening pass** (single reviewed commit). Non-smoke paths
   mechanically require a clean tree; nothing claim-ready can happen before
   this. This report can ride along or follow immediately.
2. Decide the measurement policy for method development (dev-exam-only
   iteration per §4.1, or deliberately add a clearly-labeled diagnostic
   grading mode first — a protocol decision, not a code fix).
3. Build and pin the checker artifacts (SIF + apptainer hashes; tsc) and
   provision the judge key; run `scripts/setup.sh` on a compute node; populate
   the pinned model revision in the HF cache.
4. The runner headers now request six GPUs on the current `matx` partition,
   leaving two devices free on an eight-GPU node and satisfying the L40S TP=2
   divisibility rule. Still smoke-test one episode end-to-end
   (`SB_SMOKE=1 SB_LIMIT=1`) before any real run.
5. For the next method itself: new study ID + seed, genuinely fresh
   curriculum, audit protocol snapshotted before round 1, then the two-commit
   preregistration of one control/treatment pair per docs/preregistration.md.
   Prefer a new hidden question split; any reuse of the 50 public questions
   must carry the adaptive-reuse disclosure that README already mandates.
6. Rotate any credential that was deleted from or previously used through
   `.env`; the audit inspected only its owner/type/mode metadata, not contents.

## 6. Verdict

The repository earns an unusual degree of trust: its own experiment logs
document failures, falsified predictions, and integrity incidents in more
detail than most published papers, and every checkable claim reproduced from
raw artifacts in this independent pass. The historical results are honest
negative/diagnostic results, correctly quarantined from claim-ready status by
provenance defects that the new code prevents from recurring. The hardened
pipeline passes all offline verification at the audited state.

**Go** — commit the baseline, satisfy the three external prerequisites
(checkers, judge key, cluster headers), accept the confirmatory-only
measurement discipline, and the project is well positioned to design and test
the next studying method on clean rails.
