# Historical self-quizzing: engineer handoff

This document describes the self-quizzing method that was actually run on
DSPy, rather than the cleaner procedure we originally intended. It is detailed
enough to reimplement the historical method, while calling out defects that a
new implementation must not silently reproduce.

## Executive summary

The historical method was not weight training and did not create durable
memory inside the model. It was:

> Same-model, error-conditioned construction of an external Markdown memory,
> followed by prepending that memory to later StudyBench questions.

The most important implementation fact is that the historical question
answerer had **no corpus access during ATTEMPT**. For each quiz question it made
exactly one tool-less call:

```python
dspy.Predict("note, question -> answer")(
    note=note or "(no notes yet)",
    question=item["question"],
)
```

It received the previous note and the current question. It did not receive
`grep`, `glob`, `read_file`, `run_python`, the writer's proposed answer, the
question anchors, the reference answer, or any hidden corpus text. The corpus
was opened only afterward, when a separate same-model ReAct episode constructed
a reference answer.

The defensible result is that this experiment **failed to demonstrate
superiority** over the forced-50 cheatsheet. It does not establish equivalence,
prove that self-quizzing is harmful, or show that self-quizzing in general
cannot work.

## What state could change

The model was frozen. There was no fine-tuning, optimizer, gradient update, KV
cache retained across evaluation, or other persistent parameter change. A
discarded quiz attempt had no lasting effect.

The only learned state was a cumulative Markdown note:

```text
question
  -> answer from prior note/model knowledge
  -> blind open-book reference
  -> same-model verdict
  -> correction for wrong/partial answers
  -> weak source-quote gate
  -> append correction to Markdown note
```

The attempt mattered only through the bytes selected for and written into that
external note. A precise method name is therefore **error-conditioned external
memory construction**, not durable retrieval learning.

## Model and sampling

All study phases used the same model family and sampling configuration:

- model: `Qwen/Qwen3.5-9B`, addressed through DSPy as
  `openai/Qwen/Qwen3.5-9B`;
- adapter: `dspy.ChatAdapter`;
- temperature: `1.0`;
- top-p: `0.95`;
- top-k: `20`;
- min-p: `0`;
- presence penalty: `1.5`;
- repetition penalty: `1.0`;
- maximum output per call: 32,768 tokens;
- DSPy cache disabled;
- provider retries: three;
- no explicit request/study seed; and
- no artifact-bound model revision or fingerprint.

Study jobs launched one vLLM server per visible GPU. The exact GPU identity and
model snapshot were not retained in the historical study artifacts.

## Corpus and tools

For DSPy, repository tools exposed the `dspy/` and `tests/` roots at pinned
commit `9cdb0aac28b2a04b064e40697ccd301872cf6a43`. They did not expose
StudyBench data, grades, runs, or experiment notes.

The old repository reader included every file up to 5,000,000 bytes whose first
8 KiB contained no NUL byte, not only Python files. It decoded text as UTF-8
with replacement for invalid bytes. Repository-backed ReAct phases used:

- `grep`: at most 50 matches and 240 characters per returned line;
- `glob`: at most 200 paths;
- `read_file`: at most 200 lines per call; and
- a 25,000-character observation cap.

Question generation had `grep`, `glob`, and `read_file`. Blind DSPy reference
generation additionally had `run_python`. ATTEMPT had none of these tools.

## The historical procedure

### 1. COVER: choose chapters

The scheduler summed lines in every loaded text-like file under a non-test
top-level root, including non-Python/configuration/documentation files, grouped
most files by their first two path components, excluded test/spec roots from
becoming chapters, sorted chapters by descending line count, and selected four
consecutive chapters per round. It wrapped around after the final chapter.

The 15 DSPy chapters were:

1. `dspy/teleprompt`
2. `dspy/adapters`
3. `dspy/clients`
4. `dspy/predict`
5. `dspy/primitives`
6. `dspy/utils`
7. `dspy/dsp`
8. `dspy/datasets`
9. `dspy/signatures`
10. `dspy/retrievers`
11. `dspy/streaming`
12. `dspy/evaluate`
13. `dspy/propose`
14. `dspy`
15. `dspy/experimental`

R1 therefore covered chapters 1–4, R2 covered 5–8, R3 covered 9–12, and R4
scheduled 13–15 plus a wrap to teleprompt. R4 retained questions from only the
three new chapters. No teleprompt batch or diagnostic was retained, so whether
that batch failed, was empty, or was entirely deduplicated is unknowable.

This was a breadth heuristic, not semantic coverage. A small and a large
package each received five nominal questions.

### 2. QUIZ: generate questions

For each selected chapter, one ordinary `dspy.ReAct` episode ran for at most 15
iterations. It was asked to explore the chapter and return five records:

```json
{
  "question": "...",
  "qtype": "usage | behavior | location | pitfall",
  "anchors": ["path", "..."],
  "writer_sketch": "one-line believed answer"
}
```

Important actual behavior:

- tools were repository-wide, not technically confined to the named chapter;
- the model received the chapter name, not an explicit frozen file list;
- `writer_sketch` was stored but never used downstream;
- anchors were not validated;
- exactly five valid outputs were not mechanically enforced;
- question-generation trajectories were discarded;
- near-duplicates were removed when token-set Jaccard similarity was greater
  than `0.5`; and
- concurrent chapter workers mutated one shared deduplication list without a
  manifest-bound seed.

The archived DSPy pipeline generated 75 new questions: 25 behavior, 23 usage,
16 pitfall, and 11 location. Those labels were a menu, not enforced strata.

### 3. Split train and dev

The last retained question in each chapter batch was marked dev when the batch
had more than one item. This was not a random 20% split.

Dev questions still went through the same with-note ATTEMPT and reference
pipeline. They were excluded only from immediate distillation. There was no
accumulating independent dev exam, and dev lineage was later lost when some
questions were selected as retests. This is a material train/dev leakage
defect.

### 4. ATTEMPT: answer without the corpus

Every item created one fresh LM object, reused it across that item's ATTEMPT,
reference derivation, adjudication, and optional distillation, and made exactly
one ATTEMPT call:

```python
dspy.Predict("note, question -> answer")
```

Inputs were the question and `note_(r-1)`, or the literal
`"(no notes yet)"` in R1. Every DSPy call was otherwise stateless—no dialogue
history was passed. Every question in one round saw the same prior-round note;
corrections were not incorporated online within the round.

Consequently:

- R1 attempts sampled the model's prior knowledge;
- R2 and later attempts sampled the model-plus-existing-note system; and
- ATTEMPT was **corpus-closed but note-open**.

This differs from the new proposed corpus-backed ReAct question answerer.

### 5. VERIFY A: construct a blind open-book reference

After ATTEMPT, a separate ordinary ReAct episode using the same per-item LM
object received the question but not the attempt, note, writer sketch, or
anchors. It could inspect the full corpus for at most 15 iterations and
returned:

```json
{
  "answer": "...",
  "evidence": [
    {"file": "...", "line": 123, "quote": "..."}
  ]
}
```

DSPy used one reference derivation per item, including dev items. The generic
selector chose `max(derivations, key=len(evidence))`, which was immaterial when
DSPy's ensemble size was one. `run_python` was available but optional; it used
the pinned interpreter with `python -I`, an isolated temporary working/home/tmp
directory, a 120-second timeout, and returned only the last 3,000 output
characters. Probe programs/output and ReAct trajectories were not persisted.
Evidence paths, lines, and quotes were not validated before selection. It is
therefore accurate to call this a same-model open-book reference with optional
execution access, not a reliably execution-grounded reference.

### 6. VERIFY B: adjudicate the attempt

A tool-less same-model `dspy.Predict` received the question, reference answer,
reference evidence, and original attempt. It emitted one verdict—`correct`,
`partial`, `wrong`, or `unresolved`—plus a free-text delta.

This was not an independent judge: the question writer, answerer, reference
writer, adjudicator, and distiller all used the same model family.

### 7. DISTILL: write a correction candidate

Only wrong or partial non-dev items were eligible. Another tool-less same-model
call received the question, attempt, reference answer, and reference evidence,
then produced:

```json
{
  "belief": "...",
  "correction": "...",
  "quote": "...",
  "file": "...",
  "line": 123
}
```

The distiller did not receive the adjudicator's verdict or delta. Correct and
unresolved items produced no entry.

### 8. Apply the source-quote gate

The gate took the first nonempty line of the proposed quote, removed simple
Markdown quoting/backticks, required at least six characters, and accepted the
entry if that substring appeared within plus or minus two source lines of the
claimed line.

This checked that a nearby substring existed. It did not establish that the
quote supported the answer or that the correction was true. False corrections
were admitted. For example, one entry said all ReAct callables must first be
wrapped in `dspy.Tool`, although the implementation automatically wraps
callables; a later retest added the opposite correction. Another entry claimed
a non-null `rollout_id` was removed at temperature zero, while the code removes
the key only when its value is `None`.

### 9. Render the note

The Markdown note contained:

1. a title;
2. a model-free repository map;
3. chapter-grouped belief/correction bullets; and
4. one quoted file/line snippet per bullet.

The repository map listed all syllabus chapters and their three largest files,
so even R1 exposed a lightweight map of the entire DSPy corpus. Ordinary round
construction was append-only and did not replace superseded corrections, which
allowed contradictions to accumulate.

### 10. RETEST

After R1, the driver sampled prior questions with:

```python
rng = random.Random(round_number)
n_retest = max(1, int(len(new_items) * 0.2))
sample = rng.sample(all_prior_items, n_retest)
```

Retests used the current note, were adjudicated like new questions, and could
write new entries. They were sampled from all previous records, including dev
and prior retest records, after dev lineage had been dropped. Seven of the 46
raw R4 DSPy entries came from retests; two R1 dev questions were later admitted
to the note through retesting. Retest was therefore additional training, not an
independent retention measurement.

### 11. COMPACT

Compaction was added after R3 and run manually. The same model merged entries by
chapter; outputs passed through the same weak quote gate. A stochastic guard
tested the first 12 entry-backed questions and counted both correct and partial
as success. The compact note was accepted if that score did not decrease.

The nominal 4,000-token cap did not automatically trigger compaction. Neither
compaction calls nor guard calls were included in reported study-token totals.

R4 provenance is ambiguous:

- raw R4: 46 entries and 23,238 bytes;
- compacted R4: 43 entries and 17,185 bytes, SHA-256
  `93f2510589e5ca3dd62523da76b7a073a4b022ed205eee294b15a2dcc0970408`.

The timeline strongly suggests that evaluation used the compacted note, but
evaluation episodes did not bind the note bytes or hash. A later experiment
rewrote the mutable `note-r4.md` path to the raw version. The exact evaluated
R4 note is therefore inferred, not proven.

## Persistence and resume behavior

Each round wrote:

- `questions.jsonl`;
- `items.jsonl`;
- `summary.json`; and
- `note-rN.md`.

`items.jsonl` was append-written in concurrent completion order. Resume reused
existing questions wholesale and recognized completed work by question text.
It did not validate the source revision, model revision, environment, seed,
note hash, or complete protocol. Mixed-provenance partial resumes were
therefore possible.

The artifacts did not retain request seeds, quiz/reference trajectories, a
provider-call ledger, model fingerprints, source/corpus hashes, environment
identity, per-phase usage, or immutable note binding.

## Study-cost accounting

Historical `study_tokens` counted completion/generated tokens only. Per-item
usage aggregated ATTEMPT, reference, adjudication, and optional distillation;
quiz cost was assigned to the first retained question in each chapter. The
historical `spent()` helper silently mapped missing or malformed completion
usage to zero.

It excluded prompt/prefill tokens, note-context cost, wall time, GPU time,
CPU/sandbox work, compaction, compaction guards, and failed quiz batches.

| Round | New questions | Train | Dev | Retest | New admitted entries | Cumulative raw entries | Archived raw note bytes | Round completion tokens | Corrected cumulative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 20 | 16 | 4 | 0 | 11 after recovery | 11 | 6,254 | 278,848 | 278,848 |
| R2 | 20 | 16 | 4 | 4 | 15 | 26 | 13,054 | 227,983 | 506,831 |
| R3 | 20 | 16 | 4 | 4 | 11 | 37 | 19,019 | 236,397 | 743,228 |
| R4 | 15 | 12 | 3 | 3 | 9 | 46 | 23,238 | 177,481 | 920,709 |

The previously reported approximate R4 cost of 943k was incorrect. The round
summaries total 920,709 completion tokens, still excluding compaction.

## R1 post-hoc gate change

R1 initially admitted eight DSPy entries and bounced six. After inspecting the
bounces, the quote gate was changed to normalize multiline/backtick formatting;
three DSPy entries were mechanically recovered. The final R1 note therefore
has 11 entries, while its stale summary still reports eight. This intervention
occurred before R1 evaluation and was disclosed, but it was post-hoc.

## Forced-50 cheatsheet comparison

The original forced-50 baseline was a broad, example-heavy document:

- 392 lines;
- 11,830 bytes;
- 50 recorded ReAct iterations;
- 39 `read_file` choices;
- two `glob` choices;
- nine caught premature `finish` choices;
- 41 actual repository-tool calls; and
- 64,363 generated completion tokens.

It covered architecture, signatures, prediction patterns, optimizers, data
structures, settings, evaluation, composition, and common errors. R1
self-quizzing instead produced 11 narrow correction bullets. Note breadth,
length, study compute, and inference-context cost were not matched.

Recover the original cheatsheet with:

```bash
git show 2aebd85:cheatsheets/dspy.md
```

Its SHA-256 was
`bac5b33589956aaac069be489cd7bc33b8d7e58c3396848be405c1e1b376afb7`.

## Downstream evaluation

At each milestone, the note was prepended to every official DSPy question:

```text
Reference notes on DSPy from your prior study of its repository:

<note bytes>

---

<StudyBench question>
```

The evaluator ran all 30 DSPy questions at four inference budgets—direct,
ordinary ReAct k5, ordinary ReAct k20, and forced ReAct k20—with three rollouts
per cell. That is 360 episodes per milestone. The historical ATTEMPT matched
the direct condition's tool-less answer shape, but not the tool-using
conditions.

Grading used Sakana `fugu`, binary rubric claims, and lenient weighted score.
WAUC used generated completion tokens, a 3,000-token anchor, a best-so-far
performance envelope, and strong low-token weighting. Prompt and note-prefill
costs were not priced.

## Historical DSPy results

Scores below are the historical lenient means. They retain the provenance and
grade-staleness defects described below.

| Arm | ATTEMPT corpus access | Study completion tokens | direct | k5 | k20 | k20f | WAUC |
|---|---|---:|---:|---:|---:|---:|---:|
| faithful base | not applicable | 0 | 3.57 | 16.20 | 19.60 | 29.00 | 12.31 |
| forced-50 cheatsheet | not applicable | 64,363 | 9.90 | 17.93 | 17.66 | 27.47 | 15.18 |
| selfquiz R1 | **none** | 278,848 | 5.36 | 17.42 | 24.26 | 29.02 | 13.63 |
| selfquiz R2 | **none** | 506,831 | 3.67 | 18.30 | 19.86 | 25.22 | 11.20 |
| selfquiz R4 | **none** | 920,709 plus uncounted compaction | 4.66 | 21.36 | 22.40 | 26.46 | 11.76 |

Paired WAUC differences and reported 95% intervals were:

| Comparison | Difference | 95% interval |
|---|---:|---:|
| R1 minus base | +1.21 | [-2.29, +5.01] |
| R1 minus cheatsheet | -1.66 | [-5.27, +2.11] |
| R2 minus base | -1.05 | [-4.3, +2.5] |
| R2 minus cheatsheet | -3.93 | [-8.3, +0.7] |
| R2 minus R1 | -2.26 | [-5.3, +0.7] |

All reported selfquiz comparison intervals included zero. The preregistered
criterion—selfquiz exceeding the cheatsheet with a paired 95% interval
excluding zero at a milestone—was not met at R1, R2, or R4. R8 was planned but
stopped after the earlier curve had been inspected.

## Major intended-versus-executed differences

| Intended design | Executed behavior |
|---|---|
| Random approximately 20% accumulating dev exam | Last retained item per chapter; current-round summary only |
| Dev never sees the note during ATTEMPT | Dev used the same with-note ATTEMPT |
| Dev and retest isolation | Dev identity was dropped on retest; dev items entered the note |
| Retest measures retention | Retest could distill corrections and became extra training |
| Two or three dev references | DSPy used one same-model reference for every item |
| Execution-grounded evidence | Execution was optional and probe/trajectory output was not retained |
| Source-verified corrections | One nearby substring of at least six characters was sufficient |
| Validated anchors and exactly five questions | Neither was enforced |
| Seeded fresh construction | No request seed; later generated curricula could replay |
| Automatic 4k note cap | The cap constant did not drive round construction |
| Contradiction-safe note updates | Ordinary notes were append-only and accumulated conflicts |
| Full study cost | Only completion tokens were counted |
| Immutable note provenance | Evaluation did not bind the note bytes/hash |
| Complete R1/R2/R4/R8 sequence | R8 was stopped after prior outcomes were observed |
| Human-like retrieval practice | No durable model memory or weight update existed |

## Valid interpretation

The historical data support this statement:

> Under the historical harness, Fugu judge, public DSPy question set, and known
> implementation defects, the executed self-quizzing milestones did not
> demonstrate superiority over the forced-50 cheatsheet.

They do not support any of these stronger statements:

- self-quizzing and the cheatsheet are equivalent;
- self-quizzing has no effect;
- self-quizzing is harmful;
- giving ATTEMPT corpus access cannot help;
- retrieval practice cannot help models; or
- the result will reproduce under a local Qwen judge or hardened harness.

## Reimplementation sources

The most useful source locations are:

- `experiments/005-self-quizzing-design.md` for the intended design;
- `experiments/007-selfquiz-runs.md` for the contemporaneous execution log;
- `experiments/008-repository-and-artifact-audit.md` for the artifact audit;
- `experiments/010-audit-009-disposition-and-selfquiz-review.md` for the
  first-principles retrospective;
- `study-selfquiz-run1/dspy/` for exact archived questions, attempts,
  references, verdicts, entries, and notes; and
- historical code at commit `dbbcbfa`.

Useful commands:

```bash
git show dbbcbfa:studybench/selfquiz.py
git show dbbcbfa:studybench/react.py
git show dbbcbfa:studybench/tools.py
git show 2aebd85:cheatsheets/dspy.md
```

There was no single immutable implementation revision for the whole curve.
The selfquiz loop began at `e4abbef`; name-collision, repository-map,
quote-recovery, and compaction changes were introduced during the program, and
the completed curve was committed at `2631274`. A faithful archival
reimplementation must therefore treat the historical method as a versioned
sequence, not pretend one final file generated every milestone.
