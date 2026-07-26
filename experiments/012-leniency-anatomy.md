# 012 — Anatomy of the direct→k5f leniency jump (codex logs deep-dive)

**Status: complete (2026-07-23).** Deep investigation of what accounts
for high leniency at search budgets in the codex gpt-5.4-mini runs, for
the purpose of distilling it into a k=0 study object. Sources: the 280
practice-70 teacher sessions (010) and the 720 test-30 sessions (011);
claim-level workflow `wf_c6e313da-e74` (20 readers, all returned);
deterministic passes over all grades and tool logs.

**Firewall observed throughout**: fact-level content was extracted ONLY
from practice-70 trajectories. Test-30 logs were analyzed for aggregate
structure only (command mixes, intent shares, concentration numbers) —
no test-derived facts or file identities may inform a study object.

## The headline numbers

Direct → k5f on practice-70: 22.0 → 70.5 mean lenient; **64/70 questions
improve**, 3,525 rubric-weight points flip upward. The jump is broad,
not a lottery.

**Claim-level fact classification** (12 biggest flip questions, 51
flipped claims, 1,120 weight points, each fact traced to the tool_log
step that surfaced it):

| enabling-fact category | share of flipped weight |
|---|---:|
| behavior_semantics (what the library actually does) | 39% |
| offline_harness_idiom (how to mock/prove it offline) | 17% |
| call_signature_or_arg | 17% |
| exact_api_name | 12% |
| output_format (state/serialization shapes) | 10% |
| default_value | 2% |
| file_location_only | 2% |

**Generality of those facts**: topic_general 76%, general_core_api 16%,
**question_specific only 8%**. The knowledge that flips claims is
overwhelmingly REUSABLE — the single most important finding for the
study-object thesis. (k20f marginal claims: same shape, 82%
topic-general.)

## How the five commands are spent (the search grammar)

- Command mix (350 k5f steps): 63% file reads, 30% greps, 7% locates.
- Stated intent: **68% verification** ("verify/confirm/check"), 21%
  locate, 11% learn. The strong model doesn't search to discover — it
  searches to *check* things it half-knows.
- Canonical 5-step shape: 1 broad `rg` to locate → read the
  implementation file (facts: values, semantics, signatures) → **read
  the sibling test file (the proof recipe: exact mock sets, DummyLM
  payload shapes)** → 1-2 verification reads. The decisive step is
  usually the TEST file — the repo's own tests are ready-made offline
  proof harnesses (e.g., `tests/clients/test_lm_local.py` contains the
  complete patch set the gold answer needs).
- File concentration: `dspy/utils/dummies.py` opened in **54/70
  sessions**; top-20 files = 66% of all touches; 84 distinct files.
- Reasoning budget is ~constant (direct 10.4k thinking vs k5f 8.3k):
  search does not add thought, it **redirects thought from speculation
  to grounding**. Verification is cheaper than rumination.

## Why direct fails (what the object must prevent)

- 41/70 direct sandbox failures are interface-shape errors (TypeError/
  AttributeError). One recurring fact alone — DummyLM consumes a list of
  dicts mapping output-field→value — accounts for the repeated
  `'str' object has no attribute 'items'` crashes.
- Failures are *coherent wrong-priors*, not noise: ecosystem-convention
  substitution (vLLM/OpenAI conventions where DSPy uses
  sglang/LocalProvider; `lm.api_base` attribute where DSPy writes
  `lm.kwargs['api_base']`), invented wrapper classes
  (`TracingDummyLM`, `RecordingLM`) where the shipped stub suffices.
- Some direct programs "pass" the sandbox by testing their own
  hallucination (self-consistent fiction that never imports the real
  mechanism) — sandbox exit-0 is necessary, not sufficient.

## What the bigger budgets buy

- k5f→k20f (+6.6): 28 questions improve (mean +31); the extra commands
  re-verify the SAME core surface deeper (`dspy/__init__.py` exports
  checked 21×, base_lm, chat_adapter, predict, dummies) — coverage of
  supporting claims, not new territory.
- k20f→k50f (+1.1): precision polishing — ~half the touches are
  re-reads, tail motivations are "pin the exact lines." Marginal value
  of search saturates by ~20 commands; most of it is in the first 5.

## Structural transfer to the real exam (structure only)

On the 540 test-30 search sessions (both conditions): 72% reads / ~50%
verification intent / top-20 files = 54-56% of touches — same grammar.
**9 of the 10 practice-derived core files sit in the test runs' top-20,
and those 10 files alone account for 31% of ALL file touches on the
held-out exam.** The compact core discovered from practice transfers.

## Lessons

1. **High leniency at k≥5 = verified interface truth, not more
   intelligence.** Equal thinking budgets; the difference is whether
   thought is anchored to five ground-truth observations.
2. **The unlocking knowledge is compact and bimodal**: a small universal
   core (offline-harness idiom, answer form, core API surface) plus
   topic-general behavior facts. Only ~8% of flipped weight is
   question-unique — a study object can carry the rest.
3. **Implementation files supply the facts; test files supply the
   form.** The repo's tests are pre-verified proof recipes; copying
   their mock patterns is what turns knowledge into a runnable program.
4. **Wrong priors are systematic and nameable** — a distrust-list is
   itself distillable content.
5. **Search is mostly verification, so pre-verified prompt content
   substitutes for it directly** — and also collapses the rumination
   tax at direct (10k thinking → recall).

## Proposal: the study object (evidence-backed spec for foldback v1)

Layered prompt-side object, built ONLY from practice-70 trajectories +
corpus + golds, assembled by the existing `studying/foldback` pipeline
(mine → single-pass assemble → GPT verdict-only fact-check → dev-30):

1. **Answer-form contract** (universal, ~0.5k chars): the canonical
   offline proof-program skeleton — DummyLM payload shape (list of
   dicts, output-field→string, consumed per call), configure, single
   fence, prints/asserts. Directly targets the 41/70 interface-shape
   failure class.
2. **Harness recipes** (~15 entries, from test files): per core
   mechanism, the repo's own mock/proof pattern (LocalProvider.launch
   patch set; bootstrap_trace_data(raise_on_error=False,
   capture_failed_parses=True); adapter-override pattern; ...). This is
   the 17% offline_harness_idiom weight plus the enabler for everything
   else.
3. **Behavior fact sheet** (the 58% bulk: semantics + signatures +
   defaults): per-module verified one-liners with file refs — the
   prefix-strip order, lm.kwargs write-backs, UNSAFE_LM_STATE_KEYS,
   FailedPrediction fields, adapter fallback chain, ...
4. **Navigation map + strategy priors**: mechanism → (impl file, test
   file) pairs for the core surface, plus the meta-rules the
   trajectories embody: "read the sibling test for the proof pattern",
   "distrust remembered integration details (sglang not vLLM; kwargs
   not attributes)", "verify exports in dspy/__init__.py before
   importing". At k5/k20 this converts discovery into pure targeted
   verification; at direct the trust-list substitutes for it.

Sizing guidance from the data: the object need not be huge — the core
that pays is ~10 files' worth of facts + ~15 recipes + the form
contract. Expected value concentrates at direct (45% of WAUC weight)
and k5.

Honest caveats: fact categories/generality were judged by model readers
(consistent across 18 independent agents but subjective); the 12
dissected questions are the largest flips (that is where the weight is,
but small flips may differ); rubrics are our own replication (absolute
lenient levels friendlier than StudyBench's, per 011); all analysis is
single-rollout per budget on practice-70.

## Next

Execute foldback v1 with this content spec (mine the teacher k5f logs;
the miner prompt should extract exactly the four layers above), dev-30
gate vs cheatsheet-008, one declared Qwen test shot vs 15.90.
