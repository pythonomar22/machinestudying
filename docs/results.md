# Baseline and cheatsheet evaluation record

This is the retained result record after removing the exploratory work and
generated run trees. It distinguishes published targets from historical local
measurements; no result was rerun during repository cleanup.

## Published Table 1

Cells are lenient accuracy with mean generated tokens in parentheses.

| Study-DSPy | direct | k5 | k20 | forced k20 | expertise |
|---|---:|---:|---:|---:|---:|
| base | 3.3 (4.1k) | 8.6 (7.9k) | 9.6 (8.6k) | 29.4 (34.6k) | 6.49 |
| cheatsheet | 6.3 (3.9k) | 14.4 (6.1k) | 14.1 (7.1k) | 23.1 (29.9k) | 9.65 |

| Study-OpenClaw | direct | k5 | k20 | forced k20 | expertise |
|---|---:|---:|---:|---:|---:|
| base | 2.3 (4.1k) | 6.9 (4.6k) | 15.8 (9.7k) | 17.6 (24.3k) | 7.64 |
| cheatsheet | 4.3 (3.8k) | 8.6 (6.0k) | 15.2 (9.1k) | 18.1 (20.1k) | 8.18 |

The rounded OpenClaw base cells reconstruct to approximately 7.66 rather than
the published 7.64; the paper calculated expertise from unrounded internal
token and accuracy values.

## Historical local evaluation

The faithful DSPy-ReAct runs completed on 2026-07-06 reported:

| Condition | DSPy expertise | OpenClaw expertise |
|---|---:|---:|
| local base | 12.31 | 8.45 |
| local cheatsheet | 15.18 | 10.59 |

The local OpenClaw row was close to the paper. DSPy direct and forced-20
endpoints were also close, but voluntary k5/k20 accuracy remained about twice
the published level. The local paired cheatsheet-minus-base estimates were
+2.88 DSPy and +2.36 OpenClaw; both three-rollout confidence intervals included
zero.

These values are historical records, not presently auditable artifacts. They
used Fugu rather than the paper's GPT-5.4 judge, some grades later became stale,
the evaluated cheatsheet aliases were overwritten by follow-up work, and the
raw runs were deleted at the user's request during this cleanup. They must not
be presented as a fresh replication from the current tree.

Future results should come only from the paired baseline/cheatsheet pipeline in
the root README, retain the complete study and evaluation artifacts until the
numbers are checked, and be added here only after both populations pass the
reporter's completeness and compatibility checks.

## SmallDSPy paired GPT-5.4 measurements (2026-07-16 / 2026-07-21)

All four conditions share master seed 20260715, identical per-episode seeds
(audited), and the paper-contract GPT-5.4 judge. Cells are lenient accuracy
with mean generated tokens.

| Condition | direct | k5 | k20 | forced k20 | expertise |
|---|---:|---:|---:|---:|---:|
| no-study | 7.1 (4.0k) | 14.8 (4.5k) | 16.1 (6.5k) | 26.5 (25.1k) | 12.35 |
| cheatsheet | 15.4 (2.8k) | 15.1 (4.2k) | 19.7 (5.3k) | 29.1 (21.4k) | 19.18 |
| selfquiz iter-1, final note | 11.4 (3.4k) | 14.7 (4.9k) | 17.7 (7.0k) | 21.3 (21.3k) | 13.75 |
| selfquiz iter-1, val-selected note | 4.4 (3.0k) | 12.3 (5.2k) | 15.0 (8.0k) | 38.5 (22.9k) | 13.04 |
| selfquiz iter-2, explore+quiz | 4.3 (2.5k) | 13.8 (3.9k) | 17.6 (4.7k) | 26.5 (24.8k) | 15.19 |

These are exact measurements of these artifacts, not method-level findings:
on 5 questions no pairwise expertise gap is statistically separable from
zero (paired cluster bootstrap CIs all include 0). Full context, audit
findings, and mandatory qualifiers in `experiments/006-selfquiz-smalldspy.md`
and `experiments/007-selfquiz2-smalldspy.md`.
