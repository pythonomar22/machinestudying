# Human audit contract for self-quizzing studies

Selfquiz notes are built from model-generated questions, references, verdicts,
and corrections. Automated agreement among calls to the same model is not an
independent validation of those labels. Consequently, an automated construction
manifest always has `claim_ready: false`, even when every automated gate passes.

A separate human audit can make a note eligible for confirmatory evaluation
only when the audit was pre-registered before round 1, the auditor was
independent, condition and method labels remained hidden, every cumulative
record and admitted entry was reviewed, every review passed, and the exact
construction dependency graph still hashes identically.

The code verifies artifact identity, population completeness, field values,
and the auditor's declarations. It cannot prove that a person was genuinely
independent or blinded. Those facts require an externally accountable process
(for example, named assignment and retained blinding materials) and must never
be described as cryptographically proven.

## Pre-registered protocol

Before generating any round-1 artifact, prepare a UTF-8 JSON file with exactly
this five-field contract:

```json
{
  "schema_version": 1,
  "protocol_id": "independent-audit-v1",
  "blinding": "condition_and_method_labels_hidden",
  "population": "all_train_dev_verdicts_and_admitted_entries",
  "decision_rule": "all-required-reviews-pass-v1"
}
```

`protocol_id` must match `[a-z0-9][a-z0-9._-]{2,79}`. The decision-rule value
is the exact, versioned identifier shown above; descriptive paraphrases are not
accepted. No additional fields are accepted. In particular, do not put
assignment, sampling-display, escalation, conflict, or adjudication promises in
this JSON: the software does not enforce those procedures, so accepting such
clauses would overstate what claim-ready promotion proves. Record externally
accountable operating procedures and retained blinding evidence separately;
they remain human-process evidence, not machine-validated promotion criteria.
If an immutable study already snapshotted an extended protocol, do not rewrite
it: that construction is ineligible for promotion under this schema, so begin a
new study with a new five-field pre-registration.
The study snapshots the exact bytes and SHA-256 under its immutable namespace.
Supplying a protocol after study has started is not a valid pre-registration.
`schema_version` must be the JSON integer `1`; JSON booleans are not integers
for this contract.

## Blinded review packet

Create the packet from immutable construction artifacts without showing the
auditor the arm name, method name, benchmark outcomes, milestone scores, or
whether an answer used the note. Stable opaque record IDs must be retained so
the completed audit can be joined back exactly. The packet should include only
what is needed to assess:

- whether each train/dev verdict is substantively valid;
- whether its cited evidence actually supports the assessment;
- whether question, answer, evidence, or verdict reveals benchmark content;
- whether each admitted correction follows from the repository evidence; and
- whether each stored citation is exact.

Every `items.jsonl` record from rounds 1 through the promoted round and every
`dev-exam.jsonl` record from those rounds is in the review population. Every
cumulative note entry is also in the population. This is deliberately stronger
than sampling a small convenience subset.

## Completed audit result

The completed UTF-8 JSON binds the protocol, construction manifest, and note.
Its required top-level fields are:

```json
{
  "schema_version": 1,
  "study_id": "new-study-id",
  "task": "dspy",
  "round": 1,
  "protocol_sha256": "<64 lowercase hex characters>",
  "construction_manifest_sha256": "<64 lowercase hex characters>",
  "note_sha256": "<64 lowercase hex characters>",
  "blinding_preserved": true,
  "reviewer_independent": true,
  "decision": "pass",
  "auditor_id": "auditor-opaque-id",
  "record_reviews": [
    {
      "record_id": "<exact item or dev-exam ID>",
      "verdict_valid": true,
      "evidence_valid": true,
      "leakage_free": true
    }
  ],
  "entry_reviews": [
    {
      "entry_id": "<exact admitted entry ID>",
      "correction_supported": true,
      "citation_exact": true,
      "leakage_free": true
    }
  ]
}
```

`schema_version` and `round` must be JSON integers, not booleans. There must be
exactly one review for every expected record and entry, with no extra or
duplicate IDs. Every review decision and both process declarations must be JSON
booleans. Set `decision` to `"pass"` if and only if both process declarations
and every review decision are `true`; otherwise set it to `"fail"`.

A failed review or process declaration is scientific information. A fully
bound, complete, internally consistent `"fail"` result is archived byte for
byte under `notes/audits/failed/by-sha256/`, and the promotion command then
exits without creating `note-rN.audited.manifest.json`. Malformed results,
incorrect artifact bindings, incomplete populations, and decisions that do not
follow the rule above are rejected without being archived. Do not remove failed
records, rewrite the note, or change the decision rule to obtain a pass. A
corrected method is a new study with a new pre-registration.
Before accepting any later pass, promotion scans the content-addressed failure
archive. Any previously archived, fully valid failure bound to the same study,
task, round, protocol, construction, and note permanently blocks promotion of
that construction. Malformed or unrelated files do not create a false
tombstone, while symlinked archive paths fail closed.

## Promotion and downstream binding

Promotion is an offline validation operation. It re-hashes every recorded
construction artifact and exact note and validates the complete audit
population. A passing result snapshots the protocol and result beside the note
and writes a separate `note-rN.audited.manifest.json`; a valid failing result is
archived as described above and creates no audited manifest. Neither path edits
the original construction manifest. Evaluation snapshots a passing audited
manifest, exact note, construction manifest, protocol, and result; grading
checks that entire bundle again.
The population is re-derived from every hash-verified `rN/items.jsonl` and
`rN/dev-exam.jsonl` dependency rather than trusted from summary flags. The
evaluation snapshotter and grader repeat that derivation and require exact
record and entry review coverage plus the deterministic decision rule.
Non-finite or duplicate-key JSON is rejected, and any dev or non-original
training/retest record that purports to admit a note entry invalidates the
construction instead of silently disappearing from the audit population.

After filling the completed audit JSON, promote it without contacting a model
or server. This is a command template, not a command executed during the
hardening work:

```bash
.venv-dspy/bin/python -m studybench.selfquiz \
  --task dspy --round 1 --study-id new-study-id --seed 12345 \
  --promote-human-audit path/to/completed-audit.json
```

The task, round, study ID, and seed must exactly match the immutable study
contract. Promotion fails if any construction byte, population member,
declaration, or review result differs from that contract.

Do not point a confirmatory evaluation at `note-rN.manifest.json`. That is the
automated construction record and is intentionally not claim-ready. Use only
the separate audited manifest after a real audit has passed.
