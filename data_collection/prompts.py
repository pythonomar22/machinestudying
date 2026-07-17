"""Prompt templates for the StudyBench question-construction replication.

GENERATOR_TEMPLATE, CRITIC_TEMPLATE, and RUBRIC_TEMPLATE are transcribed
verbatim from Appendix A.2-A.4 of the Machine Studying paper (docs/paper.md),
with the appendix's hard line wraps joined. The paper publishes only the
templates; every value in DSPY_VALUES fills a `{placeholder}` the paper left
library-specific, so those values are OUR RECONSTRUCTIONS (informed by the
released dataset's style), not the authors' originals. LABEL_TEMPLATE is
entirely ours: the paper describes Stage-2 cluster labeling ("GPT-5.4 assigns
a behavioral label to each cluster after reviewing 30 representative
sessions") without publishing its prompt.
"""

GENERATOR_TEMPLATE = """You are generating benchmark-grade {library_name} expert QA inside the official {library_name} repository.

## About {library_name}
{library_description}

## Mission
- Target primary label: `{label}`
- Label description: `{label_description}`
- Produce exactly `{num_candidates}` candidate QA pairs.
- The final benchmark will evaluate an answering agent that has access to the {library_name}.
- You may use both code and docs right now to understand {library_name} deeply, but every final answer must be recoverable from the code roots listed below alone (no docs at answer time).

## Available context
- {library_name} code roots the answering agent will also see:
{code_roots_bullets}
- Documentation under `{repo_docs_subpath}/` for privileged generation-time orientation only (the answering agent might or might not have these)
- Sampled community QA sessions below, which represent realistic user-question distribution for the target label

## How to use the sampled community QA
- The sampled community sessions are the **distribution we want to match**, not just a tone reference. Real {library_name} users hit real friction --- that is the question gold mine. Anchor each generated question in what a user in the seeds was actually trying to do or observing.
- Real community questions are often vague, mis-framed, or mixed with multiple issues. Your job is to **sharpen, not imitate**: keep the user framing ("I'm trying to X and I see Y"), drop the noise, and commit to one crisp locator-hard question.
- **Do not trust the community answer as ground truth.** The answers in the seed sessions were written by a weaker assistant and are frequently wrong, incomplete, or out of date. Treat them only as hints about what the user was confused about. Your gold `answer` must be re-derived from the actual {library_name} source and docs --- read the code, verify the behavior, and cite `code_evidence` by file and symbol. If the community answer contradicts the code, the code wins.
- Do **not** copy or lightly paraphrase a sampled question; upscale by sharpening the behavioral framing.

## Naming discipline (critical --- the locator is the challenge)
The answering agent has grep/glob/read over the full {library_name} repo and tests. If the question text names the specific class, method, file, or internal helper that **is** the answer, you've given away the page number and turned this into a trivia question. The whole point of the benchmark is that *locating* the right code is half the work.

**OK to name (brand-level, user-facing concepts that a real user would type):**
{ok_to_name_bullets}
- Anything the community user in the seed session typed first, at the same granularity they typed it.

**Not OK to name (these attach the question to a specific implementation and leak the locator):**
- The method or attribute on a branded class that contains the answer --- name the behavior, not the method.
- Internal adapter / handler / parser / helper classes.
- Internal helper functions, file paths, test-file names, private config flags, snake_case function names with dot-paths.
- **Do not refer to "the X example / tutorial / notebook / walkthrough / demo / README / guide".** These phrasings are awkward and underspecified --- they point at an artifact as if a shared referent exists ("in the repo's multihop RAG example, ..."). A strong question stands on its own: describe the *scenario* or *setup* itself ("in a multi-hop retrieval pipeline where the model refines its query across hops, ..."), not the artifact that demonstrates it.
- Examples for this codebase:
{not_ok_examples_bullets}

Rule of thumb: if a reader can `grep -R "<token>"` and land within a few files of the answer, the token belongs in the gold `answer` and `code_evidence`, not in the question.

**Bad (names the attachment point):**
{bad_examples_block}

**Good (forces the agent to locate):**
{good_examples_block}

Walk this line carefully: a good question is **specific enough** that a careful reader of the repo converges on one well-defined answer, and **general enough** in wording that no symbol name gives the answer away. If the question could match a dozen unrelated places in the repo, it's too generic; tighten by adding behavioral constraints, not by naming the class.

## Required quality bar
- Questions should read like a thoughtful senior user describing **what they observed or what they want to accomplish**, not like an exam asking about a specific symbol.
- Questions must be difficult enough that they require synthesis across files, abstractions, behavior, tests, edge cases, or design tradeoffs.
- Prefer questions whose answers require reading implementation and tests together.
- Gold answers should be concise but precise, and must be supported by `code_evidence` pointing into actual {library_name} source (not just paraphrased from the seed's community answer).
- `code_evidence` must cite only real files under the code roots ({code_roots_inline}), and each cited filename must match the pattern `{file_glob}` (files with other extensions are out of scope and will be rejected).
- Provide at least two evidence items per candidate.
- Difficulty must be either `hard` or `very_hard`.

## Hard bans
- No documentation-only questions.
- No questions about exact wording from docs, tutorials, README, notebooks, or guides.
- No "in the X example / tutorial / notebook / walkthrough / demo / README / guide" phrasings. Describe the scenario itself.
- No trivial "does {library_name} have X?" or single-symbol existence questions.
- No one-grep questions with an obvious single-line answer.
- No ambiguous or underspecified questions.
- No questions whose answers depend on privileged docs rather than code/tests.
- No questions that violate the naming discipline above (no internal class/method/helper names, no file paths, no `Class.method` attachment points).
- No questions whose gold answer rests on "the seed said so" rather than on verified code behavior.

## Sampling anchors
The JSON block below contains the sampled community sessions for the target label. Use it to preserve the real-world distribution while raising the quality bar sharply.

{sampled_sources_json}

Return JSON that matches the provided schema and nothing else."""

CRITIC_TEMPLATE = """You are the final critic and selector for benchmark-grade {library_name} expert QA.

## About {library_name}
{library_description}

## Benchmark reality
- Target primary label: `{label}`
- Label description: `{label_description}`
- You are selecting the final `{num_final}` items from a larger candidate set.
- Treat docs as potentially unavailable at answer time even though you may have seen them during generation.

## Selection criteria
- Keep only candidates that are clearly answerable from the code roots alone ({code_roots_inline}) --- no docs at answer time.
- Reject anything based on exact wording from documentation.
- Reject questions that are too easy, one-grep, or single-symbol lookups.
- **Reject questions that give away the locator.** The benchmark tests an agent with grep/glob/read over the repo --- locating the right code is half the challenge. If the question names a method/attribute on a class, an internal handler/adapter/parser/helper class, an internal helper, a file path, a test-file name, or a `snake_case` dotted function, it's a closed-book question. Rewrite to describe the behavior / symptom / user goal, keeping the specific symbol only in the gold `answer` and `code_evidence`. If rewriting would require fabricating a question unsupported by the seed or code, reject outright.
- **Reject "in the X example / tutorial / notebook / walkthrough / demo / README / guide" phrasings.** These are awkward and underspecified --- they point at an artifact as if a shared referent exists ("in the repo's multihop RAG example, ..."). A strong question stands on its own: rewrite to describe the *scenario* or *setup* itself (e.g., "in a multi-hop retrieval pipeline where the model refines its query across hops, ..."), or reject.
- **OK to keep**: branded user-facing concept names at the granularity a user would type:
{ok_to_name_bullets}
  The rule is "named the concept, not the attachment point." A branded class named as a concept is fine; the same class with a `.method` suffix is not.
- **Reject questions that are too generic to have one locator** (e.g., "how does {library_name} handle errors?" or "how does {library_name} do retries?"). A valid question is one where, after reading the repo, a careful expert would converge on the same specific file/symbol as the answer. Tighten generic questions by adding behavioral constraints, not by naming the class.
- **Reject questions whose gold answer rests on the seed's community answer as truth.** The community answers come from a weaker assistant and are frequently wrong. The gold answer must be supported by `code_evidence` pointing into actual {library_name} source. If the only support is "the seed said so," reject.
- Reject questions that copy or closely paraphrase sampled community questions.
- Reject anything outside the target label or overly similar to another candidate.
- Prefer diversity across subtopics within the label.
- You may rewrite the question, answer, difficulty, evidence, and note to improve quality.
- Keep final answers concise and well-grounded.
- `code_evidence` must contain real repo files under one of the code roots ({code_roots_inline}), and each filename must match the pattern `{file_glob}`; reject candidates that cite files with other extensions.

## Sampled community anchors
These are the same sampled sessions used during generation. They are for distribution anchoring only.

{sampled_sources_json}

## Candidate set to review
{candidate_json}

Return JSON that matches the provided schema and nothing else. If fewer than `{num_final}` candidates truly qualify, return fewer and explain the shortage in `selection_notes`."""

RUBRIC_TEMPLATE = """You are building a private grading rubric for one {library_name} expert QA benchmark question.
Your output is confidential and will only be used by the evaluator.

## Goal
- Turn the gold answer into 2-8 atomic grading claims.
- Claims should be small enough to score independently.
- Together, the claims should capture what a strong code-grounded answer must say.

## Rules
- Use only the provided question, gold answer, evidence references, and evidence file contents.
- Make every claim judgeable from code and tests alone.
- Use `core` for essential mechanisms or facts that define correctness.
- Use `supporting` for narrower detail, nuance, edge cases, or examples.
- Claims should be minimally overlapping.
- The claim weights must sum to exactly 100.
- `core` claims should carry most of the total weight.
- Every claim must cite 1-3 evidence spans.
- Every evidence span must come from the provided files only.
- Use exact line numbers from the numbered file dumps.
- Keep spans focused. Prefer 1-40 lines when possible, and never exceed 300 lines.
- Reuse spans across claims when that is the cleanest grounding.
- Do not include any public-release wording, benchmarking commentary, or grading instructions in the claim text.

## Inputs
- Question ID: `{question_id}`
- Label: `{label}`
- Question: `{question}`
- Gold answer:
{gold_answer}

## Evidence references
{evidence_references_json}

## Full evidence files
{evidence_files_text}

Return JSON that matches the schema exactly."""

# OUR prompt (Stage 2 labeling; the paper published no template for it).
LABEL_TEMPLATE = """You are naming one cluster of real user questions about {library_name}, an open-source library. The {num_sessions} questions below are representative members of a single behavioral cluster discovered by embedding and clustering a large pool of {library_name} GitHub issues.

Assign the cluster a behavioral label describing what its users are trying to accomplish, in the style of these examples from a related benchmark: `gepa_optimizer_usage`, `rag_and_retrieval_pipelines`, `react_agents_and_tools`, `signature_schema_and_pydantic_types`, `evaluation_metrics_and_custom_eval`.

Return JSON with:
- `label`: a short snake_case behavioral label
- `description`: 1-2 sentences describing what users in this cluster are trying to do
- `coherent`: false if the cluster mixes several unrelated behaviors, true otherwise

## Representative questions
{sessions_json}

Return JSON that matches the schema exactly."""

# ---------------------------------------------------------------------------
# RECONSTRUCTED library-specific values for DSPy. The paper does not publish
# these; they are written to match the released dataset's observable style
# (brand-level naming, runnable offline DummyLM programs, *.py evidence under
# dspy/ and tests/).
# ---------------------------------------------------------------------------

# OUR prompt (Stage 5 bundle optimization; the paper describes a Codex agent
# iterating with a syntax checker and sandbox until the reference answer
# scores 100 under its own rubric, but publishes no template for it).
REVISE_TEMPLATE = """You are repairing one {library_name} expert QA benchmark bundle (question, gold answer, grading rubric, evidence spans) that failed deterministic verification. The bundle must satisfy ALL of:
- The gold answer contains one self-contained, runnable Python program (a single fenced ```python block) that runs offline with `dspy.utils.dummies.DummyLM` (no API key) and exits cleanly.
- Grading the gold answer against its own rubric must award every claim (score 100/100).
- 2-8 atomic claims; weights are integers summing to exactly 100; `core` claims carry most of the weight; every claim cites 1-3 evidence spans; spans use exact line numbers from the numbered file dumps below, never exceeding 300 lines.
- The question text must not name file paths, internal helpers, or `Class.method` attachment points (brand-level names like `dspy.ReAct` are fine).

Fix the bundle with the smallest change that repairs the failures: prefer correcting the gold answer or rubric; change the question only if it is unavoidable. Do not weaken the rubric into vacuity to force a pass --- claims must still capture what a strong code-grounded answer must say, judgeable from code and tests alone.

## Current bundle
{bundle_json}

## Verification failures
{failure_report}

## Full evidence files (numbered)
{evidence_files_text}

Return JSON that matches the schema exactly: the complete repaired bundle (question, gold_answer, claims, spans)."""

# The held-out SmallDSPy topic (paper Table 3). The label string is the
# paper's; the description is OUR RECONSTRUCTION (the paper publishes none).
TARGET_LABEL = "react_agents_and_tools"
TARGET_DESCRIPTION = (
    "Building, configuring, and debugging tool-using ReAct-style agents in "
    "DSPy: registering Python functions or tools on an agent, controlling "
    "the reasoning/acting loop and its iteration budget, inspecting or "
    "post-processing trajectories, handling tool errors and edge cases, "
    "managing per-user or per-session conversation state, and composing "
    "agents with other DSPy modules."
)

DSPY_VALUES = {
    "library_name": "DSPy",
    "library_description": (
        "DSPy is a declarative framework for programming --- rather than "
        "prompting --- language models. Users compose typed modules (such as "
        "`dspy.Predict`, `dspy.ChainOfThought`, and `dspy.ReAct`) whose "
        "input/output behavior is declared by signatures; adapters translate "
        "signatures into provider-specific prompts and parse completions back "
        "into typed fields; optimizers tune prompts and demonstrations "
        "against metrics. Benchmark questions in this suite read like a real "
        "user describing a goal or symptom and ask for a self-contained, "
        "runnable Python program; solutions must run offline with "
        "`dspy.utils.dummies.DummyLM` (no API key) whenever a language model "
        "is needed."
    ),
    "code_roots_bullets": (
        "- `dspy/` --- the DSPy library source\n"
        "- `tests/` --- its test suite"
    ),
    "code_roots_inline": "`dspy/`, `tests/`",
    "repo_docs_subpath": "docs",
    "file_glob": "*.py",
    "ok_to_name_bullets": (
        "- Public branded module and concept names a real user would type: "
        "`dspy.ReAct`, `dspy.Predict`, `dspy.ChainOfThought`, `dspy.Module`, "
        "`dspy.Signature`, `dspy.Example`, `dspy.Tool`, `dspy.History`, "
        "`dspy.Prediction`, `dspy.Evaluate`, `DummyLM`.\n"
        "- User-facing concepts at brand granularity: signatures, "
        "input/output fields, tools, trajectories, adapters (as a concept), "
        "optimizers (as a concept), streaming, async, conversation history."
    ),
    "not_ok_examples_bullets": (
        "- Internal adapter/parser classes or their methods --- name the "
        "formatting or parsing *behavior* instead.\n"
        "- `dspy.ReAct.forward`, `dspy.Predict.acall`, or any other "
        "`Class.method` attachment point.\n"
        "- Internal helpers or dotted snake_case paths such as "
        "`dspy.predict.react._fmt`, and file paths such as "
        "`dspy/predict/react.py` or `tests/predict/test_react.py`.\n"
        "- Private flags or keyword arguments that only appear inside the "
        "implementation."
    ),
    "bad_examples_block": (
        "- \"What does `dspy.ReAct._format_trajectory` return when a tool "
        "call raises an exception?\" (names the internal helper --- the "
        "locator is given away)\n"
        "- \"In `dspy/predict/react.py`, how does the truncation logic "
        "decide which trajectory entries to drop?\" (names the file path)"
    ),
    "good_examples_block": (
        "- \"My ReAct agent's tool sometimes raises mid-episode; I want the "
        "run to survive, keep the failure visible to the model, and still "
        "produce a final typed answer. Write a runnable demo and explain "
        "what ends up in the trajectory after a failing call.\" (behavioral "
        "symptom; the agent must locate the error-handling code)\n"
        "- \"When my tool-using agent runs long, older steps seem to vanish "
        "from the prompt. Under what conditions does that happen, which "
        "steps are kept, and how do I see it in a runnable example?\" "
        "(describes observed behavior; forces locating the truncation "
        "mechanism)"
    ),
}
