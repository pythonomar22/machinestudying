"""Pure, shared contracts for StudyBench study procedures.

This module deliberately has no DSPy dependency so construction and grading
can independently derive the same prompt, sampling, seed, and tool contract.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
import json
from pathlib import PurePosixPath
import re
from typing import Any

import json_repair
import regex

from .integrity import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    sha256_text,
    stable_seed,
    strict_json_loads,
)
from .tools import DSPY_READ_MAX_LINES, dspy_tool_contract


PROTOCOL_SUMMARY_SCHEMA_VERSION = 1
SEMANTIC_SELFQUIZ_METHOD = "semantic-selfquiz-v4"
SEMANTIC_SELFQUIZ_TASK_MANIFEST_TYPE = "semantic-selfquiz-study-task"
SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE = "semantic-selfquiz-note"
SEMANTIC_SELFQUIZ_ADAPTER = "studybench.selfquiz.SemanticSelfquizAdapter"
SEMANTIC_SELFQUIZ_ADAPTER_POLICY = (
    "chat-primary-strict-json-schema-parse-or-tool-contract-repair-v1"
)
SEMANTIC_SELFQUIZ_CITATION_POLICY = (
    "uniform-answer-frozen-exact-line-pass-v1"
)
SEMANTIC_SELFQUIZ_UNRESOLVED_POLICY = (
    "abstain-no-correction-continue-v1"
)
SEMANTIC_FINAL_ROUND = 4
DSPY_SEMANTIC_CHAPTER_SYLLABUS = (
    "dspy/teleprompt",
    "dspy/adapters",
    "dspy/clients",
    "dspy/predict",
    "dspy/primitives",
    "dspy/utils",
    "dspy/dsp",
    "dspy/signatures",
    "dspy/datasets",
    "dspy/retrievers",
    "dspy/streaming",
    "dspy/evaluate",
    "dspy/propose",
    "dspy",
    "dspy/experimental",
)
SMALLDSPY_SEMANTIC_CHAPTER_SYLLABUS = (
    "dspy/adapters",
    "dspy/predict",
    "dspy/primitives",
)
STATIC_GRAPH_METHOD = "deterministic-static-call-neighborhood"
STATIC_GRAPH_TASK_MANIFEST_TYPE = "deterministic-static-graph-study-task"
STATIC_GRAPH_NOTE_MANIFEST_TYPE = "deterministic-static-graph-note"
HUMAN_AUDITED_NOTE_MANIFEST_TYPE = "human-audited-note"

FORCED50_CONFIG_SCHEMA_VERSION = 5
FORCED50_LEGACY_CONFIG_SCHEMA_VERSION = 4
FORCED50_ITERATIONS = 50
DSPY_ADAPTER_NAME = "studybench.react.ParseOnlyFallbackChatAdapter"
DSPY_ADAPTER_POLICY = "parse-only-chat-to-json-fallback-v1"
DSPY_REQUEST_AUDIT_SCHEMA_VERSION = 3
DSPY_REQUEST_AUDIT_LEGACY_SCHEMA_VERSION = 2
DSPY_REQUEST_AUDIT_SCHEMA_VERSIONS = frozenset({
    DSPY_REQUEST_AUDIT_LEGACY_SCHEMA_VERSION,
    DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
})
FORCED50_CONFIG_KEYS = frozenset({
    "schema_version",
    "study_id",
    "task",
    "method",
    "model",
    "model_revision",
    "expected_response_model",
    "sampling",
    "adapter",
    "adapter_fallback_policy",
    "dspy_request_audit_schema",
    "master_seed",
    "episode_seed",
    "study_prompt_sha256",
    "study_question_sha256",
    "tool_contract",
    "tool_schema_sha256",
    "read_max_lines",
    "forced_iterations",
    "repository_tool_scope",
    "corpus",
    "source",
    "environment",
    "claim_ready",
    "server_transport",
})
REACT_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 32_768,
    "presence_penalty": 1.5,
    "extra_body": {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
}
DSPY_REPOSITORY_TOOL_CONTRACT = dspy_tool_contract(DSPY_READ_MAX_LINES)
SEMANTIC_ATTEMPT_ACCESS_MODES = ("closed-book", "react-corpus")


def semantic_chapter_syllabus(task: str) -> tuple[str, ...]:
    """Return the frozen production-chapter syllabus for a semantic task."""

    if task == "dspy":
        return DSPY_SEMANTIC_CHAPTER_SYLLABUS
    if task == "smalldspy":
        return SMALLDSPY_SEMANTIC_CHAPTER_SYLLABUS
    raise ValueError(f"semantic selfquiz does not support task {task!r}")


_SHA256_LENGTH = 64
_PROTOCOL_SUMMARY_KEYS = frozenset({
    "schema_version",
    "task_manifest_sha256",
    "method",
    "question_mode",
    "focus",
    "attempt_access",
    "attempt_protocol_sha256",
    "resolver_contract_sha256",
    "question_bank_sha256",
    "question_bank_artifact_sha256",
})
_COMMON_TASK_KEYS = frozenset({
    "schema_version",
    "manifest_type",
    "study_id",
    "task",
    "master_seed",
    "model",
    "model_revision",
    "sampling",
    "corpus_commit",
    "corpus",
    "source",
    "environment",
    "environment_contract",
    "server_transport",
    "provenance_readiness",
    "automated_provenance_ready",
    "config",
})
_SEMANTIC_CONFIG_KEYS = frozenset({
    "chapter_syllabus",
    "chapters_per_round",
    "final_round",
    "questions_per_chapter",
    "attempt_access",
    "adapter",
    "adapter_policy",
    "citation_policy",
    "unresolved_policy",
    "smoke",
    "quiz_max_iters",
    "attempt_protocol",
    "derive_max_iters",
    "train_ensemble",
    "dev_ensemble",
    "retest_fraction",
    "freshness_near_jaccard",
    "max_freshness_near_rate",
    "concurrency",
    "provider_retries",
})
_STATIC_GRAPH_CONFIG_KEYS = frozenset({
    "method",
    "smoke",
    "concurrency",
    "attempt_protocol",
    "train_input_note_sha256",
    "train_question_count",
    "dev_question_count",
    "dev_holdout_targets",
    "provider_retries",
    "read_max_lines",
})
_EXPECTED_ATTEMPT_PROTOCOL = {
    "method": "dspy.ReAct",
    "signature": "AttemptSig",
    "adapter": "dspy.ChatAdapter",
    "max_iters": 5,
    "tools": ["grep", "glob", "read_file"],
    "termination_tool": {
        "name": "finish",
        "args": {},
        "observation": "Completed.",
    },
    "tool_contract": DSPY_REPOSITORY_TOOL_CONTRACT,
    "tool_schema_sha256": sha256_json(DSPY_REPOSITORY_TOOL_CONTRACT),
    "read_max_lines": DSPY_READ_MAX_LINES,
    "tool_corpus_scope": "complete-pinned-corpus",
}
_EXPECTED_CLOSED_BOOK_ATTEMPT_PROTOCOL = {
    "method": "dspy.Predict",
    "signature": "AttemptSig",
    "adapter": "dspy.ChatAdapter",
    "max_iters": 1,
    "tools": [],
    "termination_tool": None,
    "tool_contract": None,
    "tool_schema_sha256": None,
    "read_max_lines": None,
    "tool_corpus_scope": "none",
}
_STATIC_GRAPH_DEV_HOLDOUT_TARGETS = [
    "dspy.adapters.utils.parse_value",
    "dspy.signatures.signature._parse_signature",
    "dspy.teleprompt.bootstrap_finetune.all_predictors_have_lms",
    "dspy.teleprompt.utils.eval_candidate_program",
]

SEMANTIC_NOTE_MANIFEST_KEYS = frozenset({
    "schema_version",
    "manifest_type",
    "method",
    "protocol_summary",
    "study_id",
    "task",
    "round",
    "corpus_commit",
    "claim_ready",
    "publication_claim_ready",
    "confirmatory_claim_ready",
    "automated_claim_ready",
    "automated_readiness",
    "automated_treatment_ready",
    "automated_treatment_readiness",
    "human_audit",
    "note_sha256",
    "note_path",
    "input_note_sha256",
    "entry_ids",
    "entries",
    "construction_artifacts",
    "construction_artifacts_sha256",
    "usage",
    "round_usage",
    "cumulative_usage",
    "round_usage_by_phase",
    "cumulative_usage_by_phase",
    "round_construction_usage",
    "cumulative_construction_usage",
    "round_construction_usage_by_phase",
    "cumulative_construction_usage_by_phase",
    "note_chars",
})
SEMANTIC_READINESS_KEYS = frozenset({
    "non_smoke",
    "provenance_complete",
    "launch_environments_bound",
    "prior_rounds_automated_ready",
    "question_freshness",
    "quiz_episodes_complete",
    "training_complete",
    "dev_references_complete",
    "dev_exam_complete",
    "lineage_clean",
    "evidence_safe",
    "usage_complete",
    "adapter_audit_complete",
    "response_model_homogeneous",
    "response_model_expected",
})
SEMANTIC_TREATMENT_READINESS_KEYS = frozenset({
    "non_smoke",
    "provenance_complete",
    "launch_environments_bound",
    "prior_rounds_treatment_ready",
    "question_freshness",
    "quiz_episodes_complete",
    "training_terminal_complete",
    "dev_references_terminal_complete",
    "dev_exam_terminal_complete",
    "lineage_clean",
    "evidence_safe",
    "usage_complete",
    "adapter_audit_complete",
    "response_model_homogeneous",
    "response_model_expected",
})
STATIC_GRAPH_NOTE_MANIFEST_KEYS = frozenset({
    "schema_version",
    "manifest_type",
    "method",
    "protocol_summary",
    "study_id",
    "task",
    "round",
    "corpus_commit",
    "claim_ready",
    "publication_claim_ready",
    "confirmatory_claim_ready",
    "automated_claim_ready",
    "automated_readiness",
    "human_audit",
    "note_sha256",
    "note_path",
    "input_note_sha256",
    "entry_ids",
    "entries",
    "train_question_ids",
    "held_out_dev_question_ids",
    "resolver_contract_sha256",
    "question_bank_sha256",
    "construction_artifacts",
    "construction_artifacts_sha256",
    "usage",
    "usage_by_phase",
    "usage_audit",
    "note_chars",
    "note_bytes",
})
STATIC_GRAPH_READINESS_KEYS = frozenset({
    "non_smoke",
    "provenance_complete",
    "environment_contract_valid",
    "resolver_contract_recomputed",
    "question_bank_recomputed",
    "selection_exact",
    "training_complete",
    "training_empty_note",
    "training_scores_recomputed",
    "corrections_recomputed",
    "note_recomputed",
    "dev_holdout_isolated",
    "dev_evidence_locations_disjoint",
    "dev_pair_complete",
    "dev_pair_note_only",
    "dev_scores_recomputed",
    "usage_complete",
    "launch_environments_bound",
    "response_model_homogeneous",
    "response_model_expected",
    "construction_inventory_complete",
})


def openbook_attempt_protocol() -> dict[str, Any]:
    """Return a detached JSON copy of the exact ATTEMPT intervention."""

    value = strict_json_loads(
        canonical_json_bytes(_EXPECTED_ATTEMPT_PROTOCOL),
        label="open-book attempt protocol",
    )
    assert isinstance(value, dict)
    return value


def semantic_attempt_protocol(attempt_access: str) -> dict[str, Any]:
    """Return the exact ATTEMPT contract for one semantic-study arm.

    ``closed-book`` and ``react-corpus`` deliberately share the same signature,
    adapter, model, sampling, and answer objective. The intervention is a policy
    bundle: one tool-free :class:`dspy.Predict` call versus bounded
    :class:`dspy.ReAct` with repository access. It changes both access and the
    reasoning/call scaffold, so it is not a pure access ablation.
    """

    if attempt_access == "closed-book":
        expected = _EXPECTED_CLOSED_BOOK_ATTEMPT_PROTOCOL
    elif attempt_access == "react-corpus":
        expected = _EXPECTED_ATTEMPT_PROTOCOL
    else:
        raise ValueError(
            "semantic attempt access must be closed-book or react-corpus"
        )
    value = strict_json_loads(
        canonical_json_bytes(expected),
        label=f"{attempt_access} semantic attempt protocol",
    )
    assert isinstance(value, dict)
    value["adapter"] = SEMANTIC_SELFQUIZ_ADAPTER
    return value


class StudyProtocolError(ValueError):
    """A study task or note does not bind one recognized protocol exactly."""


def _same_json(left: object, right: object) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


_DSPY_V2_PROVIDER_CALL_KEYS = frozenset({
    "call",
    "response_model",
    "response_id",
    "system_fingerprint",
    "request_messages_sha256",
    "outputs_sha256",
    "provider_usage",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
})
_DSPY_V3_PROVIDER_CALL_KEYS = frozenset({
    *_DSPY_V2_PROVIDER_CALL_KEYS,
    "processed_response",
    "processed_response_canonical_bytes",
    "processed_response_sha256",
    "outputs",
    "outputs_canonical_bytes",
    "finish_reasons",
})
_DSPY_V3_BINDING_COMMON_KEYS = frozenset({
    "schema_version",
    "kind",
    "stage",
    "adapter",
    "fallback_used",
    "provider_calls",
    "provider_call",
    "choice",
    "outputs_sha256",
    "selected_output_canonical_bytes",
    "selected_output_sha256",
})
_DSPY_V3_PARSED_BINDING_KEYS = frozenset({
    *_DSPY_V3_BINDING_COMMON_KEYS,
    "parsed_outputs",
    "parsed_outputs_canonical_bytes",
    "parsed_outputs_sha256",
    "answer_sha256",
})
_DSPY_V3_PARSE_FAILURE_BINDING_KEYS = frozenset({
    *_DSPY_V3_BINDING_COMMON_KEYS,
    "adapter_lm_response",
    "adapter_lm_response_replay",
    "adapter_lm_response_bytes",
    "adapter_lm_response_sha256",
    "parsed_result",
    "parsed_result_canonical_bytes",
    "parsed_result_sha256",
})

_DSPY_FIELD_HEADER_PATTERN = re.compile(r"\[\[ ## (\w+) ## \]\]")
_DSPY_JSON_OBJECT_PATTERN = r"\{(?:[^{}]|(?R))*\}"
_DSPY_REACT_TOOL_NAMES = ("grep", "glob", "read_file", "finish")


class _DspyAdapterParseFailure(ValueError):
    """The independently replayed pinned adapter raised AdapterParseError."""

    def __init__(
        self,
        lm_response: str,
        parsed_result: object = None,
        *,
        lm_response_replayable: bool = True,
    ):
        super().__init__("pinned DSPy adapter parse failure")
        self.lm_response = lm_response
        self.parsed_result = parsed_result
        self.lm_response_replayable = lm_response_replayable


def _dspy_output_field_contract(stage: str) -> tuple[tuple[str, str], ...]:
    """Return the only output signatures used by the frozen local harness."""

    if stage == "direct":
        return (("answer", "str"),)
    if stage == "extract":
        return (("reasoning", "str"), ("answer", "str"))
    if stage == "react":
        return (
            ("next_thought", "str"),
            ("next_tool_name", "tool_name"),
            ("next_tool_args", "dict"),
        )
    raise StudyProtocolError("unknown DSPy adapter parse stage")


def _parse_dspy_value(value: object, field_type: str) -> object:
    """Replay pinned ``parse_value`` for this harness's three field types."""

    if field_type == "str":
        return str(value)
    if field_type == "tool_name":
        if value in _DSPY_REACT_TOOL_NAMES:
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.startswith(("Literal[", "str[")) and candidate.endswith("]"):
                candidate = candidate[candidate.find("[") + 1 : -1]
            if (
                len(candidate) > 1
                and candidate[0] == candidate[-1]
                and candidate[0] in "\"'"
            ):
                candidate = candidate[1:-1]
            if candidate in _DSPY_REACT_TOOL_NAMES:
                return candidate
        raise ValueError("invalid pinned DSPy ReAct tool name")
    if field_type != "dict":
        raise StudyProtocolError("unknown pinned DSPy output field type")
    candidate = value
    if isinstance(value, str):
        candidate = json_repair.loads(value)
        if candidate == "" and value != "":
            try:
                candidate = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                candidate = value
    if not isinstance(candidate, dict) or any(
        not isinstance(key, str) for key in candidate
    ):
        raise ValueError("invalid pinned DSPy ReAct tool arguments")
    return candidate


def _replay_dspy_chat_parse(
    completion: str,
    field_contract: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """Replay pinned ``ChatAdapter.parse`` for one harness signature."""

    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in completion.splitlines():
        match = _DSPY_FIELD_HEADER_PATTERN.match(line.strip())
        if match:
            header = match.group(1)
            remaining = line[match.end() :].strip()
            sections.append((header, [remaining] if remaining else []))
        else:
            sections[-1][1].append(line)
    normalized_sections = [
        (key, "\n".join(lines).strip()) for key, lines in sections
    ]
    expected = dict(field_contract)
    fields: dict[str, object] = {}
    for key, value in normalized_sections:
        if key not in fields and key in expected:
            try:
                fields[key] = _parse_dspy_value(value, expected[key])
            except Exception as error:
                raise _DspyAdapterParseFailure(completion) from error
    if fields.keys() != expected.keys():
        raise _DspyAdapterParseFailure(completion, fields)
    return fields


def _replay_dspy_json_parse(
    completion: str,
    field_contract: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """Replay pinned ``JSONAdapter.parse``, including repaired-object extraction."""

    fields = json_repair.loads(completion)
    if not isinstance(fields, dict):
        match = regex.search(_DSPY_JSON_OBJECT_PATTERN, completion, regex.DOTALL)
        if match:
            completion = match.group(0)
            fields = json_repair.loads(completion)
    if not isinstance(fields, dict):
        raise _DspyAdapterParseFailure(completion)
    expected = dict(field_contract)
    parsed = {key: value for key, value in fields.items() if key in expected}
    for key, value in parsed.items():
        parsed[key] = _parse_dspy_value(value, expected[key])
    if parsed.keys() != expected.keys():
        raise _DspyAdapterParseFailure(completion, parsed)
    return parsed


def _replay_dspy_adapter(
    adapter: str,
    selected_output: object,
    field_contract: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """Replay pinned ``Adapter._call_postprocess`` and its selected parser."""

    output = selected_output
    text = output.get("text") if isinstance(output, Mapping) else output
    if text:
        if not isinstance(text, str):
            raise StudyProtocolError("DSPy adapter selected non-text content")
        if adapter == "ChatAdapter":
            return _replay_dspy_chat_parse(text, field_contract)
        if adapter == "JSONAdapter":
            return _replay_dspy_json_parse(text, field_contract)
        raise StudyProtocolError("unknown DSPy adapter in extraction binding")
    if isinstance(output, Mapping) and output.get("tool_calls"):
        # None of the three frozen signatures exposes a ToolCalls output.  The
        # pinned adapter therefore provably rejects this branch before typed
        # parsing.  Its recorded AdapterParseError text remains an exact runtime
        # diagnostic, but LiteLLM SDK-object repr is not reconstructible from
        # retained JSON and is not treated as independent lineage evidence.
        raise _DspyAdapterParseFailure(
            "", lm_response_replayable=False
        )
    raise _DspyAdapterParseFailure(str(output))


def _dspy_completion_outputs(processed_response: Mapping[str, Any]) -> list[Any]:
    """Reconstruct pinned DSPy's normalized chat outputs from one response.

    The research harness uses ``model_type='chat'`` without log probabilities or
    citation-bearing providers.  Recomputing this small pinned transformation
    makes the retained normalized outputs independently checkable against the
    exact JSON-native LiteLLM response snapshot.
    """

    choices = processed_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise StudyProtocolError("DSPy provider response has no choices")
    outputs: list[Any] = []
    for index, choice in enumerate(choices):
        if (
            not isinstance(choice, Mapping)
            or type(choice.get("index")) is not int
            or choice.get("index") != index
        ):
            raise StudyProtocolError(
                f"DSPy provider response choice {index} has an invalid index"
            )
        message = choice.get("message")
        if not isinstance(message, Mapping) or "content" not in message:
            raise StudyProtocolError(
                f"DSPy provider response choice {index} has no chat message"
            )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise StudyProtocolError(
                f"DSPy provider response choice {index} has invalid content"
            )
        output: dict[str, Any] = {"text": content}
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise StudyProtocolError(
                f"DSPy provider response choice {index} has invalid reasoning"
            )
        if reasoning:
            output["reasoning_content"] = reasoning
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise StudyProtocolError(
                f"DSPy provider response choice {index} has invalid tool calls"
            )
        if tool_calls:
            output["tool_calls"] = tool_calls
        outputs.append(output)
    if all(set(output) == {"text"} for output in outputs):
        return [output["text"] for output in outputs]
    return outputs


def validate_dspy_provider_call(
    record: Mapping[str, Any],
    index: int,
    *,
    schema_version: int,
) -> None:
    """Validate one historical or current DSPy provider-call record exactly."""

    if type(index) is not int or index < 0:
        raise StudyProtocolError("DSPy provider-call expectations are invalid")
    if schema_version not in DSPY_REQUEST_AUDIT_SCHEMA_VERSIONS:
        raise StudyProtocolError("unknown DSPy request-audit schema")
    expected_keys = (
        _DSPY_V3_PROVIDER_CALL_KEYS
        if schema_version == DSPY_REQUEST_AUDIT_SCHEMA_VERSION
        else _DSPY_V2_PROVIDER_CALL_KEYS
    )
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise StudyProtocolError(f"DSPy provider call {index} has an invalid shape")
    values = [
        record.get(field)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    ]
    usage = record.get("provider_usage")
    if (
        record.get("call") != index
        or type(record.get("call")) is not int
        or not isinstance(record.get("response_model"), str)
        or not record["response_model"]
        or not isinstance(record.get("response_id"), str)
        or not record["response_id"]
        or (
            record.get("system_fingerprint") is not None
            and (
                not isinstance(record["system_fingerprint"], str)
                or not record["system_fingerprint"]
            )
        )
        or not _valid_sha256(record.get("request_messages_sha256"))
        or not _valid_sha256(record.get("outputs_sha256"))
        or record.get("request_messages_sha256") == sha256_json(None)
        or record.get("outputs_sha256") == sha256_json(None)
        or not isinstance(usage, Mapping)
        or any(type(value) is not int or value < 0 for value in values)
        or values[2] != values[0] + values[1]
        or any(
            usage.get(field) != value
            for field, value in zip(
                ("prompt_tokens", "completion_tokens", "total_tokens"),
                values,
                strict=True,
            )
        )
    ):
        raise StudyProtocolError(f"DSPy provider call {index} is incomplete")
    if schema_version == DSPY_REQUEST_AUDIT_LEGACY_SCHEMA_VERSION:
        return

    response = record.get("processed_response")
    outputs = record.get("outputs")
    finish_reasons = record.get("finish_reasons")
    if (
        not isinstance(response, Mapping)
        or set(response)
        != {"id", "model", "system_fingerprint", "choices", "usage"}
        or not isinstance(outputs, list)
    ):
        raise StudyProtocolError(
            f"DSPy provider call {index} has no retained response or outputs"
        )
    try:
        response_bytes = canonical_json_bytes(response)
        output_bytes = canonical_json_bytes(outputs)
    except (TypeError, ValueError) as error:
        raise StudyProtocolError(
            f"DSPy provider call {index} retention is not canonical JSON-native"
        ) from error
    if (
        type(record.get("processed_response_canonical_bytes")) is not int
        or record["processed_response_canonical_bytes"] != len(response_bytes)
        or record.get("processed_response_sha256") != sha256_bytes(response_bytes)
        or type(record.get("outputs_canonical_bytes")) is not int
        or record["outputs_canonical_bytes"] != len(output_bytes)
        or record.get("outputs_sha256") != sha256_bytes(output_bytes)
        or response.get("id") != record["response_id"]
        or response.get("model") != record["response_model"]
        or response.get("system_fingerprint") != record["system_fingerprint"]
        or not _same_json(response.get("usage"), usage)
    ):
        raise StudyProtocolError(
            f"DSPy provider call {index} retention linkage is inconsistent"
        )
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or len(choices) != len(outputs)
        or not isinstance(finish_reasons, list)
        or len(finish_reasons) != len(choices)
    ):
        raise StudyProtocolError(
            f"DSPy provider call {index} choice retention is incomplete"
        )
    observed_finish_reasons = []
    for choice_index, choice in enumerate(choices):
        if (
            not isinstance(choice, Mapping)
            or set(choice) != {"index", "finish_reason", "message"}
            or not isinstance(choice.get("message"), Mapping)
            or set(choice["message"])
            != {"content", "reasoning_content", "tool_calls"}
        ):
            raise StudyProtocolError(
                f"DSPy provider call {index} choice {choice_index} has an invalid shape"
            )
        finish_reason = (
            choice.get("finish_reason")
        )
        if not isinstance(finish_reason, str) or not finish_reason:
            raise StudyProtocolError(
                f"DSPy provider call {index} choice {choice_index} has no finish reason"
            )
        observed_finish_reasons.append(finish_reason)
    if finish_reasons != observed_finish_reasons:
        raise StudyProtocolError(
            f"DSPy provider call {index} finish reasons are inconsistent"
        )
    if not _same_json(outputs, _dspy_completion_outputs(response)):
        raise StudyProtocolError(
            f"DSPy provider call {index} outputs differ from its provider response"
        )


def _selected_dspy_output(record: Mapping[str, Any], choice: int) -> Any:
    outputs = record.get("outputs")
    if (
        not isinstance(outputs, list)
        or type(choice) is not int
        or choice < 0
        or choice >= len(outputs)
    ):
        raise StudyProtocolError("DSPy extraction binding selects no retained output")
    return outputs[choice]


def validate_dspy_final_binding(episode: Mapping[str, Any]) -> None:
    """Bind the final answer/non-answer to its exact final provider response.

    Every ledger response is retained, validated, and usage-accounted by the
    provider-call validator.  This narrower binding replays only the terminal
    adapter invocation; it does not partition earlier non-direct calls into a
    complete sequence of adapter invocation groups.
    """

    if (
        not isinstance(episode, Mapping)
        or episode.get("dspy_request_audit_schema")
        != DSPY_REQUEST_AUDIT_SCHEMA_VERSION
    ):
        raise StudyProtocolError("current DSPy final binding has no current schema")
    status = episode.get("status")
    binding_name = "answer_audit" if status == "ok" else "non_answer_audit"
    opposite_binding_name = (
        "non_answer_audit" if status == "ok" else "answer_audit"
    )
    binding = episode.get(binding_name)
    ledger = episode.get("usage_ledger")
    if (
        status not in {"ok", "no_answer"}
        or opposite_binding_name in episode
        or not isinstance(binding, Mapping)
    ):
        raise StudyProtocolError("current DSPy episode has no final extraction binding")
    kind = binding.get("kind")
    expected_keys = (
        _DSPY_V3_PARSED_BINDING_KEYS
        if kind in {"parsed_answer", "parsed_empty_answer"}
        else _DSPY_V3_PARSE_FAILURE_BINDING_KEYS
        if kind == "adapter_parse_failure"
        else frozenset()
    )
    provider_calls = binding.get("provider_calls")
    provider_call = binding.get("provider_call")
    choice = binding.get("choice")
    if (
        set(binding) != expected_keys
        or type(binding.get("schema_version")) is not int
        or binding.get("schema_version") != DSPY_REQUEST_AUDIT_SCHEMA_VERSION
        or not isinstance(ledger, list)
        or not ledger
        or type(provider_call) is not int
        or provider_call != len(ledger) - 1
        or not isinstance(provider_calls, list)
        or not provider_calls
        or any(type(call) is not int for call in provider_calls)
        or provider_calls != list(range(provider_calls[0], provider_call + 1))
        or provider_calls[0] < 0
        or (binding.get("stage") == "direct" and provider_calls[0] != 0)
        or type(choice) is not int
        or choice != 0
        or binding.get("stage") not in {"direct", "react", "extract"}
        or type(binding.get("fallback_used")) is not bool
        or not isinstance(binding.get("adapter"), str)
        or not binding["adapter"]
    ):
        raise StudyProtocolError("current DSPy extraction binding is malformed")
    fallback_used = binding["fallback_used"]
    # Direct/extract may add one failed structured-output JSON call before the
    # JSON-mode retry. ReAct's open-ended args mapping selects JSON mode
    # immediately, so its fallback group is exactly Chat + one JSON response.
    if (
        fallback_used != (len(provider_calls) > 1)
        or (
            fallback_used
            and binding["stage"] == "react"
            and len(provider_calls) != 2
        )
        or (
            fallback_used
            and binding["stage"] != "react"
            and len(provider_calls) not in {2, 3}
        )
        or (fallback_used and binding["adapter"] != "JSONAdapter")
        or (not fallback_used and binding["adapter"] != "ChatAdapter")
        or (
            kind == "adapter_parse_failure"
            and (not fallback_used or binding["adapter"] != "JSONAdapter")
        )
    ):
        raise StudyProtocolError("current DSPy adapter fallback binding is inconsistent")
    budget = episode.get("budget")
    stage = binding["stage"]
    if (
        (budget == "direct" and stage != "direct")
        or (budget != "direct" and stage == "direct")
        or (kind in {"parsed_answer", "parsed_empty_answer"} and stage == "react")
    ):
        raise StudyProtocolError("current DSPy extraction stage violates its budget")
    record = ledger[provider_call]
    selected = _selected_dspy_output(record, choice)
    selected_bytes = canonical_json_bytes(selected)
    if (
        binding.get("outputs_sha256") != record.get("outputs_sha256")
        or type(binding.get("selected_output_canonical_bytes")) is not int
        or binding["selected_output_canonical_bytes"] != len(selected_bytes)
        or binding.get("selected_output_sha256") != sha256_bytes(selected_bytes)
    ):
        raise StudyProtocolError("current DSPy binding differs from its selected output")
    answer = episode.get("answer")
    if not isinstance(answer, str):
        raise StudyProtocolError("current DSPy bound answer is not text")
    response = record.get("processed_response")
    if not isinstance(response, Mapping):
        raise StudyProtocolError("current DSPy binding has no provider response")
    replay_outputs = _dspy_completion_outputs(response)
    replay_selected = replay_outputs[choice]
    field_contract = _dspy_output_field_contract(stage)
    if fallback_used:
        for offset, prior_call in enumerate(provider_calls[:-1]):
            prior_record = ledger[prior_call]
            prior_response = (
                prior_record.get("processed_response")
                if isinstance(prior_record, Mapping)
                else None
            )
            if not isinstance(prior_response, Mapping):
                raise StudyProtocolError(
                    "current DSPy fallback has no prior provider response"
                )
            prior_outputs = _dspy_completion_outputs(prior_response)
            if choice >= len(prior_outputs):
                raise StudyProtocolError(
                    "current DSPy fallback prior call has no selected output"
                )
            prior_adapter = "ChatAdapter" if offset == 0 else "JSONAdapter"
            try:
                _replay_dspy_adapter(
                    prior_adapter, prior_outputs[choice], field_contract
                )
            except _DspyAdapterParseFailure:
                continue
            except StudyProtocolError:
                raise
            except Exception:
                if prior_adapter == "JSONAdapter":
                    # JSONAdapter retries structured-output calls after any
                    # parser/type exception, not only AdapterParseError.
                    continue
                raise StudyProtocolError(
                    "current DSPy primary Chat fallback is not replayable"
                )
            raise StudyProtocolError(
                "current DSPy fallback bypasses an earlier parsed provider output"
            )
    replayed: dict[str, object] | None = None
    replay_failure: _DspyAdapterParseFailure | None = None
    try:
        replayed = _replay_dspy_adapter(
            binding["adapter"], replay_selected, field_contract
        )
    except _DspyAdapterParseFailure as error:
        replay_failure = error
    except Exception as error:
        raise StudyProtocolError(
            "current DSPy adapter outcome cannot be independently replayed"
        ) from error
    if kind in {"parsed_answer", "parsed_empty_answer"}:
        parsed_outputs = binding.get("parsed_outputs")
        if not isinstance(parsed_outputs, list):
            raise StudyProtocolError("current DSPy binding has no parsed outputs")
        parsed_bytes = canonical_json_bytes(parsed_outputs)
        if (
            type(binding.get("parsed_outputs_canonical_bytes")) is not int
            or binding["parsed_outputs_canonical_bytes"] != len(parsed_bytes)
            or binding.get("parsed_outputs_sha256") != sha256_bytes(parsed_bytes)
            or len(parsed_outputs) != len(record.get("outputs", []))
            or not parsed_outputs
            or replay_failure is not None
            or replayed is None
            or not _same_json(parsed_outputs, [replayed])
            or not isinstance(parsed_outputs[choice], Mapping)
            or parsed_outputs[choice].get("answer") != answer
            or binding.get("answer_sha256") != sha256_text(answer)
            or (kind == "parsed_answer" and (status != "ok" or not answer.strip()))
            or (
                kind == "parsed_empty_answer"
                and (status != "no_answer" or answer.strip())
            )
        ):
            raise StudyProtocolError("current DSPy parsed-answer binding is inconsistent")
    else:
        if status != "no_answer" or answer.strip():
            raise StudyProtocolError("current DSPy parse failure is not a non-answer")
        adapter_lm_response = binding.get("adapter_lm_response")
        if not isinstance(adapter_lm_response, str):
            raise StudyProtocolError("current DSPy parse failure has no LM response")
        parsed_result = binding.get("parsed_result")
        parsed_result_bytes = canonical_json_bytes(parsed_result)
        replay_mode = (
            "exact"
            if replay_failure is not None and replay_failure.lm_response_replayable
            else "sdk-repr-unreplayable-tool-call-only"
        )
        if (
            type(binding.get("adapter_lm_response_bytes")) is not int
            or binding["adapter_lm_response_bytes"]
            != len(adapter_lm_response.encode("utf-8"))
            or binding.get("adapter_lm_response_sha256")
            != sha256_text(adapter_lm_response)
            or type(binding.get("parsed_result_canonical_bytes")) is not int
            or binding["parsed_result_canonical_bytes"] != len(parsed_result_bytes)
            or binding.get("parsed_result_sha256")
            != sha256_bytes(parsed_result_bytes)
            or replay_failure is None
            or binding.get("adapter_lm_response_replay") != replay_mode
            or (
                replay_mode == "exact"
                and adapter_lm_response != replay_failure.lm_response
            )
            or not _same_json(parsed_result, replay_failure.parsed_result)
        ):
            raise StudyProtocolError("current DSPy parse-failure binding is inconsistent")


def _canonical_task_manifest(
    value: bytes | Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if isinstance(value, bytes):
        parsed = strict_json_loads(value, label="study task manifest")
        if not isinstance(parsed, dict):
            raise StudyProtocolError("study task manifest is not a JSON object")
        if canonical_json_bytes(parsed) != value:
            raise StudyProtocolError("study task manifest is not canonically encoded")
        return parsed, value
    if not isinstance(value, Mapping):
        raise StudyProtocolError("study task manifest is not a JSON object")
    parsed = dict(value)
    try:
        encoded = canonical_json_bytes(parsed)
    except (TypeError, ValueError) as error:
        raise StudyProtocolError("study task manifest is not canonical JSON") from error
    return parsed, encoded


def _validate_common_task_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] <= 0
        or not isinstance(manifest.get("study_id"), str)
        or not manifest["study_id"]
        or not isinstance(manifest.get("task"), str)
        or not manifest["task"]
        or type(manifest.get("master_seed")) is not int
        or not isinstance(manifest.get("model"), str)
        or not manifest["model"]
        or not isinstance(manifest.get("model_revision"), str)
        or not manifest["model_revision"]
        or not isinstance(manifest.get("sampling"), dict)
        or not manifest["sampling"]
        or not _same_json(manifest["sampling"], REACT_SAMPLING)
        or not isinstance(manifest.get("corpus_commit"), str)
        or not manifest["corpus_commit"]
        or not isinstance(manifest.get("corpus"), dict)
        or manifest["corpus"].get("commit") != manifest["corpus_commit"]
        or not isinstance(manifest.get("source"), dict)
        or not manifest["source"]
        or not isinstance(manifest.get("environment"), dict)
        or not manifest["environment"]
        or not isinstance(config, dict)
    ):
        raise StudyProtocolError("study task manifest has an incomplete common contract")
    transport = manifest.get("server_transport")
    readiness = manifest.get("provenance_readiness")
    try:
        environment_server_count = int(manifest["environment"]["server_count"])
    except (KeyError, TypeError, ValueError):
        environment_server_count = None
    if (
        not isinstance(manifest.get("environment_contract"), dict)
        or not manifest["environment_contract"]
        or not isinstance(transport, dict)
        or set(transport)
        != {"scope", "protocol", "server_count", "assignment"}
        or transport.get("scope") != "loopback"
        or transport.get("protocol") != "openai-compatible-http"
        or type(transport.get("server_count")) is not int
        or transport["server_count"] <= 0
        or transport["server_count"] != environment_server_count
        or transport.get("assignment") not in {
            "stable_seed(master_seed, owner_id, server) modulo server_count",
            "stable_seed(master_seed, stochastic_namespace, server) modulo server_count",
        }
        or not isinstance(readiness, dict)
        or set(readiness)
        != {
            "corpus_pinned_clean",
            "source_pinned_clean",
            "environment_complete",
            "model_revision_pinned",
            "server_count_matches_environment",
        }
        or any(type(value) is not bool for value in readiness.values())
        or type(manifest.get("automated_provenance_ready")) is not bool
        or manifest["automated_provenance_ready"] != all(readiness.values())
    ):
        raise StudyProtocolError("study task provenance or transport contract is invalid")
    return config


def derive_protocol_summary(
    task_manifest: bytes | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a recognized task manifest and derive its content identity.

    Accepting either canonical bytes or a mapping lets constructors validate
    in-memory contracts while provenance consumers bind the exact archived
    bytes. The returned shape is deliberately method-independent so reports
    cannot accidentally collapse semantic and deterministic study arms.
    """

    manifest, encoded = _canonical_task_manifest(task_manifest)
    config = _validate_common_task_manifest(manifest)
    manifest_type = manifest.get("manifest_type")

    if manifest_type == SEMANTIC_SELFQUIZ_TASK_MANIFEST_TYPE:
        if set(manifest) != _COMMON_TASK_KEYS | {"method", "human_audit_protocol"}:
            raise StudyProtocolError("semantic task manifest has an unknown schema")
        if set(config) != _SEMANTIC_CONFIG_KEYS:
            raise StudyProtocolError("semantic task config has an unknown schema")
        method = manifest.get("method")
        if method != SEMANTIC_SELFQUIZ_METHOD:
            raise StudyProtocolError("semantic task manifest has the wrong method")
        attempt_access = config.get("attempt_access")
        try:
            expected_attempt_protocol = semantic_attempt_protocol(attempt_access)
        except ValueError as error:
            raise StudyProtocolError(str(error)) from error
        task = manifest.get("task")
        try:
            expected_syllabus = semantic_chapter_syllabus(task)
        except ValueError as error:
            raise StudyProtocolError(str(error)) from error
        expected_chapters_per_round = (
            1 if config.get("smoke") is True else min(4, len(expected_syllabus))
        )
        if (
            manifest.get("schema_version") != 6
            or not _same_json(
                config.get("attempt_protocol"), expected_attempt_protocol
            )
            or config.get("adapter") != SEMANTIC_SELFQUIZ_ADAPTER
            or config.get("adapter_policy") != SEMANTIC_SELFQUIZ_ADAPTER_POLICY
            or config.get("citation_policy")
            != SEMANTIC_SELFQUIZ_CITATION_POLICY
            or config.get("unresolved_policy")
            != SEMANTIC_SELFQUIZ_UNRESOLVED_POLICY
            or manifest["server_transport"].get("assignment")
            != "stable_seed(master_seed, stochastic_namespace, server) modulo server_count"
        ):
            raise StudyProtocolError(
                "semantic task method contract is invalid"
            )
        focus = None
        semantic_integer_fields = (
            "chapters_per_round",
            "final_round",
            "questions_per_chapter",
            "quiz_max_iters",
            "derive_max_iters",
            "train_ensemble",
            "dev_ensemble",
            "concurrency",
            "provider_retries",
        )
        if (
            any(type(config.get(field)) is not int for field in semantic_integer_fields)
            or (
                config["smoke"]
                and (
                    config["chapters_per_round"] != 1
                    or config["questions_per_chapter"] != 3
                )
            )
            or (
                not config["smoke"]
                and (
                    config["chapters_per_round"] != expected_chapters_per_round
                    or config["questions_per_chapter"] != 5
                )
            )
            or config["quiz_max_iters"] != 15
            or config["final_round"] != SEMANTIC_FINAL_ROUND
            or config.get("chapter_syllabus")
            != list(expected_syllabus)
            or config["derive_max_iters"] != 15
            or config["train_ensemble"] != 2
            or config["dev_ensemble"] != 2
            or config["concurrency"] <= 0
            or config["provider_retries"] != 0
            or type(config.get("smoke")) is not bool
            or config.get("retest_fraction") != 0.2
            or config.get("freshness_near_jaccard") != 0.8
            or config.get("max_freshness_near_rate") != 0.1
        ):
            raise StudyProtocolError("semantic task config differs from its protocol")
        question_mode = "semantic"
        resolver_hash = None
        question_bank_hash = None
        question_bank_artifact_hash = None
    elif manifest_type == STATIC_GRAPH_TASK_MANIFEST_TYPE:
        if set(manifest) != _COMMON_TASK_KEYS | {
            "round",
            "source_root",
            "resolver_contract",
            "resolver_contract_sha256",
            "question_bank_sha256",
            "question_bank_artifact_sha256",
        }:
            raise StudyProtocolError("static-graph task manifest has an unknown schema")
        if set(config) != _STATIC_GRAPH_CONFIG_KEYS:
            raise StudyProtocolError("static-graph task config has an unknown schema")
        method = config.get("method")
        if method != STATIC_GRAPH_METHOD:
            raise StudyProtocolError("static-graph task manifest has the wrong method")
        if (
            not _same_json(config.get("attempt_protocol"), _EXPECTED_ATTEMPT_PROTOCOL)
            or manifest["server_transport"].get("assignment")
            != "stable_seed(master_seed, owner_id, server) modulo server_count"
        ):
            raise StudyProtocolError(
                "static-graph attempt or server-assignment contract is invalid"
            )
        source_root = manifest.get("source_root")
        resolver = manifest.get("resolver_contract")
        resolver_hash = manifest.get("resolver_contract_sha256")
        question_bank_hash = manifest.get("question_bank_sha256")
        question_bank_artifact_hash = manifest.get("question_bank_artifact_sha256")
        smoke = config.get("smoke")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("task") != "dspy"
            or manifest.get("round") != 1
            or source_root != "dspy"
            or type(smoke) is not bool
            or type(config.get("concurrency")) is not int
            or config["concurrency"] <= 0
            or config.get("train_input_note_sha256") != sha256_text("")
            or type(config.get("train_question_count")) is not int
            or type(config.get("dev_question_count")) is not int
            or config.get("provider_retries") != 0
            or config.get("read_max_lines") != DSPY_READ_MAX_LINES
            or (
                smoke
                and (
                    config["train_question_count"] != 1
                    or config["dev_question_count"] != 0
                    or config.get("dev_holdout_targets") != []
                )
            )
            or (
                not smoke
                and (
                    config["train_question_count"] != 16
                    or config["dev_question_count"] != 4
                    or config.get("dev_holdout_targets")
                    != _STATIC_GRAPH_DEV_HOLDOUT_TARGETS
                )
            )
        ):
            raise StudyProtocolError("static-graph task config differs from its protocol")
        if (
            not isinstance(resolver, dict)
            or resolver_hash != sha256_json(resolver)
            or not _valid_sha256(question_bank_hash)
            or resolver.get("question_bank_sha256") != question_bank_hash
            or not _valid_sha256(question_bank_artifact_hash)
        ):
            raise StudyProtocolError(
                "static-graph resolver or question-bank identity is invalid"
            )
        question_mode = "static-call-neighborhood"
        focus = None
        attempt_access = "react-corpus"
    else:
        raise StudyProtocolError(
            f"unknown study task manifest type: {manifest_type!r}"
        )

    return {
        "schema_version": PROTOCOL_SUMMARY_SCHEMA_VERSION,
        "task_manifest_sha256": sha256_bytes(encoded),
        "method": method,
        "question_mode": question_mode,
        "focus": focus,
        "attempt_access": attempt_access,
        "attempt_protocol_sha256": sha256_json(config["attempt_protocol"]),
        "resolver_contract_sha256": resolver_hash,
        "question_bank_sha256": question_bank_hash,
        "question_bank_artifact_sha256": question_bank_artifact_hash,
    }


def validate_task_manifest_expectations(
    task_manifest: bytes | Mapping[str, Any],
    *,
    expected_task: str | None = None,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_sampling: Mapping[str, Any] | None = None,
    expected_corpus_commit: str | None = None,
    expected_corpus: Mapping[str, Any] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    expected_environment: Mapping[str, Any] | None = None,
    expected_environment_contract: Mapping[str, Any] | None = None,
    environments_compatible: Callable[[object, object], bool] | None = None,
) -> dict[str, Any]:
    """Derive a summary and bind the task to an evaluation specification."""

    manifest, _ = _canonical_task_manifest(task_manifest)
    summary = derive_protocol_summary(task_manifest)
    exact_expectations = (
        ("task", expected_task),
        ("model", expected_model),
        ("model_revision", expected_model_revision),
        ("corpus_commit", expected_corpus_commit),
    )
    for field, expected in exact_expectations:
        if expected is not None and manifest.get(field) != expected:
            raise StudyProtocolError(
                f"study task {field} does not match the evaluation specification"
            )
    json_expectations = (
        ("sampling", expected_sampling),
        ("corpus", expected_corpus),
        ("source", expected_source),
        ("environment_contract", expected_environment_contract),
    )
    for field, expected in json_expectations:
        if expected is not None and not _same_json(manifest.get(field), expected):
            raise StudyProtocolError(
                f"study task {field} does not match the evaluation specification"
            )
    if expected_environment is not None:
        compatible = environments_compatible or _same_json
        try:
            environment_matches = compatible(
                manifest.get("environment"), expected_environment
            )
        except (OSError, TypeError, ValueError):
            environment_matches = False
        if environment_matches is not True:
            raise StudyProtocolError(
                "study task environment is incompatible with the evaluation environment"
            )
    return summary


def validate_note_protocol_binding(
    note_manifest: Mapping[str, Any],
    protocol_summary: Mapping[str, Any],
    *,
    allow_human_audited: bool = False,
) -> dict[str, Any]:
    """Require a construction note to disclose the exact derived protocol."""

    if not isinstance(note_manifest, Mapping) or not isinstance(
        protocol_summary, Mapping
    ):
        raise StudyProtocolError("note protocol binding is not a JSON object")
    summary = dict(protocol_summary)
    if set(summary) != _PROTOCOL_SUMMARY_KEYS:
        raise StudyProtocolError("derived protocol summary has an invalid schema")
    method = summary.get("method")
    expected_note_type = {
        SEMANTIC_SELFQUIZ_METHOD: SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE,
        STATIC_GRAPH_METHOD: STATIC_GRAPH_NOTE_MANIFEST_TYPE,
    }.get(method)
    allowed_types = {expected_note_type}
    if allow_human_audited:
        allowed_types.add(HUMAN_AUDITED_NOTE_MANIFEST_TYPE)
    if (
        expected_note_type is None
        or note_manifest.get("method") != method
        or note_manifest.get("manifest_type") not in allowed_types
        or not _same_json(note_manifest.get("protocol_summary"), summary)
    ):
        raise StudyProtocolError(
            "note construction summary does not match its derived study protocol"
        )
    return summary


def validate_construction_protocol(
    note_manifest: Mapping[str, Any],
    construction_dependencies: Mapping[str, bytes],
    *,
    expected_task: str | None = None,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_sampling: Mapping[str, Any] | None = None,
    expected_corpus_commit: str | None = None,
    expected_corpus: Mapping[str, Any] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    expected_environment: Mapping[str, Any] | None = None,
    expected_environment_contract: Mapping[str, Any] | None = None,
    environments_compatible: Callable[[object, object], bool] | None = None,
    allow_human_audited: bool = False,
    require_final_semantic: bool = False,
) -> dict[str, Any]:
    """Validate the canonical task dependency and its note-level summary."""

    if not isinstance(note_manifest, Mapping) or not isinstance(
        construction_dependencies, Mapping
    ):
        raise StudyProtocolError("construction protocol inputs must be mappings")
    task_bytes = construction_dependencies.get("manifest.json")
    inventory = note_manifest.get("construction_artifacts")
    task_record = inventory.get("manifest.json") if isinstance(inventory, dict) else None
    if (
        not isinstance(task_bytes, bytes)
        or not isinstance(task_record, dict)
        or set(task_record) != {"sha256", "bytes"}
        or task_record.get("sha256") != sha256_bytes(task_bytes)
        or task_record.get("bytes") != len(task_bytes)
    ):
        raise StudyProtocolError(
            "construction inventory does not bind one exact task manifest"
        )
    if type(require_final_semantic) is not bool:
        raise StudyProtocolError("final-semantic requirement must be boolean")
    task_manifest, _ = _canonical_task_manifest(task_bytes)
    summary = validate_task_manifest_expectations(
        task_bytes,
        expected_task=expected_task,
        expected_model=expected_model,
        expected_model_revision=expected_model_revision,
        expected_sampling=expected_sampling,
        expected_corpus_commit=expected_corpus_commit,
        expected_corpus=expected_corpus,
        expected_source=expected_source,
        expected_environment=expected_environment,
        expected_environment_contract=expected_environment_contract,
        environments_compatible=environments_compatible,
    )
    summary = validate_note_protocol_binding(
        note_manifest,
        summary,
        allow_human_audited=allow_human_audited,
    )
    if require_final_semantic and summary.get("method") == SEMANTIC_SELFQUIZ_METHOD:
        config = task_manifest.get("config")
        expected_round = (
            1
            if isinstance(config, Mapping) and config.get("smoke") is True
            else SEMANTIC_FINAL_ROUND
        )
        if type(note_manifest.get("round")) is not int \
                or note_manifest.get("round") != expected_round:
            raise StudyProtocolError(
                "semantic evaluation requires the final construction round"
            )
    return summary


def _archive_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StudyProtocolError(f"study archive has an unsafe {label} path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise StudyProtocolError(f"study archive has an unsafe {label} path")
    return value


def _archive_json(
    dependencies: Mapping[str, bytes], path: str, *, label: str
) -> dict[str, Any]:
    data = dependencies.get(path)
    if not isinstance(data, bytes):
        raise StudyProtocolError(f"study archive is missing {label}: {path}")
    try:
        value = strict_json_loads(data, label=label)
    except ValueError as error:
        raise StudyProtocolError(f"study archive has invalid {label}: {path}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise StudyProtocolError(f"study archive has noncanonical {label}: {path}")
    return value


def _archive_jsonl(
    dependencies: Mapping[str, bytes], path: str, *, label: str
) -> list[dict[str, Any]]:
    data = dependencies.get(path)
    if not isinstance(data, bytes):
        raise StudyProtocolError(f"study archive is missing {label}: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StudyProtocolError(f"study archive has invalid UTF-8 {label}: {path}") from error
    if text and not text.endswith("\n"):
        raise StudyProtocolError(
            f"study archive has unterminated {label}: {path}"
        )
    records: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise StudyProtocolError(
                f"study archive has a blank {label} record: {path}:{index}"
            )
        try:
            value = strict_json_loads(line, label=f"{label}:{index}")
        except ValueError as error:
            raise StudyProtocolError(
                f"study archive has invalid {label} record: {path}:{index}"
            ) from error
        if not isinstance(value, dict):
            raise StudyProtocolError(
                f"study archive {label} record is not an object: {path}:{index}"
            )
        records.append(value)
    return records


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, object]:
    known: list[tuple[int, int, int]] = []
    for record in records:
        values = tuple(record.get(field) for field in (
            "prompt_tokens", "completion_tokens", "total_tokens"
        ))
        if (
            record.get("usage_reported") is True
            and all(type(value) is int and value >= 0 for value in values)
            and values[2] == values[0] + values[1]
        ):
            known.append(values)  # type: ignore[arg-type]
    complete = len(known) == len(records)
    prompt = sum(value[0] for value in known)
    generated = sum(value[1] for value in known)
    total = sum(value[2] for value in known)
    return {
        "status": "complete" if complete else "incomplete",
        "calls": len(records),
        "reported_calls": len(known),
        "prompt_tokens": prompt if complete else None,
        "generated_tokens": generated if complete else None,
        "total_tokens": total if complete else None,
        "known_prompt_tokens": prompt,
        "known_generated_tokens": generated,
        "known_total_tokens": total,
    }


def _usage_by_phase(records: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("phase", "")), []).append(record)
    return {phase: _usage_totals(grouped[phase]) for phase in sorted(grouped)}


def _record_calls(record: object) -> list[dict[str, Any]] | None:
    if not isinstance(record, dict) or not isinstance(record.get("calls"), list):
        return None
    calls = record["calls"]
    if any(not isinstance(call, dict) for call in calls):
        return None
    return calls


def _snapshot_path(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    snapshot = record.get("snapshot")
    if not isinstance(snapshot, str):
        return None
    try:
        return _archive_path(snapshot, label="environment snapshot")
    except StudyProtocolError:
        return None


def _semantic_checks_completed(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(check, Mapping) and check.get("status") == "ok"
        for check in value
    )


def _semantic_reference_phases_completed(record: Mapping[str, Any]) -> bool:
    """Reject hidden provider errors while allowing evidentiary abstentions."""

    derivations = record.get("derivations")
    checks = record.get("reference_consensus")
    if not isinstance(derivations, list) or not derivations \
            or not _semantic_checks_completed(checks):
        return False
    for derivation in derivations:
        if not isinstance(derivation, Mapping) \
                or derivation.get("status") not in {"ok", "invalid"}:
            return False
        citation = derivation.get("citation_pass")
        support = derivation.get("reference_support")
        if derivation["status"] == "ok" and (
            not isinstance(citation, Mapping)
            or citation.get("status") != "ok"
            or not isinstance(support, Mapping)
            or support.get("status") != "ok"
        ):
            return False
        if derivation["status"] == "invalid":
            if isinstance(citation, Mapping) and citation.get("status") == "error":
                return False
            if isinstance(support, Mapping) and support.get("status") == "error":
                return False
    return True


def semantic_training_item_terminal(record: object) -> bool:
    """Return whether one train/retest item reached an error-free v4 endpoint."""

    if (
        not isinstance(record, Mapping)
        or record.get("status") != "ok"
        or record.get("verdict")
        not in {"correct", "partial", "wrong", "unresolved"}
        or not isinstance(record.get("attempt"), Mapping)
        or record["attempt"].get("status") != "ok"
        or not _semantic_reference_phases_completed(record)
    ):
        return False
    adjudications = record.get("adjudications")
    if adjudications is not None and not _semantic_checks_completed(adjudications):
        return False
    bounced = record.get("entry_bounced")
    if isinstance(bounced, Mapping):
        reasons = bounced.get("reasons")
        support_checks = bounced.get("support_checks")
        if (
            not isinstance(reasons, list)
            or any(
                isinstance(reason, str) and reason.startswith("distill_error:")
                for reason in reasons
            )
            or (
                support_checks is not None
                and not _semantic_checks_completed(support_checks)
            )
        ):
            return False
    return True


def semantic_dev_reference_terminal(record: object) -> bool:
    """Accept a resolved/error-free abstaining blind dev reference."""

    return (
        isinstance(record, Mapping)
        and record.get("status") in {"ok", "unresolved"}
        and _semantic_reference_phases_completed(record)
    )


def semantic_dev_exam_terminal(record: object) -> bool:
    """Return whether one paired dev exam resolved or inherited an abstention."""

    if not isinstance(record, Mapping):
        return False
    verdicts = record.get("verdicts")
    if not isinstance(verdicts, Mapping) or set(verdicts) != {"with_note", "bare"}:
        return False
    status = record.get("status")
    if status == "reference_unresolved":
        return verdicts == {"with_note": "unresolved", "bare": "unresolved"}
    attempts = record.get("attempts")
    adjudications = record.get("adjudications")
    return (
        status == "ok"
        and set(verdicts.values())
        <= {"correct", "partial", "wrong", "unresolved"}
        and isinstance(attempts, Mapping)
        and set(attempts) == {"with_note", "bare"}
        and all(
            isinstance(attempt, Mapping) and attempt.get("status") == "ok"
            for attempt in attempts.values()
        )
        and isinstance(adjudications, Mapping)
        and set(adjudications) == {"with_note", "bare"}
        and all(
            _semantic_checks_completed(checks)
            for checks in adjudications.values()
        )
    )


def semantic_adapter_audit_complete(records: list[dict[str, Any]]) -> bool:
    """Reconstruct the semantic adapter gate without importing its runtime."""

    grouped: dict[tuple[object, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        audit = record.get("adapter_audit") if isinstance(record, Mapping) else None
        if not isinstance(audit, Mapping):
            return False
        response_format = audit.get("response_format")
        if (
            audit.get("schema_version") != 1
            or audit.get("policy") != SEMANTIC_SELFQUIZ_ADAPTER_POLICY
            or audit.get("mode")
            not in {"chat-primary", "strict-json-schema-repair"}
            or audit.get("outcome") not in {"accepted", "rejected", "error"}
            or not isinstance(audit.get("output_fields"), list)
            or not audit["output_fields"]
            or not isinstance(audit.get("finish_reasons"), list)
            or not audit["finish_reasons"]
            or any(reason != "stop" for reason in audit["finish_reasons"])
            or audit.get("response_format_sha256")
            != (sha256_json(response_format) if response_format is not None else None)
            or audit.get("provider_outputs_sha256") != record.get("outputs_sha256")
            or sha256_json(audit.get("provider_outputs"))
            != record.get("outputs_sha256")
        ):
            return False
        if audit["mode"] == "chat-primary" and response_format is not None:
            return False
        if audit["mode"] == "strict-json-schema-repair" and (
            not isinstance(response_format, Mapping)
            or response_format.get("type") != "json_schema"
        ):
            return False
        if audit["outcome"] == "accepted":
            if not isinstance(audit.get("selected_outputs_sha256"), str):
                return False
        elif not isinstance(audit.get("error_sha256"), str):
            return False
        key = (
            record.get("owner_id"),
            record.get("phase"),
            record.get("seed"),
            audit.get("logical_call_start"),
        )
        grouped.setdefault(key, []).append(audit)
    for audits in grouped.values():
        if len(audits) == 1:
            audit = audits[0]
            if audit["mode"] != "chat-primary" \
                    or audit["outcome"] not in {"accepted", "error"}:
                return False
            continue
        if len(audits) != 2:
            return False
        primary = [audit for audit in audits if audit["mode"] == "chat-primary"]
        repair = [
            audit for audit in audits
            if audit["mode"] == "strict-json-schema-repair"
        ]
        if (
            len(primary) != 1
            or len(repair) != 1
            or primary[0]["outcome"] != "rejected"
            or repair[0]["outcome"] not in {"accepted", "error"}
        ):
            return False
    return bool(records)


def _validate_archive_envelope(
    note_manifest: Mapping[str, Any],
    construction_dependencies: Mapping[str, bytes],
    note_bytes: bytes,
    *,
    expected_task: str | None,
    expected_model: str | None,
    expected_model_revision: str | None,
    expected_sampling: Mapping[str, Any] | None,
    expected_corpus_commit: str | None,
    expected_corpus: Mapping[str, Any] | None,
    expected_source: Mapping[str, Any] | None,
    expected_environment: Mapping[str, Any] | None,
    expected_environment_contract: Mapping[str, Any] | None,
    environments_compatible: Callable[[object, object], bool] | None,
    require_final_semantic: bool,
    allow_smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not isinstance(note_manifest, Mapping) or not isinstance(
        construction_dependencies, Mapping
    ) or not isinstance(note_bytes, bytes):
        raise StudyProtocolError("study archive inputs have invalid types")
    manifest = dict(note_manifest)
    manifest_type = manifest.get("manifest_type")
    expected_keys = {
        SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE: SEMANTIC_NOTE_MANIFEST_KEYS,
        STATIC_GRAPH_NOTE_MANIFEST_TYPE: STATIC_GRAPH_NOTE_MANIFEST_KEYS,
    }.get(manifest_type)
    expected_readiness_keys = {
        SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE: SEMANTIC_READINESS_KEYS,
        STATIC_GRAPH_NOTE_MANIFEST_TYPE: STATIC_GRAPH_READINESS_KEYS,
    }.get(manifest_type)
    if expected_keys is None:
        raise StudyProtocolError("study note manifest has an unknown schema")
    inventory = manifest.get("construction_artifacts")
    if (
        not isinstance(inventory, dict)
        or not inventory
        or manifest.get("construction_artifacts_sha256") != sha256_json(inventory)
        or set(inventory) != set(construction_dependencies)
    ):
        raise StudyProtocolError(
            "study construction inventory is not complete and exact"
        )
    for raw_path, data in construction_dependencies.items():
        path = _archive_path(raw_path, label="construction dependency")
        record = inventory.get(path)
        if (
            not isinstance(data, bytes)
            or not isinstance(record, dict)
            or set(record) != {"sha256", "bytes"}
            or record.get("sha256") != sha256_bytes(data)
            or type(record.get("bytes")) is not int
            or record["bytes"] != len(data)
        ):
            raise StudyProtocolError(
                f"study construction dependency record drifted: {path}"
            )
    summary = validate_construction_protocol(
        manifest,
        construction_dependencies,
        expected_task=expected_task,
        expected_model=expected_model,
        expected_model_revision=expected_model_revision,
        expected_sampling=expected_sampling,
        expected_corpus_commit=expected_corpus_commit,
        expected_corpus=expected_corpus,
        expected_source=expected_source,
        expected_environment=expected_environment,
        expected_environment_contract=expected_environment_contract,
        environments_compatible=environments_compatible,
        require_final_semantic=require_final_semantic,
    )
    if set(manifest) != expected_keys:
        raise StudyProtocolError("study note manifest has an unknown schema")
    task = _archive_json(
        construction_dependencies, "manifest.json", label="study task manifest"
    )
    config = task["config"]
    smoke = config.get("smoke")
    if type(smoke) is not bool or smoke and not allow_smoke:
        raise StudyProtocolError("smoke study notes are not accepted by this consumer")
    readiness = manifest.get("automated_readiness")
    treatment_readiness = manifest.get("automated_treatment_readiness")
    treatment_contract_valid = (
        manifest_type != SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE
        or (
            isinstance(treatment_readiness, dict)
            and set(treatment_readiness) == SEMANTIC_TREATMENT_READINESS_KEYS
            and all(type(value) is bool for value in treatment_readiness.values())
            and treatment_readiness.get("non_smoke") is (not smoke)
            and manifest.get("automated_treatment_ready")
            is all(treatment_readiness.values())
        )
    )
    if (
        not isinstance(readiness, dict)
        or set(readiness) != expected_readiness_keys
        or any(type(value) is not bool for value in readiness.values())
        or readiness.get("non_smoke") is not (not smoke)
        or manifest.get("automated_claim_ready") is not all(readiness.values())
        or not treatment_contract_valid
        or manifest.get("claim_ready") is not False
        or manifest.get("publication_claim_ready") is not False
        or manifest.get("confirmatory_claim_ready") is not False
        or manifest.get("study_id") != task.get("study_id")
        or manifest.get("task") != task.get("task")
        or manifest.get("corpus_commit") != task.get("corpus_commit")
    ):
        raise StudyProtocolError("study note readiness or identity contract is invalid")
    expected_schema = 6 if manifest_type == SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE else 1
    expected_human = (
        {
            "required": True,
            "status": "not_performed",
            "protocol": "pre-registered blinded verdict and evidence audit",
        }
        if manifest_type == SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE
        else {"required_for_publication": True, "status": "not_performed"}
    )
    note_hash = sha256_bytes(note_bytes)
    round_number = manifest.get("round")
    if (
        manifest.get("schema_version") != expected_schema
        or type(round_number) is not int
        or round_number < 1
        or manifest.get("human_audit") != expected_human
        or manifest.get("note_sha256") != note_hash
        or manifest.get("note_path") != f"by-sha256/{note_hash}.md"
        or not isinstance(manifest.get("entries"), list)
        or not isinstance(manifest.get("entry_ids"), list)
        or [entry.get("entry_id") if isinstance(entry, dict) else None
            for entry in manifest["entries"]] != manifest["entry_ids"]
        or len(set(manifest["entry_ids"])) != len(manifest["entry_ids"])
    ):
        raise StudyProtocolError("study note content identity is invalid")
    try:
        note_text = note_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StudyProtocolError("study note is not valid UTF-8") from error
    if not note_text.strip() or manifest.get("note_chars") != len(note_text):
        raise StudyProtocolError("study note character count is invalid")
    alias = f"notes/note-r{round_number}.md"
    content = f"notes/by-sha256/{note_hash}.md"
    if (
        construction_dependencies.get(alias) != note_bytes
        or construction_dependencies.get(content) != note_bytes
        or f"notes/note-r{round_number}.manifest.json" in construction_dependencies
    ):
        raise StudyProtocolError("study note aliases or self-reference are invalid")
    return task, summary, note_text


def _validate_semantic_archive(
    manifest: dict[str, Any],
    dependencies: Mapping[str, bytes],
    task: dict[str, Any],
    note_text: str,
) -> None:
    round_number = manifest["round"]
    config = task["config"]
    smoke = config["smoke"]
    if round_number > SEMANTIC_FINAL_ROUND or smoke and round_number != 1:
        raise StudyProtocolError("semantic note has an invalid construction round")
    expected: set[str] = {"manifest.json"}
    protocol = task.get("human_audit_protocol")
    if protocol is not None:
        if not isinstance(protocol, dict):
            raise StudyProtocolError("semantic task has invalid audit protocol metadata")
        expected.add(_archive_path(protocol.get("path"), label="audit protocol"))

    all_questions: list[list[dict[str, Any]]] = []
    all_items: list[dict[str, Any]] = []
    prior_manifests: list[dict[str, Any]] = []
    cumulative_calls: list[dict[str, Any]] = []
    cumulative_construction_calls: list[dict[str, Any]] = []
    all_environment_paths: set[str] = set()
    expected_reference_ids: set[str] = set()
    syllabus = config["chapter_syllabus"]
    chapters_per_round = config["chapters_per_round"]
    questions_per_chapter = config["questions_per_chapter"]

    for current_round in range(1, round_number + 1):
        prefix = f"r{current_round}"
        fixed = {
            f"{prefix}/manifest.json",
            f"{prefix}/questions.jsonl",
            f"{prefix}/freshness.json",
            f"{prefix}/items.jsonl",
            f"{prefix}/dev-exam.jsonl",
            f"{prefix}/usage.jsonl",
            f"{prefix}/cumulative-usage.jsonl",
            f"{prefix}/summary.json",
        }
        expected.update(fixed)
        round_manifest = _archive_json(
            dependencies, f"{prefix}/manifest.json", label="semantic round manifest"
        )
        planned = [
            syllabus[((current_round - 1) * chapters_per_round + index) % len(syllabus)]
            for index in range(chapters_per_round)
        ]
        if (
            round_manifest.get("study_id") != task["study_id"]
            or round_manifest.get("task") != task["task"]
            or round_manifest.get("round") != current_round
            or round_manifest.get("master_seed") != task["master_seed"]
            or round_manifest.get("chapters") != planned
            or round_manifest.get("attempt_access") != config["attempt_access"]
            or round_manifest.get("task_manifest_sha256") != sha256_json(task)
        ):
            raise StudyProtocolError("semantic round manifest lineage is invalid")
        initial_snapshot = _snapshot_path(round_manifest.get("initial_environment_snapshot"))
        if initial_snapshot is None:
            raise StudyProtocolError("semantic round has no initial environment snapshot")
        expected.add(initial_snapshot)
        all_environment_paths.add(initial_snapshot)

        questions = _archive_jsonl(
            dependencies, f"{prefix}/questions.jsonl", label="semantic questions"
        )
        all_questions.append(questions)
        originals = [record for record in questions if record.get("kind") == "quiz"]
        expected_chapters = planned[:1] if smoke else planned
        if (
            len(originals) != len(expected_chapters) * questions_per_chapter
            or any(record.get("round") != current_round for record in questions)
            or any(record.get("attempt_access") != config["attempt_access"]
                   for record in questions)
        ):
            raise StudyProtocolError("semantic question population is incomplete")
        episode_ids: set[str] = set()
        for chapter in expected_chapters:
            chapter_records = [record for record in originals
                               if record.get("chapter") == chapter]
            if (
                len(chapter_records) != questions_per_chapter
                or sum(record.get("split") == "dev" for record in chapter_records) != 1
                or sum(record.get("split") == "train" for record in chapter_records)
                != questions_per_chapter - 1
            ):
                raise StudyProtocolError("semantic chapter train/dev split is incomplete")
            values = {record.get("quiz_episode_id") for record in chapter_records}
            if len(values) != 1 or not all(isinstance(value, str) and value for value in values):
                raise StudyProtocolError("semantic quiz episode lineage is invalid")
            episode_ids.update(values)
        if {record.get("chapter") for record in originals} != set(expected_chapters):
            raise StudyProtocolError("semantic questions use the wrong chapter schedule")
        for episode_id in episode_ids:
            path = f"{prefix}/quiz-episodes/{_archive_path(episode_id, label='episode ID')}.json"
            episode = _archive_json(dependencies, path, label="semantic quiz episode")
            expected.add(path)
            snapshot = _snapshot_path(episode.get("environment_snapshot"))
            if (
                episode.get("owner_id") != episode_id
                or episode.get("status") != "ok"
                or snapshot is None
            ):
                raise StudyProtocolError("semantic quiz episode is incomplete")
            expected.add(snapshot)
            all_environment_paths.add(snapshot)

        items = _archive_jsonl(
            dependencies, f"{prefix}/items.jsonl", label="semantic training items"
        )
        expected_item_ids = {
            record.get("item_id") for record in questions if record.get("split") == "train"
        }
        observed_item_ids = {record.get("item_id") for record in items}
        if (
            None in expected_item_ids
            or observed_item_ids != expected_item_ids
            or len(observed_item_ids) != len(items)
        ):
            raise StudyProtocolError("semantic training aggregate is incomplete")
        for record in items:
            item_id = _archive_path(record.get("item_id"), label="training item ID")
            path = f"{prefix}/items/{item_id}.json"
            if _archive_json(dependencies, path, label="semantic training item") != record:
                raise StudyProtocolError("semantic training item aggregate drifted")
            expected.add(path)
            snapshot = _snapshot_path(record.get("environment_snapshot"))
            if snapshot is None or not semantic_training_item_terminal(record):
                raise StudyProtocolError("semantic training item is incomplete")
            expected.add(snapshot)
            all_environment_paths.add(snapshot)
        all_items.extend(items)

        cumulative_dev_ids = {
            record.get("item_id")
            for question_round in all_questions
            for record in question_round
            if record.get("kind") == "quiz" and record.get("split") == "dev"
        }
        dev_records = _archive_jsonl(
            dependencies, f"{prefix}/dev-exam.jsonl", label="semantic dev exam"
        )
        if (
            {record.get("origin_item_id") for record in dev_records}
            != cumulative_dev_ids
            or len({record.get("item_id") for record in dev_records}) != len(dev_records)
        ):
            raise StudyProtocolError("semantic cumulative dev exam is incomplete")
        for record in dev_records:
            item_id = _archive_path(record.get("item_id"), label="dev exam ID")
            path = f"{prefix}/dev-exam/{item_id}.json"
            if _archive_json(dependencies, path, label="semantic dev exam item") != record:
                raise StudyProtocolError("semantic dev exam aggregate drifted")
            expected.add(path)
            reference_id = _archive_path(
                record.get("reference_id"), label="dev reference ID"
            )
            expected_reference_ids.add(reference_id)
            snapshot = _snapshot_path(record.get("environment_snapshot"))
            if (
                snapshot is None
                or not semantic_dev_exam_terminal(record)
            ):
                raise StudyProtocolError("semantic dev exam record is incomplete")
            expected.add(snapshot)
            all_environment_paths.add(snapshot)

        freshness = _archive_json(
            dependencies, f"{prefix}/freshness.json", label="semantic freshness audit"
        )
        sources = freshness.get("comparison_sources")
        if (
            not isinstance(sources, list)
            or freshness.get("comparison_bundle_sha256") != sha256_json(sources)
        ):
            raise StudyProtocolError("semantic freshness source bundle is invalid")
        for source in sources:
            if not isinstance(source, dict):
                raise StudyProtocolError("semantic freshness source is invalid")
            snapshot = _archive_path(source.get("snapshot"), label="freshness snapshot")
            expected_path = f"{prefix}/{snapshot}"
            data = dependencies.get(expected_path)
            if (
                not isinstance(data, bytes)
                or sha256_bytes(data) != source.get("sha256")
                or not expected_path.startswith(f"{prefix}/freshness-sources/")
            ):
                raise StudyProtocolError("semantic freshness snapshot is incomplete")
            expected.add(expected_path)

        round_calls = _archive_jsonl(
            dependencies, f"{prefix}/usage.jsonl", label="semantic usage ledger"
        )
        recorded_cumulative = _archive_jsonl(
            dependencies,
            f"{prefix}/cumulative-usage.jsonl",
            label="semantic cumulative usage ledger",
        )
        expected_cumulative = sorted(
            cumulative_calls + round_calls,
            key=lambda call: str(call.get("call_id", "")),
        )
        if recorded_cumulative != expected_cumulative:
            raise StudyProtocolError("semantic cumulative usage ledger drifted")
        cumulative_calls = recorded_cumulative
        construction_round = [
            call for call in round_calls
            if not str(call.get("owner_id", "")).startswith(
                ("dev-exam-", "dev-reference-")
            )
        ]
        cumulative_construction_calls += construction_round

        summary = _archive_json(
            dependencies, f"{prefix}/summary.json", label="semantic round summary"
        )
        launch_records = summary.get("launch_environments")
        if not isinstance(launch_records, list):
            raise StudyProtocolError("semantic summary has no launch inventory")
        summary_environment_paths = {_snapshot_path(record) for record in launch_records}
        if None in summary_environment_paths:
            raise StudyProtocolError("semantic launch inventory is invalid")
        expected.update(summary_environment_paths)  # type: ignore[arg-type]
        all_environment_paths.update(summary_environment_paths)  # type: ignore[arg-type]
        if (
            summary.get("study_id") != task["study_id"]
            or summary.get("task") != task["task"]
            or summary.get("round") != current_round
            or summary.get("attempt_access") != config["attempt_access"]
            or summary.get("freshness") != freshness
            or summary.get("round_usage") != _usage_totals(round_calls)
            or summary.get("cumulative_usage") != _usage_totals(cumulative_calls)
            or summary.get("round_usage_by_phase") != _usage_by_phase(round_calls)
            or summary.get("cumulative_usage_by_phase")
            != _usage_by_phase(cumulative_calls)
        ):
            raise StudyProtocolError("semantic round summary lineage drifted")

        alias = f"notes/note-r{current_round}.md"
        expected.add(alias)
        if current_round < round_number:
            prior_path = f"notes/note-r{current_round}.manifest.json"
            prior = _archive_json(
                dependencies, prior_path, label="prior semantic note manifest"
            )
            expected.add(prior_path)
            prior_manifests.append(prior)
            prior_hash = prior.get("note_sha256")
            if not _valid_sha256(prior_hash):
                raise StudyProtocolError("prior semantic note has invalid identity")
            expected.add(f"notes/by-sha256/{prior_hash}.md")
            if dependencies.get(alias) != dependencies.get(
                f"notes/by-sha256/{prior_hash}.md"
            ):
                raise StudyProtocolError("prior semantic note aliases drifted")
        else:
            expected.add(f"notes/by-sha256/{manifest['note_sha256']}.md")

    for reference_id in expected_reference_ids:
        path = f"dev-references/{reference_id}.json"
        reference = _archive_json(dependencies, path, label="semantic dev reference")
        if (
            reference.get("reference_id") != reference_id
            or not semantic_dev_reference_terminal(reference)
        ):
            raise StudyProtocolError("semantic dev reference is incomplete")
        snapshot = _snapshot_path(reference.get("environment_snapshot"))
        if snapshot is None:
            raise StudyProtocolError("semantic dev reference lacks environment binding")
        expected.update({path, snapshot})
        all_environment_paths.add(snapshot)

    prior_hash = sha256_text("")
    for index, prior in enumerate(prior_manifests, 1):
        prior_readiness = prior.get("automated_readiness")
        prior_treatment_readiness = prior.get("automated_treatment_readiness")
        if (
            set(prior) != SEMANTIC_NOTE_MANIFEST_KEYS
            or prior.get("schema_version") != 6
            or prior.get("manifest_type") != SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE
            or prior.get("method") != SEMANTIC_SELFQUIZ_METHOD
            or prior.get("round") != index
            or prior.get("input_note_sha256") != prior_hash
            or not isinstance(prior_readiness, dict)
            or set(prior_readiness) != SEMANTIC_READINESS_KEYS
            or any(type(value) is not bool for value in prior_readiness.values())
            or prior.get("automated_claim_ready")
            is not all(prior_readiness.values())
            or not isinstance(prior_treatment_readiness, dict)
            or set(prior_treatment_readiness)
            != SEMANTIC_TREATMENT_READINESS_KEYS
            or any(
                type(value) is not bool
                for value in prior_treatment_readiness.values()
            )
            or prior.get("automated_treatment_ready") is not True
            or not all(prior_treatment_readiness.values())
        ):
            raise StudyProtocolError("prior semantic note readiness chain is invalid")
        prior_hash = prior["note_sha256"]
    if manifest.get("input_note_sha256") != prior_hash:
        raise StudyProtocolError("semantic input-note hash chain is invalid")

    admitted: dict[str, dict[str, Any]] = {}
    for record in all_items:
        entry = record.get("entry")
        if (
            record.get("kind") == "quiz"
            and record.get("split") == "train"
            and record.get("origin_item_id") == record.get("item_id")
            and isinstance(entry, dict)
            and isinstance(entry.get("entry_id"), str)
        ):
            admitted[entry["entry_id"]] = entry
    reconstructed_entries = [admitted[key] for key in sorted(admitted)]
    if manifest["entries"] != reconstructed_entries:
        raise StudyProtocolError("semantic note entry lineage is incomplete")

    current_summary = _archive_json(
        dependencies, f"r{round_number}/summary.json", label="semantic final summary"
    )
    current_freshness = _archive_json(
        dependencies, f"r{round_number}/freshness.json", label="semantic final freshness"
    )
    current_items = _archive_jsonl(
        dependencies, f"r{round_number}/items.jsonl", label="semantic final items"
    )
    current_devs = _archive_jsonl(
        dependencies, f"r{round_number}/dev-exam.jsonl", label="semantic final dev exam"
    )
    current_episodes = [
        _archive_json(dependencies, path, label="semantic final quiz episode")
        for path in sorted(expected)
        if path.startswith(f"r{round_number}/quiz-episodes/")
    ]
    current_refs = [
        _archive_json(dependencies, f"dev-references/{reference_id}.json",
                      label="semantic final dev reference")
        for reference_id in sorted(expected_reference_ids)
    ]
    successful_derivations = [
        derivation
        for artifact in current_items + current_refs
        for derivation in artifact.get("derivations", [])
        if isinstance(derivation, dict) and derivation.get("status") == "ok"
    ]
    calls_response_models = sorted({
        call.get("response_model") for call in cumulative_calls
        if isinstance(call.get("response_model"), str) and call.get("response_model")
    })
    expected_response_model = str(task["model"]).removeprefix("openai/")
    training_records_present = bool([
        record
        for record in current_items
        if record.get("kind") == "quiz" and record.get("split") == "train"
    ])
    derivation_evidence_safe = all(
        derivation.get("evidence_class") == "quote-only"
        and bool(derivation.get("evidence"))
        and isinstance(derivation.get("reference_support"), dict)
        and derivation["reference_support"].get("status") == "ok"
        and derivation["reference_support"].get("supported") is True
        for derivation in successful_derivations
    )
    adapter_audit_complete = semantic_adapter_audit_complete(cumulative_calls)
    recomputed_readiness = {
        "non_smoke": not smoke,
        "provenance_complete": task.get("automated_provenance_ready") is True,
        "launch_environments_bound": bool(all_environment_paths),
        "prior_rounds_automated_ready": all(
            prior.get("automated_claim_ready") is True for prior in prior_manifests
        ),
        "question_freshness": (
            current_freshness.get("fresh") is True
            and current_freshness.get("audit_complete") is True
        ),
        "quiz_episodes_complete": bool(current_episodes) and all(
            episode.get("status") == "ok" and bool(episode.get("trajectory"))
            for episode in current_episodes
        ),
        "training_complete": training_records_present and all(
            record.get("status") == "ok"
            and record.get("verdict") in {"correct", "partial", "wrong"}
            for record in current_items
        ),
        "dev_references_complete": bool(current_refs) and all(
            reference.get("status") == "ok" for reference in current_refs
        ),
        "dev_exam_complete": bool(current_devs) and all(
            record.get("status") == "ok"
            and set(record.get("verdicts", {})) == {"with_note", "bare"}
            and set(record["verdicts"].values()) <= {"correct", "partial", "wrong"}
            for record in current_devs
        ),
        "lineage_clean": manifest["entries"] == reconstructed_entries,
        "evidence_safe": bool(successful_derivations) and derivation_evidence_safe,
        "usage_complete": _usage_totals(cumulative_calls).get("status") == "complete",
        "adapter_audit_complete": adapter_audit_complete,
        "response_model_homogeneous": len(calls_response_models) == 1,
        "response_model_expected": calls_response_models == [expected_response_model],
    }
    if manifest["automated_readiness"] != recomputed_readiness:
        raise StudyProtocolError("semantic automated readiness was not reconstructed")
    recomputed_treatment_readiness = {
        "non_smoke": not smoke,
        "provenance_complete": task.get("automated_provenance_ready") is True,
        "launch_environments_bound": bool(all_environment_paths),
        "prior_rounds_treatment_ready": all(
            prior.get("automated_treatment_ready") is True
            for prior in prior_manifests
        ),
        "question_freshness": (
            current_freshness.get("fresh") is True
            and current_freshness.get("audit_complete") is True
        ),
        "quiz_episodes_complete": bool(current_episodes) and all(
            episode.get("status") == "ok" and bool(episode.get("trajectory"))
            for episode in current_episodes
        ),
        "training_terminal_complete": training_records_present
        and all(semantic_training_item_terminal(record) for record in current_items),
        "dev_references_terminal_complete": bool(current_refs)
        and all(semantic_dev_reference_terminal(record) for record in current_refs),
        "dev_exam_terminal_complete": bool(current_devs)
        and all(semantic_dev_exam_terminal(record) for record in current_devs),
        "lineage_clean": manifest["entries"] == reconstructed_entries,
        "evidence_safe": derivation_evidence_safe,
        "usage_complete": _usage_totals(cumulative_calls).get("status") == "complete",
        "adapter_audit_complete": adapter_audit_complete,
        "response_model_homogeneous": len(calls_response_models) == 1,
        "response_model_expected": calls_response_models == [expected_response_model],
    }
    if manifest["automated_treatment_readiness"] != recomputed_treatment_readiness:
        raise StudyProtocolError(
            "semantic automated treatment readiness was not reconstructed"
        )
    round_calls = _archive_jsonl(
        dependencies, f"r{round_number}/usage.jsonl", label="semantic final usage"
    )
    round_construction_calls = [
        call for call in round_calls
        if not str(call.get("owner_id", "")).startswith(
            ("dev-exam-", "dev-reference-")
        )
    ]
    if (
        manifest.get("usage") != _usage_totals(cumulative_calls)
        or manifest.get("round_usage") != _usage_totals(round_calls)
        or manifest.get("cumulative_usage") != _usage_totals(cumulative_calls)
        or manifest.get("round_usage_by_phase") != _usage_by_phase(round_calls)
        or manifest.get("cumulative_usage_by_phase") != _usage_by_phase(cumulative_calls)
        or manifest.get("round_construction_usage")
        != _usage_totals(round_construction_calls)
        or manifest.get("cumulative_construction_usage")
        != _usage_totals(cumulative_construction_calls)
        or manifest.get("round_construction_usage_by_phase")
        != _usage_by_phase(round_construction_calls)
        or manifest.get("cumulative_construction_usage_by_phase")
        != _usage_by_phase(cumulative_construction_calls)
        or current_summary.get("note_sha256") != manifest["note_sha256"]
        or current_summary.get("automated_claim_ready")
        != manifest["automated_claim_ready"]
        or current_summary.get("automated_readiness")
        != manifest["automated_readiness"]
        or current_summary.get("automated_treatment_ready")
        != manifest["automated_treatment_ready"]
        or current_summary.get("automated_treatment_readiness")
        != manifest["automated_treatment_readiness"]
    ):
        raise StudyProtocolError("semantic note usage or summary contract drifted")
    if set(dependencies) != expected:
        missing = sorted(expected - set(dependencies))
        extra = sorted(set(dependencies) - expected)
        raise StudyProtocolError(
            f"semantic construction archive is not closed; missing={missing}, extra={extra}"
        )


def _validate_graph_archive(
    manifest: dict[str, Any],
    dependencies: Mapping[str, bytes],
    task: dict[str, Any],
    note_text: str,
) -> None:
    if manifest.get("round") != 1:
        raise StudyProtocolError("static-graph note must be round 1")
    config = task["config"]
    smoke = config["smoke"]
    expected: set[str] = {
        "manifest.json",
        "r1/manifest.json",
        "r1/question-bank.json",
        "r1/items.jsonl",
        "r1/usage.jsonl",
        "r1/summary.json",
        "notes/note-r1.md",
        f"notes/by-sha256/{manifest['note_sha256']}.md",
    }
    round_manifest = _archive_json(
        dependencies, "r1/manifest.json", label="graph round manifest"
    )
    bank = _archive_json(
        dependencies, "r1/question-bank.json", label="graph question bank"
    )
    full_bank = bank.get("full_bank")
    selected_ids = bank.get("selected_question_ids")
    if not isinstance(full_bank, list) or not isinstance(selected_ids, list):
        raise StudyProtocolError("graph question bank is incomplete")
    selected_by_id = {
        question.get("id"): question
        for question in full_bank
        if isinstance(question, dict) and question.get("id") in selected_ids
    }
    if (
        len(selected_by_id) != len(selected_ids)
        or list(selected_by_id) != selected_ids
        or bank.get("resolver_contract") != task.get("resolver_contract")
        or bank.get("resolver_contract_sha256")
        != task.get("resolver_contract_sha256")
        or bank.get("question_bank_sha256") != task.get("question_bank_sha256")
        or bank.get("question_bank_artifact_sha256") != sha256_json(full_bank)
        or bank.get("selected_question_bank_sha256")
        != sha256_json(list(selected_by_id.values()))
    ):
        raise StudyProtocolError("graph question bank identity drifted")
    selected = list(selected_by_id.values())
    train = [question for question in selected if question.get("split") == "train"]
    dev = [question for question in selected if question.get("split") == "dev"]
    if (
        len(train) != config["train_question_count"]
        or len(dev) != config["dev_question_count"]
        or round_manifest.get("selected_question_ids") != selected_ids
        or round_manifest.get("train_question_ids")
        != [question["id"] for question in train]
        or round_manifest.get("dev_question_ids") != [question["id"] for question in dev]
        or round_manifest.get("selected_question_bank_sha256") != sha256_json(selected)
        or round_manifest.get("task_manifest_sha256") != sha256_json(task)
    ):
        raise StudyProtocolError("graph selected population is incomplete")

    question_artifacts: dict[str, dict[str, Any]] = {}
    for index, question in enumerate(selected):
        target = question.get("target")
        if not isinstance(target, str):
            raise StudyProtocolError("graph question target is invalid")
        path = f"r1/questions/q{index:02d}-{sha256_text(target)[:16]}.json"
        artifact = _archive_json(dependencies, path, label="graph question")
        if (
            artifact.get("question_id") != question.get("id")
            or artifact.get("target") != target
            or artifact.get("split") != question.get("split")
            or artifact.get("gold_edges") != question.get("gold_edges")
        ):
            raise StudyProtocolError("graph question artifact drifted")
        expected.add(path)
        question_artifacts[question["id"]] = artifact

    train_records = _archive_jsonl(
        dependencies, "r1/items.jsonl", label="graph training items"
    )
    if [record.get("question_id") for record in train_records] != [
        question["id"] for question in train
    ]:
        raise StudyProtocolError("graph training aggregate is incomplete")
    environment_paths: set[str] = set()
    initial_snapshot = _snapshot_path(round_manifest.get("initial_environment_snapshot"))
    if initial_snapshot is None:
        raise StudyProtocolError("graph round has no initial environment snapshot")
    environment_paths.add(initial_snapshot)
    expected.add(initial_snapshot)
    for record in train_records:
        question_id = record["question_id"]
        path = f"r1/items/{sha256_text(question_id)[:20]}.json"
        if _archive_json(dependencies, path, label="graph training item") != record:
            raise StudyProtocolError("graph training item aggregate drifted")
        expected.add(path)
        snapshot = _snapshot_path(record.get("environment_snapshot"))
        if snapshot is None:
            raise StudyProtocolError("graph training item lacks environment binding")
        expected.add(snapshot)
        environment_paths.add(snapshot)

    dev_records: list[dict[str, Any]] = []
    if dev:
        expected.add("r1/dev-exam.jsonl")
        dev_records = _archive_jsonl(
            dependencies, "r1/dev-exam.jsonl", label="graph dev exam"
        )
        if [record.get("question_id") for record in dev_records] != [
            question["id"] for question in dev
        ]:
            raise StudyProtocolError("graph dev aggregate is incomplete")
        for record in dev_records:
            question_id = record["question_id"]
            path = f"r1/dev-exam/{sha256_text(question_id)[:20]}.json"
            if _archive_json(dependencies, path, label="graph dev item") != record:
                raise StudyProtocolError("graph dev item aggregate drifted")
            expected.add(path)
            snapshot = _snapshot_path(record.get("environment_snapshot"))
            if snapshot is None:
                raise StudyProtocolError("graph dev item lacks environment binding")
            expected.add(snapshot)
            environment_paths.add(snapshot)

    entries = [record.get("entry") for record in train_records
               if record.get("entry") is not None]
    if manifest["entries"] != entries:
        raise StudyProtocolError("graph correction entry lineage is incomplete")
    if (
        manifest.get("train_question_ids") != [question["id"] for question in train]
        or manifest.get("held_out_dev_question_ids")
        != [question["id"] for question in dev]
        or manifest.get("resolver_contract_sha256")
        != task.get("resolver_contract_sha256")
        or manifest.get("question_bank_sha256") != task.get("question_bank_sha256")
        or manifest.get("input_note_sha256") != sha256_text("")
        or manifest.get("note_bytes") != len(note_text.encode("utf-8"))
    ):
        raise StudyProtocolError("graph note population identity drifted")

    calls = _archive_jsonl(dependencies, "r1/usage.jsonl", label="graph usage")
    artifact_calls = [
        call
        for record in train_records + dev_records
        for call in (_record_calls(record) or [])
    ]
    artifact_calls = sorted(artifact_calls, key=lambda call: str(call.get("call_id", "")))
    if calls != artifact_calls:
        raise StudyProtocolError("graph usage ledger is incomplete")
    summary = _archive_json(dependencies, "r1/summary.json", label="graph summary")
    response_models = sorted({
        call.get("response_model") for call in calls
        if isinstance(call.get("response_model"), str) and call.get("response_model")
    })
    train_ids = {question["id"] for question in train}
    dev_ids = {question["id"] for question in dev}
    train_evidence = {
        (edge.get("path"), edge.get("line"))
        for question in train for edge in question.get("gold_edges", [])
        if isinstance(edge, dict)
    }
    dev_evidence = {
        (edge.get("path"), edge.get("line"))
        for question in dev for edge in question.get("gold_edges", [])
        if isinstance(edge, dict)
    }
    expected_model = str(task["model"]).removeprefix("openai/")
    all_entry_text_present = all(
        isinstance(entry, dict)
        and entry.get("target") not in _STATIC_GRAPH_DEV_HOLDOUT_TARGETS
        and all(
            isinstance(entry.get(field), str) and entry[field] in note_text
            for field in ("target", "belief")
        )
        for entry in entries
    )
    recomputed = {
        "non_smoke": not smoke,
        "provenance_complete": task.get("automated_provenance_ready") is True,
        "environment_contract_valid": bool(task.get("environment_contract")),
        "resolver_contract_recomputed": (
            task.get("resolver_contract_sha256")
            == sha256_json(task.get("resolver_contract"))
        ),
        "question_bank_recomputed": (
            task.get("question_bank_sha256")
            == task.get("resolver_contract", {}).get("question_bank_sha256")
            and task.get("question_bank_artifact_sha256") == sha256_json(full_bank)
        ),
        "selection_exact": (
            len(train) == 16 and len(dev) == 4 and not smoke
        ),
        "training_complete": (
            len(train_records) == 16
            and all(record.get("attempt", {}).get("status") == "ok"
                    for record in train_records)
        ),
        "training_empty_note": all(
            record.get("input_note_sha256") == sha256_text("")
            and record.get("input_note_bytes") == 0
            for record in train_records
        ),
        "training_scores_recomputed": all(
            record.get("score_sha256") == sha256_json(record.get("score"))
            and record.get("verdict") in {"exact", "partial", "wrong"}
            for record in train_records
        ),
        "corrections_recomputed": manifest["entries"] == entries,
        "note_recomputed": all_entry_text_present,
        "dev_holdout_isolated": (
            len(dev_records) == 4
            and all(entry.get("origin_question_id") in train_ids for entry in entries)
            and all(entry.get("origin_question_id") not in dev_ids for entry in entries)
        ),
        "dev_evidence_locations_disjoint": not (train_evidence & dev_evidence),
        "dev_pair_complete": (
            len(dev_records) == 4
            and all(
                set(record.get("attempts", {})) == {"with_note", "bare"}
                and all(record["attempts"][arm].get("status") == "ok"
                        for arm in ("with_note", "bare"))
                for record in dev_records
            )
        ),
        "dev_pair_note_only": all(
            record.get("attempts", {}).get("with_note", {}).get("seed")
            == record.get("attempts", {}).get("bare", {}).get("seed")
            == record.get("paired_seed")
            and record.get("attempt_protocol", {}).get("only_manipulated_field")
            == "note"
            for record in dev_records
        ),
        "dev_scores_recomputed": all(
            record.get("score_sha256") == {
                arm: sha256_json(record.get("scores", {}).get(arm))
                for arm in ("with_note", "bare")
            }
            for record in dev_records
        ),
        "usage_complete": (
            manifest.get("usage_audit", {}).get("complete") is True
            and calls == artifact_calls
            and all(record.get("usage") == _usage_totals(_record_calls(record) or [])
                    for record in train_records + dev_records)
        ),
        "launch_environments_bound": all(
            _snapshot_path(record.get("environment_snapshot")) in environment_paths
            for record in train_records + dev_records
        ),
        "response_model_homogeneous": len(response_models) == 1,
        "response_model_expected": response_models == [expected_model],
        "construction_inventory_complete": bool(dependencies),
    }
    if manifest["automated_readiness"] != recomputed:
        raise StudyProtocolError("graph automated readiness was not reconstructed")
    if (
        manifest.get("usage") != _usage_totals(calls)
        or manifest.get("usage_by_phase") != _usage_by_phase(calls)
        or summary.get("note_sha256") != manifest["note_sha256"]
        or summary.get("usage") != _usage_totals(calls)
        or summary.get("usage_by_phase") != _usage_by_phase(calls)
    ):
        raise StudyProtocolError("graph note usage or summary contract drifted")
    # Every environment snapshot is content addressed. This admits an otherwise
    # harmless crash-orphan snapshot while rejecting every other unknown file.
    for path in dependencies:
        if path.startswith("r1/environments/"):
            match = re.fullmatch(r"r1/environments/environment-([0-9a-f]{64})\.json", path)
            if match is None or sha256_bytes(dependencies[path]) != match.group(1):
                raise StudyProtocolError("graph environment snapshot identity drifted")
            expected.add(path)
    if set(dependencies) != expected:
        missing = sorted(expected - set(dependencies))
        extra = sorted(set(dependencies) - expected)
        raise StudyProtocolError(
            f"graph construction archive is not closed; missing={missing}, extra={extra}"
        )


def _validate_study_note_archive(
    note_manifest: Mapping[str, Any],
    construction_dependencies: Mapping[str, bytes],
    note_bytes: bytes,
    *,
    expected_task: str | None = None,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_sampling: Mapping[str, Any] | None = None,
    expected_corpus_commit: str | None = None,
    expected_corpus: Mapping[str, Any] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    expected_environment: Mapping[str, Any] | None = None,
    expected_environment_contract: Mapping[str, Any] | None = None,
    environments_compatible: Callable[[object, object], bool] | None = None,
    require_final_semantic: bool = False,
    allow_smoke: bool = False,
    deep_semantics: bool = True,
) -> dict[str, Any]:
    """Validate one closed-world semantic or graph construction archive.

    Unlike :func:`validate_construction_protocol`, this function proves that the
    note manifest names the complete method-specific dependency closure and
    reconstructs its stored readiness gates. It is deliberately shared by the
    constructor, evaluation preflight, and grading reload boundary.
    """

    if type(deep_semantics) is not bool:
        raise StudyProtocolError("deep-semantics requirement must be boolean")
    task, summary, note_text = _validate_archive_envelope(
        note_manifest,
        construction_dependencies,
        note_bytes,
        expected_task=expected_task,
        expected_model=expected_model,
        expected_model_revision=expected_model_revision,
        expected_sampling=expected_sampling,
        expected_corpus_commit=expected_corpus_commit,
        expected_corpus=expected_corpus,
        expected_source=expected_source,
        expected_environment=expected_environment,
        expected_environment_contract=expected_environment_contract,
        environments_compatible=environments_compatible,
        require_final_semantic=require_final_semantic,
        allow_smoke=allow_smoke,
    )
    manifest = dict(note_manifest)
    if manifest["manifest_type"] == SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE:
        _validate_semantic_archive(manifest, construction_dependencies, task, note_text)
        if deep_semantics:
            # Imported lazily to avoid a module cycle: the constructor imports
            # these shared protocol contracts, while this downstream boundary
            # deliberately reuses the constructor's exact validators.
            from .selfquiz import validate_bundled_semantic_archive

            validate_bundled_semantic_archive(
                manifest, dict(construction_dependencies), note_bytes
            )
    elif manifest["manifest_type"] == STATIC_GRAPH_NOTE_MANIFEST_TYPE:
        _validate_graph_archive(manifest, construction_dependencies, task, note_text)
        if deep_semantics:
            from .graph_study import validate_bundled_graph_archive

            validate_bundled_graph_archive(
                manifest, dict(construction_dependencies), note_bytes
            )
    else:  # protected by the envelope; retained as a fail-closed guard.
        raise StudyProtocolError("unsupported study archive type")
    return summary


def validate_study_note_archive(
    note_manifest: Mapping[str, Any],
    construction_dependencies: Mapping[str, bytes],
    note_bytes: bytes,
    *,
    expected_task: str | None = None,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_sampling: Mapping[str, Any] | None = None,
    expected_corpus_commit: str | None = None,
    expected_corpus: Mapping[str, Any] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    expected_environment: Mapping[str, Any] | None = None,
    expected_environment_contract: Mapping[str, Any] | None = None,
    environments_compatible: Callable[[object, object], bool] | None = None,
    require_final_semantic: bool = False,
    allow_smoke: bool = False,
    deep_semantics: bool = True,
) -> dict[str, Any]:
    """Fail closed with one stable integrity exception for any archive defect."""

    try:
        return _validate_study_note_archive(
            note_manifest,
            construction_dependencies,
            note_bytes,
            expected_task=expected_task,
            expected_model=expected_model,
            expected_model_revision=expected_model_revision,
            expected_sampling=expected_sampling,
            expected_corpus_commit=expected_corpus_commit,
            expected_corpus=expected_corpus,
            expected_source=expected_source,
            expected_environment=expected_environment,
            expected_environment_contract=expected_environment_contract,
            environments_compatible=environments_compatible,
            require_final_semantic=require_final_semantic,
            allow_smoke=allow_smoke,
            deep_semantics=deep_semantics,
        )
    except StudyProtocolError:
        raise
    except (AttributeError, KeyError, TypeError, UnicodeError, ValueError) as error:
        raise StudyProtocolError(f"study archive is malformed: {error}") from error


def forced50_study_task(display: str) -> str:
    """Return the exact forced-50 study prompt for one corpus."""

    if not isinstance(display, str) or not display:
        raise ValueError("corpus display name must be a nonempty string")
    return (
        f"Study the {display} repository and write yourself a cheatsheet: a "
        f"reference document that will be prepended to every future question you are "
        f"asked about {display}. You will not see the questions in advance, "
        "but you will keep access to these repository tools when answering them. "
        "Record whatever will make you fastest and most accurate later. After your "
        f"{FORCED50_ITERATIONS} iterations of study, write the complete cheatsheet "
        "as your final answer."
    )


def forced50_study_question(display: str) -> dict[str, str]:
    return {
        "id": "cheatsheet",
        "question": forced50_study_task(display),
    }


def validate_forced50_config(
    config: Mapping[str, Any],
    *,
    corpus_display: str,
    expected_task: str | None = None,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_response_model: str | None = None,
    expected_sampling: Mapping[str, Any] | None = None,
    expected_corpus: Mapping[str, Any] | None = None,
    expected_source: Mapping[str, Any] | None = None,
    expected_environment: Mapping[str, Any] | None = None,
    environments_compatible: Callable[[object, object], bool] | None = None,
) -> dict[str, Any]:
    """Validate the exact forced-50 protocol before evaluation or grading.

    The return value is the report-safe disclosure for this method. It is
    derived from the validated config, never copied from mutable note labels.
    """

    if not isinstance(config, Mapping):
        raise StudyProtocolError("forced-50 config is not a JSON object")
    value = dict(config)
    if (
        set(value) != FORCED50_CONFIG_KEYS
        or type(value.get("schema_version")) is not int
        or (
            value.get("schema_version"),
            value.get("dspy_request_audit_schema"),
        )
        not in {
            (
                FORCED50_LEGACY_CONFIG_SCHEMA_VERSION,
                DSPY_REQUEST_AUDIT_LEGACY_SCHEMA_VERSION,
            ),
            (
                FORCED50_CONFIG_SCHEMA_VERSION,
                DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
            ),
        }
        or not isinstance(value.get("study_id"), str)
        or not value["study_id"]
        or not isinstance(value.get("task"), str)
        or not value["task"]
        or value.get("method") != "forced-50-cheatsheet"
        or not isinstance(value.get("model"), str)
        or not value["model"]
        or not isinstance(value.get("model_revision"), str)
        or not value["model_revision"]
        or not isinstance(value.get("expected_response_model"), str)
        or not value["expected_response_model"]
        or value.get("adapter") != DSPY_ADAPTER_NAME
        or value.get("adapter_fallback_policy") != DSPY_ADAPTER_POLICY
        or type(value.get("master_seed")) is not int
        or type(value.get("episode_seed")) is not int
        or value.get("forced_iterations") != FORCED50_ITERATIONS
        or value.get("repository_tool_scope") != "full-pinned-corpus"
        or not isinstance(value.get("corpus"), dict)
        or value["corpus"].get("commit") in (None, "")
        or not isinstance(value.get("source"), dict)
        or not value["source"]
        or not isinstance(value.get("environment"), dict)
        or not value["environment"]
        or value.get("claim_ready") is not True
        or not _same_json(value.get("sampling"), REACT_SAMPLING)
        or not _same_json(
            value.get("tool_contract"), DSPY_REPOSITORY_TOOL_CONTRACT
        )
        or value.get("tool_schema_sha256")
        != sha256_json(DSPY_REPOSITORY_TOOL_CONTRACT)
        or value.get("read_max_lines") != DSPY_READ_MAX_LINES
        or not _valid_sha256(value.get("study_prompt_sha256"))
        or not _valid_sha256(value.get("study_question_sha256"))
    ):
        raise StudyProtocolError("forced-50 config is incomplete or off protocol")
    exact_expectations = (
        ("task", expected_task),
        ("model", expected_model),
        ("model_revision", expected_model_revision),
        ("expected_response_model", expected_response_model),
    )
    for field, expected in exact_expectations:
        if expected is not None and value.get(field) != expected:
            raise StudyProtocolError(
                f"forced-50 {field} does not match the evaluation specification"
            )
    json_expectations = (
        ("sampling", expected_sampling),
        ("corpus", expected_corpus),
        ("source", expected_source),
    )
    for field, expected in json_expectations:
        if expected is not None and not _same_json(value.get(field), expected):
            raise StudyProtocolError(
                f"forced-50 {field} does not match the evaluation specification"
            )
    if expected_environment is not None:
        compatible = environments_compatible or _same_json
        try:
            environment_matches = compatible(
                value.get("environment"), expected_environment
            )
        except (OSError, TypeError, ValueError):
            environment_matches = False
        if environment_matches is not True:
            raise StudyProtocolError(
                "forced-50 environment is incompatible with the evaluation environment"
            )
    question = forced50_study_question(corpus_display)
    transport = value.get("server_transport")
    try:
        environment_server_count = int(value["environment"]["server_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise StudyProtocolError(
            "forced-50 environment server count is invalid"
        ) from error
    if (
        value["episode_seed"]
        != stable_seed(
            value["master_seed"],
            "cheatsheet",
            value["study_id"],
            value["task"],
        )
        or value["study_prompt_sha256"]
        != sha256_bytes(question["question"].encode("utf-8"))
        or value["study_question_sha256"] != sha256_json(question)
        or not isinstance(transport, dict)
        or set(transport)
        != {
            "scope",
            "protocol",
            "available_server_count",
            "selected_server_index",
        }
        or transport.get("scope") != "loopback"
        or transport.get("protocol") != "openai-compatible-http"
        or type(transport.get("available_server_count")) is not int
        or transport["available_server_count"] < 1
        or transport["available_server_count"] != environment_server_count
        or type(transport.get("selected_server_index")) is not int
        or transport["selected_server_index"] != 0
    ):
        raise StudyProtocolError(
            "forced-50 seed, prompt, or server transport differs from its protocol"
        )

    return {
        "schema_version": PROTOCOL_SUMMARY_SCHEMA_VERSION,
        "method": "forced-50-cheatsheet",
        "question_mode": "forced-50-cheatsheet",
        "focus": None,
        "protocol_config_sha256": sha256_json(value),
    }


def validate_forced50_provider_identity(
    episode: Mapping[str, Any],
    *,
    expected_response_model: str,
) -> dict[str, Any]:
    """Validate the provider-returned identity and usage of a study episode."""

    if not isinstance(episode, Mapping):
        raise StudyProtocolError("forced-50 episode is not a JSON object")
    ledger = episode.get("usage_ledger")
    audit_schema = episode.get("dspy_request_audit_schema")
    if (
        not isinstance(expected_response_model, str)
        or not expected_response_model
        or audit_schema not in DSPY_REQUEST_AUDIT_SCHEMA_VERSIONS
        or not isinstance(ledger, list)
        or not ledger
        or type(episode.get("n_lm_calls")) is not int
        or episode.get("n_lm_calls") != len(ledger)
    ):
        raise StudyProtocolError("forced-50 episode has no complete provider ledger")
    fingerprints: set[str] = set()
    response_ids: set[str] = set()
    missing_fingerprints = 0
    totals = {field: 0 for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
    for index, record in enumerate(ledger):
        try:
            validate_dspy_provider_call(
                record,
                index,
                schema_version=audit_schema,
            )
        except StudyProtocolError as error:
            raise StudyProtocolError(
                f"forced-50 provider call {index} has invalid identity or usage"
            ) from error
        if record.get("response_model") != expected_response_model:
            raise StudyProtocolError(
                f"forced-50 provider call {index} has an unexpected response model"
            )
        if record["response_id"] in response_ids:
            raise StudyProtocolError("forced-50 provider response IDs are not unique")
        response_ids.add(record["response_id"])
        fingerprint = record.get("system_fingerprint")
        if fingerprint is None:
            missing_fingerprints += 1
        elif not isinstance(fingerprint, str) or not fingerprint:
            raise StudyProtocolError(
                f"forced-50 provider call {index} has an invalid system fingerprint"
            )
        else:
            fingerprints.add(fingerprint)
        for field in totals:
            amount = record.get(field)
            if (
                type(amount) is not int
                or amount < 0
                or type(record["provider_usage"].get(field)) is not int
                or record["provider_usage"].get(field) != amount
            ):
                raise StudyProtocolError(
                    f"forced-50 provider call {index} has invalid {field}"
                )
            totals[field] += amount
        if record["total_tokens"] != (
            record["prompt_tokens"] + record["completion_tokens"]
        ):
            raise StudyProtocolError(
                f"forced-50 provider call {index} has inconsistent token totals"
            )
    if any(episode.get(field) != total for field, total in totals.items()):
        raise StudyProtocolError(
            "forced-50 episode totals do not match its provider usage ledger"
        )
    if audit_schema == DSPY_REQUEST_AUDIT_SCHEMA_VERSION:
        validate_dspy_final_binding(episode)
    return {
        "response_models": [expected_response_model],
        "system_fingerprints": sorted(fingerprints),
        "missing_system_fingerprint_calls": missing_fingerprints,
        "provider_call_count": len(ledger),
    }


def validate_forced50_episode(
    episode: bytes | Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    expected_note_sha256: str,
) -> dict[str, Any]:
    """Validate one canonical forced-50 study episode and provider ledger."""

    if isinstance(episode, bytes):
        parsed = strict_json_loads(episode, label="forced-50 study episode")
        if not isinstance(parsed, dict) or canonical_json_bytes(parsed) != episode:
            raise StudyProtocolError(
                "forced-50 study episode is not a canonical JSON object"
            )
    elif isinstance(episode, Mapping):
        parsed = dict(episode)
    else:
        raise StudyProtocolError("forced-50 study episode is not a JSON object")
    if not isinstance(config, Mapping) or not _valid_sha256(expected_note_sha256):
        raise StudyProtocolError("forced-50 episode expectations are invalid")
    value = dict(config)
    integer_fields = (
        "rollout",
        "seed",
        "n_react_iters",
        "n_tool_iters",
        "finish_catches",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "gen_tokens",
    )
    turns = parsed.get("turns")
    observed_tool_iters = 0
    observed_finish_catches = 0
    if not isinstance(turns, list) or len(turns) != FORCED50_ITERATIONS:
        raise StudyProtocolError(
            "forced-50 study episode does not contain exactly 50 recorded turns"
        )
    for index, turn in enumerate(turns):
        if (
            not isinstance(turn, dict)
            or set(turn) != {"reasoning", "tool_calls", "observations"}
            or not isinstance(turn.get("reasoning"), str)
            or not isinstance(turn.get("tool_calls"), list)
            or len(turn["tool_calls"]) != 1
            or not isinstance(turn.get("observations"), list)
            or len(turn["observations"]) != 1
            or not isinstance(turn["observations"][0], str)
        ):
            raise StudyProtocolError(
                f"forced-50 study turn {index} has invalid recorded evidence"
            )
        call = turn["tool_calls"][0]
        if (
            not isinstance(call, dict)
            or set(call) != {"name", "arguments"}
            or call.get("name") not in {"grep", "glob", "read_file", "finish"}
            or not isinstance(call.get("arguments"), str)
        ):
            raise StudyProtocolError(
                f"forced-50 study turn {index} has an invalid tool call"
            )
        try:
            arguments = strict_json_loads(
                call["arguments"], label=f"forced-50 turn {index} arguments"
            )
        except ValueError as error:
            raise StudyProtocolError(
                f"forced-50 study turn {index} arguments are not JSON"
            ) from error
        if not isinstance(arguments, dict):
            raise StudyProtocolError(
                f"forced-50 study turn {index} arguments are not an object"
            )
        if call["name"] == "finish":
            observed_finish_catches += 1
        else:
            observed_tool_iters += 1
    if (
        any(type(parsed.get(field)) is not int for field in integer_fields)
        or parsed.get("task") != value.get("task")
        or parsed.get("qid") != "cheatsheet"
        or parsed.get("budget") != "s50"
        or parsed.get("rollout") != 0
        or parsed.get("harness") != "dspy.ReAct"
        or parsed.get("dspy_request_audit_schema")
        != value.get("dspy_request_audit_schema")
        or parsed.get("status") != "ok"
        or not isinstance(parsed.get("started"), str)
        or not parsed["started"]
        or not isinstance(parsed.get("finished"), str)
        or not parsed["finished"]
        or "error" in parsed
        or "invalid_final_status" in parsed
        or parsed.get("model") != value.get("model")
        or parsed.get("model_revision") != value.get("model_revision")
        or parsed.get("seed") != value.get("episode_seed")
        or parsed.get("study_intent_sha256") != sha256_json(value)
        or parsed.get("question_sha256") != value.get("study_question_sha256")
        or not isinstance(parsed.get("answer"), str)
        or sha256_text(parsed["answer"]) != expected_note_sha256
        or parsed.get("n_react_iters") != FORCED50_ITERATIONS
        or parsed.get("n_react_iters") != len(turns)
        or parsed.get("n_tool_iters") != observed_tool_iters
        or parsed.get("finish_catches") != observed_finish_catches
        or parsed.get("prompt_tokens") < 0
        or parsed.get("completion_tokens") < 0
        or parsed.get("total_tokens")
        != parsed.get("prompt_tokens") + parsed.get("completion_tokens")
        or parsed.get("gen_tokens") != parsed.get("completion_tokens")
    ):
        raise StudyProtocolError(
            "forced-50 study episode identity or iteration contract is invalid"
        )
    return validate_forced50_provider_identity(
        parsed,
        expected_response_model=value.get("expected_response_model"),
    )
