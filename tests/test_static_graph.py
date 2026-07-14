from __future__ import annotations

import json
import unittest

from studybench.dataset import CORPORA
from studybench.static_graph import (
    CONTRACT_VERSION,
    FROZEN_DEV_TARGETS,
    FROZEN_TRAIN_TARGETS,
    CallEdge,
    analyze_calls,
    build_question_bank,
    contract_sha256,
    dspy_python_texts,
    excluded_call_candidates,
    parse_prediction,
    render_question,
    resolver_contract_record,
    verify_prediction,
)
from studybench.tools import RepoTools


SYNTHETIC_SOURCES = {
    "dspy/adapters/__init__.py": "",
    "dspy/adapters/utils.py": """\
def target():
    target()

def other():
    return None

class HasMethod:
    def target(self):
        return None

def outer_shadow():
    target = lambda: None
    def inner():
        target()
    return inner
""",
    "dspy/adapters/consumer.py": """\
from dspy.adapters.utils import target as imported_target
import dspy.adapters.utils as utils_module
from dspy.adapters.utils import target as rebound_target
rebound_target = lambda: None

def caller():
    imported_target()
    imported_target(); imported_target(); utils_module.other()
    utils_module.target()
    indirect = imported_target
    indirect()

def local_import():
    from .utils import target as local_target
    local_target()

def local_module_import():
    import dspy.adapters.utils as local_utils
    local_utils.target()

def local_rebound():
    from .utils import target as local_target
    local_target = lambda: None
    local_target()

def before_import():
    late_target()
    from .utils import target as late_target
    late_target()

def shadow_parameter(imported_target):
    imported_target()

def dynamic_attribute(obj):
    obj.target()

def rebound_module_alias():
    utils_module = object()
    utils_module.target()

def calls_rebound_module_name():
    rebound_target()

def wildcard():
    from dspy.adapters.utils import *
    target()

class Consumer:
    def method(self):
        imported_target()
        self.target()

def outer_import_shadow():
    imported_target = lambda: None
    def inner():
        imported_target()
    return inner
""",
}


class StaticResolverTests(unittest.TestCase):
    def target_edges(self) -> list[CallEdge]:
        return [
            edge
            for edge in analyze_calls(SYNTHETIC_SOURCES)
            if edge.callee == "dspy.adapters.utils.target"
        ]

    def test_resolves_aliases_local_imports_recursion_and_method_callers(self):
        locations = {
            (edge.caller, edge.path, edge.line)
            for edge in self.target_edges()
        }
        self.assertEqual(
            locations,
            {
                ("dspy.adapters.consumer.caller", "dspy/adapters/consumer.py", 7),
                ("dspy.adapters.consumer.caller", "dspy/adapters/consumer.py", 8),
                ("dspy.adapters.consumer.caller", "dspy/adapters/consumer.py", 9),
                (
                    "dspy.adapters.consumer.local_import",
                    "dspy/adapters/consumer.py",
                    15,
                ),
                (
                    "dspy.adapters.consumer.local_module_import",
                    "dspy/adapters/consumer.py",
                    19,
                ),
                (
                    "dspy.adapters.consumer.before_import",
                    "dspy/adapters/consumer.py",
                    29,
                ),
                (
                    "dspy.adapters.consumer.Consumer.method",
                    "dspy/adapters/consumer.py",
                    50,
                ),
                (
                    "dspy.adapters.utils.target",
                    "dspy/adapters/utils.py",
                    2,
                ),
            },
        )

    def test_only_identical_same_line_sites_collapse(self):
        caller_edges = [
            edge
            for edge in self.target_edges()
            if edge.caller == "dspy.adapters.consumer.caller"
        ]
        self.assertEqual([edge.line for edge in caller_edges], [7, 8, 9])
        same_line = {
            edge.callee
            for edge in analyze_calls(SYNTHETIC_SOURCES)
            if edge.caller == "dspy.adapters.consumer.caller" and edge.line == 8
        }
        self.assertEqual(
            same_line,
            {
                "dspy.adapters.utils.other",
                "dspy.adapters.utils.target",
            },
        )

    def test_fails_closed_for_rebinding_shadowing_wildcards_and_dynamic_calls(self):
        callers = {edge.caller for edge in self.target_edges()}
        excluded = {
            "dspy.adapters.consumer.local_rebound",
            "dspy.adapters.consumer.shadow_parameter",
            "dspy.adapters.consumer.dynamic_attribute",
            "dspy.adapters.consumer.rebound_module_alias",
            "dspy.adapters.consumer.calls_rebound_module_name",
            "dspy.adapters.consumer.wildcard",
            "dspy.adapters.consumer.outer_import_shadow.<locals>.inner",
            "dspy.adapters.utils.outer_shadow.<locals>.inner",
        }
        self.assertTrue(callers.isdisjoint(excluded))
        method_edges = [
            edge
            for edge in self.target_edges()
            if edge.caller == "dspy.adapters.consumer.Consumer.method"
        ]
        self.assertEqual(len(method_edges), 1)

    def test_rebound_top_level_function_is_not_an_eligible_callee(self):
        sources = {
            "dspy/adapters/rebound.py": """\
def target():
    return None
target = lambda: None
def caller():
    target()
"""
        }
        self.assertEqual(analyze_calls(sources), ())

    def test_input_scope_and_parse_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            analyze_calls({"tests/outside.py": "pass\n"})
        with self.assertRaisesRegex(ValueError, "outside"):
            analyze_calls({"dspy/adapters/x\\y.py": "pass\n"})
        with self.assertRaisesRegex(ValueError, "cannot parse"):
            analyze_calls({"dspy/adapters/x.py": "def broken(:\n"})

    def test_annotations_are_excluded_but_annotated_values_are_runtime_calls(self):
        sources = {"dspy/adapters/annotations.py": """\
def target():
    return None
def caller():
    annotation_only: target()
    runtime_value: object = target()
"""}
        self.assertEqual(
            analyze_calls(sources),
            (CallEdge(
                "dspy.adapters.annotations.caller",
                "dspy.adapters.annotations.target",
                "dspy/adapters/annotations.py",
                5,
            ),),
        )

    def test_definition_metadata_and_lambda_calls_are_excluded(self):
        sources = {"dspy/adapters/metadata.py": """\
def target():
    return None
@target()
def caller(value: target() = target()) -> target():
    annotation_only: target()
    deferred = lambda: target()
    target()
"""}
        self.assertEqual(
            analyze_calls(sources),
            (CallEdge(
                "dspy.adapters.metadata.caller",
                "dspy.adapters.metadata.target",
                "dspy/adapters/metadata.py",
                7,
            ),),
        )

    def test_nested_function_beneath_control_flow_is_analyzed(self):
        sources = {"dspy/adapters/nested.py": """\
def target():
    return None
def outer(flag):
    if flag:
        def nested():
            target()
    return None
"""}
        self.assertEqual(
            analyze_calls(sources),
            (CallEdge(
                "dspy.adapters.nested.outer.<locals>.nested",
                "dspy.adapters.nested.target",
                "dspy/adapters/nested.py",
                6,
            ),),
        )

    def test_function_beneath_module_body_control_flow_is_outside_relation(self):
        sources = {"dspy/adapters/conditional.py": """\
def target():
    return None
if enabled:
    def conditional():
        target()
def caller():
    conditional()
"""}
        self.assertEqual(analyze_calls(sources), ())

    def test_class_header_and_body_calls_are_excluded_but_methods_are_callers(self):
        sources = {"dspy/adapters/class_scope.py": """\
def target():
    return None
def outer():
    class Inner(target()):
        target()
        def method(self):
            target()
"""}
        self.assertEqual(
            analyze_calls(sources),
            (CallEdge(
                "dspy.adapters.class_scope.outer.<locals>.Inner.method",
                "dspy.adapters.class_scope.target",
                "dspy/adapters/class_scope.py",
                7,
            ),),
        )

    def test_caller_local_import_must_be_on_an_earlier_source_line(self):
        sources = {"dspy/adapters/local_line.py": """\
def target():
    return None
def caller():
    from .local_line import target as local_target; local_target()
"""}
        self.assertEqual(analyze_calls(sources), ())

    def test_match_capture_shadows_module_function(self):
        sources = {"dspy/adapters/match_capture.py": """\
def target():
    return None
def caller(value):
    match value:
        case {"x": target}:
            pass
    target()
"""}
        self.assertEqual(analyze_calls(sources), ())

    def test_global_and_nonlocal_declarations_are_excluded(self):
        sources = {"dspy/adapters/declarations.py": """\
def target():
    return None
def global_caller():
    global target
    target()
def outer():
    target = lambda: None
    def nonlocal_caller():
        nonlocal target
        target()
"""}
        self.assertEqual(analyze_calls(sources), ())

    def test_every_unresolved_function_body_call_is_persistable(self):
        sources = {"dspy/adapters/exclusions.py": """\
def target():
    return None
def caller(obj):
    target()
    obj.method()
    (lambda: None)()
"""}
        records = [candidate.to_dict() for candidate in excluded_call_candidates(sources)]
        self.assertEqual(
            [(record["line"], record["reason"]) for record in records],
            [(5, "unresolved-attribute-callee"),
             (6, "dynamic-callee-expression")],
        )
        self.assertTrue(all(record["caller"] == "dspy.adapters.exclusions.caller"
                            for record in records))


class StaticVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gold = [
            {
                "caller": "dspy.adapters.a.caller",
                "callee": "dspy.adapters.b.target",
                "path": "dspy/adapters/a.py",
                "line": 4,
                "direction": "incoming",
            },
            {
                "caller": "dspy.adapters.b.target",
                "callee": "dspy.adapters.b.helper",
                "path": "dspy/adapters/b.py",
                "line": 9,
                "direction": "outgoing",
            },
        ]

    def encode(self, edges) -> str:
        return json.dumps({"edges": edges}, separators=(",", ":"))

    def test_exact_and_continuous_scores(self):
        exact = verify_prediction(self.encode(self.gold), self.gold)
        self.assertTrue(exact["schema_valid"])
        self.assertTrue(exact["exact"])
        self.assertEqual((exact["tp"], exact["fp"], exact["fn"]), (2, 0, 0))
        self.assertEqual(exact["f1"], 1.0)
        self.assertEqual(exact["predicted_edges"], self.gold)
        self.assertEqual(exact["missing_edges"], [])
        self.assertEqual(exact["spurious_edges"], [])

        false_edge = dict(self.gold[1], line=10)
        partial = verify_prediction(
            self.encode([self.gold[0], false_edge]),
            self.gold,
        )
        self.assertEqual((partial["tp"], partial["fp"], partial["fn"]), (1, 1, 1))
        self.assertEqual(partial["precision"], 0.5)
        self.assertEqual(partial["recall"], 0.5)
        self.assertEqual(partial["f1"], 0.5)
        self.assertFalse(partial["exact"])
        self.assertEqual(partial["missing_edges"], [self.gold[1]])
        self.assertEqual(partial["spurious_edges"], [false_edge])

    def test_duplicate_predictions_are_deduplicated_and_logged(self):
        result = verify_prediction(
            self.encode([*self.gold, self.gold[0]]),
            self.gold,
        )
        self.assertTrue(result["exact"])
        self.assertEqual(result["predicted_count"], 2)
        self.assertEqual(result["duplicate_predictions"], 1)

    def test_schema_errors_score_zero(self):
        invalid = [
            "```json\n{\"edges\":[]}\n```",
            "{\"edge\":[]}",
            "{\"edges\":{},\"extra\":1}",
            self.encode([dict(self.gold[0], line=True)]),
            self.encode([dict(self.gold[0], surprise=1)]),
            self.encode([dict(self.gold[0], direction=[])]),
            '{"edges":[],"edges":[]}',
            '{"edges":[{"caller":"x","callee":"y","path":"../x.py",'
            '"line":1,"direction":"incoming"}]}',
            '{"edges":[{"caller":"x","callee":"y",'
            '"path":"dspy/adapters/x\\\\y.py","line":1,'
            '"direction":"incoming"}]}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                result = verify_prediction(raw, self.gold)
                self.assertFalse(result["schema_valid"])
                self.assertEqual(result["f1"], 0.0)
                self.assertEqual(result["tp"], 0)
                self.assertEqual(result["fn"], 2)
                self.assertFalse(result["exact"])

    def test_empty_prediction_and_empty_oracle_are_exact(self):
        result = verify_prediction('{"edges":[]}', [])
        self.assertTrue(result["exact"])
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_invalid_source_locations_are_logged_and_score_as_false_positives(self):
        invalid = dict(
            self.gold[0],
            path="dspy/adapters/missing.py",
            line=999,
        )
        source_view = {
            "dspy/adapters/a.py": "\n" * 4,
            "dspy/adapters/b.py": "\n" * 9,
        }
        result = verify_prediction(
            self.encode([self.gold[0], invalid]),
            self.gold,
            source_view=source_view,
        )
        self.assertTrue(result["schema_valid"])
        self.assertTrue(result["locations_validated"])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["invalid_location_edges"], [invalid])
        self.assertEqual(result["spurious_edges"], [invalid])
        self.assertEqual(result["missing_edges"], [self.gold[1]])

    def test_parser_returns_canonical_edges_and_duplicate_count(self):
        edges, duplicates = parse_prediction(
            self.encode([self.gold[1], self.gold[0], self.gold[1]])
        )
        self.assertEqual(duplicates, 1)
        self.assertEqual(
            [edge.path for edge in edges],
            ["dspy/adapters/a.py", "dspy/adapters/b.py"],
        )


class PinnedDspyGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_tools = RepoTools(CORPORA["dspy"])
        cls.questions = build_question_bank(cls.repo_tools)

    def test_frozen_target_order_split_and_definition_anchors(self):
        self.assertEqual(
            tuple(question["target"] for question in self.questions),
            FROZEN_TRAIN_TARGETS + FROZEN_DEV_TARGETS,
        )
        self.assertEqual(
            [question["split"] for question in self.questions],
            ["train"] * 16 + ["dev"] * 4,
        )
        self.assertEqual(
            [question["neighborhood_edge_count"] for question in self.questions],
            [10, 6, 1, 4, 9, 4, 2, 5, 9, 6, 9, 5, 8, 7, 7, 7,
             6, 5, 5, 5],
        )
        self.assertTrue(all(
            question["target_definition"]["path"] == question["anchors"][0]
            and question["neighborhood_edge_count"] == len(question["gold_edges"])
            for question in self.questions
        ))

    def test_frozen_neighborhood_sizes_and_format_field_value_gold(self):
        self.assertEqual(
            [len(question["gold_edges"]) for question in self.questions],
            [10, 6, 1, 4, 9, 4, 2, 5, 9, 6, 9, 5, 8, 7, 7, 7,
             6, 5, 5, 5],
        )
        format_question = next(
            question for question in self.questions
            if question["target"] == "dspy.adapters.utils.format_field_value"
        )
        self.assertEqual(
            [
                (
                    edge["caller"],
                    edge["callee"],
                    edge["path"],
                    edge["line"],
                    edge["direction"],
                )
                for edge in format_question["gold_edges"]
            ],
            [
                (
                    "dspy.adapters.baml_adapter.BAMLAdapter.format_user_message_content",
                    "dspy.adapters.utils.format_field_value",
                    "dspy/adapters/baml_adapter.py",
                    260,
                    "incoming",
                ),
                (
                    "dspy.adapters.chat_adapter.ChatAdapter.format_user_message_content",
                    "dspy.adapters.utils.format_field_value",
                    "dspy/adapters/chat_adapter.py",
                    156,
                    "incoming",
                ),
                (
                    "dspy.adapters.chat_adapter.ChatAdapter.format_field_with_value",
                    "dspy.adapters.utils.format_field_value",
                    "dspy/adapters/chat_adapter.py",
                    263,
                    "incoming",
                ),
                (
                    "dspy.adapters.json_adapter.JSONAdapter.format_field_with_value",
                    "dspy.adapters.utils.format_field_value",
                    "dspy/adapters/json_adapter.py",
                    197,
                    "incoming",
                ),
                (
                    "dspy.adapters.utils.format_field_value",
                    "dspy.adapters.utils._format_input_list_field_value",
                    "dspy/adapters/utils.py",
                    59,
                    "outgoing",
                ),
                (
                    "dspy.adapters.utils.format_field_value",
                    "dspy.adapters.utils.serialize_for_json",
                    "dspy/adapters/utils.py",
                    61,
                    "outgoing",
                ),
                (
                    "dspy.adapters.xml_adapter.XMLAdapter.format_field_with_value",
                    "dspy.adapters.utils.format_field_value",
                    "dspy/adapters/xml_adapter.py",
                    20,
                    "incoming",
                ),
            ],
        )

    def test_full_source_strata_and_dev_evidence_are_frozen(self):
        train = [question for question in self.questions if question["split"] == "train"]
        dev = [question for question in self.questions if question["split"] == "dev"]
        base = [
            question for question in train
            if question["selection_stage"] == "base-stratum-train"
        ]
        self.assertEqual(
            {question["stratum"] for question in base},
            {
                "adapters", "clients", "datasets", "dsp", "evaluate",
                "predict", "primitives", "propose", "signatures",
                "streaming", "teleprompt", "utils",
            },
        )
        self.assertEqual(len(base), 12)
        train_locations = {
            (edge["path"], edge["line"])
            for question in train for edge in question["gold_edges"]
        }
        dev_locations = {
            (edge["path"], edge["line"])
            for question in dev for edge in question["gold_edges"]
        }
        self.assertFalse(train_locations & dev_locations)
        self.assertEqual(sum(len(question["gold_edges"]) for question in train), 99)
        self.assertEqual(sum(len(question["gold_edges"]) for question in dev), 21)

    def test_contract_and_question_bank_hashes_are_persistable(self):
        record = resolver_contract_record(self.repo_tools)
        self.assertEqual(record["version"], CONTRACT_VERSION)
        self.assertEqual(record["source_root"], "dspy")
        self.assertEqual(record["contract_sha256"], contract_sha256())
        self.assertEqual(
            record["contract_sha256"],
            "999daa8224ecb80fbb7d2f9c24b87463b6d59d09db5ed1691fb0c1be9c2ac142",
        )
        self.assertEqual(
            record["source_sha256"],
            "375cf6fc8379ac2db817046eca8032cc83093bee207a4f4ba605cb72298932d5",
        )
        self.assertEqual(
            record["question_bank_sha256"],
            "1ec5df5730e920ed57decc8ff5dc03481da401c93011091ca7a88409afe4c20b",
        )
        self.assertEqual(record["train_question_count"], 16)
        self.assertEqual(record["dev_question_count"], 4)
        self.assertEqual(len(record["source_files"]), 139)
        self.assertTrue(
            all(
                set(item) == {"path", "sha256", "line_count"}
                for item in record["source_files"]
            )
        )
        self.assertRegex(record["python_version"], r"^3\.\d+\.\d+$")
        self.assertEqual(record["analyzer_module"], "studybench.static_graph")
        self.assertRegex(record["analyzer_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(record["corpus_name"], "dspy")
        self.assertEqual(record["corpus_commit"], CORPORA["dspy"].commit)
        self.assertGreater(record["excluded_candidate_count"], 0)
        self.assertEqual(
            record["excluded_candidate_count"], len(record["excluded_candidates"])
        )
        self.assertRegex(record["excluded_candidates_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(record["candidate_count"], 230)
        self.assertEqual(record["eligible_candidate_count"], 187)
        self.assertEqual(
            record["target_selection_sha256"],
            "1f5de691f0864297c009f32e0c38c6371cece304540f61d4b53ea2773e287876",
        )
        self.assertEqual(
            record["candidate_inventory_sha256"],
            "0e9cd2f6d8ded34265f14b0e30494dfc3772442c37390228f4e28fa526b119cf",
        )
        if record["python_version"] == "3.12.11":
            self.assertEqual(record["excluded_candidate_count"], 5776)
            self.assertEqual(
                record["excluded_candidates_sha256"],
                "42d386c2e22eaae3bf5f0b337d10332de1b5367d57af0b2b9cdf7536830778c6",
            )
        self.assertTrue(all(
            set(candidate) == {
                "caller", "path", "line", "column", "callee_syntax", "reason"
            }
            for candidate in record["excluded_candidates"]
        ))

    def test_repo_tools_selection_and_public_validation(self):
        selected = dspy_python_texts(self.repo_tools)
        self.assertTrue(selected)
        self.assertTrue(
            all(
                path.startswith("dspy/")
                and path.endswith(".py")
                for path in selected
            )
        )
        self.assertTrue(any(path.startswith("dspy/teleprompt/") for path in selected))
        with self.assertRaisesRegex(ValueError, "fully qualified"):
            render_question("format_field_value")

        prompt = self.questions[0]["question"]
        for clause in (
            "local name has exactly one module-scope binding",
            "methods and nested functions are not eligible callees",
            "inserting `<locals>` between a function scope",
            "explicit absolute or relative `from ... import ...` binding",
            "explicit `import module as alias` binding",
            "on an earlier source line than the call",
            "self/cls/dynamic attributes",
            "closures over enclosing local imports",
            "names declared global/nonlocal",
            "calls through values or higher-order arguments",
            "class-definition headers",
            "decorators, defaults, annotations, or lambdas",
            "definitions beneath module-body control flow",
            "identical caller/callee/path/line sites",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, prompt)


if __name__ == "__main__":
    unittest.main()
