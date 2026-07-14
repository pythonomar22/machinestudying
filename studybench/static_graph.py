"""Exact questions over a conservative, deterministic Python call relation.

This is deliberately not a runtime call graph. It recognizes only a small,
auditable subset of calls whose callee follows from syntax and explicit imports.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


CONTRACT_VERSION = "dspy-source-static-call-neighborhood-v1"
SOURCE_ROOT = "dspy"
SOURCE_PREFIX = f"{SOURCE_ROOT}/"
MIN_TARGET_EDGES = 1
MAX_TARGET_EDGES = 10
BASE_TRAIN_PER_STRATUM = 1
EXTRA_TRAIN_TARGETS = 4
DEV_TARGETS = 4

# These are not hand-selected targets.  They are the expected output of the
# corpus-only selector frozen below at the pinned DSPy commit.  Keeping the
# expected result explicit makes an implementation or corpus drift fail closed.
FROZEN_TRAIN_TARGETS = (
    "dspy.adapters.utils.get_field_description_string",
    "dspy.clients.lm._add_dspy_identifier_to_headers",
    "dspy.datasets.gsm8k.gsm8k_metric",
    "dspy.dsp.utils.utils.print_message",
    "dspy.evaluate.metrics.normalize_text",
    "dspy.predict.predict.serialize_object",
    "dspy.primitives.python_interpreter._jsonrpc_notification",
    "dspy.propose.dataset_summary_generator.create_dataset_summary",
    "dspy.signatures.signature.ensure_signature",
    "dspy.streaming.messages.sync_send_to_stream",
    "dspy.teleprompt.utils.get_signature",
    "dspy.utils.inspect_history._red",
    "dspy.adapters.utils.translate_field_type",
    "dspy.adapters.baml_adapter._render_type_str",
    "dspy.adapters.types.image.encode_image",
    "dspy.adapters.utils.format_field_value",
)
FROZEN_DEV_TARGETS = (
    "dspy.adapters.utils.parse_value",
    "dspy.signatures.signature._parse_signature",
    "dspy.teleprompt.bootstrap_finetune.all_predictors_have_lms",
    "dspy.teleprompt.utils.eval_candidate_program",
)
EDGE_FIELDS = frozenset({"caller", "callee", "path", "line", "direction"})
DIRECTIONS = frozenset({"incoming", "outgoing", "self"})

CONTRACT: dict[str, Any] = {
    "version": CONTRACT_VERSION,
    "language": "python",
    "source_scope": "tracked UTF-8 .py files beneath dspy/",
    "eligible_callees": (
        "undecorated sync or async function definitions which are direct statements "
        "in a module body and whose local name has exactly one module-scope binding; "
        "any other module-scope binding makes the name ineligible"
    ),
    "callers": (
        "function bodies reached from direct module-body function or class "
        "definitions, including methods and nested functions; module bodies, class "
        "definition headers and bodies, decorators, defaults, annotations, lambda "
        "bodies, and definitions beneath module-body control flow are excluded"
    ),
    "caller_symbol": (
        "module-qualified lexical names; class and method components are dot joined, "
        "and <locals> is inserted between a function scope and each function or "
        "class defined inside it"
    ),
    "resolved_calls": [
        "unshadowed same-module bare names",
        "unambiguously bound explicit absolute or relative from-import aliases",
        "attributes of unambiguously bound explicit import-module-as aliases",
        "direct function-body imports on an earlier source line than the call and "
        "whose local name has no other binding",
    ],
    "excluded_calls": [
        "methods as callees, self/cls/dynamic attributes, wildcard imports, "
        "implicit package attributes, closures over enclosing local imports, and "
        "names declared global/nonlocal, ambiguous names, and rebound names",
        "runtime dispatch, monkey-patching, decorator replacement, reflection, "
        "and calls through values or higher-order arguments",
    ],
    "excluded_candidate_record": (
        "every function-body ast.Call outside the resolved subset is retained "
        "with caller, path, line, zero-indexed column, attribute-free ast.dump "
        "callee syntax, and one "
        "of unresolved-bare-callee, unresolved-attribute-callee, or "
        "dynamic-callee-expression"
    ),
    "edge_identity": ["caller", "callee", "path", "line"],
    "line_granularity": (
        "calls with the same caller, callee, path, and line collapse to one edge"
    ),
    "question_neighborhood": (
        "all resolved calls to the target plus all resolved calls made directly "
        "by the target; target recursion is one self edge"
    ),
    "target_selection": {
        "benchmark_independent": True,
        "candidate": (
            "an eligible callee with between 1 and 10 inclusive resolved "
            "neighborhood edges"
        ),
        "stratum": (
            "first module component beneath dspy; __root__ for a function "
            "defined directly in dspy/__init__.py"
        ),
        "rank": "descending neighborhood edge count, then target symbol",
        "base_train": "highest-ranked candidate in every stratum",
        "extra_train": (
            "four highest-ranked remaining candidates whose (path,line) "
            "evidence locations are disjoint from prior training evidence"
        ),
        "dev": (
            "four highest-ranked remaining candidates whose (path,line) "
            "evidence locations are disjoint from all training and prior dev "
            "evidence"
        ),
        "minimum_edges": MIN_TARGET_EDGES,
        "maximum_edges": MAX_TARGET_EDGES,
        "base_train_per_stratum": BASE_TRAIN_PER_STRATUM,
        "extra_train_count": EXTRA_TRAIN_TARGETS,
        "dev_count": DEV_TARGETS,
    },
    "frozen_targets": {
        "train": list(FROZEN_TRAIN_TARGETS),
        "dev": list(FROZEN_DEV_TARGETS),
    },
    "question_record": (
        "id, fully qualified target, question, canonical source-path anchors, "
        "target definition path/line, gold edges, and frozen split"
    ),
    "prediction_schema": {
        "top_level": {"edges": "array"},
        "edge_fields": ["caller", "callee", "path", "line", "direction"],
        "additional_fields": False,
        "directions": ["incoming", "outgoing", "self"],
    },
    "scoring": (
        "set precision/recall/F1 over exact question edges; malformed JSON or "
        "schema violations score zero; duplicate predictions are deduplicated "
        "and counted; when a source view is supplied, nonexistent paths and "
        "out-of-range lines remain schema-valid but are logged and scored as "
        "false positives"
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def contract_sha256() -> str:
    """Hash the complete analyzer and scoring contract."""
    return _sha256_json(CONTRACT)


@dataclass(frozen=True, order=True)
class CallEdge:
    """One line-granular call to an eligible direct module-body function."""

    caller: str
    callee: str
    path: str
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "callee": self.callee,
            "path": self.path,
            "line": self.line,
        }


@dataclass(frozen=True, order=True)
class QuestionEdge:
    """A call edge labeled relative to one question's target."""

    caller: str
    callee: str
    path: str
    line: int
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "callee": self.callee,
            "path": self.path,
            "line": self.line,
            "direction": self.direction,
        }


@dataclass(frozen=True, order=True)
class ExcludedCallCandidate:
    """One visited function-body call outside the conservative resolver."""

    caller: str
    path: str
    line: int
    column: int
    callee_syntax: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "callee_syntax": self.callee_syntax,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StaticQuestion:
    id: str
    target: str
    prompt: str
    oracle: tuple[QuestionEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "prompt": self.prompt,
            "oracle": [edge.to_dict() for edge in self.oracle],
        }


@dataclass(frozen=True)
class QuestionBank:
    contract_version: str
    contract_sha256: str
    source_sha256: str
    questions: tuple[StaticQuestion, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "contract_sha256": self.contract_sha256,
            "source_sha256": self.source_sha256,
            "questions": [question.to_dict() for question in self.questions],
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class VerificationResult:
    schema_valid: bool
    schema_error: str | None
    duplicate_predictions: int
    predicted_count: int
    oracle_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    exact: bool
    locations_validated: bool
    predicted_edges: tuple[QuestionEdge, ...]
    missing_edges: tuple[QuestionEdge, ...]
    spurious_edges: tuple[QuestionEdge, ...]
    invalid_location_edges: tuple[QuestionEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_valid": self.schema_valid,
            "schema_error": self.schema_error,
            "duplicate_predictions": self.duplicate_predictions,
            "predicted_count": self.predicted_count,
            "oracle_count": self.oracle_count,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact": self.exact,
            "locations_validated": self.locations_validated,
            "predicted_edges": [
                edge.to_dict() for edge in self.predicted_edges
            ],
            "missing_edges": [
                edge.to_dict() for edge in self.missing_edges
            ],
            "spurious_edges": [
                edge.to_dict() for edge in self.spurious_edges
            ],
            "invalid_location_edges": [
                edge.to_dict() for edge in self.invalid_location_edges
            ],
        }


@dataclass(frozen=True)
class _Function:
    module: str
    path: str
    node: ast.FunctionDef | ast.AsyncFunctionDef

    @property
    def symbol(self) -> str:
        return f"{self.module}.{self.node.name}"


@dataclass
class _Module:
    module: str
    path: str
    tree: ast.Module
    eligible: dict[str, _Function]
    imports: dict[str, str]
    module_aliases: dict[str, str]


class _Bindings(ast.NodeVisitor):
    """Collect bindings in one lexical scope without entering child scopes."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.global_or_nonlocal: set[str] = set()

    def visit_arg(self, node: ast.arg) -> None:
        self.counts[node.arg] += 1

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.counts[node.id] += 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.counts[node.name] += 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.counts[node.name] += 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.counts[node.name] += 1

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.counts[alias.asname or alias.name.split(".", 1)[0]] += 1

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.counts[alias.asname or alias.name] += 1

    def visit_Global(self, node: ast.Global) -> None:
        self.global_or_nonlocal.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.global_or_nonlocal.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.counts[node.name] += 1
        for statement in node.body:
            self.visit(statement)

    # Structural pattern captures bind local names but are not ast.Name(Store)
    # nodes. Missing these would falsely resolve a captured name to a module
    # function everywhere in the callable.
    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.counts[node.name] += 1
        if node.pattern is not None:
            self.visit(node.pattern)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.counts[node.name] += 1

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.counts[node.rest] += 1
        for pattern in node.patterns:
            self.visit(pattern)


def _scope_bindings(
    body: Iterable[ast.stmt], arguments: ast.arguments | None = None
) -> _Bindings:
    bindings = _Bindings()
    if arguments is not None:
        for arg in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            bindings.visit(arg)
        if arguments.vararg:
            bindings.visit(arguments.vararg)
        if arguments.kwarg:
            bindings.visit(arguments.kwarg)
    for statement in body:
        bindings.visit(statement)
    return bindings


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import_module(
    module: str, path: str, node: ast.ImportFrom
) -> str | None:
    if not node.level:
        return node.module
    package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    remove = node.level - 1
    if remove >= len(parts):
        return None
    base = parts[: len(parts) - remove]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _direct_import_candidates(
    body: Iterable[ast.stmt], module: str, path: str
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    imported: dict[str, str] = {}
    module_aliases: dict[str, str] = {}
    lines: dict[str, int] = {}
    ambiguous: set[str] = set()

    def add(destination: dict[str, str], local: str, value: str, line: int) -> None:
        if local in imported or local in module_aliases:
            ambiguous.add(local)
            return
        destination[local] = value
        lines[local] = line

    for statement in body:
        if isinstance(statement, ast.ImportFrom):
            source = _resolve_import_module(module, path, statement)
            if source is None:
                continue
            for alias in statement.names:
                if alias.name != "*":
                    add(
                        imported,
                        alias.asname or alias.name,
                        f"{source}.{alias.name}",
                        statement.lineno,
                    )
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                # `import a.b` without `as` binds `a`, not the leaf module.
                if alias.asname:
                    add(
                        module_aliases,
                        alias.asname,
                        alias.name,
                        statement.lineno,
                    )
    for local in ambiguous:
        imported.pop(local, None)
        module_aliases.pop(local, None)
        lines.pop(local, None)
    return imported, module_aliases, lines


def _validated_sources(texts: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(texts, Mapping):
        raise TypeError("texts must be a path-to-source mapping")
    sources: dict[str, str] = {}
    for path, source in texts.items():
        if not isinstance(path, str) or not isinstance(source, str):
            raise TypeError("source paths and contents must be strings")
        logical = PurePosixPath(path)
        if (
            not path.startswith(SOURCE_PREFIX)
            or path != path.strip()
            or "\\" in path
            or "\x00" in path
            or logical.suffix != ".py"
            or logical.is_absolute()
            or logical.as_posix() != path
            or any(part in ("", ".", "..") for part in logical.parts)
        ):
            raise ValueError(f"source is outside the DSPy Python scope: {path!r}")
        sources[path] = source
    if not sources:
        raise ValueError("DSPy source mapping is empty")
    return dict(sorted(sources.items()))


def dspy_python_texts(repo_tools: Any) -> dict[str, str]:
    """Select production DSPy Python sources from validated `RepoTools`."""
    texts = getattr(repo_tools, "text", None)
    if not isinstance(texts, dict):
        raise TypeError("repo_tools must expose its validated text mapping")
    return _validated_sources(
        {
            path: source
            for path, source in texts.items()
            if path.startswith(SOURCE_PREFIX) and path.endswith(".py")
        }
    )


def _build_modules(
    sources: Mapping[str, str],
) -> tuple[dict[str, _Module], dict[str, _Function]]:
    modules: dict[str, _Module] = {}
    functions: dict[str, _Function] = {}
    for path, source in sources.items():
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            raise ValueError(f"cannot parse DSPy source {path}: {exc}") from exc
        module_name = _module_name(path)
        if not module_name or module_name in modules:
            raise ValueError(f"duplicate or invalid module for path: {path}")
        bindings = _scope_bindings(tree.body)
        eligible: dict[str, _Function] = {}
        for statement in tree.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # A decorator may replace the binding at runtime.
            if statement.decorator_list or bindings.counts[statement.name] != 1:
                continue
            function = _Function(module_name, path, statement)
            eligible[statement.name] = function
            functions[function.symbol] = function
        imports, aliases, _ = _direct_import_candidates(
            tree.body, module_name, path
        )
        imports = {
            local: symbol
            for local, symbol in imports.items()
            if bindings.counts[local] == 1
            and local not in bindings.global_or_nonlocal
        }
        aliases = {
            local: target
            for local, target in aliases.items()
            if bindings.counts[local] == 1
            and local not in bindings.global_or_nonlocal
        }
        modules[module_name] = _Module(
            module_name, path, tree, eligible, imports, aliases
        )
    return modules, functions


class _Calls(ast.NodeVisitor):
    """Resolve calls in exactly one callable body."""

    def __init__(
        self,
        *,
        info: _Module,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        caller: str,
        eligible_symbols: set[str],
        enclosing_bound: frozenset[str],
    ) -> None:
        self.info = info
        self.caller = caller
        self.eligible_symbols = eligible_symbols
        self.enclosing_bound = enclosing_bound
        self.edges: set[CallEdge] = set()
        self.excluded_candidates: set[ExcludedCallCandidate] = set()
        self.bindings = _scope_bindings(function.body, function.args)
        imports, aliases, import_lines = _direct_import_candidates(
            function.body, info.module, info.path
        )
        self.local_imports = {
            local: symbol
            for local, symbol in imports.items()
            if self.bindings.counts[local] == 1
            and local not in self.bindings.global_or_nonlocal
            and symbol in eligible_symbols
        }
        self.local_aliases = {
            local: module
            for local, module in aliases.items()
            if self.bindings.counts[local] == 1
            and local not in self.bindings.global_or_nonlocal
        }
        self.import_lines = import_lines

    def _bare(self, name: str, line: int) -> str | None:
        if (
            name in self.bindings.global_or_nonlocal
            or name in self.enclosing_bound
        ):
            return None
        if name in self.local_imports:
            return self.local_imports[name] if self.import_lines[name] < line else None
        if self.bindings.counts[name]:
            return None
        local = self.info.eligible.get(name)
        if local:
            return local.symbol
        imported = self.info.imports.get(name)
        return imported if imported in self.eligible_symbols else None

    def _attribute(self, node: ast.Attribute, line: int) -> str | None:
        if not isinstance(node.value, ast.Name):
            return None
        name = node.value.id
        if (
            name in self.bindings.global_or_nonlocal
            or name in self.enclosing_bound
        ):
            return None
        if name in self.local_aliases:
            if self.import_lines[name] >= line:
                return None
            module = self.local_aliases[name]
        elif self.bindings.counts[name]:
            return None
        else:
            module = self.info.module_aliases.get(name)
        candidate = f"{module}.{node.attr}" if module else None
        return candidate if candidate in self.eligible_symbols else None

    def visit_Call(self, node: ast.Call) -> None:
        callee = None
        if isinstance(node.func, ast.Name):
            callee = self._bare(node.func.id, node.lineno)
        elif isinstance(node.func, ast.Attribute):
            callee = self._attribute(node.func, node.lineno)
        if callee:
            self.edges.add(
                CallEdge(self.caller, callee, self.info.path, node.lineno)
            )
        else:
            if isinstance(node.func, ast.Name):
                reason = "unresolved-bare-callee"
            elif isinstance(node.func, ast.Attribute):
                reason = "unresolved-attribute-callee"
            else:
                reason = "dynamic-callee-expression"
            self.excluded_candidates.add(ExcludedCallCandidate(
                caller=self.caller,
                path=self.info.path,
                line=node.lineno,
                column=node.col_offset,
                callee_syntax=ast.dump(
                    node.func, annotate_fields=True, include_attributes=False
                ),
                reason=reason,
            ))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Assignment targets and values execute in the function body; the
        # annotation is explicitly outside this relation.
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        # Python 3.12 `type` statements are annotation-scope declarations.
        return

    # These bodies execute in different lexical scopes. They are analyzed
    # separately, while lambda calls are deliberately outside the contract.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class _LexicalChildren(ast.NodeVisitor):
    """Find child callables beneath control flow without entering their scopes."""

    def __init__(self) -> None:
        self.nodes: list[
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.nodes.append(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.nodes.append(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.nodes.append(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _lexical_children(
    body: Iterable[ast.stmt],
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, ...]:
    visitor = _LexicalChildren()
    for statement in body:
        visitor.visit(statement)
    return tuple(visitor.nodes)


def _callable_edges(
    info: _Module,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualifier: tuple[str, ...],
    eligible_symbols: set[str],
    enclosing_bound: frozenset[str],
    excluded_candidates: set[ExcludedCallCandidate] | None = None,
) -> set[CallEdge]:
    caller = f"{info.module}.{'.'.join(qualifier)}"
    visitor = _Calls(
        info=info,
        function=node,
        caller=caller,
        eligible_symbols=eligible_symbols,
        enclosing_bound=enclosing_bound,
    )
    for statement in node.body:
        visitor.visit(statement)
    edges = set(visitor.edges)
    if excluded_candidates is not None:
        excluded_candidates.update(visitor.excluded_candidates)
    child_enclosing = enclosing_bound | frozenset(visitor.bindings.counts)
    for statement in _lexical_children(node.body):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            edges.update(
                _callable_edges(
                    info,
                    statement,
                    (*qualifier, "<locals>", statement.name),
                    eligible_symbols,
                    child_enclosing,
                    excluded_candidates,
                )
            )
        elif isinstance(statement, ast.ClassDef):
            edges.update(
                _class_edges(
                    info,
                    statement,
                    (*qualifier, "<locals>", statement.name),
                    eligible_symbols,
                    child_enclosing,
                    excluded_candidates,
                )
            )
    return edges


def _class_edges(
    info: _Module,
    node: ast.ClassDef,
    qualifier: tuple[str, ...],
    eligible_symbols: set[str],
    enclosing_bound: frozenset[str],
    excluded_candidates: set[ExcludedCallCandidate] | None = None,
) -> set[CallEdge]:
    edges: set[CallEdge] = set()
    for statement in _lexical_children(node.body):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            edges.update(
                _callable_edges(
                    info,
                    statement,
                    (*qualifier, statement.name),
                    eligible_symbols,
                    enclosing_bound,
                    excluded_candidates,
                )
            )
        elif isinstance(statement, ast.ClassDef):
            edges.update(
                _class_edges(
                    info,
                    statement,
                    (*qualifier, statement.name),
                    eligible_symbols,
                    enclosing_bound,
                    excluded_candidates,
                )
            )
    return edges


def _analyze_calls(
    texts: Mapping[str, str],
) -> tuple[tuple[CallEdge, ...], tuple[ExcludedCallCandidate, ...]]:
    sources = _validated_sources(texts)
    modules, functions = _build_modules(sources)
    eligible_symbols = set(functions)
    edges: set[CallEdge] = set()
    excluded: set[ExcludedCallCandidate] = set()
    for info in modules.values():
        for statement in info.tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                edges.update(
                    _callable_edges(
                        info,
                        statement,
                        (statement.name,),
                        eligible_symbols,
                        frozenset(),
                        excluded,
                    )
                )
            elif isinstance(statement, ast.ClassDef):
                edges.update(
                    _class_edges(
                        info,
                        statement,
                        (statement.name,),
                        eligible_symbols,
                        frozenset(),
                        excluded,
                    )
                )
    resolved = tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.path,
                edge.line,
                edge.caller,
                edge.callee,
            ),
        )
    )
    return resolved, tuple(sorted(excluded))


def analyze_calls(texts: Mapping[str, str]) -> tuple[CallEdge, ...]:
    """Return every statically resolved internal edge under `CONTRACT`."""
    return _analyze_calls(texts)[0]


def excluded_call_candidates(
    texts: Mapping[str, str],
) -> tuple[ExcludedCallCandidate, ...]:
    """Return every visited call rejected by the conservative resolver."""
    return _analyze_calls(texts)[1]


def _question_edges(
    target: str, edges: Iterable[CallEdge]
) -> tuple[QuestionEdge, ...]:
    selected: set[QuestionEdge] = set()
    for edge in edges:
        if edge.caller == target and edge.callee == target:
            direction = "self"
        elif edge.callee == target:
            direction = "incoming"
        elif edge.caller == target:
            direction = "outgoing"
        else:
            continue
        selected.add(
            QuestionEdge(
                edge.caller,
                edge.callee,
                edge.path,
                edge.line,
                direction,
            )
        )
    return tuple(
        sorted(
            selected,
            key=lambda edge: (
                edge.path,
                edge.line,
                edge.caller,
                edge.callee,
                edge.direction,
            ),
        )
    )


def render_question(target: str) -> str:
    """Render the canonical question text for a fully qualified target."""
    if not isinstance(target, str) or not target.startswith("dspy."):
        raise ValueError("target must be a fully qualified dspy symbol")
    return (
        f"Enumerate the conservative static call neighborhood of `{target}` in "
        "the pinned `dspy/**/*.py` sources. Include every resolved "
        "internal call to the target (direction `incoming`) and every resolved "
        "call made directly by the target to another eligible DSPy function "
        "(direction `outgoing`). Label direct recursion `self`. Use "
        "the exact fully qualified caller and callee, repository-relative path, "
        "and 1-indexed call line. This is the bounded syntactic relation defined "
        f"by contract `{CONTRACT_VERSION}`, not a runtime call graph. Eligible "
        "callees are undecorated sync or async functions defined as direct "
        "statements in a DSPy module body whose local name has exactly one "
        "module-scope binding. Callers are bodies of direct "
        "module-body functions plus methods and nested functions reached from "
        "direct module-body function or class definitions; methods and nested "
        "functions are not eligible callees. Caller symbols use module-qualified "
        "dotted lexical names, inserting `<locals>` between a function scope and "
        "each function or class defined inside it. Resolve an eligible callee only via "
        "an unshadowed same-module bare name, an unambiguous explicit absolute or "
        "relative `from ... import ...` binding, or an attribute of an unambiguous "
        "explicit `import module as alias` binding. Caller-local imports use those "
        "same forms, must be direct statements in the caller body, and must occur "
        "on an earlier source line than the call. Exclude self/cls/dynamic "
        "attributes, constructors, wildcards, implicit package attributes, "
        "closures over enclosing local imports, ambiguous or rebound names, and "
        "names declared global/nonlocal. Exclude calls through values or "
        "higher-order arguments. Module bodies, class-definition headers and "
        "bodies, and "
        "calls in decorators, defaults, annotations, or lambdas are excluded; "
        "definitions beneath module-body control flow are also outside the "
        "relation. Collapse "
        "identical caller/callee/path/line sites. Return "
        "only strict JSON shaped as {\"edges\":[{\"caller\":\"...\","
        "\"callee\":\"...\",\"path\":\"...\",\"line\":1,"
        "\"direction\":\"incoming|outgoing|self\"}]} with no additional keys."
    )


def _build_question_bank_record(
    texts: Mapping[str, str],
    target_symbols: Iterable[str],
) -> QuestionBank:
    """Build a hash-bound question bank for exact eligible target symbols."""
    sources = _validated_sources(texts)
    _, functions = _build_modules(sources)
    edges = analyze_calls(sources)
    requested = tuple(target_symbols)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("target_symbols must be nonempty and unique")
    questions: list[StaticQuestion] = []
    for target in requested:
        if not isinstance(target, str) or not target:
            raise TypeError("target symbols must be nonempty strings")
        if target not in functions:
            raise ValueError(f"target is not one exact eligible definition: {target!r}")
        questions.append(
            StaticQuestion(
                id=f"static-call-neighborhood::{target}",
                target=target,
                prompt=render_question(target),
                oracle=_question_edges(target, edges),
            )
        )
    source_payload = [
        {
            "path": path,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        for path, source in sources.items()
    ]
    return QuestionBank(
        contract_version=CONTRACT_VERSION,
        contract_sha256=contract_sha256(),
        source_sha256=_sha256_json(source_payload),
        questions=tuple(questions),
    )


def _target_stratum(target: str) -> str:
    parts = target.split(".")
    if len(parts) < 2 or parts[0] != SOURCE_ROOT or any(not part for part in parts):
        raise ValueError(f"invalid fully qualified DSPy target: {target!r}")
    return parts[1] if len(parts) > 2 else "__root__"


def _edge_locations(edges: Iterable[QuestionEdge]) -> set[tuple[str, int]]:
    return {(edge.path, edge.line) for edge in edges}


def _selection_records(
    sources: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the complete candidate inventory and frozen train/dev selection."""

    validated = _validated_sources(sources)
    _, functions = _build_modules(validated)
    edges = analyze_calls(validated)
    oracles = {
        target: _question_edges(target, edges)
        for target in functions
    }
    eligible_by_stratum: dict[str, list[str]] = {}
    inventory: list[dict[str, Any]] = []
    for target, function in sorted(functions.items()):
        edge_count = len(oracles[target])
        eligible = MIN_TARGET_EDGES <= edge_count <= MAX_TARGET_EDGES
        if edge_count < MIN_TARGET_EDGES:
            exclusion = "below-minimum-neighborhood"
        elif edge_count > MAX_TARGET_EDGES:
            exclusion = "above-maximum-neighborhood"
        else:
            exclusion = None
        stratum = _target_stratum(target)
        if eligible:
            eligible_by_stratum.setdefault(stratum, []).append(target)
        inventory.append({
            "target": target,
            "stratum": stratum,
            "definition_path": function.path,
            "definition_line": function.node.lineno,
            "neighborhood_edge_count": edge_count,
            "eligible": eligible,
            "exclusion": exclusion,
            "stratum_rank": None,
            "global_rank": None,
        })

    rank_key = lambda target: (-len(oracles[target]), target)
    for targets in eligible_by_stratum.values():
        targets.sort(key=rank_key)
    if not eligible_by_stratum:
        raise ValueError("target selector found no eligible strata")
    global_targets = sorted(
        (target for targets in eligible_by_stratum.values() for target in targets),
        key=rank_key,
    )
    stratum_ranks = {
        target: rank
        for targets in eligible_by_stratum.values()
        for rank, target in enumerate(targets, start=1)
    }
    global_ranks = {
        target: rank for rank, target in enumerate(global_targets, start=1)
    }
    for record in inventory:
        target = record["target"]
        if record["eligible"]:
            record["stratum_rank"] = stratum_ranks[target]
            record["global_rank"] = global_ranks[target]

    selected: list[dict[str, Any]] = []
    selected_targets: set[str] = set()
    train_locations: set[tuple[str, int]] = set()

    def admit(target: str, split: str, stage: str) -> None:
        selected_targets.add(target)
        locations = _edge_locations(oracles[target])
        if split == "train":
            train_locations.update(locations)
        selected.append({
            "target": target,
            "stratum": _target_stratum(target),
            "split": split,
            "selection_stage": stage,
            "neighborhood_edge_count": len(oracles[target]),
            "stratum_rank": stratum_ranks[target],
            "global_rank": global_ranks[target],
        })

    for stratum in sorted(eligible_by_stratum):
        candidates = eligible_by_stratum[stratum]
        if len(candidates) < BASE_TRAIN_PER_STRATUM:
            raise ValueError(f"stratum has no base training target: {stratum}")
        for target in candidates[:BASE_TRAIN_PER_STRATUM]:
            admit(target, "train", "base-stratum-train")

    extra_train_admitted = 0
    for target in global_targets:
        if target in selected_targets:
            continue
        locations = _edge_locations(oracles[target])
        if locations & train_locations:
            continue
        admit(target, "train", "global-extra-train")
        extra_train_admitted += 1
        if extra_train_admitted == EXTRA_TRAIN_TARGETS:
            break
    if extra_train_admitted != EXTRA_TRAIN_TARGETS:
        raise ValueError("not enough location-disjoint extra training targets")

    dev_locations: set[tuple[str, int]] = set()
    dev_admitted = 0
    for target in global_targets:
        if target in selected_targets:
            continue
        locations = _edge_locations(oracles[target])
        if locations & train_locations or locations & dev_locations:
            continue
        admit(target, "dev", "global-held-out-dev")
        dev_locations.update(locations)
        dev_admitted += 1
        if dev_admitted == DEV_TARGETS:
            break
    if dev_admitted != DEV_TARGETS:
        raise ValueError("not enough location-disjoint development targets")

    train = tuple(item["target"] for item in selected if item["split"] == "train")
    dev = tuple(item["target"] for item in selected if item["split"] == "dev")
    if train != FROZEN_TRAIN_TARGETS or dev != FROZEN_DEV_TARGETS:
        raise ValueError(
            "corpus-only target selection drifted from the frozen DSPy split"
        )
    return inventory, selected


def _sources_from_view(view: Any) -> dict[str, str]:
    if isinstance(view, Mapping):
        return _validated_sources(view)
    return dspy_python_texts(view)


def build_question_bank(repo_tools: Any) -> list[dict[str, Any]]:
    """Return the frozen full-source 16-train/four-dev question bank."""
    sources = _sources_from_view(repo_tools)
    _, selection = _selection_records(sources)
    bank = _build_question_bank_record(
        sources, (item["target"] for item in selection)
    )
    _, functions = _build_modules(sources)
    selection_by_target = {item["target"]: item for item in selection}
    records: list[dict[str, Any]] = []
    for question in bank.questions:
        function = functions[question.target]
        selected = selection_by_target[question.target]
        records.append(
            {
                "id": question.id,
                "target": question.target,
                "question": question.prompt,
                "anchors": [function.path],
                "target_definition": {
                    "path": function.path,
                    "line": function.node.lineno,
                },
                "gold_edges": [
                    edge.to_dict() for edge in question.oracle
                ],
                "split": selected["split"],
                "stratum": selected["stratum"],
                "selection_stage": selected["selection_stage"],
                "neighborhood_edge_count": selected["neighborhood_edge_count"],
                "stratum_rank": selected["stratum_rank"],
                "global_rank": selected["global_rank"],
            }
        )
    train_locations = {
        (edge["path"], edge["line"])
        for question in records
        if question["split"] == "train"
        for edge in question["gold_edges"]
    }
    dev_locations = {
        (edge["path"], edge["line"])
        for question in records
        if question["split"] == "dev"
        for edge in question["gold_edges"]
    }
    overlap = sorted(train_locations & dev_locations)
    if overlap:
        raise ValueError(
            "development evidence locations overlap training evidence: "
            f"{overlap}"
        )
    return records


def resolver_contract_record(repo_tools: Any) -> dict[str, Any]:
    """Return hashes sufficient to persist and later reconstruct the oracle."""
    sources = _sources_from_view(repo_tools)
    candidate_inventory, selection = _selection_records(sources)
    bank = _build_question_bank_record(
        sources, (item["target"] for item in selection)
    )
    questions = build_question_bank(sources)
    excluded_candidates = [
        candidate.to_dict() for candidate in excluded_call_candidates(sources)
    ]
    source_files = [
        {
            "path": path,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "line_count": len(source.splitlines()),
        }
        for path, source in sources.items()
    ]
    corpus = getattr(repo_tools, "corpus", None)
    return {
        "version": CONTRACT_VERSION,
        "source_root": SOURCE_ROOT,
        "contract_sha256": bank.contract_sha256,
        "source_sha256": bank.source_sha256,
        "source_files": source_files,
        "question_bank_sha256": _sha256_json(questions),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "analyzer_module": "studybench.static_graph",
        "analyzer_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "corpus_name": getattr(corpus, "name", None),
        "corpus_commit": getattr(corpus, "commit", None),
        "target_symbols": [item["target"] for item in selection],
        "target_selection": selection,
        "target_selection_sha256": _sha256_json(selection),
        "candidate_inventory": candidate_inventory,
        "candidate_inventory_sha256": _sha256_json(candidate_inventory),
        "candidate_count": len(candidate_inventory),
        "eligible_candidate_count": sum(
            item["eligible"] for item in candidate_inventory
        ),
        "train_question_count": sum(
            question["split"] == "train" for question in questions
        ),
        "dev_question_count": sum(
            question["split"] == "dev" for question in questions
        ),
        "excluded_candidates": excluded_candidates,
        "excluded_candidates_sha256": _sha256_json(excluded_candidates),
        "excluded_candidate_count": len(excluded_candidates),
    }


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def parse_prediction(text: str) -> tuple[tuple[QuestionEdge, ...], int]:
    """Parse strict answer JSON; return unique edges and duplicate count."""
    if not isinstance(text, str):
        raise ValueError("prediction must be a JSON string")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"edges"}:
        raise ValueError("top level must contain exactly the `edges` key")
    raw_edges = value["edges"]
    if not isinstance(raw_edges, list):
        raise ValueError("`edges` must be an array")
    parsed: list[QuestionEdge] = []
    for index, item in enumerate(raw_edges):
        if not isinstance(item, dict) or set(item) != EDGE_FIELDS:
            raise ValueError(
                f"edge {index} must contain exactly {sorted(EDGE_FIELDS)}"
            )
        caller, callee, path = item["caller"], item["callee"], item["path"]
        line, direction = item["line"], item["direction"]
        if not all(
            isinstance(value, str) and value
            for value in (caller, callee, path)
        ):
            raise ValueError(
                f"edge {index} caller, callee, and path must be nonempty strings"
            )
        logical = PurePosixPath(path)
        if (
            not path.startswith(SOURCE_PREFIX)
            or path != path.strip()
            or "\\" in path
            or "\x00" in path
            or logical.suffix != ".py"
            or logical.is_absolute()
            or logical.as_posix() != path
            or any(part in ("", ".", "..") for part in logical.parts)
        ):
            raise ValueError(
                f"edge {index} path is outside the DSPy Python scope"
            )
        if type(line) is not int or line < 1:
            raise ValueError(f"edge {index} line must be a positive integer")
        if not isinstance(direction, str) or direction not in DIRECTIONS:
            raise ValueError(f"edge {index} has an invalid direction")
        parsed.append(
            QuestionEdge(caller, callee, path, line, direction)
        )
    unique = set(parsed)
    return (
        tuple(
            sorted(
                unique,
                key=lambda edge: (
                    edge.path,
                    edge.line,
                    edge.caller,
                    edge.callee,
                    edge.direction,
                ),
            )
        ),
        len(parsed) - len(unique),
    )


def _coerce_oracle(
    oracle: Iterable[QuestionEdge | Mapping[str, Any]],
) -> set[QuestionEdge]:
    edges: set[QuestionEdge] = set()
    for index, value in enumerate(oracle):
        if isinstance(value, QuestionEdge):
            edges.add(value)
            continue
        if not isinstance(value, Mapping) or set(value) != EDGE_FIELDS:
            raise ValueError(f"oracle edge {index} violates the edge schema")
        try:
            encoded = _canonical_json({"edges": [dict(value)]})
            parsed, _ = parse_prediction(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid oracle edge {index}: {exc}") from exc
        edges.add(parsed[0])
    return edges


def _sorted_question_edges(
    edges: Iterable[QuestionEdge],
) -> tuple[QuestionEdge, ...]:
    return tuple(
        sorted(
            set(edges),
            key=lambda edge: (
                edge.path,
                edge.line,
                edge.caller,
                edge.callee,
                edge.direction,
            ),
        )
    )


def _invalid_locations(
    predicted: Iterable[QuestionEdge],
    source_view: Any,
) -> set[QuestionEdge]:
    sources = _sources_from_view(source_view)
    line_counts = {
        path: len(source.splitlines())
        for path, source in sources.items()
    }
    return {
        edge
        for edge in predicted
        if edge.path not in line_counts or edge.line > line_counts[edge.path]
    }


def verify_prediction(
    raw_answer: str,
    gold_edges: Iterable[QuestionEdge | Mapping[str, Any]],
    source_view: Any | None = None,
) -> dict[str, Any]:
    """Score exact edge F1, optionally validating locations against source.

    A syntactically well-formed edge at a nonexistent path or line is an
    interpretable wrong answer, not malformed output. It is therefore logged in
    `invalid_location_edges` and counted as a false positive.
    """
    oracle_set = _coerce_oracle(gold_edges)
    try:
        predicted, duplicate_count = parse_prediction(raw_answer)
    except ValueError as exc:
        return VerificationResult(
            schema_valid=False,
            schema_error=str(exc),
            duplicate_predictions=0,
            predicted_count=0,
            oracle_count=len(oracle_set),
            tp=0,
            fp=0,
            fn=len(oracle_set),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            exact=False,
            locations_validated=source_view is not None,
            predicted_edges=(),
            missing_edges=_sorted_question_edges(oracle_set),
            spurious_edges=(),
            invalid_location_edges=(),
        ).to_dict()
    predicted_set = set(predicted)
    invalid = (
        _invalid_locations(predicted_set, source_view)
        if source_view is not None
        else set()
    )
    matched = (predicted_set - invalid) & oracle_set
    missing = oracle_set - matched
    spurious = predicted_set - matched
    tp = len(matched)
    fp = len(spurious)
    fn = len(missing)
    precision = (
        tp / len(predicted_set)
        if predicted_set
        else (1.0 if not oracle_set else 0.0)
    )
    recall = (
        tp / len(oracle_set)
        if oracle_set
        else (1.0 if not predicted_set else 0.0)
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return VerificationResult(
        schema_valid=True,
        schema_error=None,
        duplicate_predictions=duplicate_count,
        predicted_count=len(predicted_set),
        oracle_count=len(oracle_set),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        exact=not fp and not fn,
        locations_validated=source_view is not None,
        predicted_edges=_sorted_question_edges(predicted_set),
        missing_edges=_sorted_question_edges(missing),
        spurious_edges=_sorted_question_edges(spurious),
        invalid_location_edges=_sorted_question_edges(invalid),
    ).to_dict()
