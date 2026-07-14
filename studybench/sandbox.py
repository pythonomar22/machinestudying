"""Fail-closed deterministic checks for generated answer programs.

Python source is syntax-checked in-process, but it is never executed directly on
the host. Execution is enabled only when all of the following are configured:

* ``STUDYBENCH_PYTHON_IMAGE`` is a canonical absolute path to a regular ``.sif``;
* ``STUDYBENCH_PYTHON_IMAGE_SHA256`` pins the exact image bytes; and
* an Apptainer executable is available (or is named by the absolute
  ``STUDYBENCH_APPTAINER_BIN`` path) and
  ``STUDYBENCH_APPTAINER_SHA256`` pins its exact bytes.

The verified image runs with ``--containall --cleanenv --no-home --net
--network none``; the per-answer work directory is the only explicit/user bind.
Apptainer may retain runtime-managed system mounts. The outer process also has
wall-clock and POSIX resource limits. This is container isolation, not a VM: the
Apptainer runtime, image, system mounts, and host kernel remain trusted.

Tree-sitter is a TypeScript *syntax* check only. Strict compilation fails closed
unless ``STUDYBENCH_TYPESCRIPT_CHECKER`` and its SHA-256 pin an absolute
executable, while ``STUDYBENCH_TYPESCRIPT_CHECKER_BUNDLE`` and its SHA-256 pin a
canonical manifest of the checker, compiler, and all supporting artifacts. The
checker contract is ``CHECKER SOURCE_PATH typescript|tsx``. Before it is treated
as ready, the checker must pass the built-in well-typed, type-error, and
import-dependent calibration protocol. This rejects an unconditional-success
wrapper; the manifest still requires human review for completeness and for the
checker command's suitability to the target repository.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading

from .dataset import ROOT
from .integrity import canonical_json_bytes, read_artifact_bytes, strict_json_loads

PYTHON_BIN = ROOT / ".venv-dspy/bin/python"
RESOURCE_LAUNCHER_VERSION = "3.12.11"
RUN_TIMEOUT = 240
CPU_SECONDS = 180
ADDRESS_SPACE_BYTES = 8 * 1024**3
OUTPUT_FILE_BYTES = 16 * 1024**2
OUTPUT_DETAIL_BYTES = 2_000
OPEN_FILES = 256
PROCESSES = 64

PYTHON_IMAGE_ENV = "STUDYBENCH_PYTHON_IMAGE"
PYTHON_IMAGE_SHA256_ENV = "STUDYBENCH_PYTHON_IMAGE_SHA256"
APPTAINER_BIN_ENV = "STUDYBENCH_APPTAINER_BIN"
APPTAINER_SHA256_ENV = "STUDYBENCH_APPTAINER_SHA256"
TYPESCRIPT_CHECKER_ENV = "STUDYBENCH_TYPESCRIPT_CHECKER"
TYPESCRIPT_CHECKER_SHA256_ENV = "STUDYBENCH_TYPESCRIPT_CHECKER_SHA256"
TYPESCRIPT_CHECKER_BUNDLE_ENV = "STUDYBENCH_TYPESCRIPT_CHECKER_BUNDLE"
TYPESCRIPT_CHECKER_BUNDLE_SHA256_ENV = (
    "STUDYBENCH_TYPESCRIPT_CHECKER_BUNDLE_SHA256"
)
TYPESCRIPT_CALIBRATION_PROTOCOL = "studybench-typescript-checker-v1"

FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)
SHA256 = re.compile(r"[0-9a-f]{64}")
LANG_TAGS = {
    "python": {"python", "python3", "py", ""},
    "typescript": {"typescript", "ts", "tsx", "javascript", "js", ""},
}

CONTAINMENT = (
    "Apptainer SIF with containall, cleanenv, no-home, isolated network, "
    "one explicit writable work bind, and host rlimits; runtime system mounts "
    "and the shared host kernel remain trusted"
)


@dataclass(frozen=True)
class PythonContainer:
    runtime: Path
    runtime_sha256: str
    image: Path
    image_sha256: str


class CheckerConfigurationChanged(RuntimeError):
    """The deterministic checker no longer matches the frozen grade contract."""


def _configuration_sha256(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_HASH_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}
_HASH_LOCK = threading.Lock()
_TS_CALIBRATION_CACHE: dict[
    tuple[str, str, str, str], tuple[dict, str | None]
] = {}
_TS_CALIBRATION_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    """Hash a file, reusing a digest only while all useful stat fields match."""

    stat = path.stat()
    key = (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    with _HASH_LOCK:
        cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    with _HASH_LOCK:
        _HASH_CACHE[key] = observed
    return observed


def _resource_launcher_record() -> tuple[dict | None, str | None]:
    """Identify the trusted Python used only to install host resource limits."""

    try:
        resolved = PYTHON_BIN.resolve(strict=True)
    except OSError as exc:
        return None, f"resource-limit launcher Python is unavailable: {exc}"
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return None, f"resource-limit launcher is not executable: {resolved}"
    config = PYTHON_BIN.parent.parent / "pyvenv.cfg"
    if not config.is_file() or config.is_symlink():
        return None, f"resource-limit launcher config is unavailable: {config}"
    try:
        fields = {
            key.strip(): value.strip()
            for line in config.read_text(encoding="utf-8").splitlines()
            if "=" in line
            for key, value in (line.split("=", 1),)
        }
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, f"cannot read resource-limit launcher metadata: {exc}"
    release = fields.get("version_info", "").strip()
    if release != RESOURCE_LAUNCHER_VERSION:
        return None, (
            f"resource-limit launcher uses Python {release or 'unknown'}, "
            f"expected {RESOURCE_LAUNCHER_VERSION}"
        )
    return {
        "resource_launcher": str(resolved),
        "resource_launcher_version": release,
        "resource_launcher_sha256": _sha256_file(resolved),
        "resource_launcher_config": str(config.resolve()),
        "resource_launcher_config_sha256": _sha256_file(config),
    }, None


def extract_code(answer: str, language: str) -> list[tuple[str, str]]:
    """Return fenced code blocks whose tag is compatible with *language*."""

    if language not in LANG_TAGS:
        return []
    return [
        (tag.lower(), body)
        for tag, body in FENCE.findall(answer)
        if tag.lower() in LANG_TAGS[language] and body.strip()
    ]


def check(
    answer: str,
    language: str,
    *,
    expected_configuration: dict | None = None,
) -> dict:
    """Run one check while proving its checker configuration stayed frozen.

    ``expected_configuration`` is the exact record included in the grading
    specification.  Re-reading it both before and after the check makes a
    changed image, compiler, launcher, runtime, or containment contract fatal
    instead of silently attaching a result to an obsolete specification.
    """

    before = configuration_record(language)
    expected = before if expected_configuration is None else expected_configuration
    if not isinstance(expected, dict) or before != expected:
        raise CheckerConfigurationChanged(
            "deterministic checker configuration changed before execution"
        )

    if language not in LANG_TAGS:
        result = {
            "compile_ok": False,
            "syntax_ok": False,
            "sandboxed": False,
            "check_level": "syntax-only",
            "detail": f"unsupported answer language: {language!r}",
        }
    else:
        blocks = extract_code(answer, language)
        if not blocks:
            result = {
                "compile_ok": False,
                "syntax_ok": False,
                "sandboxed": False,
                "check_level": "syntax-only",
                "detail": "no matching fenced code block found in answer",
            }
        else:
            # Prefer an explicitly tagged program; untagged blocks often contain
            # output or configuration snippets. Within that class, take the
            # largest block.
            tagged = [block for block in blocks if block[0]]
            tag, program = max(tagged or blocks, key=lambda block: len(block[1]))
            if language == "python":
                result = _check_python(program)
            else:
                result = _check_typescript(program, tsx=tag == "tsx")

    after = configuration_record(language)
    if after != expected:
        raise CheckerConfigurationChanged(
            "deterministic checker configuration changed during execution"
        )
    result["configuration_sha256"] = _configuration_sha256(expected)
    return result


def _child_environment(directory: str) -> dict[str, str]:
    """Return a minimal environment with no inherited credentials or GPU access."""

    return {
        "PATH": "/usr/bin:/bin",
        "HOME": directory,
        "TMPDIR": directory,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "none",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def _limit_launcher() -> str:
    # Limits are set in a single-threaded child before exec. Using preexec_fn in
    # the multi-threaded grader process would be unsafe.
    return f"""\
import os
import resource
import sys

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, ({CPU_SECONDS}, {CPU_SECONDS}))
resource.setrlimit(resource.RLIMIT_AS, ({ADDRESS_SPACE_BYTES}, {ADDRESS_SPACE_BYTES}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({OUTPUT_FILE_BYTES}, {OUTPUT_FILE_BYTES}))
resource.setrlimit(resource.RLIMIT_NOFILE, ({OPEN_FILES}, {OPEN_FILES}))
if hasattr(resource, "RLIMIT_NPROC"):
    resource.setrlimit(resource.RLIMIT_NPROC, ({PROCESSES}, {PROCESSES}))
os.execv(sys.argv[1], sys.argv[1:])
"""


def _tail(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - OUTPUT_DETAIL_BYTES))
        return handle.read().decode("utf-8", errors="replace")


def _run_limited(
    command: list[str], scratch: Path
) -> tuple[int | None, str, bool, bool]:
    """Return ``(code, output_tail, timed_out, launched)`` for *command*."""

    launcher_record, launcher_error = _resource_launcher_record()
    if launcher_error:
        return (
            None,
            launcher_error,
            False,
            False,
        )
    assert launcher_record is not None
    launcher = scratch / "resource_runner.py"
    output = scratch / "checker-output.log"
    launcher.write_text(_limit_launcher(), encoding="utf-8")
    try:
        with output.open("wb") as stream:
            proc = subprocess.Popen(
                [
                    launcher_record["resource_launcher"],
                    "-I",
                    str(launcher),
                    *command,
                ],
                # Avoid exposing the scratch directory as Apptainer's implicit cwd.
                cwd="/",
                env=_child_environment(str(scratch)),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            try:
                code = proc.wait(timeout=RUN_TIMEOUT)
                timed_out = False
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                code = None
                timed_out = True
    except OSError as exc:
        return None, f"checker launch failed: {type(exc).__name__}: {exc}", False, False
    return code, _tail(output), timed_out, True


def _canonical_executable(raw: str, label: str) -> tuple[Path | None, str | None]:
    path = Path(raw)
    if not path.is_absolute():
        return None, f"{label} must be an absolute path"
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return None, f"configured executable is unavailable: {path}: {exc}"
    if resolved != path or not path.is_file() or not os.access(path, os.X_OK):
        return None, f"configured executable is not a canonical executable file: {path}"
    if path.stat().st_mode & 0o022:
        return None, f"configured executable must not be group/world writable: {path}"
    return path, None


def _canonical_regular_file(raw: str, label: str) -> tuple[Path | None, str | None]:
    path = Path(raw)
    if not path.is_absolute():
        return None, f"{label} must be an absolute path"
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return None, f"configured file is unavailable: {path}: {exc}"
    if resolved != path or path.is_symlink() or not path.is_file():
        return None, f"configured file is not a canonical regular file: {path}"
    if path.stat().st_mode & 0o022:
        return None, f"configured file must not be group/world writable: {path}"
    return path, None


def _python_container_configuration() -> tuple[PythonContainer | None, dict, str | None]:
    """Validate the configured runtime/image and return provenance on all paths."""

    raw_image = os.environ.get(PYTHON_IMAGE_ENV, "")
    expected = os.environ.get(PYTHON_IMAGE_SHA256_ENV, "").lower()
    metadata = {
        "image": raw_image or None,
        "image_sha256": None,
        "expected_image_sha256": expected or None,
        "checker": None,
        "checker_sha256": None,
        "expected_checker_sha256": os.environ.get(APPTAINER_SHA256_ENV, "").lower() or None,
        "resource_launcher": None,
        "resource_launcher_version": None,
        "resource_launcher_sha256": None,
        "resource_launcher_config": None,
        "resource_launcher_config_sha256": None,
    }
    launcher, launcher_error = _resource_launcher_record()
    if launcher_error:
        return None, metadata, launcher_error
    assert launcher is not None
    metadata.update(launcher)
    if not raw_image:
        return None, metadata, f"{PYTHON_IMAGE_ENV} is not configured"
    image = Path(raw_image)
    if not image.is_absolute():
        return None, metadata, f"{PYTHON_IMAGE_ENV} must be an absolute path"
    if image.suffix.lower() != ".sif":
        return None, metadata, f"{PYTHON_IMAGE_ENV} must name a .sif image"
    try:
        resolved = image.resolve(strict=True)
    except OSError as exc:
        return None, metadata, f"configured Python image is unavailable: {image}: {exc}"
    if resolved != image or image.is_symlink() or not image.is_file():
        return None, metadata, f"Python image is not a canonical regular file: {image}"
    if image.stat().st_mode & 0o022:
        return None, metadata, f"Python image must not be group/world writable: {image}"
    if not SHA256.fullmatch(expected):
        return None, metadata, f"{PYTHON_IMAGE_SHA256_ENV} must be exactly 64 hex digits"
    observed = _sha256_file(image)
    metadata["image"] = str(image)
    metadata["image_sha256"] = observed
    if observed != expected:
        return None, metadata, (
            f"Python image hash mismatch: observed {observed}, expected {expected}"
        )

    raw_runtime = os.environ.get(APPTAINER_BIN_ENV, "")
    if raw_runtime:
        runtime, error = _canonical_executable(raw_runtime, APPTAINER_BIN_ENV)
    else:
        found = shutil.which("apptainer", path="/usr/bin:/bin:/usr/local/bin")
        if found is None:
            runtime, error = None, "Apptainer executable is unavailable"
        else:
            resolved_runtime = str(Path(found).resolve())
            runtime, error = _canonical_executable(resolved_runtime, "apptainer")
    if error:
        return None, metadata, error
    assert runtime is not None
    metadata["checker"] = str(runtime)
    expected_runtime = os.environ.get(APPTAINER_SHA256_ENV, "").lower()
    if not SHA256.fullmatch(expected_runtime):
        return None, metadata, f"{APPTAINER_SHA256_ENV} must be exactly 64 hex digits"
    observed_runtime = _sha256_file(runtime)
    metadata["checker_sha256"] = observed_runtime
    if observed_runtime != expected_runtime:
        return None, metadata, (
            f"Apptainer hash mismatch: observed {observed_runtime}, "
            f"expected {expected_runtime}"
        )
    return (
        PythonContainer(runtime, observed_runtime, image, observed),
        metadata,
        None,
    )


def _python_result(
    *, syntax_ok: bool, detail: str, metadata: dict | None = None
) -> dict:
    return {
        "compile_ok": False,
        "syntax_ok": syntax_ok,
        "run_ok": False,
        "sandboxed": False,
        "check_level": "syntax-only",
        "isolation": "no generated Python was executed",
        **(metadata or {}),
        "detail": detail[-OUTPUT_DETAIL_BYTES:],
    }


def _check_python(program: str) -> dict:
    try:
        compile(program, "<answer>", "exec")
    except SyntaxError as exc:
        _, metadata, _ = _python_container_configuration()
        return _python_result(
            syntax_ok=False,
            metadata=metadata,
            detail=f"SyntaxError: {exc}",
        )

    container, metadata, error = _python_container_configuration()
    if error:
        return _python_result(syntax_ok=True, metadata=metadata, detail=error)
    assert container is not None

    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        work = scratch / "work"
        work.mkdir(mode=0o700)
        source = work / "answer.py"
        source.write_text(program, encoding="utf-8")
        bind = f"{work.resolve()}:/work:rw"
        command = [
            str(container.runtime),
            "exec",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--net",
            "--network",
            "none",
            "--bind",
            bind,
            str(container.image),
            "python3",
            "-I",
            "/work/answer.py",
        ]
        code, output, timed_out, launched = _run_limited(command, scratch)
    if timed_out:
        detail = f"contained execution timed out after {RUN_TIMEOUT}s"
    elif code is None:
        detail = output
    elif code < 0:
        detail = f"terminated by signal {-code}" + (f"\n{output}" if output else "")
    else:
        detail = output or ("contained execution passed" if code == 0 else f"exit {code}")
    passed = code == 0 and not timed_out
    return {
        "compile_ok": passed,
        "syntax_ok": True,
        "run_ok": passed,
        "sandboxed": launched,
        "check_level": "contained-execution",
        "isolation": CONTAINMENT,
        **metadata,
        "detail": detail[-OUTPUT_DETAIL_BYTES:],
    }


_TS_PARSERS: dict[bool, object] = {}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _module_binary_record(module, distribution: str) -> dict:
    root = Path(module.__file__).resolve().parent
    binaries = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".so", ".pyd", ".dylib"}
    )
    if len(binaries) != 1:
        raise RuntimeError(
            f"expected exactly one binary artifact for {distribution}, found {binaries}"
        )
    artifact = binaries[0]
    return {
        "path": str(artifact),
        "version": _package_version(distribution),
        "sha256": _sha256_file(artifact),
    }


def _typescript_parser_record() -> dict:
    import tree_sitter
    import tree_sitter_typescript

    core = _module_binary_record(tree_sitter, "tree-sitter")
    grammar = _module_binary_record(
        tree_sitter_typescript, "tree-sitter-typescript"
    )
    return {
        "syntax_checker": grammar["path"],
        "syntax_checker_version": grammar["version"],
        "syntax_checker_sha256": grammar["sha256"],
        "syntax_core": core["path"],
        "syntax_core_version": core["version"],
        "syntax_core_sha256": core["sha256"],
    }


def _parse_typescript(program: str, *, tsx: bool) -> dict:
    try:
        import tree_sitter_typescript as tst
        from tree_sitter import Language, Parser
        provenance = _typescript_parser_record()
    except (ImportError, OSError, RuntimeError) as exc:
        return {
            "syntax_ok": False,
            "checker": "tree-sitter-typescript",
            "checker_version": None,
            "checker_sha256": None,
            "syntax_core": None,
            "syntax_core_version": None,
            "syntax_core_sha256": None,
            "detail": f"TypeScript syntax checker unavailable: {exc}",
        }
    if tsx not in _TS_PARSERS:
        language = tst.language_tsx() if tsx else tst.language_typescript()
        _TS_PARSERS[tsx] = Parser(Language(language))
    tree = _TS_PARSERS[tsx].parse(program.encode("utf-8"))
    errors = []

    def walk(node) -> None:
        if node.type == "ERROR" or node.is_missing:
            errors.append(f"line {node.start_point[0] + 1}: {node.type}")
        elif node.has_error:
            for child in node.children:
                walk(child)

    walk(tree.root_node)
    return {
        "syntax_ok": not errors,
        "checker": provenance["syntax_checker"],
        "checker_version": provenance["syntax_checker_version"],
        "checker_sha256": provenance["syntax_checker_sha256"],
        "syntax_core": provenance["syntax_core"],
        "syntax_core_version": provenance["syntax_core_version"],
        "syntax_core_sha256": provenance["syntax_core_sha256"],
        "detail": (
            "; ".join(errors[:5])
            if errors
            else "tree-sitter TypeScript syntax parses cleanly"
        ),
    }


def _configured_typescript_checker(
    explicit: str | Path | None,
    explicit_sha256: str | None,
) -> tuple[Path | None, str | None, str | None]:
    raw = str(explicit) if explicit is not None else os.environ.get(TYPESCRIPT_CHECKER_ENV, "")
    expected = (
        explicit_sha256
        if explicit_sha256 is not None
        else os.environ.get(TYPESCRIPT_CHECKER_SHA256_ENV, "")
    ).lower()
    if not raw:
        return None, None, None
    checker, error = _canonical_executable(raw, TYPESCRIPT_CHECKER_ENV)
    if error:
        return None, None, error
    if not SHA256.fullmatch(expected):
        return None, None, (
            f"{TYPESCRIPT_CHECKER_SHA256_ENV} must be exactly 64 hex digits"
        )
    assert checker is not None
    observed = _sha256_file(checker)
    if observed != expected:
        return None, observed, (
            f"TypeScript checker hash mismatch: observed {observed}, expected {expected}"
        )
    return checker, observed, None


_TYPESCRIPT_BUNDLE_ROLES = {
    "checker",
    "compiler",
    "configuration",
    "dependency",
    "library",
    "runtime",
}
_TYPESCRIPT_CALIBRATION_CASES = (
    {
        "name": "well_typed",
        "mode": "typescript",
        "filename": "answer.ts",
        "source": "export const studybenchValue: number = 1;\n",
        "support": (),
        "expected": "accept",
    },
    {
        "name": "type_error",
        "mode": "typescript",
        "filename": "answer.ts",
        "source": 'export const studybenchValue: number = "not a number";\n',
        "support": (),
        "expected": "reject",
    },
    {
        "name": "relative_import",
        "mode": "tsx",
        "filename": "answer.tsx",
        "source": (
            'import { dependency } from "./dependency";\n'
            "declare const React: { createElement(...args: unknown[]): unknown };\n"
            "declare namespace JSX {\n"
            "  interface IntrinsicElements { div: { value: number } }\n"
            "}\n"
            "export const studybenchElement = <div value={dependency} />;\n"
        ),
        "support": (("dependency.ts", "export const dependency: number = 1;\n"),),
        "expected": "accept",
    },
)


def _typescript_bundle_configuration(
    checker: Path,
    checker_sha256: str,
) -> tuple[dict, str | None]:
    raw = os.environ.get(TYPESCRIPT_CHECKER_BUNDLE_ENV, "")
    expected = os.environ.get(TYPESCRIPT_CHECKER_BUNDLE_SHA256_ENV, "").lower()
    metadata = {
        "checker_bundle": raw or None,
        "checker_bundle_sha256": None,
        "expected_checker_bundle_sha256": expected or None,
        "checker_bundle_artifacts": None,
        "checker_bundle_artifacts_sha256": None,
    }
    if not raw:
        return metadata, f"{TYPESCRIPT_CHECKER_BUNDLE_ENV} is not configured"
    if not SHA256.fullmatch(expected):
        return metadata, (
            f"{TYPESCRIPT_CHECKER_BUNDLE_SHA256_ENV} must be exactly 64 hex digits"
        )

    bundle, error = _canonical_regular_file(raw, TYPESCRIPT_CHECKER_BUNDLE_ENV)
    if error:
        return metadata, error
    assert bundle is not None
    try:
        bundle_bytes = read_artifact_bytes(bundle)
    except (OSError, ValueError) as exc:
        return metadata, f"cannot read TypeScript checker bundle: {exc}"
    observed_bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    metadata["checker_bundle"] = str(bundle)
    metadata["checker_bundle_sha256"] = observed_bundle_sha256
    if observed_bundle_sha256 != expected:
        return metadata, (
            "TypeScript checker bundle hash mismatch: observed "
            f"{observed_bundle_sha256}, expected {expected}"
        )

    try:
        manifest = strict_json_loads(
            bundle_bytes,
            label="TypeScript checker bundle",
        )
    except ValueError as exc:
        return metadata, str(exc)
    if not isinstance(manifest, dict):
        return metadata, "TypeScript checker bundle must be a JSON object"
    if set(manifest) != {"artifacts", "calibration_protocol", "schema_version"}:
        return metadata, (
            "TypeScript checker bundle must contain exactly artifacts, "
            "calibration_protocol, and schema_version"
        )
    if canonical_json_bytes(manifest) != bundle_bytes:
        return metadata, "TypeScript checker bundle is not canonical JSON"
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        return metadata, "TypeScript checker bundle schema_version must be 1"
    if manifest["calibration_protocol"] != TYPESCRIPT_CALIBRATION_PROTOCOL:
        return metadata, (
            "TypeScript checker bundle calibration_protocol must be "
            f"{TYPESCRIPT_CALIBRATION_PROTOCOL!r}"
        )

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        return metadata, "TypeScript checker bundle artifacts must be a non-empty list"
    validated: list[dict] = []
    for index, artifact in enumerate(artifacts):
        label = f"TypeScript checker bundle artifact {index}"
        if not isinstance(artifact, dict) or set(artifact) != {
            "path", "roles", "sha256",
        }:
            return metadata, f"{label} must contain exactly path, roles, and sha256"
        raw_path = artifact["path"]
        declared_sha256 = artifact["sha256"]
        roles = artifact["roles"]
        if not isinstance(raw_path, str) or not raw_path:
            return metadata, f"{label} path must be a non-empty string"
        if not isinstance(declared_sha256, str) or not SHA256.fullmatch(
            declared_sha256
        ):
            return metadata, f"{label} sha256 must be exactly 64 lowercase hex digits"
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) for role in roles)
            or roles != sorted(set(roles))
            or not set(roles).issubset(_TYPESCRIPT_BUNDLE_ROLES)
        ):
            return metadata, (
                f"{label} roles must be a sorted unique non-empty subset of "
                f"{sorted(_TYPESCRIPT_BUNDLE_ROLES)}"
            )
        validator = (
            _canonical_executable
            if "checker" in roles or "runtime" in roles
            else _canonical_regular_file
        )
        path, path_error = validator(raw_path, f"{label} path")
        if path_error:
            return metadata, path_error
        assert path is not None
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != declared_sha256:
            return metadata, (
                f"{label} hash mismatch: observed {observed_sha256}, "
                f"expected {declared_sha256}"
            )
        validated.append(
            {"path": str(path), "roles": roles, "sha256": observed_sha256}
        )

    paths = [artifact["path"] for artifact in validated]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        return metadata, (
            "TypeScript checker bundle artifacts must have unique paths sorted "
            "lexicographically"
        )
    checker_artifacts = [
        artifact for artifact in validated if "checker" in artifact["roles"]
    ]
    if len(checker_artifacts) != 1:
        return metadata, (
            "TypeScript checker bundle must identify exactly one checker artifact"
        )
    if (
        checker_artifacts[0]["path"] != str(checker)
        or checker_artifacts[0]["sha256"] != checker_sha256
    ):
        return metadata, (
            "TypeScript checker bundle checker artifact does not match the "
            "configured checker"
        )
    if not any("compiler" in artifact["roles"] for artifact in validated):
        return metadata, (
            "TypeScript checker bundle must identify at least one compiler artifact"
        )

    metadata["checker_bundle_artifacts"] = validated
    metadata["checker_bundle_artifacts_sha256"] = hashlib.sha256(
        canonical_json_bytes(validated)
    ).hexdigest()
    return metadata, None


def _typescript_checker_calibration(
    checker: Path,
    checker_sha256: str,
    bundle_metadata: dict,
) -> tuple[dict, str | None]:
    bundle_sha256 = bundle_metadata["checker_bundle_sha256"]
    artifacts_sha256 = bundle_metadata["checker_bundle_artifacts_sha256"]
    assert isinstance(bundle_sha256, str)
    assert isinstance(artifacts_sha256, str)
    key = (str(checker), checker_sha256, bundle_sha256, artifacts_sha256)

    with _TS_CALIBRATION_LOCK:
        cached = _TS_CALIBRATION_CACHE.get(key)
        if cached is not None:
            record, error = cached
            return deepcopy(record), error

        results = []
        with tempfile.TemporaryDirectory(
            prefix="studybench-ts-calibration-"
        ) as temporary:
            root = Path(temporary)
            for case in _TYPESCRIPT_CALIBRATION_CASES:
                scratch = root / case["name"]
                scratch.mkdir(mode=0o700)
                source = scratch / case["filename"]
                source.write_text(case["source"], encoding="utf-8")
                support_records = []
                for filename, contents in case["support"]:
                    support = scratch / filename
                    support.write_text(contents, encoding="utf-8")
                    support_records.append(
                        {
                            "path": filename,
                            "sha256": hashlib.sha256(
                                contents.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
                code, _output, timed_out, launched = _run_limited(
                    [str(checker), str(source), case["mode"]],
                    scratch,
                )
                if not launched or code is None or timed_out:
                    observed = "infrastructure-error"
                elif code < 0:
                    observed = "signal"
                else:
                    observed = "accept" if code == 0 else "reject"
                passed = observed == case["expected"]
                results.append(
                    {
                        "expected": case["expected"],
                        "mode": case["mode"],
                        "name": case["name"],
                        "observed": observed,
                        "passed": passed,
                        "source_sha256": hashlib.sha256(
                            case["source"].encode("utf-8")
                        ).hexdigest(),
                        "support_artifacts": support_records,
                    }
                )
                if observed in {"infrastructure-error", "signal"}:
                    break

        passed = len(results) == len(_TYPESCRIPT_CALIBRATION_CASES) and all(
            result["passed"] for result in results
        )
        record = {
            "cases": results,
            "passed": passed,
            "protocol": TYPESCRIPT_CALIBRATION_PROTOCOL,
        }
        if not passed:
            failures = [
                f"{result['name']} expected {result['expected']} but observed "
                f"{result['observed']}"
                for result in results
                if not result["passed"]
            ]
            if len(results) != len(_TYPESCRIPT_CALIBRATION_CASES):
                failures.append("calibration stopped after an infrastructure failure")
            error = (
                "TypeScript checker semantic calibration failed: "
                + "; ".join(failures)
            )
            _TS_CALIBRATION_CACHE[key] = (deepcopy(record), error)
            return record, error

        _TS_CALIBRATION_CACHE[key] = (deepcopy(record), None)
        return record, None


def _typescript_checker_configuration(
    explicit: str | Path | None,
    explicit_sha256: str | None,
) -> tuple[Path | None, dict, str | None]:
    raw = (
        str(explicit)
        if explicit is not None
        else os.environ.get(TYPESCRIPT_CHECKER_ENV, "")
    )
    expected = (
        explicit_sha256
        if explicit_sha256 is not None
        else os.environ.get(TYPESCRIPT_CHECKER_SHA256_ENV, "")
    ).lower()
    metadata = {
        "configured_checker": raw or None,
        "configured_checker_sha256": None,
        "expected_checker_sha256": expected or None,
        "checker_bundle": os.environ.get(TYPESCRIPT_CHECKER_BUNDLE_ENV) or None,
        "checker_bundle_sha256": None,
        "expected_checker_bundle_sha256": (
            os.environ.get(TYPESCRIPT_CHECKER_BUNDLE_SHA256_ENV, "").lower() or None
        ),
        "checker_bundle_artifacts": None,
        "checker_bundle_artifacts_sha256": None,
        "checker_calibration": None,
    }
    checker, observed_sha256, error = _configured_typescript_checker(
        explicit,
        explicit_sha256,
    )
    metadata["configured_checker_sha256"] = observed_sha256
    if error:
        return None, metadata, error
    if checker is None or observed_sha256 is None:
        return None, metadata, f"{TYPESCRIPT_CHECKER_ENV} is not configured"

    bundle_metadata, error = _typescript_bundle_configuration(
        checker,
        observed_sha256,
    )
    metadata.update(bundle_metadata)
    if error:
        return None, metadata, error
    calibration, error = _typescript_checker_calibration(
        checker,
        observed_sha256,
        bundle_metadata,
    )
    metadata["checker_calibration"] = calibration
    if error:
        return None, metadata, error
    return checker, metadata, None


def configuration_record(language: str) -> dict:
    """Return the immutable checker identity without evaluating generated code."""

    if language == "python":
        container, metadata, error = _python_container_configuration()
        ready = container is not None
        return {
            "language": language,
            "ready": ready,
            "check_level": "contained-execution" if ready else "syntax-only",
            "sandboxed": ready,
            "isolation": CONTAINMENT if ready else "no generated Python will execute",
            **metadata,
            "error": error,
        }
    if language == "typescript":
        try:
            parser = _typescript_parser_record()
        except (ImportError, OSError, RuntimeError) as exc:
            return {
                "language": language,
                "ready": False,
                "check_level": "syntax-only",
                "sandboxed": False,
                "checker": None,
                "checker_version": None,
                "checker_sha256": None,
                "error": f"TypeScript syntax checker unavailable: {exc}",
            }
        configured, checker_metadata, error = _typescript_checker_configuration(
            None,
            None,
        )
        launcher, launcher_error = _resource_launcher_record()
        ready = configured is not None and launcher is not None
        return {
            "language": language,
            "ready": ready,
            "check_level": "configured-compiler" if ready else "syntax-only",
            "sandboxed": False,
            "checker": str(configured) if ready else parser["syntax_checker"],
            "checker_version": None if ready else parser["syntax_checker_version"],
            "checker_sha256": (
                checker_metadata["configured_checker_sha256"]
                if ready
                else parser["syntax_checker_sha256"]
            ),
            **checker_metadata,
            **parser,
            **(launcher or {
                "resource_launcher": None,
                "resource_launcher_version": None,
                "resource_launcher_sha256": None,
                "resource_launcher_config": None,
                "resource_launcher_config_sha256": None,
            }),
            "error": launcher_error or error,
        }
    return {
        "language": language,
        "ready": False,
        "check_level": "syntax-only",
        "sandboxed": False,
        "checker": None,
        "checker_version": None,
        "checker_sha256": None,
        "error": f"unsupported answer language: {language!r}",
    }


def _check_typescript(
    program: str,
    tsx: bool = False,
    *,
    checker: str | Path | None = None,
    checker_sha256: str | None = None,
) -> dict:
    syntax = _parse_typescript(program, tsx=tsx)
    base = {
        "compile_ok": False,
        "syntax_ok": syntax["syntax_ok"],
        "sandboxed": False,
        "check_level": "syntax-only",
        "checker": syntax["checker"],
        "checker_version": syntax["checker_version"],
        "checker_sha256": syntax["checker_sha256"],
        "syntax_core": syntax["syntax_core"],
        "syntax_core_version": syntax["syntax_core_version"],
        "syntax_core_sha256": syntax["syntax_core_sha256"],
    }
    if not syntax["syntax_ok"]:
        return {**base, "detail": syntax["detail"]}

    configured, checker_metadata, error = _typescript_checker_configuration(
        checker, checker_sha256
    )
    if error:
        return {
            **base,
            **checker_metadata,
            "detail": f"{syntax['detail']}; {error}",
        }
    assert configured is not None

    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        source = scratch / ("answer.tsx" if tsx else "answer.ts")
        source.write_text(program, encoding="utf-8")
        code, output, timed_out, launched = _run_limited(
            [str(configured), str(source), "tsx" if tsx else "typescript"],
            scratch,
        )
    if timed_out:
        detail = f"configured TypeScript checker timed out after {RUN_TIMEOUT}s"
    elif code is None:
        detail = output
    else:
        detail = output or (
            "configured checker passed" if code == 0 else f"checker exited {code}"
        )
    return {
        "compile_ok": code == 0 and not timed_out,
        "syntax_ok": True,
        "sandboxed": False,
        "check_level": "configured-compiler",
        "checker": str(configured),
        "checker_version": None,
        "checker_sha256": checker_metadata["configured_checker_sha256"],
        "checker_launched": launched,
        **checker_metadata,
        "syntax_checker": syntax["checker"],
        "syntax_checker_version": syntax["checker_version"],
        "syntax_checker_sha256": syntax["checker_sha256"],
        "syntax_core": syntax["syntax_core"],
        "syntax_core_version": syntax["syntax_core_version"],
        "syntax_core_sha256": syntax["syntax_core_sha256"],
        "detail": detail[-OUTPUT_DETAIL_BYTES:],
    }
