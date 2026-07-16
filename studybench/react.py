"""Study and evaluate Qwen3.5-9B with the paper's DSPy ReAct harness."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import dspy
from dspy.predict.react import _fmt_exc

from .artifacts import read_json, sha256_json, sha256_text, stable_seed, write_json, write_text
from .dataset import CORPORA, NOTE_PREFIX, ROOT, load_questions, verify_corpus
from .tools import (
    GLOB_MAX_PATHS,
    GREP_MAX_MATCHES,
    OBSERVATION_MAX_CHARS,
    READ_MAX_LINES,
    RepoTools,
)

MODEL = "openai/Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BUDGETS = {
    "direct": (0, False),
    "k5": (5, False),
    "k20": (20, False),
    "k20f": (20, True),
}
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 32_768,
    "presence_penalty": 1.5,
    "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
}
TOOL_CONFIG = {
    "names": ["grep", "glob", "read_file"],
    "read_max_lines": READ_MAX_LINES,
    "grep_max_matches": GREP_MAX_MATCHES,
    "glob_max_paths": GLOB_MAX_PATHS,
    "observation_max_chars": OBSERVATION_MAX_CHARS,
}
log = logging.getLogger("studybench.react")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def study_prompt(library: str, iterations: int) -> str:
    return (
        f"Study the {library} repository and write yourself a cheatsheet: a reference "
        f"document that will be prepended to every future question about {library}. "
        "You will not see those questions in advance, and repository tools will remain "
        f"available when you answer them. Study for exactly {iterations} iterations, then "
        "write the complete cheatsheet as your final answer."
    )


def make_tools(repository: RepoTools) -> list:
    def grep(pattern: str, path: str = "") -> str:
        """Search repository code for a case-sensitive regular expression.

        Returns path:line_number:line. ``path`` may restrict the search to one
        file or directory.
        """

        return repository.grep(pattern, path)

    def glob(pattern: str) -> str:
        """List repository files matching a glob pattern, including ``**``."""

        return repository.glob(pattern)

    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a 1-indexed line range from a repository file, capped at 200 lines."""

        return repository.read_file(path, start_line, end_line)

    return [grep, glob, read_file]


class ForcedTrajectoryError(RuntimeError):
    def __init__(self, message: str, trajectory: dict):
        super().__init__(message)
        self.trajectory = trajectory


class ForcedReAct(dspy.ReAct):
    """ReAct where a selected ``finish`` consumes the step but cannot stop early."""

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for index in range(max_iters):
            try:
                prediction = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args
                )
            except ValueError as error:
                raise ForcedTrajectoryError(
                    f"forced trajectory stopped at iteration {index}: {_fmt_exc(error)}",
                    trajectory,
                ) from error
            trajectory[f"thought_{index}"] = prediction.next_thought
            trajectory[f"tool_name_{index}"] = prediction.next_tool_name
            trajectory[f"tool_args_{index}"] = prediction.next_tool_args
            if prediction.next_tool_name == "finish":
                trajectory[f"observation_{index}"] = (
                    "You cannot finish yet — keep searching and choose another repository tool."
                )
                continue
            try:
                observation = self.tools[prediction.next_tool_name](**prediction.next_tool_args)
            except Exception as error:
                observation = f"Execution error in {prediction.next_tool_name}: {_fmt_exc(error)}"
            trajectory[f"observation_{index}"] = observation
        try:
            extraction = self._call_with_potential_trajectory_truncation(
                self.extract, trajectory, **input_args
            )
        except Exception as error:
            raise ForcedTrajectoryError(
                f"forced answer extraction failed: {_fmt_exc(error)}", trajectory
            ) from error
        return dspy.Prediction(trajectory=trajectory, **extraction)


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _usage(history: list[dict]) -> tuple[list[dict], int, int, int]:
    ledger, prompt_tokens, completion_tokens, total_tokens = [], 0, 0, 0
    for index, entry in enumerate(history):
        usage = _jsonable(entry.get("usage"))
        if not isinstance(usage, dict):
            raise ValueError(f"model call {index} has no usage record")
        values = [usage.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"model call {index} has incomplete token usage: {usage}")
        if values[2] != values[0] + values[1]:
            raise ValueError(f"model call {index} has inconsistent token usage: {usage}")
        prompt_tokens += values[0]
        completion_tokens += values[1]
        total_tokens += values[2]
        ledger.append({"call": index, **usage})
    return ledger, prompt_tokens, completion_tokens, total_tokens


def run_episode(
    *,
    corpus,
    tools: list,
    question: dict,
    condition: str,
    budget: str,
    rollout: int,
    seed: int,
    base_url: str,
    max_iters: int,
    forced: bool,
    debug: bool,
) -> dict:
    lm = dspy.LM(
        MODEL,
        api_base=base_url,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        model_type="chat",
        cache=False,
        num_retries=0,
        seed=seed,
        **SAMPLING,
    )
    episode = {
        "task": corpus.name,
        "qid": question["id"],
        "condition": condition,
        "budget": budget,
        "rollout": rollout,
        "seed": seed,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "harness": "dspy.ReAct",
        "question_sha256": sha256_text(question["question"]),
        "started": utc_now(),
        "status": "ok",
        "answer": "",
        "turns": [],
    }
    trajectory = {}
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        try:
            if budget == "direct":
                prediction = dspy.Predict("question -> answer")(question=question["question"])
            else:
                module_type = ForcedReAct if forced else dspy.ReAct
                module = module_type("question -> answer", tools=list(tools), max_iters=max_iters)
                prediction = module(question=question["question"])
                trajectory = dict(prediction.trajectory)
            episode["answer"] = prediction.answer or ""
            if not episode["answer"].strip():
                episode["status"] = "no_answer"
        except ForcedTrajectoryError as error:
            trajectory = error.trajectory
            episode["status"] = "forced_short"
            episode["error"] = str(error)
        except Exception as error:
            episode["status"] = "error"
            episode["error"] = f"{type(error).__name__}: {str(error)[:1000]}"

    indices = sorted(
        int(key.rsplit("_", 1)[1]) for key in trajectory if key.startswith("thought_")
    )
    for index in indices:
        episode["turns"].append(
            {
                "reasoning": trajectory.get(f"thought_{index}"),
                "tool": trajectory.get(f"tool_name_{index}"),
                "arguments": trajectory.get(f"tool_args_{index}"),
                "observation": str(trajectory.get(f"observation_{index}")),
            }
        )
    episode["react_iterations"] = len(indices)
    episode["finish_catches"] = sum(turn["tool"] == "finish" for turn in episode["turns"])
    episode["repository_tool_calls"] = len(indices) - episode["finish_catches"]
    if forced and episode["status"] == "ok" and len(indices) != max_iters:
        episode["status"] = "forced_short"
        episode["error"] = f"completed {len(indices)} of {max_iters} forced iterations"
    try:
        ledger, prompt_tokens, completion_tokens, total_tokens = _usage(lm.history)
        episode.update(
            usage=ledger,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            gen_tokens=completion_tokens,
            lm_calls=len(ledger),
        )
    except ValueError as error:
        episode["status"] = "error"
        episode["error"] = str(error)
        episode.update(
            usage=[],
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            gen_tokens=0,
            lm_calls=len(lm.history),
        )
    if debug:
        episode["debug_history"] = [
            {
                "prompt": _jsonable(entry.get("prompt")),
                "messages": _jsonable(entry.get("messages")),
                "outputs": _jsonable(entry.get("outputs")),
                "usage": _jsonable(entry.get("usage")),
                "response_model": entry.get("response_model"),
            }
            for entry in lm.history
        ]
    episode["finished"] = utc_now()
    return episode


def _source_state(smoke: bool) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if dirty and not smoke:
        raise SystemExit("full evaluations require a clean committed source tree")
    return commit, dirty


def _base_urls(value: str) -> list[str]:
    urls = value.split(",")
    if not urls or any(urlparse(url).hostname not in {"localhost", "127.0.0.1"} for url in urls):
        raise SystemExit("--base-urls must contain only loopback vLLM endpoints")
    return urls


def _valid_episode(path: Path, identity: dict, forced_iterations: int | None) -> bool:
    if not path.exists():
        return False
    try:
        episode = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if any(episode.get(key) != value for key, value in identity.items()):
        raise SystemExit(f"episode identity mismatch; use a new run ID: {path}")
    if episode.get("status") not in {"ok", "no_answer"}:
        return False
    if type(episode.get("gen_tokens")) is not int or episode["gen_tokens"] < 0:
        return False
    if forced_iterations is not None and episode.get("react_iterations") != forced_iterations:
        return False
    return True


def _study(args, corpus, tools, url: str, root: Path, config: dict) -> tuple[str, dict]:
    iterations = 2 if args.smoke else 50
    seed = stable_seed(args.seed, "study", args.task)
    question = {"id": "cheatsheet", "question": study_prompt(corpus.display, iterations)}
    episode_path, note_path = root / "study.json", root / "cheatsheet.md"
    identity = {
        "task": args.task,
        "qid": "cheatsheet",
        "condition": "cheatsheet",
        "budget": "study",
        "rollout": 0,
        "seed": seed,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "question_sha256": sha256_text(question["question"]),
        "study_config_sha256": sha256_json({**config, "iterations": iterations}),
    }
    if _valid_episode(episode_path, identity, iterations):
        episode = read_json(episode_path)
        if not note_path.exists():
            write_text(note_path, episode["answer"])
        elif note_path.read_text(encoding="utf-8") != episode["answer"]:
            raise SystemExit(f"study note does not match its episode: {note_path}")
    else:
        episode = run_episode(
            corpus=corpus,
            tools=tools,
            question=question,
            condition="cheatsheet",
            budget="study",
            rollout=0,
            seed=seed,
            base_url=url,
            max_iters=iterations,
            forced=True,
            debug=args.debug,
        )
        episode["study_config_sha256"] = identity["study_config_sha256"]
        write_json(episode_path, episode)
        if episode["status"] != "ok" or not episode["answer"].strip():
            raise SystemExit(f"cheatsheet study failed: {episode['status']}")
        write_text(note_path, episode["answer"])
    note = note_path.read_text(encoding="utf-8")
    return note, {
        "iterations": iterations,
        "seed": seed,
        "episode_sha256": sha256_json(episode),
        "generated_tokens": episode["gen_tokens"],
        "repository_tool_calls": episode["repository_tool_calls"],
        "finish_catches": episode["finish_catches"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=CORPORA)
    parser.add_argument("--condition", required=True, choices=("baseline", "cheatsheet"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budgets", default=",".join(BUDGETS))
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-urls", default="http://localhost:8100/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-id must contain only letters, digits, '-' and '_'")
    if args.rollouts < 1 or args.concurrency < 1 or args.limit < 0:
        parser.error("rollouts/concurrency must be positive and limit nonnegative")
    if args.smoke != (args.limit > 0):
        parser.error("smoke runs require --smoke and a positive --limit; full runs use neither")
    budgets = args.budgets.split(",")
    if not budgets or len(budgets) != len(set(budgets)) or any(item not in BUDGETS for item in budgets):
        parser.error(f"--budgets must be unique values from {','.join(BUDGETS)}")

    urls = _base_urls(args.base_urls)
    source_commit, source_dirty = _source_state(args.smoke)
    corpus = CORPORA[args.task]
    verify_corpus(corpus)
    repository = RepoTools(corpus)
    repository_tools = make_tools(repository)
    tool_config = {
        **TOOL_CONFIG,
        "corpus_roots": list(corpus.roots),
        "corpus_file_count": len(repository.files),
        "corpus_snapshot_sha256": repository.snapshot_sha256,
    }
    run_root = ROOT / "runs" / args.run_id / args.task
    runtime = {
        "python": platform.python_version(),
        "dspy": getattr(dspy, "__version__", "unknown"),
        "vllm": os.environ.get("SB_VLLM_VERSION", "unknown"),
        "gpu_names": os.environ.get("SB_GPU_NAMES", "unknown"),
        "gpu_memory_mib": os.environ.get("SB_GPU_MEMORY_MIB", "unknown"),
        "gpu_count": os.environ.get("SB_NGPU", "unknown"),
        "tensor_parallel": os.environ.get("SB_TP_EFFECTIVE", "unknown"),
        "servers": len(urls),
    }

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / f"{args.run_id}-{args.task}.log"),
        ],
    )

    note, study = "", None
    if args.condition == "cheatsheet":
        study_config = {
            "schema_version": 2,
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "corpus_commit": corpus.commit,
            "corpus_display": corpus.display,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "harness": "dspy.ReAct",
            "runtime": runtime,
            "sampling": SAMPLING,
            "tools": tool_config,
            "debug": args.debug,
        }
        note, study = _study(
            args, corpus, repository_tools, urls[0], run_root, study_config
        )

    questions = list(load_questions(args.task))[: args.limit or None]
    prefix = NOTE_PREFIX.format(library=corpus.display, note=note) if note else ""
    manifest = {
        "schema_version": 2,
        "run_id": args.run_id,
        "task": args.task,
        "condition": args.condition,
        "smoke": args.smoke,
        "debug": args.debug,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "corpus_commit": corpus.commit,
        "corpus_display": corpus.display,
        "corpus_file_count": len(repository.files),
        "corpus_snapshot_sha256": repository.snapshot_sha256,
        "dataset_sha256": corpus.dataset_sha256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "harness": "dspy.ReAct",
        "runtime": runtime,
        "sampling": SAMPLING,
        "tools": tool_config,
        "budgets": budgets,
        "rollouts": args.rollouts,
        "master_seed": args.seed,
        "question_ids": [question["id"] for question in questions],
        "note_sha256": sha256_text(note) if note else None,
        "note_prefix_sha256": sha256_text(prefix) if prefix else None,
        "study": study,
    }
    manifest_path = run_root / "run.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise SystemExit(f"run configuration changed; use a new --run-id: {manifest_path}")
    write_json(manifest_path, manifest)
    config_hash = sha256_json(manifest)

    cases = []
    for budget in budgets:
        max_iters, forced = BUDGETS[budget]
        for rollout in range(args.rollouts):
            for raw_question in questions:
                question = {**raw_question, "question": prefix + raw_question["question"]}
                seed = stable_seed(args.seed, "eval", args.task, question["id"], budget, rollout)
                path = run_root / "episodes" / budget / f"r{rollout}" / f"{question['id']}.json"
                identity = {
                    "run_config_sha256": config_hash,
                    "task": args.task,
                    "qid": question["id"],
                    "condition": args.condition,
                    "budget": budget,
                    "rollout": rollout,
                    "seed": seed,
                    "model": MODEL,
                    "model_revision": MODEL_REVISION,
                    "question_sha256": sha256_text(question["question"]),
                }
                cases.append((question, budget, rollout, seed, path, identity, max_iters, forced))

    pending = [
        case for case in cases
        if not _valid_episode(case[4], case[5], case[6] if case[7] else None)
    ]
    log.info("%d/%d evaluation episodes pending", len(pending), len(cases))

    def run_case(index: int, case):
        question, budget, rollout, seed, path, identity, max_iters, forced = case
        episode = run_episode(
            corpus=corpus,
            tools=repository_tools,
            question=question,
            condition=args.condition,
            budget=budget,
            rollout=rollout,
            seed=seed,
            base_url=urls[index % len(urls)],
            max_iters=max_iters,
            forced=forced,
            debug=args.debug,
        )
        episode["run_config_sha256"] = config_hash
        write_json(path, episode)
        return path, episode

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_case, index, case) for index, case in enumerate(pending)]
        for completed, future in enumerate(as_completed(futures), 1):
            path, episode = future.result()
            log.info(
                "[%d/%d] %s status=%s iters=%d calls=%d gen_tokens=%d",
                completed,
                len(futures),
                path.relative_to(run_root),
                episode["status"],
                episode["react_iterations"],
                episode["lm_calls"],
                episode["gen_tokens"],
            )

    failures = []
    for _, _, _, _, path, identity, max_iters, forced in cases:
        if not _valid_episode(path, identity, max_iters if forced else None):
            failures.append(str(path))
    if failures:
        log.error("%d episodes failed or are missing; rerun the same command", len(failures))
        raise SystemExit(1)
    log.info("evaluation complete: %s/%s", args.run_id, args.task)


if __name__ == "__main__":
    main()
