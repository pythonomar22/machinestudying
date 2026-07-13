"""The paper's harness, faithfully: dspy.ReAct over the pinned DSPy checkout.

Author-confirmed details (docs/jacob.md):
- harness = dspy.ReAct ("I was using dspy.ReAct"); model re-thinks fresh every
  step (each react step is a stateless LM call);
- forced-20 / forced-50: "Just catch the finish and return something like you
  gotta keep searching type of logic, no need to remove that specific turn";
- direct = dspy.Predict;
- tools = grep, glob, read_file (line ranges, capped at 200 lines);
- lenient = pure weighted claim sum (grading side).

Run inside .venv-dspy (the pinned corpora/dspy install). New episodes live in
the immutable namespace chosen with --run-id and are bound to its manifest.
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
from dspy.predict.react import _fmt_exc

from .dataset import CORPORA, ROOT, load_questions
from .integrity import (
    exclusive_process_lock,
    load_json_artifact,
    read_artifact_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
    stable_seed,
    utc_now,
    write_immutable_json,
    write_immutable_text,
)
from .provenance import (
    corpus_record,
    environment_record,
    environment_is_claim_ready,
    episode_identity,
    prepare_run,
    source_record,
    validate_id,
    validate_local_server_urls,
    write_episode_result,
)
from .rollout import (
    BUDGETS,
    _reject_invalid_final_episode,
    _validate_final_episode,
    _validated_resumable_episode,
)
from .tools import TOOL_SCHEMAS, RepoTools

MODEL_ID = "openai/Qwen/Qwen3.5-9B"  # openai-compatible provider -> our vLLM
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
READ_MAX_LINES = 200  # author: "read file (lines, Capped at 200lines)"

# The cheatsheet study task (replication inference, carried over from the native
# study loop — see experiments/002 §inferences). The forced-50 mechanism is the
# author's ("same question for the cheatsheet study loop" -> catch finish).
def study_task(corpus) -> str:
    return (
        f"Study the {corpus.display} repository and write yourself a cheatsheet: a "
        f"reference document that will be prepended to every future question you are "
        f"asked about {corpus.display}. You will not see the questions in advance, "
        "but you will keep access to these repository tools when answering them. "
        "Record whatever will make you fastest and most accurate later. After your "
        "50 iterations of study, write the complete cheatsheet as your final answer."
    )
SAMPLING = dict(  # paper §B; passed through litellm to vLLM
    temperature=1.0, top_p=0.95, max_tokens=32768, presence_penalty=1.5,
    extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
)

log = logging.getLogger("react")


def _artifact_inventory(root: Path, relatives: tuple[str, ...]) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for relative in relatives:
        path = root / relative
        # Both forced-50 dependencies are JSON. This read also rejects
        # non-regular files and symlinks anywhere in the artifact path.
        load_json_artifact(path)
        inventory[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return inventory


def _validate_completed_study(
    manifest_path: Path,
    intent_path: Path,
    out: Path,
    config: dict[str, object],
) -> None:
    """Revalidate every dependency before treating an immutable study as complete."""

    try:
        manifest = load_json_artifact(manifest_path)
        intent = load_json_artifact(intent_path)
        episode_path = out / "episode.json"
        episode = load_json_artifact(episode_path)
        if not all(isinstance(value, dict) for value in (manifest, intent, episode)):
            raise ValueError("study artifacts must be JSON objects")
        if intent != config or sha256_file(intent_path) != sha256_json(config):
            raise ValueError("study intent differs from its canonical configuration")

        answer = episode.get("answer")
        if (episode.get("status") != "ok" or not isinstance(answer, str)
                or not answer.strip()):
            raise ValueError("study episode is not a successful nonempty answer")
        intent_sha256 = sha256_json(config)
        if (episode.get("study_intent_sha256") != intent_sha256
                or episode.get("question_sha256") != config["study_question_sha256"]
                or episode.get("model") != config["model"]
                or episode.get("model_revision") != config["model_revision"]
                or episode.get("seed") != config["episode_seed"]
                or sha256_file(episode_path) != sha256_json(episode)):
            raise ValueError("study episode does not bind to the exact intent")

        _validate_final_episode(
            episode,
            {
                "study_intent_sha256": intent_sha256,
                "question_sha256": config["study_question_sha256"],
                "task": config["task"],
                "qid": "cheatsheet",
                "budget": "s50",
                "rollout": 0,
                "seed": config["episode_seed"],
            },
            expected_model=config["model"],
            expected_model_revision=config["model_revision"],
            expected_harness="dspy.ReAct",
            expected_response_model=config["expected_response_model"],
        )
        if config.get("forced_iterations") != BUDGETS["s50"][0]:
            raise ValueError("study intent has a drifted forced-iteration budget")

        token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(type(episode.get(field)) is not int or episode[field] < 0
               for field in token_fields):
            raise ValueError("study episode has invalid token accounting")
        if episode["total_tokens"] != episode["prompt_tokens"] + episode["completion_tokens"]:
            raise ValueError("study episode token accounting is inconsistent")

        note_sha256 = sha256_text(answer)
        note_name = f"note-{note_sha256}.md"
        note_path = out / note_name
        note_bytes = read_artifact_bytes(note_path)
        if (sha256_file(note_path) != note_sha256
                or note_bytes.decode("utf-8") != answer):
            raise ValueError("study note differs from the exact episode answer")

        inventory = _artifact_inventory(out, ("intent.json", "episode.json"))
        expected = {
            "manifest_schema": 1,
            "manifest_type": "forced-50-cheatsheet",
            "claim_ready": config["claim_ready"],
            "study_id": config["study_id"],
            "task": config["task"],
            "corpus_commit": config["corpus"]["commit"],
            "config": config,
            "note_sha256": note_sha256,
            "note_path": note_name,
            "episode_sha256": sha256_json(episode),
            "intent_sha256": intent_sha256,
            "study_generated_tokens": episode["completion_tokens"],
            "study_prompt_tokens": episode["prompt_tokens"],
            "study_total_tokens": episode["total_tokens"],
            "construction_artifacts": inventory,
            "construction_artifacts_sha256": sha256_json(inventory),
        }
        if manifest != expected:
            raise ValueError("study manifest or dependency inventory has drifted")
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(
            "completed study failed immutable dependency validation; "
            "preserve it and choose a new --study-id"
        ) from exc


def make_tools(rt: RepoTools):
    def grep(pattern: str, path: str = "") -> str:
        """Search the repository code for a regular expression (case-sensitive).
        Returns matching lines as path:line_number:line. `path` optionally
        restricts the search to a file or directory."""
        args = {"pattern": pattern, **({"path": path} if path else {})}
        return rt.dispatch("grep", json.dumps(args))

    def glob(pattern: str) -> str:
        """List repository files matching a glob pattern, e.g. 'dspy/**/*.py'."""
        return rt.dispatch("glob", json.dumps({"pattern": pattern}))

    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file from the repository by line range (1-indexed; at most
        200 lines per call). end_line=0 reads from start_line to the cap."""
        args = {"path": path, "start_line": start_line,
                **({"end_line": end_line} if end_line else {})}
        return rt.dispatch("read_file", json.dumps(args))

    return [grep, glob, read_file]


class ForcedReAct(dspy.ReAct):
    """dspy.ReAct with no early stopping: finish selections are caught and
    answered with a keep-searching observation; the turn stays in the
    trajectory and the loop runs its full max_iters, then extract runs."""

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args)
            except ValueError as err:
                raise ForcedTrajectoryError(
                    f"forced trajectory failed at iteration {idx}: {_fmt_exc(err)}",
                    trajectory,
                    status="forced_short",
                ) from err
            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args
            if pred.next_tool_name == "finish":
                trajectory[f"observation_{idx}"] = (
                    "You cannot finish yet — you gotta keep searching. "
                    "Pick another tool call.")
                continue
            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](
                    **pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = (
                    f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}")
        try:
            extract = self._call_with_potential_trajectory_truncation(
                self.extract, trajectory, **input_args)
        except Exception as err:
            raise ForcedTrajectoryError(
                f"forced answer extraction failed: {_fmt_exc(err)}",
                trajectory,
                status="error",
            ) from err
        return dspy.Prediction(trajectory=trajectory, **extract)


class ForcedTrajectoryError(RuntimeError):
    def __init__(self, message: str, trajectory: dict, *, status: str):
        super().__init__(message)
        self.trajectory = trajectory
        self.status = status


def _response_field(response: object, field: str) -> object:
    return response.get(field) if isinstance(response, dict) \
        else getattr(response, field, None)


def _dspy_usage_record(history: object, index: int) -> dict:
    """Build one exact provider ledger record without inventing token counts."""

    if not isinstance(history, dict):
        raise ValueError("DSPy history entry is not an object")
    usage = history.get("usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    if not isinstance(usage, dict):
        raise ValueError("DSPy history entry has no provider usage")
    values = [usage.get(field) for field in (
        "prompt_tokens", "completion_tokens", "total_tokens")]
    if (any(type(value) is not int or value < 0 for value in values)
            or values[2] != values[0] + values[1]):
        raise ValueError("DSPy provider usage is incomplete or inconsistent")
    messages = history.get("messages")
    outputs = history.get("outputs")
    if messages is None or outputs is None:
        raise ValueError("DSPy request or output ledger is unavailable")
    response = history.get("response")
    return {
        "call": index,
        "response_model": history.get("response_model")
        or _response_field(response, "model"),
        "response_id": _response_field(response, "id"),
        "system_fingerprint": _response_field(response, "system_fingerprint"),
        "request_messages_sha256": sha256_json(messages),
        "outputs_sha256": sha256_json(outputs),
        "provider_usage": usage,
        "prompt_tokens": values[0],
        "completion_tokens": values[1],
        "total_tokens": values[2],
    }


def run_episode(corpus, tools_fns, q: dict, budget: str, rollout: int,
                base_url: str, *, seed: int,
                identity: dict[str, object] | None = None) -> dict:
    max_iters, forced = BUDGETS[budget]
    api_key = os.environ.get("SB_VLLM_API_KEY")
    if not api_key:
        raise RuntimeError("authenticated local server key is unavailable")
    lm = dspy.LM(MODEL_ID, api_base=base_url, api_key=api_key, model_type="chat",
                 cache=False, num_retries=0,
                 **{**SAMPLING, "seed": seed})
    ep = {
        **(identity or {}),
        "task": corpus.name, "qid": q["id"], "budget": budget, "rollout": rollout,
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "harness": "dspy.ReAct", "seed": seed, "started": utc_now(),
        "turns": [], "answer": "", "n_react_iters": 0, "n_tool_iters": 0,
        "finish_catches": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "gen_tokens": 0, "status": "ok",
    }
    trajectory = {}
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        try:
            if budget == "direct":
                pred = dspy.Predict("question -> answer")(question=q["question"])
            else:
                cls = ForcedReAct if forced else dspy.ReAct
                module = cls("question -> answer", tools=list(tools_fns),
                             max_iters=max_iters)
                pred = module(question=q["question"])
                trajectory = dict(pred.trajectory)
            ep["answer"] = pred.answer or ""
            if not ep["answer"].strip():
                ep["status"] = "no_answer"
        except ForcedTrajectoryError as error:
            trajectory = error.trajectory
            ep["status"] = error.status
            ep["error"] = str(error)[:500]
        except Exception as e:
            ep["status"] = "error"
            ep["error"] = f"{type(e).__name__}: {str(e)[:500]}"

    steps = sorted(int(k.rsplit("_", 1)[1]) for k in trajectory
                   if k.startswith("thought_"))
    for i in steps:
        name = trajectory.get(f"tool_name_{i}")
        ep["turns"].append({
            "reasoning": trajectory.get(f"thought_{i}"),
            "tool_calls": [{"name": name,
                            "arguments": json.dumps(trajectory.get(f"tool_args_{i}"))}],
            "observations": [str(trajectory.get(f"observation_{i}"))],
        })
        if name == "finish":
            ep["finish_catches"] += 1
        else:
            ep["n_tool_iters"] += 1
    ep["n_react_iters"] = len(steps)
    if forced and ep["status"] == "ok" and len(steps) != max_iters:
        ep["status"] = "forced_short"
        ep["error"] = f"forced trajectory has {len(steps)} of {max_iters} required iterations"
    ep["n_lm_calls"] = len(lm.history)
    ep["usage_ledger"] = []
    for index, history in enumerate(lm.history):
        try:
            record = _dspy_usage_record(history, index)
        except (TypeError, ValueError) as error:
            if ep["status"] in {"ok", "no_answer"}:
                ep["invalid_final_status"] = ep["status"]
            ep["status"] = "error"
            ep["error"] = f"DSPy usage ledger validation failed: {str(error)[:450]}"
            break
        ep["usage_ledger"].append(record)
        ep["prompt_tokens"] += record["prompt_tokens"]
        ep["completion_tokens"] += record["completion_tokens"]
        ep["total_tokens"] += record["total_tokens"]
    ep["gen_tokens"] = ep["completion_tokens"]
    ep["finished"] = utc_now()
    expected_identity = {
        **(identity or {}),
        "task": corpus.name,
        "qid": q["id"],
        "budget": budget,
        "rollout": rollout,
        "seed": seed,
    }
    _reject_invalid_final_episode(
        ep,
        expected_identity,
        expected_model=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_harness="dspy.ReAct",
        expected_response_model=MODEL_ID.removeprefix("openai/"),
    )
    return ep


def _run_study_locked(args, corpus, tools_fns, urls: list[str], out: Path) -> None:
    if not args.study_id:
        raise SystemExit("--study requires --study-id")
    validate_id(args.study_id, "study ID")
    manifest_path = out / "manifest.json"
    seed = stable_seed(args.seed, "cheatsheet", args.study_id, args.task)
    corpus_info = corpus_record(corpus)
    source = source_record()
    environment = environment_record()
    if corpus_info["dirty"] or corpus_info["commit"] != corpus.commit:
        raise SystemExit("corpus is dirty or not at its pinned commit")
    if source["dirty"] and not (args.allow_dirty or args.smoke):
        raise SystemExit("research source files are dirty; commit before studying")
    packages = environment.get("packages")
    environment_ready = bool(
        environment_is_claim_ready(environment)
        and isinstance(packages, dict)
        and packages.get("dspy")
    )
    if not environment_ready and not (args.allow_dirty or args.smoke):
        raise SystemExit("study environment is incomplete; use the pinned server launcher")
    question = {"id": "cheatsheet", "question": study_task(corpus)}
    config = {
        "schema_version": 1,
        "study_id": args.study_id,
        "task": args.task,
        "method": "forced-50-cheatsheet",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "expected_response_model": MODEL_ID.removeprefix("openai/"),
        "sampling": SAMPLING,
        "master_seed": args.seed,
        "episode_seed": seed,
        "study_prompt_sha256": sha256_text(question["question"]),
        "study_question_sha256": sha256_json(question),
        "tool_schema_sha256": sha256_json(TOOL_SCHEMAS),
        "read_max_lines": READ_MAX_LINES,
        "forced_iterations": BUDGETS["s50"][0],
        "corpus": corpus_info,
        "source": source,
        "environment": environment,
        "claim_ready": not args.smoke and not source["dirty"] and environment_ready,
        "server_transport": {
            "scope": "loopback",
            "protocol": "openai-compatible-http",
            "available_server_count": len(urls),
            "selected_server_index": 0,
        },
    }
    intent_path = out / "intent.json"
    if manifest_path.exists():
        _validate_completed_study(manifest_path, intent_path, out, config)
        log.info("study already complete: %s", manifest_path)
        return
    write_immutable_json(intent_path, config)
    episode_path = out / "episode.json"
    if episode_path.exists():
        raise SystemExit(
            "an earlier study attempt ended before a claim-ready manifest was written; "
            "preserve it and choose a new --study-id"
        )

    intent_sha256 = sha256_json(config)
    ep = run_episode(
        corpus,
        tools_fns,
        question,
        "s50",
        0,
        urls[0],
        seed=seed,
        identity={
            "study_intent_sha256": intent_sha256,
            "question_sha256": sha256_json(question),
        },
    )
    log.info("study: status=%s react_iters=%d tool_iters=%d catches=%d gen_tokens=%d note_chars=%d",
             ep["status"], ep["n_react_iters"], ep["n_tool_iters"],
             ep["finish_catches"], ep["gen_tokens"], len(ep["answer"]))
    _reject_invalid_final_episode(
        ep,
        {
            "study_intent_sha256": intent_sha256,
            "question_sha256": sha256_json(question),
            "task": args.task,
            "qid": "cheatsheet",
            "budget": "s50",
            "rollout": 0,
            "seed": seed,
        },
        expected_model=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_harness="dspy.ReAct",
        expected_response_model=config["expected_response_model"],
    )
    write_immutable_json(episode_path, ep)
    if ep["status"] != "ok" or not ep["answer"].strip():
        raise SystemExit(f"study episode failed: {ep['status']}")
    note_hash = sha256_text(ep["answer"])
    note_name = f"note-{note_hash}.md"
    write_immutable_text(out / note_name, ep["answer"])
    manifest = {
        "manifest_schema": 1,
        "manifest_type": "forced-50-cheatsheet",
        "claim_ready": config["claim_ready"],
        "study_id": args.study_id,
        "task": args.task,
        "corpus_commit": corpus_info["commit"],
        "config": config,
        "note_sha256": note_hash,
        "note_path": note_name,
        "episode_sha256": sha256_json(ep),
        "intent_sha256": intent_sha256,
        "study_generated_tokens": ep["completion_tokens"],
        "study_prompt_tokens": ep["prompt_tokens"],
        "study_total_tokens": ep["total_tokens"],
    }
    construction_artifacts = _artifact_inventory(out, ("intent.json", "episode.json"))
    manifest["construction_artifacts"] = construction_artifacts
    manifest["construction_artifacts_sha256"] = sha256_json(construction_artifacts)
    write_immutable_json(manifest_path, manifest)
    log.info("wrote immutable study manifest %s", manifest_path)


def _run_study(args, corpus, tools_fns, urls: list[str]) -> None:
    if not args.study_id:
        raise SystemExit("--study requires --study-id")
    validate_id(args.study_id, "study ID")
    out = ROOT / "studies" / "cheatsheet" / args.study_id / args.task
    lock = ROOT / "studies" / "cheatsheet" / ".locks" / args.study_id / f"{args.task}.lock"
    with exclusive_process_lock(lock):
        _run_study_locked(args, corpus, tools_fns, urls, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--run-id", help="unique immutable evaluation namespace")
    p.add_argument("--seed", required=True, type=int,
                   help="master seed for study or deterministic evaluation episodes")
    p.add_argument("--seed-group",
                   help="evaluation pairing ID shared by baseline and treatment arms")
    p.add_argument("--budgets", default="direct,k5,k20,k20f")
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--base-urls", default="http://localhost:8100/v1")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--note", type=Path,
                   help="exact note to prepend; never inferred from a mutable alias")
    p.add_argument("--note-manifest", type=Path,
                   help="construction manifest whose note_sha256 matches --note")
    purpose = p.add_mutually_exclusive_group()
    purpose.add_argument("--smoke", action="store_true",
                         help="isolate output under runs/smoke and mark it non-claim-ready")
    purpose.add_argument(
        "--exploratory", action="store_true",
        help="run the full evaluation grid but mark it non-claim-ready",
    )
    p.add_argument(
        "--preregistration", type=Path,
        help="canonical committed preregistrations/*.json contract",
    )
    p.add_argument(
        "--preregistration-role", choices=("control", "treatment"),
        help="arm in --preregistration bound to this evaluation",
    )
    p.add_argument("--allow-dirty", action="store_true",
                   help="diagnostic only; dirty inputs make artifacts non-claim-ready")
    p.add_argument("--study", action="store_true",
                   help="run one immutable forced-50 cheatsheet study episode")
    p.add_argument("--study-id", help="unique namespace required by --study")
    args = p.parse_args()

    if args.rollouts <= 0 or args.concurrency <= 0 or args.limit < 0:
        p.error("--rollouts and --concurrency must be positive; --limit must be nonnegative")
    if args.allow_dirty and not args.smoke:
        p.error("--allow-dirty is diagnostic and requires --smoke")
    if not args.study and args.smoke and args.limit <= 0:
        p.error("evaluation --smoke requires a positive --limit")

    if args.study and (
        args.run_id
        or args.seed_group
        or args.note
        or args.note_manifest
        or args.exploratory
        or args.preregistration
        or args.preregistration_role
    ):
        raise SystemExit("--study cannot be combined with evaluation --run-id/--note options")
    if not args.study and args.study_id:
        raise SystemExit("--study-id is only valid with --study")
    if not args.study and not args.seed_group:
        raise SystemExit("evaluation requires --seed-group")
    if not args.study and (
        (args.preregistration is None) != (args.preregistration_role is None)
    ):
        p.error("--preregistration and --preregistration-role are required together")
    if not args.study and not (
        args.smoke or args.exploratory or args.preregistration is not None
    ):
        p.error("confirmatory runs require --preregistration; otherwise use --exploratory")
    log_id = args.study_id if args.study else args.run_id
    if not log_id:
        raise SystemExit("--study-id is required for study; --run-id is required for evaluation")
    validate_id(log_id, "log namespace")
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"react-{log_id}-{args.task}.log")])
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    corpus = CORPORA[args.task]
    tools_fns = make_tools(RepoTools(corpus, read_max_lines=READ_MAX_LINES))
    declared_servers = None
    if not (args.smoke or args.allow_dirty):
        try:
            declared_servers = int(os.environ["SB_NSERVE"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("claim-ready runs require a valid SB_NSERVE") from exc
    urls = validate_local_server_urls(args.base_urls, expected_count=declared_servers)

    if args.study:
        _run_study(args, corpus, tools_fns, urls)
        return

    if not args.run_id:
        raise SystemExit("evaluation requires --run-id")
    if args.limit and not args.smoke:
        raise SystemExit("--limit is diagnostic only and requires --smoke")
    questions = load_questions(args.task)[: args.limit or None]
    episode_contract = {
        "expected_model": MODEL_ID,
        "expected_model_revision": MODEL_REVISION,
        "expected_harness": "dspy.ReAct",
        "expected_response_model": MODEL_ID.removeprefix("openai/"),
    }
    budgets = args.budgets.split(",")
    if any(budget not in BUDGETS or budget == "s50" for budget in budgets):
        raise SystemExit(f"invalid evaluation budgets: {args.budgets}")
    context = prepare_run(
        run_id=args.run_id,
        task=args.task,
        corpus=corpus,
        questions=questions,
        budgets=budgets,
        rollouts=args.rollouts,
        harness="dspy.ReAct",
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        sampling=SAMPLING,
        master_seed=args.seed,
        seed_namespace="dspy-react",
        seed_group=args.seed_group,
        note_path=args.note,
        note_manifest_path=args.note_manifest,
        note_prefix_template=(
            f"Reference notes on {corpus.display} from your prior study of its repository:\n\n"
            "{note}\n\n---\n\n"
        ),
        smoke=args.smoke,
        exploratory=args.exploratory,
        allow_dirty=args.allow_dirty,
        preregistration_path=args.preregistration,
        preregistration_role=args.preregistration_role,
        extra={
            "model_revision": MODEL_REVISION,
            "expected_response_model": MODEL_ID.removeprefix("openai/"),
            "model_context_tokens": 262_144,
            "tool_iterations": BUDGETS,
            "tool_schema_sha256": sha256_json(TOOL_SCHEMAS),
            "read_max_lines": READ_MAX_LINES,
            "signature": "question -> answer",
            "adapter": "dspy.ChatAdapter",
            "server_transport": {
                "scope": "loopback",
                "protocol": "openai-compatible-http",
                "server_count": len(urls),
            },
            "concurrency": args.concurrency,
        },
    )
    prompted = [
        (q, {**q, "question": context.prompt_prefix + q["question"]})
        for q in questions
    ]

    cases = []
    pending = []
    for budget in budgets:
        for rollout in range(args.rollouts):
            for raw_q, q in prompted:
                out = context.root / budget / f"r{rollout}" / f"{q['id']}.json"
                seed = stable_seed(
                    args.seed, "dspy-react", args.seed_group, args.task,
                    q["id"], budget, rollout,
                )
                identity = episode_identity(
                    context, q=raw_q, prompt=q["question"], budget=budget,
                    rollout=rollout, seed=seed,
                )
                cases.append((raw_q, q, budget, rollout, out, seed, identity))
                existing = _validated_resumable_episode(
                    out, identity, context=context, **episode_contract
                )
                if existing:
                    if existing.get("status") in ("ok", "no_answer"):
                        continue
                    raise ValueError(f"non-final artifact occupies expected episode path: {out}")
                pending.append((raw_q, q, budget, rollout, out, seed, identity))
    log.info("%d episodes pending (task=%s, harness=dspy.ReAct)", len(pending), args.task)

    done = 0

    def one(i, raw_q, q, budget, rollout, out, seed, identity):
        nonlocal done
        try:
            lock = context.root / "locks" / out.relative_to(context.root).with_suffix(".lock")
            with exclusive_process_lock(lock):
                if _validated_resumable_episode(
                    out, identity, context=context, **episode_contract
                ) is not None:
                    return
                ep = run_episode(
                    corpus, tools_fns, q, budget, rollout, urls[i % len(urls)],
                    seed=seed, identity=identity,
                )
                _reject_invalid_final_episode(
                    ep, identity, **episode_contract
                )
                artifact = write_episode_result(context, out, ep)
        except Exception:
            log.exception("episode %s/%s/r%d failed", budget, q["id"], rollout)
            return
        done += 1
        log.info("[%d/%d] %s/%s/r%d: status=%s iters=%d catches=%d calls=%d gen_tokens=%d",
                 done, len(pending), budget, q["id"], rollout, ep["status"],
                 ep["n_react_iters"], ep["finish_catches"], ep["n_lm_calls"],
                 ep["gen_tokens"])
        if artifact != out:
            log.warning("retained failed attempt at %s", artifact)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(lambda t: one(*t), [(i, *p) for i, p in enumerate(pending)]))

    statuses = {}
    for _, _, _, _, out, _, identity in cases:
        existing = _validated_resumable_episode(
            out, identity, context=context, **episode_contract
        )
        s = existing.get("status", "missing") if existing else "missing"
        statuses[s] = statuses.get(s, 0) + 1
    log.info("all done: %s", statuses)
    if statuses.keys() - {"ok", "no_answer"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
