"""Generation: ReAct rollouts of Qwen3.5-9B over StudyBench, one JSON per episode.

Replicates the paper's four inference budgets: direct answer (no tools), up to 5 or 20
voluntary tool iterations, and a forced 20 iterations with no early stopping. Sampling
parameters are the paper's (= Qwen3.5 thinking-mode defaults). Grading happens
in a separate judge-API stage; this module writes only inside the immutable
--run-id namespace.
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

from .dataset import CORPORA, ROOT, load_questions
from .grade import GradeIntegrityError, episode_provider_identity, validate_episode
from .integrity import (
    canonical_json_bytes,
    exclusive_process_lock,
    sha256_json,
    stable_seed,
    utc_now,
)
from .provenance import (
    episode_identity,
    prepare_run,
    validate_id,
    validate_local_server_urls,
    validate_resumable_episode,
    write_episode_result,
)
from .tools import TOOL_SCHEMAS, RepoTools

MODEL = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MAX_TOKENS_PER_TURN = 32_768  # paper: maximum generation length per ReAct turn
SAMPLING = {  # paper §B: official Qwen3.5 thinking-mode sampling parameters
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 1.5,
    "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
}
BUDGETS = {  # name -> (max tool iterations, forced: no early stopping)
    "direct": (0, False),
    "k5": (5, False),
    "k20": (20, False),
    "k20f": (20, True),
    "s50": (50, True),  # the cheatsheet study loop (paper §B Table 4); study.py only
}
MAX_NUDGES = 2  # retries when a turn produces neither a tool call nor answer text

log = logging.getLogger("rollout")


def system_prompt(corpus, budget: str) -> str:
    base = (
        f"You are an expert on {corpus.display}, answering a user's question about it. "
        f"A checkout of the {corpus.display} repository is available; its code lives under "
        f"these top-level directories: {', '.join(corpus.roots)}."
    )
    max_iters, forced = BUDGETS[budget]
    if max_iters == 0:
        return base + " Answer the user's question directly."
    base += (
        " Use the tools (grep, glob, read_file) to explore the code as needed, "
        "then give a complete, self-contained answer to the question."
    )
    if forced:
        base += (
            f" You must use tools for exactly {max_iters} iterations to research the "
            "question before giving your final answer; do not answer early."
        )
    return base


def episode_seed(master_seed: int, seed_group: str, task: str, qid: str,
                 budget: str, rollout: int) -> int:
    return stable_seed(master_seed, "native-react", seed_group, task, qid, budget, rollout)


def _validate_final_episode(
    episode: dict,
    identity: dict[str, object],
    *,
    expected_model: str,
    expected_model_revision: str,
    expected_harness: str,
    expected_response_model: str,
) -> None:
    """Validate one final producer artifact before persistence or resumption.

    Grading owns the canonical turn, token, provider, and request-ledger rules.
    Producers additionally bind those rules to the exact invocation and model
    contract so an invalid ``ok``/``no_answer`` artifact never becomes durable.
    Launch-environment compatibility remains owned by
    :func:`validate_resumable_episode`; it deliberately permits a compatible
    allocation change when a run resumes.
    """

    if not isinstance(episode, dict) or episode.get("status") not in {"ok", "no_answer"}:
        raise ValueError("episode is not a final outcome")
    if not isinstance(identity, dict):
        raise ValueError("episode identity is not an object")
    try:
        for field, expected in identity.items():
            if field == "environment_snapshot":
                continue
            if (field not in episode
                    or canonical_json_bytes(episode[field])
                    != canonical_json_bytes(expected)):
                raise ValueError(f"episode identity mismatch for {field}")
        declared = {
            "model": expected_model,
            "model_revision": expected_model_revision,
            "harness": expected_harness,
        }
        if any(
            not isinstance(expected, str)
            or not expected
            or episode.get(field) != expected
            for field, expected in declared.items()
        ):
            raise ValueError("episode model or harness identity is invalid")
        if (not isinstance(episode.get("started"), str) or not episode["started"]
                or not isinstance(episode.get("finished"), str)
                or not episode["finished"]
                or "error" in episode
                or "invalid_final_status" in episode):
            raise ValueError("final episode lifecycle is invalid")

        validate_episode(episode, {"id": episode.get("qid")})
        provider = episode_provider_identity(episode)
        if (not isinstance(expected_response_model, str)
                or not expected_response_model
                or provider["response_models"] != [expected_response_model]):
            raise ValueError(
                "provider response model does not match the producer contract"
            )

        budget = episode.get("budget")
        if budget not in BUDGETS:
            raise ValueError("episode has an unknown inference budget")
        max_iters, forced = BUDGETS[budget]
        if "n_react_iters" in episode:
            observed_iters = episode["n_react_iters"]
        else:
            observed_iters = episode.get("n_tool_iters", 0) \
                + episode.get("finish_catches", 0)
        if (type(observed_iters) is not int or observed_iters < 0
                or observed_iters > max_iters
                or (forced and observed_iters != max_iters)):
            raise ValueError("episode iteration counters violate the declared budget")

        native = "usage_ledger" not in episode
        records = episode["turns"] if native else episode["usage_ledger"]
        response_ids = [record.get("response_id") for record in records]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("episode repeats a provider response ID")

        if native:
            required_turn = {
                "response_id", "response_model", "system_fingerprint",
                "reasoning", "content", "tool_calls", "observations",
                "finish_reason", "prompt_tokens", "completion_tokens",
                "total_tokens",
            }
            for index, turn in enumerate(records):
                if (not required_turn.issubset(turn)
                        or (turn["reasoning"] is not None
                            and not isinstance(turn["reasoning"], str))
                        or (turn["content"] is not None
                            and not isinstance(turn["content"], str))
                        or not isinstance(turn["finish_reason"], str)
                        or not turn["finish_reason"]):
                    raise ValueError(f"native turn {index} is incomplete")
            final_turn = records[-1]
            if final_turn["tool_calls"]:
                raise ValueError("final native provider turn still contains a tool call")
            if (episode["status"] == "ok"
                    and final_turn["content"] != episode["answer"]):
                raise ValueError("native answer differs from the final provider turn")

            attempts = episode.get("request_attempts")
            if len(attempts) > 4 * len(records):
                raise ValueError("native request audit exceeds the retry contract")
            for index, attempt in enumerate(attempts):
                status = attempt.get("status")
                if status == "response":
                    required = {
                        "logical_call", "attempt", "status", "request_sha256",
                        "response_id", "response_model",
                    }
                elif status == "transport_error":
                    required = {
                        "logical_call", "attempt", "status", "request_sha256",
                        "error_type", "error", "usage",
                    }
                else:
                    raise ValueError(f"native request attempt {index} has invalid status")
                if not required.issubset(attempt) or attempt["attempt"] > 4:
                    raise ValueError(f"native request attempt {index} is incomplete")
        else:
            required_call = {
                "call", "response_model", "response_id", "system_fingerprint",
                "request_messages_sha256", "outputs_sha256", "provider_usage",
                "prompt_tokens", "completion_tokens", "total_tokens",
            }
            null_hash = sha256_json(None)
            for index, record in enumerate(records):
                if (not required_call.issubset(record)
                        or record["request_messages_sha256"] == null_hash
                        or record["outputs_sha256"] == null_hash):
                    raise ValueError(f"DSPy usage call {index} is incomplete")
    except GradeIntegrityError as error:
        raise ValueError(f"episode failed canonical validation: {error}") from error


def _reject_invalid_final_episode(
    episode: dict,
    identity: dict[str, object],
    **contract: str,
) -> bool:
    """Turn an invalid claimed final result into a retained failure attempt."""

    if episode.get("status") not in {"ok", "no_answer"}:
        return False
    try:
        _validate_final_episode(episode, identity, **contract)
    except (TypeError, ValueError) as error:
        episode["invalid_final_status"] = episode["status"]
        episode["status"] = "error"
        episode["error"] = f"producer validation failed: {str(error)[:450]}"
        return False
    return True


def _validated_resumable_episode(
    path: Path,
    identity: dict[str, object],
    *,
    context,
    **contract: str,
) -> dict | None:
    """Load an existing episode and fully validate any claimed final result."""

    episode = validate_resumable_episode(path, identity, context=context)
    if episode is not None and episode.get("status") in {"ok", "no_answer"}:
        try:
            _validate_final_episode(episode, identity, **contract)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid final existing episode: {path}") from error
    return episode


async def run_episode(client: AsyncOpenAI, corpus, tools: RepoTools, q: dict,
                      budget: str, rollout: int, think_history: bool = True,
                      system: str | None = None, *, seed: int,
                      identity: dict[str, object]) -> dict:
    max_iters, forced = BUDGETS[budget]
    messages = [
        {"role": "system", "content": system or system_prompt(corpus, budget)},
        {"role": "user", "content": q["question"]},
    ]
    ep = {
        **identity,
        "task": corpus.name, "qid": q["id"], "budget": budget, "rollout": rollout,
        "model": MODEL, "model_revision": MODEL_REVISION,
        "harness": "native-react" if think_history else "native-react-no-think-history",
        "seed": seed,
        "started": utc_now(), "turns": [], "answer": "", "n_tool_iters": 0,
        "request_attempts": [],
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        # Backward-compatible name; expertise uses generated/completion tokens.
        "gen_tokens": 0, "status": "ok",
    }
    iters = nudges = 0
    while True:
        allow_tools = iters < max_iters
        if not allow_tools and max_iters and messages[-1]["role"] == "tool":
            # the tool budget was just exhausted; the model cannot see that tools
            # are gone, so ask for the answer explicitly
            messages.append({"role": "user", "content":
                             "You have used all your tool calls. Now give your "
                             "final, complete answer to the original question."})
        kwargs = {}
        if allow_tools:
            kwargs["tools"] = TOOL_SCHEMAS
            kwargs["tool_choice"] = "required" if forced else "auto"
            # one action per ReAct iteration (paper: "up to 5 or 20 tool iterations")
            kwargs["parallel_tool_calls"] = False
        try:
            resp = await request_with_retry(
                client, messages=messages, model=MODEL, seed=seed,
                max_tokens=MAX_TOKENS_PER_TURN, audit=ep["request_attempts"],
                logical_call=len(ep["turns"]), **SAMPLING, **kwargs)
        except Exception as e:
            ep["status"], ep["error"] = "error", f"{type(e).__name__}: {e}"
            break
        try:
            if not isinstance(resp.id, str) or not resp.id:
                raise ValueError("provider response has no response ID")
            if resp.model != MODEL:
                raise ValueError(
                    f"provider response model {resp.model!r} does not match {MODEL!r}"
                )
            if not isinstance(resp.choices, list) or len(resp.choices) != 1:
                raise ValueError("provider response must contain exactly one choice")
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []
            if (not isinstance(tool_calls, list)
                    or (msg.content is not None and not isinstance(msg.content, str))
                    or not isinstance(resp.choices[0].finish_reason, str)
                    or not resp.choices[0].finish_reason):
                raise ValueError("provider response message is malformed")
            for tool_call in tool_calls:
                if (not isinstance(tool_call.id, str) or not tool_call.id
                        or not isinstance(tool_call.function.name, str)
                        or not tool_call.function.name
                        or not isinstance(tool_call.function.arguments, str)):
                    raise ValueError("provider response has a malformed tool call")
            usage = resp.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            if any(type(value) is not int or value < 0 for value in (
                prompt_tokens, completion_tokens, total_tokens
            )) or total_tokens != prompt_tokens + completion_tokens:
                raise ValueError("provider response has incomplete token accounting")
            extra = msg.model_extra or {}
            if not isinstance(extra, dict):
                raise ValueError("provider response message metadata is malformed")
            reasoning = extra.get("reasoning") or extra.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("provider response reasoning is malformed")
        except Exception as error:
            ep["status"] = "error"
            ep["error"] = f"response validation failed: {type(error).__name__}: {error}"
            break
        ep["prompt_tokens"] += prompt_tokens
        ep["completion_tokens"] += completion_tokens
        ep["total_tokens"] += total_tokens
        ep["gen_tokens"] = ep["completion_tokens"]
        turn = {
            "response_id": resp.id,
            "response_model": resp.model,
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "reasoning": reasoning,
            "content": msg.content,
            "tool_calls": [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in tool_calls
            ],
            "observations": [],
            "finish_reason": resp.choices[0].finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        ep["turns"].append(turn)

        if msg.tool_calls:
            if not allow_tools or len(msg.tool_calls) != 1:
                ep["status"] = "error"
                ep["error"] = (
                    "provider violated the one-tool-call ReAct contract: "
                    f"allow_tools={allow_tools}, tool_calls={len(msg.tool_calls)}"
                )
                break
            tool_call = msg.tool_calls[0]
            if not isinstance(tool_call.id, str) or not tool_call.id:
                ep["status"] = "error"
                ep["error"] = "provider tool call has no stable ID"
                break
            iters += 1
            ep["n_tool_iters"] = iters
            assistant = {
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
            if think_history and turn["reasoning"]:
                # interleaved thinking: the chat template renders prior turns'
                # <think> blocks within the current tool loop (empty ones
                # otherwise, which conditions the model to stop thinking)
                assistant["reasoning"] = turn["reasoning"]
            messages.append(assistant)
            obs = await asyncio.to_thread(
                tools.dispatch, tool_call.function.name, tool_call.function.arguments
            )
            turn["observations"].append(obs)
            messages.append({
                "role": "tool", "tool_call_id": tool_call.id, "content": obs,
            })
            continue

        content = (msg.content or "").strip()
        tool_shaped = "<tool_call>" in content or "<function=" in content
        early = forced and iters < max_iters
        if content and not tool_shaped and not early:
            ep["answer"] = msg.content
            break

        # Not an acceptable answer: empty (e.g. cut mid-thinking), a tool call
        # written as plain text (the model trying to explore past its budget), or
        # a forced episode answering before its 20 iterations.
        nudges += 1
        if nudges > MAX_NUDGES:
            if early and content:
                ep["answer"], ep["status"] = msg.content, "forced_short"
            elif content:
                # A serialized tool call is not an answer. Preserve it for audit,
                # but score the genuine model no-answer outcome as zero.
                ep["last_non_answer_content"] = msg.content
                ep["status"] = "no_answer"
            else:
                ep["status"] = "no_answer"
            break
        assistant = {"role": "assistant", "content": msg.content or ""}
        if think_history and turn["reasoning"]:
            assistant["reasoning"] = turn["reasoning"]
        messages.append(assistant)
        if early:
            nudge = "Please continue researching with a tool call."
        elif tool_shaped and allow_tools:
            nudge = ("Your last message contained a tool call written as plain "
                     "text, which cannot be executed. Use the tool-calling "
                     "interface, or give your final answer.")
        elif tool_shaped:
            nudge = ("You cannot make any more tool calls. Give your final, "
                     "complete answer to the original question now, based on "
                     "what you have learned.")
        else:
            nudge = "Please give your final answer now."
        messages.append({"role": "user", "content": nudge})

    ep["finished"] = utc_now()
    expected_identity = {
        **identity,
        "task": corpus.name,
        "qid": q["id"],
        "budget": budget,
        "rollout": rollout,
        "seed": seed,
    }
    _reject_invalid_final_episode(
        ep,
        expected_identity,
        expected_model=MODEL,
        expected_model_revision=MODEL_REVISION,
        expected_harness=(
            "native-react" if think_history else "native-react-no-think-history"
        ),
        expected_response_model=MODEL,
    )
    return ep


async def request_with_retry(
    client: AsyncOpenAI,
    *,
    audit: list[dict],
    logical_call: int,
    **kwargs,
):
    """Make one logical request and retain every transport attempt."""

    request_sha256 = sha256_json(kwargs)
    delay = 5.0
    for attempt in range(4):
        try:
            response = await client.chat.completions.create(**kwargs)
            audit.append({
                "logical_call": logical_call,
                "attempt": attempt + 1,
                "status": "response",
                "request_sha256": request_sha256,
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
            })
            return response
        except Exception as e:
            audit.append({
                "logical_call": logical_call,
                "attempt": attempt + 1,
                "status": "transport_error",
                "request_sha256": request_sha256,
                "error_type": type(e).__name__,
                "error": str(e)[:500],
                "usage": "unknown",
            })
            transient = "maximum context length" not in str(e).lower() and attempt < 3
            if not transient:
                raise
            log.warning("retrying after %s: %s", type(e).__name__, str(e)[:200])
            await asyncio.sleep(delay)
            delay *= 2


async def main_async(args):
    corpus = CORPORA[args.task]
    tools = RepoTools(corpus)
    all_questions = load_questions(args.task)
    if args.limit and not args.smoke:
        raise SystemExit("--limit is diagnostic only and requires --smoke")
    questions = all_questions[: args.limit or None]
    declared_servers = None
    if not (args.smoke or args.allow_dirty):
        try:
            declared_servers = int(os.environ["SB_NSERVE"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("claim-ready runs require a valid SB_NSERVE") from exc
    urls = validate_local_server_urls(args.base_urls, expected_count=declared_servers)
    api_key = os.environ.get("SB_VLLM_API_KEY")
    if not api_key:
        raise SystemExit("authenticated local server key is unavailable")
    clients = [AsyncOpenAI(base_url=u, api_key=api_key, timeout=3600, max_retries=0)
               for u in urls]
    think_history = not args.no_think_history
    episode_contract = {
        "expected_model": MODEL,
        "expected_model_revision": MODEL_REVISION,
        "expected_harness": (
            "native-react" if think_history else "native-react-no-think-history"
        ),
        "expected_response_model": MODEL,
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
        harness="native-react-no-think-history" if not think_history else "native-react",
        model=MODEL,
        model_revision=MODEL_REVISION,
        sampling={**SAMPLING, "max_tokens": MAX_TOKENS_PER_TURN},
        master_seed=args.seed,
        seed_namespace="native-react",
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
            "expected_response_model": MODEL,
            "model_context_tokens": 262_144,
            "tool_iterations": BUDGETS,
            "tool_schema_sha256": sha256_json(TOOL_SCHEMAS),
            "system_prompts": {budget: system_prompt(corpus, budget) for budget in budgets},
            "think_history": think_history,
            "server_transport": {
                "scope": "loopback",
                "protocol": "openai-compatible-http",
                "server_count": len(urls),
            },
            "concurrency": args.concurrency,
        },
    )
    # Repository tools remain available, preserving open-book evaluation.
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
                seed = episode_seed(
                    args.seed, args.seed_group, corpus.name, q["id"], budget, rollout
                )
                identity = episode_identity(
                    context, q=raw_q, prompt=q["question"], budget=budget,
                    rollout=rollout, seed=seed,
                )
                cases.append((raw_q, q, budget, rollout, out, seed, identity))
                # "ok" and "no_answer" are genuine model outcomes; only infra
                # failures ("error", "forced_short") are retried on resume.
                existing = _validated_resumable_episode(
                    out, identity, context=context, **episode_contract
                )
                if existing:
                    if existing.get("status") in ("ok", "no_answer"):
                        continue
                    raise ValueError(f"non-final artifact occupies expected episode path: {out}")
                pending.append((raw_q, q, budget, rollout, out, seed, identity))
    log.info("%d episodes pending (task=%s)", len(pending), args.task)

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(i, raw_q, q, budget, rollout, out, seed, identity):
        nonlocal done
        async with sem:
            try:
                lock = context.root / "locks" / out.relative_to(context.root).with_suffix(".lock")
                with exclusive_process_lock(lock):
                    if _validated_resumable_episode(
                        out, identity, context=context, **episode_contract
                    ) is not None:
                        return
                    ep = await run_episode(clients[i % len(clients)], corpus, tools, q,
                                           budget, rollout, think_history, seed=seed,
                                           identity=identity)
                    _reject_invalid_final_episode(
                        ep, identity, **episode_contract
                    )
                    artifact = write_episode_result(context, out, ep)
            except Exception:
                log.exception("episode %s/%s/r%d failed outside run_episode",
                              budget, q["id"], rollout)
                return
            done += 1
            log.info("[%d/%d] %s/%s/r%d %s: status=%s iters=%d gen_tokens=%d",
                     done, len(pending), budget, q["id"], rollout, ep["finished"],
                     ep["status"], ep["n_tool_iters"], ep["gen_tokens"])
            if artifact != out:
                log.warning("retained failed attempt at %s", artifact)
            if args.debug:
                log.debug("answer for %s/%s/r%d:\n%s", budget, q["id"], rollout,
                          ep["answer"][:2000])

    await asyncio.gather(*(one(i, *p) for i, p in enumerate(pending)))

    statuses: dict[str, int] = {}
    for _, _, _, _, out, _, identity in cases:
        existing = _validated_resumable_episode(
            out, identity, context=context, **episode_contract
        )
        s = existing.get("status", "missing") if existing else "missing"
        statuses[s] = statuses.get(s, 0) + 1
    log.info("all done: %s", statuses)
    if statuses.keys() - {"ok", "no_answer"}:
        log.error("some episodes failed; rerun to retry them")
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--run-id", required=True,
                   help="unique immutable run namespace (never reuse across arms)")
    p.add_argument("--seed", required=True, type=int,
                   help="master seed; deterministic per-episode seeds are recorded")
    p.add_argument("--seed-group", required=True,
                   help="pairing ID shared by baseline and treatment arms")
    p.add_argument("--budgets", default="direct,k5,k20,k20f")
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--base-urls", default="http://localhost:8100/v1")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="first N questions; only valid with --smoke")
    purpose = p.add_mutually_exclusive_group()
    purpose.add_argument("--smoke", action="store_true",
                         help="write under runs/smoke and mark artifacts non-claim-ready")
    purpose.add_argument(
        "--exploratory", action="store_true",
        help="run the full grid but explicitly mark it non-claim-ready",
    )
    p.add_argument(
        "--preregistration", type=Path,
        help="canonical committed preregistrations/*.json contract",
    )
    p.add_argument(
        "--preregistration-role", choices=("control", "treatment"),
        help="arm in --preregistration bound to this run ID and note",
    )
    p.add_argument("--note", type=Path,
                   help="exact study note to prepend; never inferred from a mutable alias")
    p.add_argument("--note-manifest", type=Path,
                   help="construction manifest whose note_sha256 matches --note")
    p.add_argument("--no-think-history", action="store_true")
    p.add_argument("--allow-dirty", action="store_true",
                   help="diagnostic only: record dirty source and mark the run non-claim-ready")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    validate_id(args.run_id)
    if args.rollouts <= 0 or args.concurrency <= 0 or args.limit < 0:
        p.error("--rollouts and --concurrency must be positive; --limit must be nonnegative")
    if args.allow_dirty and not args.smoke:
        p.error("--allow-dirty is diagnostic and requires --smoke")
    if args.smoke and args.limit <= 0:
        p.error("--smoke requires a positive --limit to bound diagnostic cost")
    if (args.preregistration is None) != (args.preregistration_role is None):
        p.error("--preregistration and --preregistration-role are required together")
    if not (args.smoke or args.exploratory or args.preregistration is not None):
        p.error("confirmatory runs require --preregistration; otherwise use --exploratory")
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(
                      ROOT / "logs" / f"rollout-{args.run_id}-{args.task}.log"
                  )],
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
