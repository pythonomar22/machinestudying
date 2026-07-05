"""Generation: ReAct rollouts of Qwen3.5-9B over StudyBench, one JSON per episode.

Replicates the paper's four inference budgets: direct answer (no tools), up to 5 or 20
voluntary tool iterations, and a forced 20 iterations with no early stopping. Sampling
parameters are the paper's (= Qwen3.5 thinking-mode defaults). Grading happens offline
(grade.py); this module only produces runs/{task}/{budget}/r{rollout}/{qid}.json.
"""

import argparse
import asyncio
import json
import logging
import time
import zlib
from pathlib import Path

from openai import AsyncOpenAI

from .dataset import CORPORA, ROOT, load_questions
from .tools import TOOL_SCHEMAS, RepoTools

MODEL = "Qwen/Qwen3.5-9B"
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


def episode_seed(task: str, qid: str, budget: str, rollout: int) -> int:
    return zlib.crc32(f"{task}/{qid}/{budget}/{rollout}".encode()) & 0x7FFFFFFF


async def run_episode(client: AsyncOpenAI, corpus, tools: RepoTools, q: dict,
                      budget: str, rollout: int) -> dict:
    max_iters, forced = BUDGETS[budget]
    seed = episode_seed(corpus.name, q["id"], budget, rollout)
    messages = [
        {"role": "system", "content": system_prompt(corpus, budget)},
        {"role": "user", "content": q["question"]},
    ]
    ep = {
        "task": corpus.name, "qid": q["id"], "budget": budget, "rollout": rollout,
        "model": MODEL, "seed": seed, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "turns": [], "answer": "", "n_tool_iters": 0, "gen_tokens": 0, "status": "ok",
    }
    iters = nudges = 0
    while True:
        allow_tools = iters < max_iters
        kwargs = {}
        if allow_tools:
            kwargs["tools"] = TOOL_SCHEMAS
            kwargs["tool_choice"] = "required" if forced else "auto"
            # one action per ReAct iteration (paper: "up to 5 or 20 tool iterations")
            kwargs["parallel_tool_calls"] = False
        try:
            resp = await request_with_retry(
                client, messages=messages, model=MODEL, seed=seed,
                max_tokens=MAX_TOKENS_PER_TURN, **SAMPLING, **kwargs)
        except Exception as e:
            ep["status"], ep["error"] = "error", f"{type(e).__name__}: {e}"
            break
        msg = resp.choices[0].message
        usage = resp.usage
        ep["gen_tokens"] += usage.completion_tokens
        turn = {
            "reasoning": (msg.model_extra or {}).get("reasoning_content"),
            "content": msg.content,
            "tool_calls": [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
            "finish_reason": resp.choices[0].finish_reason,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
        ep["turns"].append(turn)

        if msg.tool_calls:
            iters += 1
            ep["n_tool_iters"] = iters
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            turn["observations"] = []
            for tc in msg.tool_calls:
                obs = await asyncio.to_thread(tools.dispatch, tc.function.name, tc.function.arguments)
                turn["observations"].append(obs)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
            continue

        answered = bool(msg.content and msg.content.strip())
        if answered and not (forced and iters < max_iters):
            ep["answer"] = msg.content
            break

        # Either an empty turn (e.g. cut mid-thinking), or a forced episode trying to
        # answer before its 20 iterations (tool_choice="required" should prevent the
        # latter; this is a guard against tool-parser failures).
        nudges += 1
        if nudges > MAX_NUDGES:
            if answered:
                ep["answer"], ep["status"] = msg.content, "forced_short"
            else:
                ep["status"] = "no_answer"
            break
        messages.append({"role": "assistant", "content": msg.content or ""})
        messages.append({"role": "user", "content":
                         "Please continue researching with a tool call."
                         if forced and iters < max_iters
                         else "Please give your final answer now."})

    ep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return ep


async def request_with_retry(client: AsyncOpenAI, **kwargs):
    delay = 5.0
    for attempt in range(4):
        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as e:
            transient = "maximum context length" not in str(e).lower() and attempt < 3
            if not transient:
                raise
            log.warning("retrying after %s: %s", type(e).__name__, str(e)[:200])
            await asyncio.sleep(delay)
            delay *= 2


async def main_async(args):
    corpus = CORPORA[args.task]
    tools = RepoTools(corpus)
    questions = load_questions(args.task)[: args.limit or None]
    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=3600, max_retries=0)
               for u in args.base_urls.split(",")]

    pending = []
    for budget in args.budgets.split(","):
        assert budget in BUDGETS, f"unknown budget {budget}"
        for rollout in range(args.rollouts):
            for q in questions:
                out = ROOT / "runs" / args.task / budget / f"r{rollout}" / f"{q['id']}.json"
                # "ok" and "no_answer" are genuine model outcomes; only infra
                # failures ("error", "forced_short") are retried on resume.
                if out.exists() and json.loads(out.read_text()).get("status") in ("ok", "no_answer"):
                    continue
                pending.append((q, budget, rollout, out))
    log.info("%d episodes pending (task=%s)", len(pending), args.task)

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(i, q, budget, rollout, out):
        nonlocal done
        async with sem:
            try:
                ep = await run_episode(clients[i % len(clients)], corpus, tools, q, budget, rollout)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                tmp.write_text(json.dumps(ep, indent=2))
                tmp.rename(out)
            except Exception:
                log.exception("episode %s/%s/r%d failed outside run_episode",
                              budget, q["id"], rollout)
                return
            done += 1
            log.info("[%d/%d] %s/%s/r%d %s: status=%s iters=%d gen_tokens=%d",
                     done, len(pending), budget, q["id"], rollout, ep["finished"],
                     ep["status"], ep["n_tool_iters"], ep["gen_tokens"])
            if args.debug:
                log.debug("answer for %s/%s/r%d:\n%s", budget, q["id"], rollout,
                          ep["answer"][:2000])

    await asyncio.gather(*(one(i, *p) for i, p in enumerate(pending)))

    statuses = {}
    for _, budget, rollout, out in pending:
        s = json.loads(out.read_text())["status"] if out.exists() else "missing"
        statuses[s] = statuses.get(s, 0) + 1
    log.info("all done: %s", statuses)
    if statuses.keys() - {"ok", "no_answer"}:
        log.error("some episodes failed; rerun to retry them")
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--budgets", default="direct,k5,k20,k20f")
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--base-urls", default="http://localhost:8100/v1")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke tests)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"rollout-{args.task}.log")],
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
