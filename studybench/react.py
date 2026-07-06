"""The paper's harness, faithfully: dspy.ReAct over the pinned DSPy checkout.

Author-confirmed details (docs/jacob.md):
- harness = dspy.ReAct ("I was using dspy.ReAct"); model re-thinks fresh every
  step (each react step is a stateless LM call);
- forced-20 / forced-50: "Just catch the finish and return something like you
  gotta keep searching type of logic, no need to remove that specific turn";
- direct = dspy.Predict;
- tools = grep, glob, read_file (line ranges, capped at 200 lines);
- lenient = pure weighted claim sum (grading side).

Run inside .venv-dspy (the pinned corpora/dspy install). Episodes land in
runs/react/{task}/{budget}/r{rollout}/{qid}.json with the same schema as the
native harness, so grade.py/report.py work unchanged (--variant react).
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import dspy
from dspy.predict.react import _fmt_exc

from .dataset import CORPORA, ROOT, load_questions
from .rollout import BUDGETS
from .tools import RepoTools

MODEL_ID = "openai/Qwen/Qwen3.5-9B"  # openai-compatible provider -> our vLLM
READ_MAX_LINES = 200  # author: "read file (lines, Capped at 200lines)"
SAMPLING = dict(  # paper §B; passed through litellm to vLLM
    temperature=1.0, top_p=0.95, max_tokens=32768, presence_penalty=1.5,
    extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
)

log = logging.getLogger("react")


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
                log.warning("ending forced trajectory early: %s", _fmt_exc(err))
                break
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
        extract = self._call_with_potential_trajectory_truncation(
            self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)


def run_episode(corpus, tools_fns, q: dict, budget: str, rollout: int,
                base_url: str) -> dict:
    max_iters, forced = BUDGETS[budget]
    lm = dspy.LM(MODEL_ID, api_base=base_url, api_key="EMPTY", model_type="chat",
                 cache=False, num_retries=3, **SAMPLING)
    ep = {
        "task": corpus.name, "qid": q["id"], "budget": budget, "rollout": rollout,
        "model": MODEL_ID, "harness": "dspy.ReAct",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "turns": [], "answer": "", "n_tool_iters": 0, "finish_catches": 0,
        "gen_tokens": 0, "status": "ok",
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
            "observations": [str(trajectory.get(f"observation_{i}"))[:2000]],
        })
        if name == "finish":
            ep["finish_catches"] += 1
        else:
            ep["n_tool_iters"] += 1
    ep["n_lm_calls"] = len(lm.history)
    ep["gen_tokens"] = sum((h.get("usage") or {}).get("completion_tokens") or 0
                           for h in lm.history)
    ep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return ep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--budgets", default="direct,k5,k20,k20f")
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--base-urls", default="http://localhost:8100/v1")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"react-{args.task}.log")])
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    corpus = CORPORA[args.task]
    tools_fns = make_tools(RepoTools(corpus, read_max_lines=READ_MAX_LINES))
    questions = load_questions(args.task)[: args.limit or None]
    urls = args.base_urls.split(",")

    pending = []
    for budget in args.budgets.split(","):
        assert budget in BUDGETS and budget != "s50"
        for rollout in range(args.rollouts):
            for q in questions:
                out = ROOT / "runs" / "react" / args.task / budget / f"r{rollout}" / f"{q['id']}.json"
                if out.exists() and json.loads(out.read_text()).get("status") in ("ok", "no_answer"):
                    continue
                pending.append((q, budget, rollout, out))
    log.info("%d episodes pending (task=%s, harness=dspy.ReAct)", len(pending), args.task)

    done = 0

    def one(i, q, budget, rollout, out):
        nonlocal done
        try:
            ep = run_episode(corpus, tools_fns, q, budget, rollout, urls[i % len(urls)])
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(ep, indent=2))
            tmp.rename(out)
        except Exception:
            log.exception("episode %s/%s/r%d failed", budget, q["id"], rollout)
            return
        done += 1
        log.info("[%d/%d] %s/%s/r%d: status=%s iters=%d catches=%d calls=%d gen_tokens=%d",
                 done, len(pending), budget, q["id"], rollout, ep["status"],
                 ep["n_tool_iters"], ep["finish_catches"], ep["n_lm_calls"],
                 ep["gen_tokens"])

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(lambda t: one(*t), [(i, *p) for i, p in enumerate(pending)]))

    statuses = {}
    for _, budget, rollout, out in pending:
        s = json.loads(out.read_text())["status"] if out.exists() else "missing"
        statuses[s] = statuses.get(s, 0) + 1
    log.info("all done: %s", statuses)
    if statuses.keys() - {"ok", "no_answer"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
