"""The CHEATSHEET study procedure (paper §B Table 4): a forced ReAct study loop of
50 no-early-return tool calls over the corpus, whose final answer is a cheatsheet.
The cheatsheet is saved to cheatsheets/{task}.md and prepended to every eval
question by `rollout --variant cheatsheet`. Study tokens are not counted on the
eval token axis (confirmed by the first author, docs/jacob.md).

The study prompts are a replication inference — the paper does not give them. They
mirror the eval prompts' structure and tell the agent only that its document will
be prepended to future questions about the library (no hints about the hidden task
distribution).
"""

import argparse
import asyncio
import json
import logging

from openai import AsyncOpenAI

from .dataset import CORPORA, ROOT
from .rollout import run_episode
from .tools import RepoTools

log = logging.getLogger("study")


def study_system(corpus) -> str:
    return (
        f"You are studying {corpus.display} to become an expert on it. "
        f"A checkout of the {corpus.display} repository is available; its code lives "
        f"under these top-level directories: {', '.join(corpus.roots)}. "
        "Use the tools (grep, glob, read_file) to explore the code. You must use "
        "tools for exactly 50 iterations to study before writing your final document."
    )


def study_task(corpus) -> str:
    return (
        f"Study the {corpus.display} repository and write yourself a cheatsheet: a "
        f"reference document that will be prepended to every future question you are "
        f"asked about {corpus.display}. You will not see the questions in advance, "
        "but you will keep access to these repository tools when answering them. "
        "Record whatever will make you fastest and most accurate later. After your "
        "50 iterations of study, write the complete cheatsheet as your final answer."
    )


async def main_async(args):
    corpus = CORPORA[args.task]
    tools = RepoTools(corpus)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY",
                         timeout=3600, max_retries=0)
    q = {"id": "cheatsheet", "question": study_task(corpus)}
    ep = await run_episode(client, corpus, tools, q, "s50", args.rollout,
                           system=study_system(corpus))
    out = ROOT / "cheatsheets"
    out.mkdir(exist_ok=True)
    (out / f"{args.task}.episode.json").write_text(json.dumps(ep, indent=2))
    log.info("study episode: status=%s iters=%d gen_tokens=%d cheatsheet_chars=%d",
             ep["status"], ep["n_tool_iters"], ep["gen_tokens"], len(ep["answer"]))
    if ep["status"] != "ok" or not ep["answer"].strip():
        raise SystemExit(f"study episode failed: status={ep['status']}")
    (out / f"{args.task}.md").write_text(ep["answer"])
    log.info("wrote cheatsheets/%s.md", args.task)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--base-url", default="http://localhost:8100/v1")
    p.add_argument("--rollout", type=int, default=0, help="seed index for the study episode")
    args = p.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "logs" / f"study-{args.task}.log")])
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
