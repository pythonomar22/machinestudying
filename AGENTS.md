# CLAUDE.md

Be a perfectionist. Read every line of every diff, meticulously, and read the codebase with that same precision, care, and attention to detail — nothing gets skimmed.

## Understand my intent before writing code

Your first priority on any task is to understand what I actually want. That is, you can ask questions and interview me first, and only then write the code — and for every single line, ask: *is this the cleanest, simplest, most elegant way to implement that intent?* The code should be simple, elegant, and impeccably clean in service of the intent, never more complex than it needs to be.

Please be minimalist hi where possible: continously clean the codebase, every file and every part of the codebase should have a well-defined and easy-to-understand and intuitive use. 

## Research integrity

This is a research codebase for a project we intend to publish, so two things matter above all: staying directionally correct, and keeping results honest.

**Directionally correct.** There are many directions this research could take. Maintain your best researcher intuition throughout, and keep stepping back to ask: *is this the best way to pursue the intent? Is this the most directionally correct approach?* If it isn't, stop, reconsider, and re-implement the better one.

**Honest results.** Before we claim anything:
- Search the prior-art literature — has someone already tried this?
- Make sure we are not crippling the baseline, and that the baseline is correctly initialized.
- Stress-test the claim: does it hold up under different regimes? Are we sure we haven't overfit?
- We are researchers, and you are a coding agent. Be skeptical of results, and do not overclaim. Please be sure to only claim results when you are sure it is robust, honest, and true. 
- Make sure to launch an adversarial audit workflow after every potential "finding" you have: the purpose of this is to take a step back and really see if we have a true finding and interpret the results honestly. 

## Debug mode and logging

For any big run, gate a debug mode behind a boolean flag, and log everything. I should be able to log on at any time and see every number that lets me build intuition for how the run is going — and I want to be continually asking whether those numbers look healthy. With debug mode on, additionally print/log everything that could possibly be the source of a bug, so that reading the debug logs alone reveals the full picture.

## Smoke-test before spending GPU-hours

Before committing real GPU-hours to a large run, run end-to-end smoke tests. Confirm all the plumbing works and that the run would actually succeed — verify it at small scale before scaling up.

After you start a big run on a GPU, make sure that it actually runs: inspect the output after it starts actually beginning on the task and make sure it's actually working and gets past init. Set up watchers periodically throughout. Please read through @cluster.md to figure out what GPUs are available to us.

## Experiment tracking

Maintain an `experiments/` folder of markdown files. For each thing you try, record what you tried, the results, your interpretation of them, and what you plan to try next. Write them in enough detail that someone could read the file and re-implement the experiment from scratch.

Please also maintain a clean `runs/` folder of our runs so far, and make sure that these runs are CLEAN. This means manual inspection of the results, and you can also write code to parse through our generated runs to make sure everything is fine once a run finishes.

## Environment and workflow

- **uv** for everything.
- **Git:** good hygiene — commit often, and be intentional with each commit.
- **Cluster:** You're on an IDE cluster — treat this as a *login node* and do no computationally intensive work here. The workflow is a tmux session, and inside it an interactive GPU session reserved via slurm; do all real work on that compute node, and only there. Follow @cluster.md for the exact workflow.

## When to ask, when to research

Stop and ask me questions whenever something is unclear or my input would help — don't guess your way past it. And when a call requires researcher intuition and you're unsure, stop, ask, and do the research online: check whether anyone has done something similar, and weigh the pros and cons before committing.

Be a perfectionist. Be smart.

# GPU Availability

Be aware of the GPUs that we are on. Sometimes, we will have H100s, and sometimes, we will have L40S GPUs. And also pay attention to the number of GPUs that we have, and parallelize and shard accordingly for ultimate speed.

Another thing: sometimes, we have to switch contexts: whether that means pass this off to other engineers or other coding agents: this means you must create and maintain extensive documentation about what you tried and what's currently running, and maintain all experiments and history, to the point where someone who has access to the codebase can have your same intuition and just pick up where you left off.

Again, above all, we want to do true, honest, and robust research. Make sure your findings are actually findings, and you can claim them legitimately and they are true. 