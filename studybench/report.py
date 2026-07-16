"""Print the baseline and cheatsheet Table-1 rows and their expertise scores."""

from __future__ import annotations

import argparse
from statistics import fmean

from .artifacts import read_json
from .dataset import CORPORA, ROOT, load_questions
from .grade import (
    BUDGETS,
    artifact_path,
    load_run,
    validate_grade,
    validate_grade_manifest,
    weighted_auc,
)

PAPER = {
    ("dspy", "baseline"): ([3.3, 8.6, 9.6, 29.4], [4.1, 7.9, 8.6, 34.6], 6.49),
    ("dspy", "cheatsheet"): ([6.3, 14.4, 14.1, 23.1], [3.9, 6.1, 7.1, 29.9], 9.65),
    ("openclaw", "baseline"): ([2.3, 6.9, 15.8, 17.6], [4.1, 4.6, 9.7, 24.3], 7.64),
    ("openclaw", "cheatsheet"): ([4.3, 8.6, 15.2, 18.1], [3.8, 6.0, 9.1, 20.1], 8.18),
}
COMPATIBLE = (
    "smoke",
    "debug",
    "source_commit",
    "corpus_commit",
    "dataset_sha256",
    "model",
    "model_revision",
    "harness",
    "runtime",
    "sampling",
    "tools",
    "budgets",
    "rollouts",
    "master_seed",
    "question_ids",
)


def load_grades(run_id: str, task: str, grade_id: str):
    manifest, rows, episodes = load_run(run_id, task)
    root = ROOT / "grades" / run_id / grade_id / task
    grade_manifest = read_json(root / "grade.json")
    try:
        validate_grade_manifest(manifest, grade_manifest, grade_id)
    except ValueError as error:
        raise ValueError(
            f"grade manifest does not match run: {run_id}/{task}"
        ) from error
    expected = {artifact_path(root, *key) for key in episodes}
    actual = set((root / "episodes").rglob("*.json")) if (root / "episodes").exists() else set()
    if actual != expected:
        raise ValueError(
            f"grade population is incomplete: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected"
        )
    grades = {}
    for key, episode in episodes.items():
        grade = read_json(artifact_path(root, *key))
        validate_grade(rows[key[2]], episode, grade, grade_manifest)
        grades[key] = grade
    return manifest, episodes, grades, grade_manifest


def aggregate(grades: dict) -> tuple[list[tuple[float, float]], float]:
    points = []
    for budget in BUDGETS:
        population = [grade for key, grade in grades.items() if key[0] == budget]
        points.append(
            (
                fmean(grade["gen_tokens"] for grade in population),
                fmean(grade["lenient"] for grade in population),
            )
        )
    return points, weighted_auc(points)


def check_pair(base, cheat, base_episodes, cheat_episodes) -> None:
    if base["condition"] != "baseline" or cheat["condition"] != "cheatsheet":
        raise ValueError("--baseline-run and --cheatsheet-run have the wrong conditions")
    different = [field for field in COMPATIBLE if base[field] != cheat[field]]
    if different:
        raise ValueError("paired runs differ in: " + ", ".join(different))
    if set(base_episodes) != set(cheat_episodes):
        raise ValueError("paired runs have different episode grids")
    for key in base_episodes:
        if base_episodes[key]["seed"] != cheat_episodes[key]["seed"]:
            raise ValueError(f"paired seeds differ at {key}")


def print_row(label: str, points, wauc: float) -> None:
    cells = "  ".join(f"{accuracy:5.1f}%/{tokens / 1000:5.1f}k" for tokens, accuracy in points)
    print(f"{label:22}  {cells}  {wauc:6.2f}")


def report(base_id: str, cheat_id: str, grade_id: str) -> None:
    for task in ("dspy", "openclaw"):
        corpus = CORPORA[task]
        base, base_episodes, base_grades, base_grade_manifest = load_grades(
            base_id, task, grade_id
        )
        cheat, cheat_episodes, cheat_grades, cheat_grade_manifest = load_grades(
            cheat_id, task, grade_id
        )
        if (
            base_grade_manifest["judge"] != cheat_grade_manifest["judge"]
            or base_grade_manifest["judge"]["provider"] != "gpt"
            or base_grade_manifest["judge"]["tier"] != "paper"
            or base_grade_manifest["judge"]["contract"] != "paper"
        ):
            raise ValueError(
                "paper comparison requires matching GPT-5.4 grade configurations"
            )
        if (
            base["smoke"] is not False
            or base["rollouts"] != 3
            or base["budgets"] != list(BUDGETS)
            or base["question_ids"] != [row["id"] for row in load_questions(task)]
        ):
            raise ValueError(f"{task} is not the full paper StudyBench population")
        check_pair(base, cheat, base_episodes, cheat_episodes)
        base_points, base_wauc = aggregate(base_grades)
        cheat_points, cheat_wauc = aggregate(cheat_grades)
        print(f"\n{corpus.display}: lenient accuracy / generated tokens")
        print(f"{'condition':22}  {'direct':>13}  {'k5':>13}  {'k20':>13}  {'k20f':>13}    WAUC")
        print_row(f"{base_id} baseline", base_points, base_wauc)
        base_acc, base_tok, target = PAPER[(task, "baseline")]
        print_row("paper baseline", list(zip([value * 1000 for value in base_tok], base_acc)), target)
        print_row(f"{cheat_id} cheatsheet", cheat_points, cheat_wauc)
        cheat_acc, cheat_tok, target = PAPER[(task, "cheatsheet")]
        print_row("paper cheatsheet", list(zip([value * 1000 for value in cheat_tok], cheat_acc)), target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--cheatsheet-run", required=True)
    parser.add_argument("--grade-id", default="gpt-5-4")
    args = parser.parse_args()
    try:
        report(args.baseline_run, args.cheatsheet_run, args.grade_id)
    except (OSError, ValueError) as error:
        raise SystemExit(f"report error: {error}") from error


if __name__ == "__main__":
    main()
