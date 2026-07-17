"""Snapshot every stanfordnlp/dspy GitHub issue as a user-question session.

The paper's Stage 1 "begins with a snapshot of real user-question sessions
for each library"; their DSPy sessions are private community conversations,
so ours are the public GitHub issues (the paper's own source for OpenClaw).
One session = the issue author's opening question (title + body) plus the
conversation that followed (comments). Every issue is captured regardless
of state or date - Stage 1 filtering happens downstream, explicitly.

Usage:
    uv run --frozen python data_collection/1_scrape_sessions.py

Requires GITHUB_PAT in .env. Writes artifacts/sessions.json: one JSON
object with the collection manifest and the full list of sessions.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

OWNER, REPO = "stanfordnlp", "dspy"
PAGE_SIZE, COMMENT_CAP = 50, 20
DC = Path(__file__).resolve().parent
ROOT = DC.parent
OUTPUT = DC / "artifacts" / "1_scrape_sessions" / "1_sessions.json"

QUERY = """
query($cursor: String) {
  repository(owner: "%s", name: "%s") {
    issues(first: %d, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body state createdAt closedAt url
        author { login __typename }
        labels(first: 10) { nodes { name } }
        comments(first: %d) {
          totalCount
          nodes { body createdAt author { login __typename } }
        }
      }
    }
  }
}
""" % (OWNER, REPO, PAGE_SIZE, COMMENT_CAP)


def github_token() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition("=")
        if key == "GITHUB_PAT" and value:
            return value.strip().strip("'\"")
    return os.environ["GITHUB_PAT"]


def graphql(token: str, variables: dict) -> dict:
    request = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": QUERY, "variables": variables}).encode(),
        headers={"Authorization": f"bearer {token}"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
            if "errors" in payload:
                raise RuntimeError(f"GraphQL errors: {payload['errors']}")
            return payload["data"]
        except (OSError, RuntimeError) as error:
            if attempt == 4:
                raise
            print(f"retrying after error: {error}")
            time.sleep(2 ** (attempt + 1))
    raise AssertionError


def session(node: dict) -> dict:
    author = node["author"] or {}
    return {
        "number": node["number"],
        "title": node["title"],
        "body": node["body"],
        "state": node["state"],
        "created_at": node["createdAt"],
        "closed_at": node["closedAt"],
        "url": node["url"],
        "labels": [label["name"] for label in node["labels"]["nodes"]],
        "author": author.get("login"),
        "author_type": author.get("__typename"),
        "comment_count": node["comments"]["totalCount"],
        "comments": [
            {
                "author": (comment["author"] or {}).get("login"),
                "author_type": (comment["author"] or {}).get("__typename"),
                "created_at": comment["createdAt"],
                "body": comment["body"],
            }
            for comment in node["comments"]["nodes"]
        ],
    }


def main() -> None:
    token = github_token()
    sessions, cursor = [], None
    while True:
        page = graphql(token, {"cursor": cursor})["repository"]["issues"]
        sessions.extend(session(node) for node in page["nodes"])
        print(f"collected {len(sessions)} sessions", flush=True)
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]

    numbers = [row["number"] for row in sessions]
    if len(numbers) != len(set(numbers)):
        raise RuntimeError("duplicate issue numbers in snapshot")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "repository": f"{OWNER}/{REPO}",
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session_count": len(sessions),
                "states": {
                    state: sum(row["state"] == state for row in sessions)
                    for state in sorted({row["state"] for row in sessions})
                },
                "comment_cap_per_session": COMMENT_CAP,
                "sessions": sessions,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(sessions)} sessions to {OUTPUT}")


if __name__ == "__main__":
    main()
