"""s0: snapshot every stanfordnlp/dspy GitHub issue (no pull requests).

Paper Stage 1 sources "real user friction traces"; for DSPy we substitute
GitHub issues (the paper's OpenClaw recipe) because its community sessions
are not public. Collects ALL issue states so the raw snapshot is complete;
state filtering happens explicitly in s1.

Usage: uv run --frozen python data_collection/s0_collect_issues.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

from common import ARTIFACTS, GITHUB_OWNER, GITHUB_REPO, load_env, write_json, write_jsonl

QUERY = """
query($cursor: String) {
  repository(owner: "%s", name: "%s") {
    issues(first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body state createdAt closedAt url
        author { login __typename }
        labels(first: 10) { nodes { name } }
        comments(first: 20) {
          totalCount
          nodes { body createdAt author { login __typename } }
        }
      }
    }
  }
}
""" % (GITHUB_OWNER, GITHUB_REPO)


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


def main() -> None:
    load_env()
    token = os.environ["GITHUB_PAT"]
    rows, cursor = [], None
    while True:
        data = graphql(token, {"cursor": cursor})
        page = data["repository"]["issues"]
        for node in page["nodes"]:
            author = node.pop("author") or {}
            node["author"] = author.get("login")
            node["author_type"] = author.get("__typename")
            node["labels"] = [label["name"] for label in node.pop("labels")["nodes"]]
            comments = node.pop("comments")
            node["comment_count"] = comments["totalCount"]
            node["comments"] = [
                {
                    "author": (comment["author"] or {}).get("login"),
                    "author_type": (comment["author"] or {}).get("__typename"),
                    "created_at": comment["createdAt"],
                    "body": comment["body"],
                }
                for comment in comments["nodes"]
            ]
            rows.append(node)
        print(f"collected {len(rows)} issues", flush=True)
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]

    numbers = [row["number"] for row in rows]
    if len(numbers) != len(set(numbers)):
        raise RuntimeError("duplicate issue numbers in snapshot")
    write_jsonl(ARTIFACTS / "issues_raw.jsonl", rows)
    write_json(
        ARTIFACTS / "collect_manifest.json",
        {
            "repository": f"{GITHUB_OWNER}/{GITHUB_REPO}",
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "issue_count": len(rows),
            "states": {
                state: sum(row["state"] == state for row in rows)
                for state in {"OPEN", "CLOSED"}
            },
            "comment_cap_per_issue": 20,
            "query": QUERY,
        },
    )
    print(f"wrote {len(rows)} issues to {ARTIFACTS / 'issues_raw.jsonl'}")


if __name__ == "__main__":
    main()
