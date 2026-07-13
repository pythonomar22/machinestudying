from __future__ import annotations

from types import SimpleNamespace
import unittest

from studybench.rollout import MODEL, MODEL_REVISION, _validate_final_episode, run_episode


def tool_call(identifier: str, name: str = "read_file") -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments='{"path":"pkg/a.py"}')
    payload = {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": function.arguments},
    }
    return SimpleNamespace(
        id=identifier,
        function=function,
        model_dump=lambda: payload,
    )


def response(tool_calls: list[SimpleNamespace], *, content=None,
             model=MODEL, total_tokens=13) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_extra={},
    )
    return SimpleNamespace(
        id="response-1",
        model=model,
        system_fingerprint="fingerprint",
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=3, total_tokens=total_tokens),
    )


class FakeCompletions:
    def __init__(self, value: SimpleNamespace):
        self.value = value

    async def create(self, **_kwargs):
        return self.value


class FakeTools:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def dispatch(self, name: str, arguments: str) -> str:
        self.calls.append((name, arguments))
        return "observation"


class RolloutIntegrityTests(unittest.IsolatedAsyncioTestCase):
    async def run_with(self, budget: str, calls: list[SimpleNamespace]):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions(response(calls)))
        )
        tools = FakeTools()
        corpus = SimpleNamespace(name="dspy", display="DSPy", roots=("dspy",))
        episode = await run_episode(
            client,
            corpus,
            tools,
            {"id": "q1", "question": "Question?"},
            budget,
            0,
            seed=7,
            identity={"manifest_sha256": "m", "question_sha256": "q"},
        )
        return episode, tools

    async def test_final_answer_passes_full_producer_validation(self) -> None:
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=FakeCompletions(response([], content="answer"))))
        corpus = SimpleNamespace(name="dspy", display="DSPy", roots=("dspy",))
        identity = {"manifest_sha256": "m", "question_sha256": "q"}
        episode = await run_episode(
            client, corpus, FakeTools(), {"id": "q1", "question": "Question?"},
            "direct", 0, seed=7, identity=identity,
        )
        self.assertEqual(episode["status"], "ok")
        _validate_final_episode(
            episode,
            {**identity, "task": "dspy", "qid": "q1", "budget": "direct",
             "rollout": 0, "seed": 7},
            expected_model=MODEL,
            expected_model_revision=MODEL_REVISION,
            expected_harness="native-react",
            expected_response_model=MODEL,
        )

    async def test_malformed_usage_or_provider_model_is_a_failure(self) -> None:
        corpus = SimpleNamespace(name="dspy", display="DSPy", roots=("dspy",))
        for value in (
            response([], content="answer", total_tokens=99),
            response([], content="answer", model="wrong-model"),
        ):
            with self.subTest(value=value):
                client = SimpleNamespace(chat=SimpleNamespace(
                    completions=FakeCompletions(value)))
                episode = await run_episode(
                    client, corpus, FakeTools(),
                    {"id": "q1", "question": "Question?"}, "direct", 0,
                    seed=7, identity={"manifest_sha256": "m"},
                )
                self.assertEqual(episode["status"], "error")

    async def test_multiple_parallel_calls_are_preserved_but_never_executed(self) -> None:
        episode, tools = await self.run_with(
            "k5", [tool_call("call-1"), tool_call("call-2")]
        )
        self.assertEqual(episode["status"], "error")
        self.assertIn("one-tool-call", episode["error"])
        self.assertEqual(episode["n_tool_iters"], 0)
        self.assertEqual(len(episode["turns"][0]["tool_calls"]), 2)
        self.assertEqual(tools.calls, [])

    async def test_tool_call_outside_the_declared_budget_is_never_executed(self) -> None:
        episode, tools = await self.run_with("direct", [tool_call("call-1")])
        self.assertEqual(episode["status"], "error")
        self.assertIn("allow_tools=False", episode["error"])
        self.assertEqual(episode["n_tool_iters"], 0)
        self.assertEqual(tools.calls, [])


if __name__ == "__main__":
    unittest.main()
