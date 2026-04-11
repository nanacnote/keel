"""
Chaos Tests — verifying the three-level self-correction state machine.

These tests mock the LLM to return intentionally broken output and verify
that the Engine's retry loop catches errors, injects the right correction
prompts, and either self-heals or produces a structured FailureReport.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from keel.core.dispatcher import Dispatcher
from keel.core.engine import Engine, FailureReport
from keel.core.middleware import MiddlewareHook
from keel.core.registry import BaseTool, ToolRegistry
from tests.conftest import MockLLM


# ---------------------------------------------------------------------------
# Re-usable helpers
# ---------------------------------------------------------------------------


def _make_engine(
    responses: list[str],
    max_retries: int = 3,
    middleware: list[MiddlewareHook] | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[Engine, MockLLM]:
    if registry is None:
        registry = _build_registry()
    llm = MockLLM(responses)
    dispatcher = Dispatcher(registry)
    engine = Engine(
        llm=llm,
        registry=registry,
        dispatcher=dispatcher,
        max_retries=max_retries,
        middleware=middleware or [],
    )
    return engine, llm


def _build_registry() -> ToolRegistry:
    class EchoTool(BaseTool):
        """Echoes back the provided message."""

        tool_name = "Echo"
        namespace = "test"
        message: str = Field(description="The message to echo.")

        def execute(self) -> Any:
            return self.message

    class AddTool(BaseTool):
        """Adds two integers."""

        tool_name = "Add"
        namespace = "math"
        a: int = Field(description="First operand.")
        b: int = Field(description="Second operand.")

        def execute(self) -> Any:
            return self.a + self.b

    reg = ToolRegistry()
    reg.register(EchoTool)
    reg.register(AddTool)
    return reg


GOOD_ECHO = '{"tool": "Echo", "message": "hello"}'
GOOD_ADD = '{"tool": "Add", "a": 3, "b": 4}'


# ---------------------------------------------------------------------------
# Chaos 1: LLM always returns plain text → FailureReport
# ---------------------------------------------------------------------------


class TestChaos1AlwaysSyntaxError:
    def test_returns_failure_report_after_max_retries(self):
        bad_responses = ["I cannot do that.", "Sorry, no JSON.", "Still no JSON."]
        engine, llm = _make_engine(bad_responses, max_retries=3)

        result = engine.run("say hello")

        assert isinstance(result, FailureReport)
        assert result.attempts == 3
        assert all(lvl == "SYNTAX" for lvl in result.error_levels)

    def test_failure_report_contains_last_raw_output(self):
        bad_responses = ["bad1", "bad2", "bad3"]
        engine, _ = _make_engine(bad_responses, max_retries=3)
        result = engine.run("say hello")

        assert isinstance(result, FailureReport)
        assert result.last_raw_output == "bad3"

    def test_llm_called_exactly_max_retries_times(self):
        bad_responses = ["bad"] * 5
        engine, llm = _make_engine(bad_responses, max_retries=4)
        engine.run("do something")
        assert llm.call_count == 4


# ---------------------------------------------------------------------------
# Chaos 2: Schema error twice, then correct on 3rd → returns valid tool
# ---------------------------------------------------------------------------


class TestChaos2SelfHealsOnThirdAttempt:
    def test_returns_tool_not_failure_report(self):
        responses = [
            '{"tool": "Add", "a": "not-a-number", "b": 2}',  # schema error
            '{"tool": "Add", "a": "still-bad", "b": 2}',  # schema error
            GOOD_ADD,  # correct
        ]
        engine, llm = _make_engine(responses, max_retries=3)
        result = engine.run("add numbers")

        assert isinstance(result, BaseTool)
        assert result.tool_name == "Add"
        assert llm.call_count == 3

    def test_error_levels_reflect_correction_history(self):
        responses = [
            "I'll try: not json",  # syntax
            '{"tool": "Add", "a": "x", "b": 1}',  # schema
            GOOD_ADD,  # success
        ]
        engine, llm = _make_engine(responses, max_retries=3)
        # The engine succeeds on the third attempt, so no FailureReport — but
        # we can inspect via a side-channel (mock) if we want. Here we verify
        # success itself.
        result = engine.run("add numbers")
        assert isinstance(result, BaseTool)
        assert llm.call_count == 3


# ---------------------------------------------------------------------------
# Chaos 3: LLM hallucinates tool name → correction message lists available tools
# ---------------------------------------------------------------------------


class TestChaos3HallucinatedToolName:
    def test_correction_message_contains_available_tools(self):
        responses = [
            '{"tool": "NonExistentTool", "x": 1}',  # hallucinated
            GOOD_ECHO,  # correct
        ]
        engine, llm = _make_engine(responses, max_retries=3)
        result = engine.run("do something")

        assert isinstance(result, BaseTool)
        # The second message sent to LLM should contain available tool names.
        second_call_messages = llm.received_messages[1]
        full_text = " ".join(m["content"] for m in second_call_messages)
        assert "Echo" in full_text
        assert "Add" in full_text

    def test_failure_report_records_unknown_tool_level(self):
        responses = ["{'tool': 'Ghost'}"] * 3  # always hallucinated, never self-heals
        # Make it always a hallucination — we need valid json pointing to ghost
        responses = ['{"tool": "Ghost"}'] * 3
        engine, _ = _make_engine(responses, max_retries=3)
        result = engine.run("do something")
        assert isinstance(result, FailureReport)
        assert "UNKNOWN_TOOL" in result.error_levels


# ---------------------------------------------------------------------------
# Chaos 4: Schema error on specific field → correction references that field
# ---------------------------------------------------------------------------


class TestChaos4FieldLevelCriticism:
    def test_correction_message_names_the_bad_field(self):
        responses = [
            '{"tool": "Add", "a": "bad", "b": 2}',  # 'a' is wrong type
            GOOD_ADD,  # self-healed
        ]
        engine, llm = _make_engine(responses, max_retries=3)
        result = engine.run("add numbers")

        assert isinstance(result, BaseTool)
        # Second LLM call should include a message criticising field 'a'.
        second_call_messages = llm.received_messages[1]
        full_text = " ".join(m["content"] for m in second_call_messages)
        assert "`a`" in full_text or "field `a`" in full_text or "'a'" in full_text

    def test_failure_report_errors_contain_field_loc(self):
        responses = ['{"tool": "Add", "a": "bad", "b": 2}'] * 3  # always schema error
        engine, _ = _make_engine(responses, max_retries=3)
        result = engine.run("add")
        assert isinstance(result, FailureReport)
        # All accumulated errors should mention 'a'
        all_locs = [
            str(e.get("loc", ""))
            for e in result.errors
            if isinstance(e.get("loc"), (list, str))
        ]
        assert any("a" in str(loc) for loc in all_locs)


# ---------------------------------------------------------------------------
# Chaos 5: max_retries=1 → FailureReport after single attempt
# ---------------------------------------------------------------------------


class TestChaos5SingleRetry:
    def test_failure_report_after_one_attempt(self):
        engine, llm = _make_engine(["no json here"], max_retries=1)
        result = engine.run("say hello")

        assert isinstance(result, FailureReport)
        assert result.attempts == 1
        assert llm.call_count == 1

    def test_succeeds_on_first_attempt_with_max_retries_1(self):
        engine, llm = _make_engine([GOOD_ECHO], max_retries=1)
        result = engine.run("echo hello")

        assert isinstance(result, BaseTool)
        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# Chaos 6: Middleware fires on success, silent on failure
# ---------------------------------------------------------------------------


class _RecordingHook:
    """Middleware hook that records what it was called with."""

    def __init__(self):
        self.pre_calls: list[BaseTool] = []
        self.post_calls: list[tuple[BaseTool, Any]] = []

    def pre_execute(self, tool: BaseTool) -> None:
        self.pre_calls.append(tool)

    def post_execute(self, tool: BaseTool, result: Any) -> None:
        self.post_calls.append((tool, result))


class TestChaos6Middleware:
    def test_middleware_fires_on_success(self):
        hook = _RecordingHook()
        engine, _ = _make_engine([GOOD_ECHO], max_retries=3, middleware=[hook])
        result = engine.run("echo hello")

        assert isinstance(result, BaseTool)
        assert len(hook.pre_calls) == 1
        assert hook.pre_calls[0].tool_name == "Echo"
        assert len(hook.post_calls) == 1

    def test_middleware_silent_on_failure(self):
        hook = _RecordingHook()
        engine, _ = _make_engine(
            ["bad", "bad", "bad"], max_retries=3, middleware=[hook]
        )
        result = engine.run("fail")

        assert isinstance(result, FailureReport)
        assert len(hook.pre_calls) == 0
        assert len(hook.post_calls) == 0

    def test_post_execute_receives_execute_result(self):
        hook = _RecordingHook()
        engine, _ = _make_engine([GOOD_ECHO], max_retries=3, middleware=[hook])
        engine.run("echo hello")

        _, result = hook.post_calls[0]
        assert result == "hello"
