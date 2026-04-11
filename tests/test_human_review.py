"""
Tests for the human-review escalation system.

Covers:
- HumanReviewTool.execute() return value
- model-requested escalation (model calls HumanReviewTool)
- failure-driven escalation (engine exhausts retries with queue attached)
- LoggingHumanReviewQueue does not raise
- no queue attached — FailureReport returned normally, no error
- HumanReviewRequest field values (user_message, attempts, error_levels)
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from keel.core.dispatcher import Dispatcher
from keel.core.engine import Engine, FailureReport
from keel.core.human_review import (
    HumanReviewQueue,
    HumanReviewRequest,
    HumanReviewTool,
    LoggingHumanReviewQueue,
)
from keel.core.registry import BaseTool, ToolRegistry
from tests.conftest import MockLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hr_registry() -> ToolRegistry:
    """Registry that contains a real workable tool and HumanReviewTool."""

    class EchoTool(BaseTool):
        """Echoes back the provided message."""

        tool_name = "Echo"
        namespace = "test"
        message: str = Field(description="The message to echo.")

        def execute(self) -> Any:
            return self.message

    reg = ToolRegistry()
    reg.register(EchoTool)
    reg.register(HumanReviewTool)
    return reg


class _CapturingQueue(HumanReviewQueue):
    """Test double — records every submitted request without side-effects."""

    def __init__(self) -> None:
        self.received: list[HumanReviewRequest] = []

    def submit(self, request: HumanReviewRequest) -> None:
        self.received.append(request)


def _make_engine(
    responses: list[str],
    max_retries: int = 3,
    registry: ToolRegistry | None = None,
    human_review_queue: HumanReviewQueue | None = None,
) -> tuple[Engine, MockLLM]:
    if registry is None:
        registry = _make_hr_registry()
    llm = MockLLM(responses)
    dispatcher = Dispatcher(registry)
    engine = Engine(
        llm=llm,
        registry=registry,
        dispatcher=dispatcher,
        max_retries=max_retries,
        human_review_queue=human_review_queue,
    )
    return engine, llm


# ---------------------------------------------------------------------------
# HumanReviewTool unit tests
# ---------------------------------------------------------------------------


def test_human_review_tool_execute_returns_structured_dict() -> None:
    tool = HumanReviewTool(
        reason="task is ambiguous",
        confidence="low",
    )
    result = tool.execute()
    assert result == {
        "escalated": True,
        "reason": "task is ambiguous",
        "confidence": "low",
    }


def test_human_review_tool_confidence_defaults_to_low() -> None:
    tool = HumanReviewTool(reason="unclear task")
    assert tool.confidence == "low"


def test_human_review_tool_medium_confidence() -> None:
    tool = HumanReviewTool(reason="probably web search", confidence="medium")
    assert tool.confidence == "medium"


# ---------------------------------------------------------------------------
# LoggingHumanReviewQueue
# ---------------------------------------------------------------------------


def test_logging_queue_does_not_raise() -> None:
    queue = LoggingHumanReviewQueue()
    request = HumanReviewRequest(
        trigger="model_requested",
        user_message="do something",
        reason="not sure which tool",
        attempts=1,
    )
    # Must not raise
    queue.submit(request)


def test_logging_queue_does_not_raise_on_failure_report() -> None:
    queue = LoggingHumanReviewQueue()
    request = HumanReviewRequest(
        trigger="failure_report",
        user_message="broken input",
        reason="exhausted retries",
        attempts=3,
        error_levels=["SYNTAX", "SYNTAX", "SYNTAX"],
        last_raw_output="garbage garbage garbage",
    )
    queue.submit(request)


# ---------------------------------------------------------------------------
# Model-requested escalation
# ---------------------------------------------------------------------------


MODEL_REQUESTS_REVIEW = (
    '{"tool": "HumanReview", "reason": "task is ambiguous", "confidence": "low"}'
)


def test_model_requested_escalation_calls_queue_submit() -> None:
    queue = _CapturingQueue()
    engine, _ = _make_engine(
        responses=[MODEL_REQUESTS_REVIEW],
        human_review_queue=queue,
    )
    result = engine.run("do something vague")

    assert isinstance(result, HumanReviewTool)
    assert len(queue.received) == 1
    req = queue.received[0]
    assert req.trigger == "model_requested"
    assert req.reason == "task is ambiguous"


def test_model_requested_escalation_request_fields() -> None:
    queue = _CapturingQueue()
    engine, _ = _make_engine(
        responses=[MODEL_REQUESTS_REVIEW],
        human_review_queue=queue,
    )
    user_message = "what should I do here?"
    engine.run(user_message)

    req = queue.received[0]
    assert req.user_message == user_message
    assert req.attempts == 1
    assert req.trigger == "model_requested"


def test_model_requested_no_queue_does_not_raise() -> None:
    """When no queue is attached, HumanReviewTool is still returned cleanly."""
    engine, _ = _make_engine(
        responses=[MODEL_REQUESTS_REVIEW],
        human_review_queue=None,
    )
    result = engine.run("ambiguous task")
    assert isinstance(result, HumanReviewTool)


# ---------------------------------------------------------------------------
# Failure-driven escalation
# ---------------------------------------------------------------------------

BAD_RESPONSE = "this is not json at all"


def test_failure_report_escalation_calls_queue_submit() -> None:
    queue = _CapturingQueue()
    engine, _ = _make_engine(
        responses=[BAD_RESPONSE, BAD_RESPONSE, BAD_RESPONSE, BAD_RESPONSE],
        max_retries=3,
        human_review_queue=queue,
    )
    result = engine.run("impossible task")

    assert isinstance(result, FailureReport)
    assert len(queue.received) == 1
    req = queue.received[0]
    assert req.trigger == "failure_report"


def test_failure_report_escalation_request_fields() -> None:
    queue = _CapturingQueue()
    engine, _ = _make_engine(
        responses=[BAD_RESPONSE] * 4,
        max_retries=3,
        human_review_queue=queue,
    )
    user_message = "this task will always fail"
    engine.run(user_message)

    req = queue.received[0]
    assert req.user_message == user_message
    assert req.attempts == 3
    assert req.error_levels.count("SYNTAX") == 3
    assert req.last_raw_output == BAD_RESPONSE


def test_failure_report_no_queue_returns_failure_report_without_error() -> None:
    engine, _ = _make_engine(
        responses=[BAD_RESPONSE] * 4,
        max_retries=3,
        human_review_queue=None,
    )
    result = engine.run("this will fail")
    assert isinstance(result, FailureReport)


# ---------------------------------------------------------------------------
# Queue is called exactly once per run, not on successful normal tools
# ---------------------------------------------------------------------------


def test_successful_tool_call_does_not_trigger_queue() -> None:
    queue = _CapturingQueue()
    engine, _ = _make_engine(
        responses=['{"tool": "Echo", "message": "hello"}'],
        human_review_queue=queue,
    )
    result = engine.run("echo hello")

    assert isinstance(result, BaseTool)
    assert not isinstance(result, HumanReviewTool)
    assert len(queue.received) == 0
