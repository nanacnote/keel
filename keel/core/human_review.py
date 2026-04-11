"""
Human review escalation — fallback when the model can't confidently select a tool.

Two escalation triggers exist and both flow through the same ``HumanReviewQueue``:

1. **Model-requested** — The model explicitly calls ``HumanReviewTool`` when it
   is uncertain which tool to use or the task is ambiguous.  The tool appears in
   the registry like any other, so the prompt teaches the model it can ask for
   help.

2. **Automatic / failure-driven** — ``Engine`` exhausts ``max_retries`` and
   returns a ``FailureReport``.  If a queue is attached to the engine, the
   failure is automatically forwarded without any extra call-site code.

Admin integration
-----------------
Subclass ``HumanReviewQueue`` and implement ``submit()``.  Pass an instance to
``Engine(human_review_queue=...)``.  The default ``LoggingHumanReviewQueue``
writes structured WARNING lines — useful during development and as a fallback.

Example (Slack webhook)::

    class SlackQueue(HumanReviewQueue):
        def __init__(self, webhook_url: str) -> None:
            self._url = webhook_url

        def submit(self, request: HumanReviewRequest) -> None:
            import urllib.request, json
            body = json.dumps({
                "text": (
                    f":hand: *Human review requested*\\n"
                    f"*Trigger:* {request.trigger}\\n"
                    f"*User message:* {request.user_message}\\n"
                    f"*Reason:* {request.reason}\\n"
                    f"*Attempts:* {request.attempts}"
                )
            }).encode()
            urllib.request.urlopen(urllib.request.Request(
                self._url, data=body,
                headers={"Content-Type": "application/json"},
            ))

    engine = Engine(..., human_review_queue=SlackQueue("https://hooks.slack.com/..."))
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from keel.core.registry import BaseTool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review request — the payload sent to every queue implementation
# ---------------------------------------------------------------------------


class HumanReviewRequest(BaseModel):
    """Structured payload delivered to ``HumanReviewQueue.submit()``.

    Fields:
        trigger:        ``"model_requested"`` — the model called ``HumanReviewTool``
                        explicitly.  ``"failure_report"`` — all retries were
                        exhausted by the engine.
        user_message:   The original user turn that could not be resolved.
        reason:         Plain-text explanation.  For model-requested reviews this
                        comes from the model itself; for failure reports it is
                        auto-generated.
        attempts:       How many LLM calls were made before escalation.
        error_levels:   Ordered list of correction levels hit, e.g.
                        ``["SYNTAX", "SCHEMA"]``.
        last_raw_output: The final raw string the LLM returned.
        extra:          Any additional structured data (e.g. per-field schema
                        errors from ``SchemaDispatchError``).
    """

    trigger: Literal["model_requested", "failure_report"]
    user_message: str
    reason: str
    attempts: int
    error_levels: list[str] = Field(default_factory=list)
    last_raw_output: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Queue ABC
# ---------------------------------------------------------------------------


class HumanReviewQueue(ABC):
    """Abstract sink for human review requests.

    Implement ``submit()`` to route requests to your admin surface — a
    database table, a Slack channel, a PagerDuty incident, etc.

    ``submit()`` must not raise.  Log and swallow any delivery errors so that
    a broken notification channel never crashes the main execution path.
    """

    @abstractmethod
    def submit(self, request: HumanReviewRequest) -> None:  # pragma: no cover
        """Deliver *request* to admins."""
        ...


# ---------------------------------------------------------------------------
# Built-in implementations
# ---------------------------------------------------------------------------


class LoggingHumanReviewQueue(HumanReviewQueue):
    """Default queue — writes structured WARNING lines via ``logging``.

    Useful during development and as a safe fallback before a real
    notification channel is wired up.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger(__name__)

    def submit(self, request: HumanReviewRequest) -> None:
        self._log.warning(
            "HUMAN REVIEW REQUESTED  trigger=%s  attempts=%d  reason=%r  "
            "error_levels=%s  user_message=%r",
            request.trigger,
            request.attempts,
            request.reason,
            request.error_levels,
            request.user_message,
        )
        if request.last_raw_output:
            self._log.warning("  last_raw_output=%r", request.last_raw_output[:300])


# ---------------------------------------------------------------------------
# HumanReviewTool — the model's explicit escalation path
# ---------------------------------------------------------------------------


class HumanReviewTool(BaseTool):
    """Escalate this task to a human reviewer.
    Use when the task is ambiguous, unclear, or does not match any available tool."""

    tool_name = "HumanReview"
    namespace = "escalation"

    reason: str = Field(
        description=(
            "In 1-2 sentences, describe what is unclear or missing from the user's request."
        )
    )
    confidence: Literal["low", "medium"] = Field(
        default="low",
        description="Your confidence level in tool selection: 'low' or 'medium'.",
    )

    def execute(self) -> dict[str, Any]:
        """Return a structured escalation dict.

        The ``Engine`` intercepts this before ``execute()`` is called in the
        normal middleware path, so this method is a safe fallback for callers
        who invoke ``tool.execute()`` directly.
        """
        return {
            "escalated": True,
            "reason": self.reason,
            "confidence": self.confidence,
        }
