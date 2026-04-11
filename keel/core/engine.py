"""
Engine — the LLM-agnostic orchestration layer with a three-level self-correction loop.

State machine (per attempt):

    ┌──────────────────────────────────────────────────────────────┐
    │  llm.complete(messages) → raw_output                         │
    │                                                              │
    │  dispatcher.dispatch(raw_output)                             │
    │     ├─ SyntaxDispatchError  → syntax correction              │
    │     ├─ UnknownToolError     → unknown tool correction        │
    │     ├─ SchemaDispatchError  → schema correction              │
    │     └─ success              → run middleware, return tool    │
    │                                                              │
    │  if attempt >= max_retries  → return FailureReport           │
    └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from keel.core.dispatcher import (
    Dispatcher,
    SchemaDispatchError,
    SyntaxDispatchError,
    UnknownToolError,
)
from keel.core.human_review import (
    HumanReviewQueue,
    HumanReviewRequest,
    HumanReviewTool,
)
from keel.core.middleware import MiddlewareChain, MiddlewareHook
from keel.core import prompts
from keel.core.registry import BaseTool, ToolRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Interface (the single decoupling seam)
# ---------------------------------------------------------------------------


class LLMInterface(ABC):
    """Abstract base class for any LLM backend.

    Implementations must be **synchronous**.  To use async providers, wrap
    them with ``asyncio.run()`` inside ``complete()``.

    The ``messages`` list follows the standard chat format::

        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]

    Set ``flat_input = True`` on a subclass to have the engine serialise
    ``messages`` to a flat string before calling ``complete()``.  The engine
    then logs the serialised string, so debug output always matches what the
    model receives.
    """

    flat_input: bool = False

    @abstractmethod
    def complete(self, messages: list[dict[str, str]] | str) -> str:  # pragma: no cover
        """Send *messages* to the LLM and return the raw response string.

        Receives a flat string when ``flat_input = True``, otherwise the
        standard chat-format list.
        """
        ...


# ---------------------------------------------------------------------------
# FailureReport
# ---------------------------------------------------------------------------


class FailureReport(BaseModel):
    """Structured report returned when all retry attempts are exhausted.

    Fields:
        tool_name:       The tool the LLM was trying to call (if determinable).
        attempts:        Total number of ``llm.complete()`` calls made.
        error_levels:    Ordered list of correction levels hit, e.g.
                         ``["SYNTAX", "SCHEMA", "SYNTAX"]``.
        last_raw_output: The final raw string returned by the LLM.
        errors:          Accumulated structured error dicts from Pydantic or the
                         dispatch pipeline.
    """

    tool_name: str | None = None
    attempts: int
    error_levels: list[str]
    last_raw_output: str
    errors: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_ROLE_SYSTEM = "system"
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"


class Engine:
    """Orchestrates LLM calls, dispatch, and the self-correction state machine.

    Args:
        llm:                 An ``LLMInterface`` implementation.
        registry:            The ``ToolRegistry`` containing all available tools.
        dispatcher:          A configured ``Dispatcher`` instance.
        max_retries:         Maximum number of LLM calls before giving up.
                             Each failed attempt counts as one retry. Defaults to 3.
        middleware:          Optional list of ``MiddlewareHook`` instances to run
                             around successful ``execute()`` calls.
        human_review_queue:  Optional ``HumanReviewQueue``.  When set, any
                             ``HumanReviewTool`` result or ``FailureReport`` is
                             automatically forwarded to admins.
    """

    def __init__(
        self,
        llm: LLMInterface,
        registry: ToolRegistry,
        dispatcher: Dispatcher,
        max_retries: int = 3,
        middleware: list[MiddlewareHook] | None = None,
        human_review_queue: HumanReviewQueue | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._dispatcher = dispatcher
        self._max_retries = max_retries
        self._middleware = MiddlewareChain(middleware or [])
        self._human_review_queue = human_review_queue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_message: str,
        system_prompt: str | None = None,
    ) -> BaseTool | FailureReport:
        """Run the user's request through the self-correction loop.

        If *system_prompt* is ``None``, the registry's schema block is
        automatically injected as the system message.

        Returns a ``BaseTool`` instance on success, or a ``FailureReport``
        if ``max_retries`` is exhausted.
        """
        effective_system = system_prompt or prompts.system_prompt(self._registry)
        messages: list[dict[str, str]] = [
            {"role": _ROLE_SYSTEM, "content": effective_system},
            {"role": _ROLE_USER, "content": user_message},
        ]
        state = _LoopState()
        state.user_message = user_message
        log.debug("run  user=%r  max_retries=%d", user_message, self._max_retries)
        result = self._attempt_loop(messages, state)
        self._maybe_enqueue(result, state)
        if isinstance(result, FailureReport):
            log.warning(
                "run exhausted %d/%d retries  user=%r",
                result.attempts,
                self._max_retries,
                user_message,
            )
        elif isinstance(result, HumanReviewTool):
            log.info("run escalated to human review  reason=%r", result.reason)
        else:
            log.info("run complete  tool=%r", result.tool_name)
        return result

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _attempt_loop(
        self,
        messages: list[dict[str, str]],
        state: _LoopState,
    ) -> BaseTool | FailureReport:
        """Recursive retry loop — one call per attempt."""
        if state.attempts >= self._max_retries:
            log.debug(
                "max_retries=%d reached — returning FailureReport", self._max_retries
            )
            return FailureReport(
                tool_name=state.last_tool_name,
                attempts=state.attempts,
                error_levels=state.error_levels,
                last_raw_output=state.last_raw_output,
                errors=state.accumulated_errors,
            )

        log.debug("attempt %d/%d", state.attempts + 1, self._max_retries)
        payload: list[dict[str, str]] | str = (
            prompts.serialize_messages(messages) if self._llm.flat_input else messages
        )
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "\n── messages ────────────────────────────────\n%s\n────────────────────────────────────────────",
                payload if isinstance(payload, str) else json.dumps(payload, indent=2),
            )
        raw_output = self._llm.complete(payload)
        state.attempts += 1
        state.last_raw_output = raw_output
        log.debug(
            "\n── raw output ──────────────────────────────\n%s\n────────────────────────────────────────────",
            raw_output,
        )

        # Append the model's answer to the conversation so the next turn
        # sees the full history (important for self-correction).
        messages = messages + [{"role": _ROLE_ASSISTANT, "content": raw_output}]

        try:
            tool = self._dispatcher.dispatch(raw_output)
        except SyntaxDispatchError as exc:
            state.error_levels.append("SYNTAX")
            state.accumulated_errors.append({"level": "SYNTAX", "raw": exc.raw})
            log.debug("dispatch SYNTAX  attempt=%d", state.attempts)
            correction = prompts.syntax_correction(exc.raw)
            return self._attempt_loop(
                messages + [{"role": _ROLE_USER, "content": correction}],
                state,
            )
        except UnknownToolError as exc:
            state.error_levels.append("UNKNOWN_TOOL")
            state.last_tool_name = exc.attempted
            state.accumulated_errors.append(
                {
                    "level": "UNKNOWN_TOOL",
                    "attempted": exc.attempted,
                    "available": exc.available,
                }
            )
            log.debug(
                "dispatch UNKNOWN_TOOL  attempted=%r  attempt=%d",
                exc.attempted,
                state.attempts,
            )
            correction = prompts.unknown_tool_correction(exc.attempted, exc.available)
            return self._attempt_loop(
                messages + [{"role": _ROLE_USER, "content": correction}],
                state,
            )
        except SchemaDispatchError as exc:
            state.error_levels.append("SCHEMA")
            state.last_tool_name = exc.tool_name
            state.accumulated_errors.extend(exc.errors)
            log.debug(
                "dispatch SCHEMA  tool=%r  n_errors=%d  attempt=%d",
                exc.tool_name,
                len(exc.errors),
                state.attempts,
            )
            correction = prompts.schema_correction(exc.tool_name, exc.errors)
            return self._attempt_loop(
                messages + [{"role": _ROLE_USER, "content": correction}],
                state,
            )

        # --- Success path ---
        log.debug("dispatch OK  tool=%r  attempt=%d", tool.tool_name, state.attempts)
        self._middleware.run_pre(tool)
        result = tool.execute()
        self._middleware.run_post(tool, result)
        return tool

    def _maybe_enqueue(
        self,
        result: BaseTool | FailureReport,
        state: _LoopState,
    ) -> None:
        """Forward *result* to the human review queue when appropriate.

        Called after every ``run()`` regardless of outcome.  Does nothing
        when no queue is configured or when the result is a normal tool.
        """
        if self._human_review_queue is None:
            return

        if isinstance(result, HumanReviewTool):
            log.info(
                "enqueuing human review  trigger=model_requested  reason=%r",
                result.reason,
            )
            self._human_review_queue.submit(
                HumanReviewRequest(
                    trigger="model_requested",
                    user_message=state.user_message,
                    reason=result.reason,
                    attempts=state.attempts,
                    error_levels=state.error_levels,
                    last_raw_output=state.last_raw_output,
                )
            )
        elif isinstance(result, FailureReport):
            log.warning(
                "enqueuing human review  trigger=failure_report  attempts=%d  error_levels=%s",
                result.attempts,
                result.error_levels,
            )
            self._human_review_queue.submit(
                HumanReviewRequest(
                    trigger="failure_report",
                    user_message=state.user_message,
                    reason=prompts.failure_reason(result.attempts),
                    attempts=result.attempts,
                    error_levels=result.error_levels,
                    last_raw_output=result.last_raw_output,
                    extra={"errors": result.errors},
                )
            )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Internal state container
# ---------------------------------------------------------------------------


class _LoopState:
    """Mutable accumulator passed through the recursive loop."""

    __slots__ = (
        "attempts",
        "error_levels",
        "last_raw_output",
        "last_tool_name",
        "accumulated_errors",
        "user_message",
    )

    def __init__(self) -> None:
        self.attempts: int = 0
        self.error_levels: list[str] = []
        self.last_raw_output: str = ""
        self.last_tool_name: str | None = None
        self.accumulated_errors: list[dict[str, Any]] = []
        self.user_message: str = ""
