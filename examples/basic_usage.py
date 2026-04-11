"""
Basic usage example — the "Definition of Done" demonstration.

A developer adds a brand-new, multi-parameter tool by writing ONLY a Pydantic
class.  The framework handles prompting, validation, and error-correction with
no additional code required.

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import Field

from keel.core.dispatcher import Dispatcher
from keel.core.engine import Engine, FailureReport, LLMInterface
from keel.core.human_review import (
    HumanReviewQueue,
    HumanReviewRequest,
    HumanReviewTool,
)
from keel.core.middleware import MiddlewareHook
from keel.core.registry import BaseTool, ToolRegistry
from keel.utils.logging import configure_debug_logging

configure_debug_logging(level=logging.INFO)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub LLM — cycles through canned responses to simulate an LLM conversation.
#            In production, swap this for a real LLMInterface implementation.
# ---------------------------------------------------------------------------


class StubLLM(LLMInterface):
    """Cycles through a list of canned responses to simulate an LLM conversation."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        log.debug("llm attempt %d → %r", self._index, response)
        return response


class LoggingMiddleware(MiddlewareHook):
    def pre_execute(self, tool: BaseTool) -> None:
        log.debug("middleware pre   %s  args=%s", tool.tool_name, tool.model_dump())

    def post_execute(self, tool: BaseTool, result: Any) -> None:
        log.debug("middleware post  %s  result=%r", tool.tool_name, result)


# ---------------------------------------------------------------------------
# Step 1: Define a custom registry so this example stays isolated.
# ---------------------------------------------------------------------------

example_registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Step 2: Define the tools.
# ---------------------------------------------------------------------------


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class DatabaseQuery(BaseTool):
    """Query a relational database table with optional filters and ordering.

    Use this tool when the user wants to retrieve records from a named table,
    optionally applying a WHERE condition, setting a row limit, or changing
    the sort order.
    """

    tool_name = "DatabaseQuery"
    namespace = "database"

    table: str = Field(description="The name of the table to query.")
    where: str | None = Field(
        default=None,
        description="SQL WHERE clause (without the WHERE keyword), e.g. 'age > 30'.",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Maximum number of rows to return. Must be between 1 and 10 000.",
    )
    order_by: str | None = Field(
        default=None,
        description="Column to sort by.",
    )
    order: SortOrder = Field(
        default=SortOrder.ASC,
        description="Sort direction: 'asc' or 'desc'.",
    )

    def execute(self) -> Any:
        # In production this would hit a real DB.  Here we just echo back
        # the generated query so the demo output is readable.
        parts = [f"SELECT * FROM {self.table}"]
        if self.where:
            parts.append(f"WHERE {self.where}")
        if self.order_by:
            parts.append(f"ORDER BY {self.order_by} {self.order.value.upper()}")
        parts.append(f"LIMIT {self.limit}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Step 3: Register the tools.
# ---------------------------------------------------------------------------

example_registry.register(HumanReviewTool)
example_registry.register(DatabaseQuery)


# ---------------------------------------------------------------------------
# Step 4: Wire it all together and run the scenarios.
# ---------------------------------------------------------------------------


def run_scenario(
    title: str,
    responses: list[str],
    user_message: str,
    review_queue: HumanReviewQueue | None = None,
) -> None:
    log.info("=" * 60)
    log.info("  Scenario: %s", title)
    log.info("  User:     %r", user_message)
    log.info("=" * 60)

    llm = StubLLM(responses)
    dispatcher = Dispatcher(example_registry)
    engine = Engine(
        llm=llm,
        registry=example_registry,
        dispatcher=dispatcher,
        max_retries=3,
        middleware=[LoggingMiddleware()],
        human_review_queue=review_queue,
    )

    result = engine.run(user_message)

    if isinstance(result, HumanReviewTool):
        log.info(
            "ESCALATED  reason=%r  confidence=%s", result.reason, result.confidence
        )
    elif isinstance(result, FailureReport):
        log.warning(
            "FAILURE  attempts=%d  error_levels=%s  last_output=%r",
            result.attempts,
            result.error_levels,
            result.last_raw_output,
        )
    else:
        log.info("SUCCESS  tool=%s", result.tool_name)


if __name__ == "__main__":
    # --- Scenario A: happy path (first attempt succeeds) ---
    run_scenario(
        title="Happy path — valid JSON on first try",
        responses=[
            '{"tool": "DatabaseQuery", "table": "users", "where": "age > 30", "limit": 50}'
        ],
        user_message="query the users table where age > 30, max 50 rows",
    )

    # --- Scenario B: self-correction (syntax error then schema error then success) ---
    run_scenario(
        title="Self-correction — two failures then success",
        responses=[
            # Attempt 1: plain text (syntax error)
            "I'll query the database for you.",
            # Attempt 2: valid JSON but wrong type for limit (schema error)
            '{"tool": "DatabaseQuery", "table": "orders", "limit": "not-a-number"}',
            # Attempt 3: correct
            '{"tool": "DatabaseQuery", "table": "orders", "where": "status = \'pending\'", "limit": 20}',
        ],
        user_message="show pending orders",
    )

    # --- Scenario C: exhausted retries (always bad JSON) ---
    run_scenario(
        title="All retries exhausted — FailureReport returned",
        responses=["no json", "still no json", "never json"],
        user_message="this will fail",
    )

    # --- Scenario D: model signals low confidence → human review ---
    # The model can call HumanReviewTool when it isn't sure which tool fits.
    # The engine forwards the request to whatever HumanReviewQueue is attached.
    # Here we use LoggingHumanReviewQueue (the default) which writes WARNING lines.
    class _CapturingQueue(HumanReviewQueue):
        """Logs submitted review requests at INFO level."""

        def submit(self, request: HumanReviewRequest) -> None:
            log.info(
                "review queue  trigger=%r  attempts=%d  reason=%r  user=%r",
                request.trigger,
                request.attempts,
                request.reason,
                request.user_message,
            )

    run_scenario(
        title="Ambiguous task — model requests human review",
        responses=[
            '{"tool": "HumanReview", "reason": "the request does not match any available tool", "confidence": "low"}'
        ],
        user_message="can you just figure it out yourself?",
        review_queue=_CapturingQueue(),
    )

    log.info("done — add any BaseTool subclass and it is immediately dispatchable")
