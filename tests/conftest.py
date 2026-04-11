"""
Shared fixtures for the test suite.
"""

from __future__ import annotations

from typing import Any

import pytest

from keel.core.dispatcher import Dispatcher
from keel.core.engine import LLMInterface
from keel.core.registry import BaseTool, ToolRegistry
from pydantic import Field


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


class SearchTool(BaseTool):
    """Search the web for a query."""

    tool_name = "Search"
    namespace = "search"

    query: str = Field(description="The search query string.")

    def execute(self) -> Any:
        return f"results:{self.query}"


class CalculateTool(BaseTool):
    """Perform a basic arithmetic calculation."""

    tool_name = "Calculate"
    namespace = "math"

    expression: str = Field(description="A mathematical expression to evaluate.")
    precision: int = Field(default=2, description="Number of decimal places.")

    def execute(self) -> Any:
        return eval(self.expression)  # noqa: S307 — test only


class StrictTypeTool(BaseTool):
    """Tool with an integer field to test type coercion."""

    tool_name = "StrictType"
    namespace = "default"

    count: int = Field(description="A numeric count.")

    def execute(self) -> Any:
        return self.count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    """A fresh ToolRegistry pre-loaded with three test tools."""
    reg = ToolRegistry()
    reg.register(SearchTool)
    reg.register(CalculateTool)
    reg.register(StrictTypeTool)
    return reg


@pytest.fixture()
def dispatcher(registry: ToolRegistry) -> Dispatcher:
    """A Dispatcher in lax (default) mode."""
    return Dispatcher(registry)


@pytest.fixture()
def strict_dispatcher(registry: ToolRegistry) -> Dispatcher:
    """A Dispatcher in strict mode — no type coercion."""
    return Dispatcher(registry, strict=True)


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM(LLMInterface):
    """A mock LLM that returns responses from a pre-loaded queue.

    Args:
        responses: Ordered list of strings to return from ``complete()``.
                   Raises ``StopIteration`` if exhausted.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.call_count = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.received_messages.append(messages)
        self.call_count += 1
        if self._index >= len(self._responses):
            raise StopIteration("MockLLM has no more responses.")
        response = self._responses[self._index]
        self._index += 1
        return response


@pytest.fixture()
def mock_llm_factory():
    """Factory fixture: call it with a list of response strings."""

    def factory(responses: list[str]) -> MockLLM:
        return MockLLM(responses)

    return factory
