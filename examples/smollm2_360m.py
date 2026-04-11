"""
Example: using SmolLM2-360M via the onnx-inference gRPC SDK.

Requires the onnx-inference SDK and a running gateway:
    pip install -e ".[onnx]"
    # gateway running at localhost:50050 (or set ONNX_GATEWAY env var)

Run:
    python examples/smollm2_360m.py
"""

from __future__ import annotations

import logging
import os
from typing import Any

from keel.utils.logging import configure_debug_logging
from pydantic import Field

from keel.core.dispatcher import Dispatcher
from keel.core.engine import Engine, FailureReport, LLMInterface
from keel.core.human_review import (
    HumanReviewQueue,
    HumanReviewTool,
    LoggingHumanReviewQueue,
)
from keel.core.registry import BaseTool, ToolRegistry

configure_debug_logging()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter — wraps InferenceClient behind LLMInterface
# ---------------------------------------------------------------------------


class OnnxLLM(LLMInterface):
    """``LLMInterface`` adapter for the onnx-inference gRPC SDK.

    Args:
        address:        Host:port of the running gateway.
                        Defaults to the ``ONNX_GATEWAY`` env var, then
                        ``localhost:50050``.
        max_new_tokens: Token budget per call. 0 = server default (256).
        temperature:    Sampling temperature.
    """

    flat_input = True

    def __init__(
        self,
        address: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0,
    ) -> None:
        self._address = address or os.environ.get("ONNX_GATEWAY", "localhost:50050")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def complete(self, messages: str) -> str:
        """Call ``client.generate()`` with the serialised *messages* string."""
        from onnx_inference import InferenceClient, InferenceError  # noqa: PLC0415

        with InferenceClient(self._address) as client:
            try:
                reply = client.generate(
                    messages,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                )
                return reply
            except InferenceError:
                raise


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

onnx_registry = ToolRegistry()


class WebSearch(BaseTool):
    """Search the web and return relevant results for a query.
    Use when the task asks to search, find, look up, or retrieve information from the internet."""

    tool_name = "WebSearch"
    namespace = "search"

    query: str = Field(description="The search query string.")
    max_results: int = Field(
        default=5, ge=1, le=20, description="Number of results to return."
    )

    def execute(self) -> Any:
        # Replace with a real search client (e.g. SerpAPI, Brave Search).
        return f"[stub] top {self.max_results} results for: {self.query!r}"


class Summarise(BaseTool):
    """Summarise a block of text to a given target length.
    Use when the task provides text to be condensed, summarised, or shortened."""

    tool_name = "Summarise"
    namespace = "text"

    text: str = Field(description="The text to summarise.")
    max_sentences: int = Field(
        default=3, ge=1, description="Target summary length in sentences."
    )

    def execute(self) -> Any:
        # Replace with a real summarisation call.
        preview = self.text[:120].replace("\n", " ")
        return f"[stub] {self.max_sentences}-sentence summary of: {preview!r}..."


class HumanReviewToolRegistered(HumanReviewTool):
    """Escalate to a human reviewer when you are unsure which tool to call.
    Use when the task is ambiguous, unclear, or does not match any available tool."""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

onnx_registry.register(WebSearch)
onnx_registry.register(Summarise)
onnx_registry.register(HumanReviewToolRegistered)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    user_message: str,
    review_queue: HumanReviewQueue | None = None,
    **llm_kwargs: Any,
) -> None:
    """Wire everything up and run a single turn.

    Args:
        user_message:  The task to send to the engine.
        review_queue:  Optional ``HumanReviewQueue`` implementation. Receives
                       both explicit ``HumanReview`` tool calls and automatic
                       ``FailureReport`` escalations. Defaults to
                       ``LoggingHumanReviewQueue`` when ``None``.
    """
    queue = review_queue or LoggingHumanReviewQueue()
    llm = OnnxLLM(**llm_kwargs)
    dispatcher = Dispatcher(onnx_registry)
    engine = Engine(
        llm=llm,
        registry=onnx_registry,
        dispatcher=dispatcher,
        max_retries=3,
        human_review_queue=queue,
    )

    log.info("user: %s", user_message)
    result = engine.run(user_message)

    if isinstance(result, FailureReport):
        log.warning(
            "FAILURE after %d attempt(s) — routed to human review queue",
            result.attempts,
        )
        log.warning("  error levels : %s", result.error_levels)
        log.warning("  last output  : %r", result.last_raw_output)
    elif isinstance(result, HumanReviewTool):
        log.info("HUMAN REVIEW requested by model — routed to human review queue")
        log.info("  reason     : %s", result.reason)
        log.info("  confidence : %s", result.confidence)
    else:
        log.info("SUCCESS  %s -> %r", result.tool_name, result.execute())


if __name__ == "__main__":
    gateway = os.environ.get("ONNX_GATEWAY", "localhost:50050")
    log.info("using gateway: %s", gateway)

    # --- Normal tool use ---
    run("search for recent papers on ONNX quantisation")
    # run("summarise the following in 2 sentences: The ONNX runtime is a cross-platform engine.")
    # run("Find recent papers on ONNX quantisation and summarise the top one in 2 sentences")

    # --- Ambiguous task: model should call HumanReview ---
    # run("please do the needful regarding the project")
