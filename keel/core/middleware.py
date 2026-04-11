"""
Middleware system for pre/post-execution hooks.

Hooks run synchronously before and after a tool's ``execute()`` call.
Common uses: structured logging, latency tracking, cost estimation, security sandboxing.

Example::

    class AuditLogger(MiddlewareHook):
        def pre_execute(self, tool):
            print(f"[AUDIT] About to execute: {tool.tool_name}")

        def post_execute(self, tool, result):
            print(f"[AUDIT] Finished: {tool.tool_name} -> {result!r}")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from keel.core.registry import BaseTool


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MiddlewareHook(Protocol):
    """Protocol for pre/post execution middleware.

    Both methods are required.  Implementations must not raise exceptions
    during ``post_execute`` — any exception there would shadow the tool result.
    """

    def pre_execute(self, tool: BaseTool) -> None:
        """Called immediately before ``tool.execute()``."""
        ...  # pragma: no cover

    def post_execute(self, tool: BaseTool, result: Any) -> None:
        """Called immediately after ``tool.execute()`` with the returned *result*."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


class MiddlewareChain:
    """Ordered chain of ``MiddlewareHook`` implementations.

    The chain is iterated in insertion order for both ``run_pre`` and
    ``run_post``.  Each hook is called even if a previous hook raises, so that
    all observers get a chance to record state.

    Usage::

        chain = MiddlewareChain([AuditLogger(), CostTracker()])
        chain.run_pre(tool_instance)
        result = tool_instance.execute()
        chain.run_post(tool_instance, result)
    """

    def __init__(self, hooks: list[MiddlewareHook] | None = None) -> None:
        self._hooks: list[MiddlewareHook] = list(hooks or [])

    def add(self, hook: MiddlewareHook) -> None:
        """Append *hook* to the end of the chain."""
        if not isinstance(hook, MiddlewareHook):
            raise TypeError(f"{hook!r} does not implement the MiddlewareHook protocol.")
        self._hooks.append(hook)

    def run_pre(self, tool: BaseTool) -> None:
        """Call ``pre_execute`` on every hook in order."""
        errors: list[Exception] = []
        for hook in self._hooks:
            try:
                hook.pre_execute(tool)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise ExceptionGroup("pre_execute middleware errors", errors)

    def run_post(self, tool: BaseTool, result: Any) -> None:
        """Call ``post_execute`` on every hook in order."""
        errors: list[Exception] = []
        for hook in self._hooks:
            try:
                hook.post_execute(tool, result)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise ExceptionGroup("post_execute middleware errors", errors)

    def __len__(self) -> int:
        return len(self._hooks)
