"""
Tool Registry — the source of truth for all registered tools.

Key concepts:
- ``BaseTool``: All tools inherit from this Pydantic model and implement ``execute()``.
- ``ToolRegistry``: Stores tool classes, indexes them by namespace, exposes JSON schema generation.
- ``@register_tool``: Decorator that registers a class in the module-level default registry.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RegistryError(Exception):
    """Base exception for registry-related errors."""


class UnknownToolError(RegistryError):
    """Raised when a tool name is not found in the registry."""

    def __init__(self, attempted: str, available: list[str]) -> None:
        self.attempted = attempted
        self.available = available
        super().__init__(
            f"Tool '{attempted}' is not registered. Available tools: {available}"
        )


class DuplicateToolError(RegistryError):
    """Raised when a tool with the same name is registered twice."""


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------


class BaseTool(BaseModel):
    """Base class for all tools.

    Subclasses must:
    - Set ``tool_name`` (a ClassVar string).  If not set, the class name is used.
    - Optionally set ``namespace`` to group related tools (default: ``"default"``).
    - Implement ``execute()`` with the actual side-effectful logic.

    All fields should use ``Field(description=...)`` so the registry can surface
    human-readable argument descriptions in the LLM system prompt.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- ClassVar meta ---
    tool_name: ClassVar[str]
    namespace: ClassVar[str] = "default"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-derive tool_name from class name if not explicitly set.
        if "tool_name" not in cls.__dict__:
            cls.tool_name = cls.__name__

    @abstractmethod
    def execute(self) -> Any:  # pragma: no cover
        """Run the tool's logic and return its result."""
        ...

    @classmethod
    def get_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for this tool's arguments."""
        return cls.model_json_schema()


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Thread-safe (read-heavy) registry mapping tool names to their classes.

    Usage::

        registry = ToolRegistry()
        registry.register(MyTool)
        tool_cls = registry.get("MyTool")
        instance = tool_cls(arg="value")
    """

    def __init__(self) -> None:
        self._tools: dict[str, type[BaseTool]] = {}
        self._namespaces: dict[str, list[str]] = {}

    # --- Registration ---

    def register(self, tool_cls: type[BaseTool]) -> None:
        """Register *tool_cls* in the registry.

        Raises:
            TypeError: If *tool_cls* is not a subclass of ``BaseTool``.
            DuplicateToolError: If a tool with the same name is already registered.
        """
        if not (isinstance(tool_cls, type) and issubclass(tool_cls, BaseTool)):
            raise TypeError(f"{tool_cls!r} must be a subclass of BaseTool.")
        if tool_cls is BaseTool:
            raise TypeError("Cannot register BaseTool itself.")

        name = tool_cls.tool_name
        if name in self._tools:
            raise DuplicateToolError(
                f"Tool '{name}' is already registered. "
                "Use a unique tool_name or a separate ToolRegistry."
            )

        self._tools[name] = tool_cls
        ns = tool_cls.namespace
        self._namespaces.setdefault(ns, []).append(name)

    # --- Lookup ---

    def get(self, name: str) -> type[BaseTool]:
        """Return the tool class for *name*.

        Raises:
            UnknownToolError: If *name* is not registered (includes available list).
        """
        if name not in self._tools:
            raise UnknownToolError(attempted=name, available=self.list_tools())
        return self._tools[name]

    def list_tools(self, namespace: str | None = None) -> list[str]:
        """Return sorted tool names, optionally filtered by *namespace*."""
        if namespace is not None:
            return sorted(self._namespaces.get(namespace, []))
        return sorted(self._tools.keys())

    # --- Schema / prompt generation ---

    def generate_system_prompt_block(self) -> str:
        """Build a structured block describing all registered tools.

        Delegates to ``framework.core.prompts.system_prompt`` — edit that
        module to change what the LLM sees.
        """
        from keel.core import prompts  # local import avoids circular dep

        return prompts.system_prompt(self)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Module-level default registry + decorator
# ---------------------------------------------------------------------------

#: The default registry used by ``@register_tool``.
_default_registry: ToolRegistry = ToolRegistry()


def register_tool(
    cls: type[BaseTool] | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> Any:
    """Class decorator that registers a ``BaseTool`` subclass.

    Can be used with or without arguments::

        @register_tool
        class Search(BaseTool):
            ...

        @register_tool(registry=my_registry)
        class Search(BaseTool):
            ...

    Args:
        cls: The tool class (when used without parentheses).
        registry: Target registry. Defaults to the module-level
            ``_default_registry``.
    """
    target_registry = registry if registry is not None else _default_registry

    def decorator(tool_cls: type[BaseTool]) -> type[BaseTool]:
        target_registry.register(tool_cls)
        return tool_cls

    if cls is not None:
        # Used as @register_tool (no parentheses)
        return decorator(cls)

    # Used as @register_tool(...) — return the decorator
    return decorator
