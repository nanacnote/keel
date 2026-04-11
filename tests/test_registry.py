"""
Tests for framework/core/registry.py — ToolRegistry, BaseTool, @register_tool.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import Field

from keel.core.registry import (
    BaseTool,
    DuplicateToolError,
    ToolRegistry,
    UnknownToolError,
    _default_registry,
    register_tool,
)


# ---------------------------------------------------------------------------
# Helper tool classes (local to this module, not using _default_registry)
# ---------------------------------------------------------------------------


class _LocalRegistry:
    """Provides a fresh ToolRegistry + a couple of test tools per test."""

    @staticmethod
    def make() -> ToolRegistry:
        return ToolRegistry()


def _make_tool(name: str, ns: str = "default") -> type[BaseTool]:
    class _T(BaseTool):
        tool_name = name
        namespace = ns
        value: str = Field(default="x")

        def execute(self) -> Any:
            return self.value

    _T.__name__ = name
    return _T


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_get(self):
        reg = ToolRegistry()
        T = _make_tool("Alpha")
        reg.register(T)
        assert reg.get("Alpha") is T

    def test_auto_tool_name_from_class_name(self):
        class AutoNamed(BaseTool):
            def execute(self) -> Any:
                return None

        assert AutoNamed.tool_name == "AutoNamed"

    def test_duplicate_raises(self):
        reg = ToolRegistry()
        T = _make_tool("Dup")
        reg.register(T)
        T2 = _make_tool("Dup")
        with pytest.raises(DuplicateToolError):
            reg.register(T2)

    def test_register_base_tool_raises_type_error(self):
        reg = ToolRegistry()
        with pytest.raises(TypeError):
            reg.register(BaseTool)  # type: ignore[arg-type]

    def test_register_non_tool_raises_type_error(self):
        reg = ToolRegistry()
        with pytest.raises(TypeError):
            reg.register(str)  # type: ignore[arg-type]

    def test_len(self):
        reg = ToolRegistry()
        reg.register(_make_tool("A"))
        reg.register(_make_tool("B"))
        assert len(reg) == 2

    def test_contains(self):
        reg = ToolRegistry()
        T = _make_tool("C")
        reg.register(T)
        assert "C" in reg
        assert "Z" not in reg


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


class TestNamespace:
    def test_namespace_indexing(self):
        reg = ToolRegistry()
        reg.register(_make_tool("WebSearch", ns="search"))
        reg.register(_make_tool("ImageSearch", ns="search"))
        reg.register(_make_tool("WriteFile", ns="filesystem"))

        assert set(reg.list_tools("search")) == {"WebSearch", "ImageSearch"}
        assert reg.list_tools("filesystem") == ["WriteFile"]

    def test_unknown_namespace_returns_empty(self):
        reg = ToolRegistry()
        assert reg.list_tools("nonexistent") == []

    def test_list_all_tools_sorted(self):
        reg = ToolRegistry()
        reg.register(_make_tool("Zebra"))
        reg.register(_make_tool("Apple"))
        assert reg.list_tools() == ["Apple", "Zebra"]


# ---------------------------------------------------------------------------
# UnknownToolError
# ---------------------------------------------------------------------------


class TestUnknownToolError:
    def test_raises_with_available_list(self):
        reg = ToolRegistry()
        reg.register(_make_tool("Foo"))
        reg.register(_make_tool("Bar"))

        with pytest.raises(UnknownToolError) as exc_info:
            reg.get("NotHere")

        err = exc_info.value
        assert err.attempted == "NotHere"
        assert "Foo" in err.available
        assert "Bar" in err.available

    def test_error_message_contains_available(self):
        reg = ToolRegistry()
        reg.register(_make_tool("MyTool"))
        with pytest.raises(UnknownToolError, match="MyTool"):
            reg.get("Ghost")


# ---------------------------------------------------------------------------
# System prompt block
# ---------------------------------------------------------------------------


class TestSystemPromptBlock:
    def test_contains_tool_name(self):
        reg = ToolRegistry()
        T = _make_tool("Prompt Tool")
        reg.register(T)
        block = reg.generate_system_prompt_block()
        assert "Prompt Tool" in block

    def test_contains_field_description(self):
        class RichTool(BaseTool):
            tool_name = "Rich"
            namespace = "test"
            query: str = Field(description="A very specific search query.")

            def execute(self) -> Any:
                return None

        reg = ToolRegistry()
        reg.register(RichTool)
        block = reg.generate_system_prompt_block()
        assert "A very specific search query." in block

    def test_contains_json_format_hint(self):
        reg = ToolRegistry()
        reg.register(_make_tool("AnyTool"))
        block = reg.generate_system_prompt_block()
        assert '"tool"' in block

    def test_contains_namespace(self):
        reg = ToolRegistry()
        reg.register(_make_tool("NsTool", ns="multimedia"))
        block = reg.generate_system_prompt_block()
        assert "multimedia" in block

    def test_docstring_in_prompt(self):
        class DocTool(BaseTool):
            """Searches external databases for relevant records."""

            tool_name = "DocTool"
            namespace = "default"
            q: str = Field(default="x")

            def execute(self) -> Any:
                return None

        reg = ToolRegistry()
        reg.register(DocTool)
        block = reg.generate_system_prompt_block()
        assert "Searches external databases" in block


# ---------------------------------------------------------------------------
# @register_tool decorator
# ---------------------------------------------------------------------------


class TestRegisterToolDecorator:
    def test_decorator_without_args_registers_in_default(self):
        # Use a unique name to avoid colliding with other tests in the same process.
        @register_tool
        class _DecoratorTest(BaseTool):
            tool_name = "_DecoratorTestTool_unique_97531"
            namespace = "test"

            def execute(self) -> Any:
                return None

        assert "_DecoratorTestTool_unique_97531" in _default_registry

    def test_decorator_with_custom_registry(self):
        custom_reg = ToolRegistry()

        @register_tool(registry=custom_reg)
        class _CustomRegTool(BaseTool):
            tool_name = "_CustomRegTool_unique_13579"
            namespace = "test"

            def execute(self) -> Any:
                return None

        assert "_CustomRegTool_unique_13579" in custom_reg
        assert "_CustomRegTool_unique_13579" not in _default_registry

    def test_decorator_returns_original_class(self):
        custom_reg = ToolRegistry()

        @register_tool(registry=custom_reg)
        class _ReturnCheck(BaseTool):
            tool_name = "_ReturnCheck_unique_24680"
            namespace = "test"

            def execute(self) -> Any:
                return None

        assert _ReturnCheck.__name__ == "_ReturnCheck"
        assert issubclass(_ReturnCheck, BaseTool)
