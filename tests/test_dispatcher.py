"""
Tests for framework/core/dispatcher.py — Dispatcher and its typed exceptions.
"""

from __future__ import annotations

import pytest

from keel.core.dispatcher import (
    SchemaDispatchError,
    SyntaxDispatchError,
    UnknownToolError,
)


# The `dispatcher`, `strict_dispatcher`, and `registry` fixtures come from conftest.py.


class TestSuccessfulDispatch:
    def test_bare_json_dispatched(self, dispatcher):
        tool = dispatcher.dispatch('{"tool": "Search", "query": "hello"}')
        assert tool.tool_name == "Search"
        assert tool.query == "hello"

    def test_json_with_default_field(self, dispatcher):
        # precision has a default of 2
        tool = dispatcher.dispatch('{"tool": "Calculate", "expression": "1+1"}')
        assert tool.expression == "1+1"
        assert tool.precision == 2

    def test_json_in_markdown_fence(self, dispatcher):
        raw = '```json\n{"tool": "Search", "query": "fenced"}\n```'
        tool = dispatcher.dispatch(raw)
        assert tool.query == "fenced"

    def test_json_buried_in_text(self, dispatcher):
        raw = 'I will now call: {"tool": "Search", "query": "buried"} Done.'
        tool = dispatcher.dispatch(raw)
        assert tool.query == "buried"

    def test_execute_returns_result(self, dispatcher):
        tool = dispatcher.dispatch('{"tool": "Search", "query": "exec test"}')
        assert tool.execute() == "results:exec test"


class TestLaxTypeCoercion:
    def test_string_to_int_coercion(self, dispatcher):
        # "5" should be coerced to 5 in lax mode
        tool = dispatcher.dispatch('{"tool": "StrictType", "count": "5"}')
        assert tool.count == 5

    def test_float_to_int_coercion(self, dispatcher):
        tool = dispatcher.dispatch('{"tool": "StrictType", "count": 3.0}')
        assert tool.count == 3


class TestStrictModeRejectsCoercion:
    def test_string_rejected_for_int(self, strict_dispatcher):
        with pytest.raises(SchemaDispatchError) as exc_info:
            strict_dispatcher.dispatch('{"tool": "StrictType", "count": "5"}')
        err = exc_info.value
        assert err.tool_name == "StrictType"
        assert any("count" in str(e["loc"]) for e in err.errors)

    def test_float_rejected_for_int(self, strict_dispatcher):
        with pytest.raises(SchemaDispatchError):
            strict_dispatcher.dispatch('{"tool": "StrictType", "count": 3.7}')


class TestSyntaxErrors:
    def test_plain_text_raises_syntax_error(self, dispatcher):
        with pytest.raises(SyntaxDispatchError) as exc_info:
            dispatcher.dispatch("I cannot do that.")
        assert exc_info.value.raw == "I cannot do that."

    def test_truncated_json_raises_syntax_error(self, dispatcher):
        with pytest.raises(SyntaxDispatchError):
            dispatcher.dispatch('{"tool": "Search", "query":')

    def test_empty_string_raises_syntax_error(self, dispatcher):
        with pytest.raises(SyntaxDispatchError):
            dispatcher.dispatch("")

    def test_missing_tool_key_raises_syntax_error(self, dispatcher):
        # Valid JSON but no "tool" key
        with pytest.raises(SyntaxDispatchError):
            dispatcher.dispatch('{"query": "no tool key"}')

    def test_array_root_raises_syntax_error(self, dispatcher):
        with pytest.raises(SyntaxDispatchError):
            dispatcher.dispatch("[1, 2, 3]")


class TestUnknownTool:
    def test_unknown_tool_raises(self, dispatcher):
        with pytest.raises(UnknownToolError) as exc_info:
            dispatcher.dispatch('{"tool": "GhostTool", "x": 1}')
        err = exc_info.value
        assert err.attempted == "GhostTool"
        assert "Search" in err.available

    def test_error_lists_all_available_tools(self, dispatcher, registry):
        with pytest.raises(UnknownToolError) as exc_info:
            dispatcher.dispatch('{"tool": "Nonexistent"}')
        available = exc_info.value.available
        for name in registry.list_tools():
            assert name in available


class TestSchemaErrors:
    def test_missing_required_field(self, dispatcher):
        # 'query' is required for Search
        with pytest.raises(SchemaDispatchError) as exc_info:
            dispatcher.dispatch('{"tool": "Search"}')
        err = exc_info.value
        assert err.tool_name == "Search"
        assert len(err.errors) >= 1

    def test_error_has_loc_msg_input(self, dispatcher):
        with pytest.raises(SchemaDispatchError) as exc_info:
            dispatcher.dispatch('{"tool": "Search"}')
        for e in exc_info.value.errors:
            assert "loc" in e
            assert "msg" in e
            assert "input" in e

    def test_wrong_type_in_strict_mode_has_field_loc(self, strict_dispatcher):
        with pytest.raises(SchemaDispatchError) as exc_info:
            strict_dispatcher.dispatch(
                '{"tool": "StrictType", "count": "not-a-number"}'
            )
        locs = [e["loc"] for e in exc_info.value.errors]
        assert any("count" in loc for loc in locs)
