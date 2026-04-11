"""
Tests for framework/utils/parser.py — JSON extraction logic.
"""

from __future__ import annotations


from keel.utils.parser import extract_json


class TestDirectJson:
    def test_bare_object(self):
        result = extract_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_bare_object_with_whitespace(self):
        result = extract_json('  {"key": 42}  ')
        assert result == '{"key": 42}'

    def test_nested_object(self):
        raw = '{"outer": {"inner": 1}}'
        result = extract_json(raw)
        assert result == raw

    def test_array_at_root_not_extracted(self):
        # Arrays are not tool calls; we only extract objects.
        # The parser should return None or the raw array (implementation-defined).
        # Our parser looks for { } blocks — arrays not supported, returns None.
        result = extract_json("[1, 2, 3]")
        assert result is None


class TestMarkdownFence:
    def test_json_fence(self):
        raw = '```json\n{"tool": "Search", "query": "hello"}\n```'
        result = extract_json(raw)
        assert result == '{"tool": "Search", "query": "hello"}'

    def test_plain_fence(self):
        raw = '```\n{"tool": "Search", "query": "hello"}\n```'
        result = extract_json(raw)
        assert result == '{"tool": "Search", "query": "hello"}'

    def test_fence_with_extra_whitespace(self):
        raw = '```json\n  {"a": 1}  \n```'
        result = extract_json(raw)
        assert result == '{"a": 1}'

    def test_fence_with_nested_braces(self):
        raw = '```json\n{"outer": {"inner": "val"}}\n```'
        result = extract_json(raw)
        assert result == '{"outer": {"inner": "val"}}'


class TestBracketScan:
    def test_json_buried_in_text(self):
        raw = 'Sure, I will call the tool. {"tool": "Search", "query": "test"} Let me proceed.'
        result = extract_json(raw)
        assert result == '{"tool": "Search", "query": "test"}'

    def test_first_object_extracted(self):
        raw = 'First: {"a": 1}. Second: {"b": 2}.'
        result = extract_json(raw)
        assert result == '{"a": 1}'

    def test_nested_object_correct_boundary(self):
        raw = 'Here: {"tool": "X", "args": {"a": 1, "b": {"c": 2}}}'
        result = extract_json(raw)
        assert result == '{"tool": "X", "args": {"a": 1, "b": {"c": 2}}}'

    def test_string_containing_brace(self):
        # Braces inside string values must not confuse the scanner.
        raw = '{"key": "value with { brace }"}'
        result = extract_json(raw)
        assert result == raw

    def test_escaped_quote_in_string(self):
        raw = '{"key": "say \\"hello\\""}'
        result = extract_json(raw)
        assert result == raw


class TestEdgeCases:
    def test_empty_string(self):
        assert extract_json("") is None

    def test_none_input(self):
        assert extract_json(None) is None  # type: ignore[arg-type]

    def test_plain_text_no_json(self):
        assert extract_json("I cannot help with that.") is None

    def test_unclosed_brace(self):
        assert extract_json('{"key": "value"') is None

    def test_single_brace(self):
        assert extract_json("{") is None

    def test_valid_json_integer(self):
        # A bare integer is valid JSON but not an object; expect None.
        result = extract_json("42")
        assert result is None
