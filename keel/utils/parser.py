"""
Robust JSON extraction from raw LLM output.

Handles three cases:
1. The response IS valid JSON already.
2. The JSON is wrapped in a markdown code fence (```json ... ``` or ``` ... ```).
3. The JSON is buried in conversational text — found via bracket-balancing scan.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> str | None:
    """Return the first valid JSON object string found in *text*, or ``None``.

    The returned value is a raw string, not a parsed dict. Callers are
    responsible for calling ``json.loads()`` on the result.

    Strategy (tried in order, stops at first success):
    1. Strip whitespace and attempt a direct parse.
    2. Strip a markdown code fence and attempt a direct parse.
    3. Bracket-balancing scan to isolate the outermost ``{...}`` block.
    """
    if not text or not isinstance(text, str):
        return None

    stripped = text.strip()

    # --- Fast path: the entire string is already valid JSON object ---
    if _is_valid_json_object(stripped):
        return stripped

    # --- Markdown fence: ```json ... ``` or ``` ... ``` ---
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        if _is_valid_json_object(candidate):
            return candidate

    # --- Bracket-balancing scan ---
    candidate = _bracket_scan(stripped)
    if candidate is not None and _is_valid_json_object(candidate):
        return candidate

    return None


def _is_valid_json_object(text: str) -> bool:
    """Return True if *text* is parseable as a JSON object (dict)."""
    try:
        return isinstance(json.loads(text), dict)
    except (json.JSONDecodeError, ValueError):
        return False


def _bracket_scan(text: str) -> str | None:
    """Find the outermost ``{...}`` block in *text* using bracket counting.

    Correctly handles nested objects and strings containing braces.
    Returns the raw substring, or ``None`` if no balanced block is found.
    """
    start: int | None = None
    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]

    return None
