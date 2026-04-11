"""
Dispatcher — turns raw LLM output into a validated BaseTool instance.

Pipeline (each step raises a typed exception on failure):
1. ``extract_json()``      — find and isolate a JSON block.
2. ``json.loads()``        — parse the JSON.
3. Extract ``"tool"`` key  — identify which tool was called.
4. ``registry.get(name)``  — look up the tool class (hallucination guard).
5. ``ToolClass(**args)``   — Pydantic validation; type coercion if strict=False.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from keel.core.registry import BaseTool, ToolRegistry, UnknownToolError
from keel.utils.parser import extract_json


# ---------------------------------------------------------------------------
# Exceptions  (re-export UnknownToolError so callers can import from here)
# ---------------------------------------------------------------------------

__all__ = [
    "Dispatcher",
    "DispatchError",
    "SyntaxDispatchError",
    "SchemaDispatchError",
    "UnknownToolError",
]


class DispatchError(Exception):
    """Base class for all dispatch failures."""


class SyntaxDispatchError(DispatchError):
    """The raw LLM output did not contain extractable, parseable JSON.

    Carries the original *raw* string so the engine can show it back to the
    LLM in the correction prompt.
    """

    def __init__(self, raw: str, message: str = "") -> None:
        self.raw = raw
        super().__init__(
            message or f"Could not extract valid JSON from output: {raw!r}"
        )


class SchemaDispatchError(DispatchError):
    """JSON was valid but failed Pydantic schema validation for *tool_name*.

    ``errors`` is a list of dicts, each containing:
    - ``loc``   — tuple of field path components (e.g. ``("query",)``).
    - ``msg``   — human-readable description of the problem.
    - ``input`` — the value that was actually provided.
    - ``type``  — the Pydantic error type string.
    """

    def __init__(self, tool_name: str, errors: list[dict[str, Any]]) -> None:
        self.tool_name = tool_name
        self.errors = errors
        field_summaries = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in errors
        )
        super().__init__(
            f"Schema validation failed for tool '{tool_name}': {field_summaries}"
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class Dispatcher:
    """Converts raw LLM text into a ready-to-execute ``BaseTool`` instance.

    Args:
        registry:    The ``ToolRegistry`` to look up tool classes from.
        strict:      When ``True``, Pydantic will reject type coercions (e.g.
                     string ``"5"`` for an ``int`` field).  When ``False``
                     (the default), lax mode coerces compatible types.
        tool_key:    The JSON key used to identify the tool name.
                     Defaults to ``"tool"``.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        strict: bool = False,
        tool_key: str = "tool",
    ) -> None:
        self._registry = registry
        self._strict = strict
        self._tool_key = tool_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(self, raw: str) -> BaseTool:
        """Parse *raw* and return a validated ``BaseTool`` instance.

        Raises:
            SyntaxDispatchError:  No valid JSON found, or JSON is malformed.
            UnknownToolError:     The tool name is not in the registry.
            SchemaDispatchError:  JSON is valid but fails Pydantic validation.
        """
        # --- Step 1 & 2: extract + parse JSON ---
        json_str = extract_json(raw)
        if json_str is None:
            raise SyntaxDispatchError(raw=raw)

        try:
            payload: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise SyntaxDispatchError(
                raw=raw,
                message=f"JSON parse error: {exc}",
            ) from exc

        if not isinstance(payload, dict):
            raise SyntaxDispatchError(
                raw=raw,
                message="Expected a JSON object (dict) at the top level.",
            )

        # --- Step 3: identify tool name ---
        tool_name = payload.get(self._tool_key)
        if not tool_name or not isinstance(tool_name, str):
            raise SyntaxDispatchError(
                raw=raw,
                message=(
                    f"JSON object must contain a '{self._tool_key}' key with a "
                    f"non-empty string value. Got: {tool_name!r}"
                ),
            )

        # --- Step 4: registry lookup ---
        tool_cls = self._registry.get(tool_name)  # raises UnknownToolError if absent

        # --- Step 5: build arg dict (everything except the tool key) ---
        args = {k: v for k, v in payload.items() if k != self._tool_key}

        # --- Step 6: Pydantic instantiation + validation ---
        try:
            if self._strict:
                instance = tool_cls.model_validate(args, strict=True)
            else:
                instance = tool_cls.model_validate(args)
        except ValidationError as exc:
            errors = self._extract_pydantic_errors(exc)
            raise SchemaDispatchError(tool_name=tool_name, errors=errors) from exc

        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pydantic_errors(exc: ValidationError) -> list[dict[str, Any]]:
        """Normalise a ``ValidationError`` into a list of structured dicts.

        Each dict has:
        - ``loc``   — list of path components (converted from tuple).
        - ``msg``   — the human-readable error message.
        - ``input`` — the value that failed validation.
        - ``type``  — the Pydantic error type string.
        """
        result = []
        for error in exc.errors():
            result.append(
                {
                    "loc": list(error.get("loc", [])),
                    "msg": error.get("msg", ""),
                    "input": error.get("input"),
                    "type": error.get("type", ""),
                }
            )
        return result
