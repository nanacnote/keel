"""
Prompt templates — the single source of truth for all natural language sent to LLMs.

Every string that influences model behaviour lives here.  To tune how the
framework communicates with a model, change this file only.  No other module
contains user-visible natural language.

Design principles (optimised for small models like SmolLM, but work with any):
- Concrete over abstract: per-tool examples, not just templates.
- Flat text: no markdown formatting that small models try to reproduce.
- Brevity: human-readable field lists, not raw JSON Schema dumps.
- Closing anchors: core constraint repeated at the end of every prompt.
- Schema flattening: $ref / allOf / anyOf resolved to plain {type, enum, …}.

Sections
--------
1. system_prompt           — chat-format system message (tool list + call format).
2. serialize_messages      — flatten chat-format messages into a single prompt string.
3. syntax_correction       — correction after a response with no valid JSON.
4. unknown_tool_correction — correction after the model names a non-existent tool.
5. schema_correction       — correction after Pydantic field validation fails.
6. failure_reason          — reason string for HumanReviewRequest on retry exhaustion.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from keel.core.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Internal helpers — JSON Schema flattening
# ---------------------------------------------------------------------------
# Pydantic v2 emits $defs / $ref / allOf / anyOf for enums, Optionals, and
# Literal types.  Small models cannot follow these indirections.  The helpers
# below resolve everything into flat {type, description, default, enum, …}
# property dicts so the prompt never contains $ref.
# ---------------------------------------------------------------------------


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a ``$ref`` string like ``#/$defs/Foo`` into the definition."""
    parts = ref.lstrip("#/").split("/")
    node: Any = {"$defs": defs}
    for p in parts:
        node = node[p]
    return dict(node)


def _resolve_property(prop: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Flatten a single JSON Schema property, resolving $ref / allOf / anyOf."""
    # Direct $ref → inline the definition.
    if "$ref" in prop:
        resolved = _resolve_ref(prop["$ref"], defs)
        result = _resolve_property(resolved, defs)
        for k in ("default", "description"):
            if k in prop:
                result[k] = prop[k]
        return result

    # allOf — Pydantic uses this for enums with defaults.
    if "allOf" in prop:
        merged: dict[str, Any] = {}
        for sub in prop["allOf"]:
            merged.update(_resolve_property(sub, defs))
        for k in ("default", "description", "title"):
            if k in prop:
                merged[k] = prop[k]
        return merged

    # anyOf — Pydantic uses this for Optional[X]: anyOf: [{type}, {type: null}].
    if "anyOf" in prop:
        for sub in prop["anyOf"]:
            if sub.get("type") != "null":
                resolved = _resolve_property(sub, defs)
                for k in ("default", "description"):
                    if k in prop:
                        resolved[k] = prop[k]
                return resolved
        return {
            "type": "null",
            **{k: prop[k] for k in ("default", "description") if k in prop},
        }

    # Already a flat property — keep the fields we care about.
    return {
        k: prop[k]
        for k in (
            "type",
            "description",
            "default",
            "enum",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "const",
            "title",
        )
        if k in prop
    }


def _flatten_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve all $ref / allOf / anyOf in a Pydantic JSON Schema to flat dicts."""
    defs = schema.get("$defs", {})
    return {
        "properties": {
            name: _resolve_property(prop, defs)
            for name, prop in schema.get("properties", {}).items()
        },
        "required": schema.get("required", []),
        "type": "object",
    }


# ---------------------------------------------------------------------------
# Internal helpers — field and example rendering
# ---------------------------------------------------------------------------

_TYPE_EXAMPLES: dict[str, Any] = {
    "integer": 1,
    "number": 0.0,
    "boolean": False,
}


def _field_line(name: str, prop: dict[str, Any], is_required: bool) -> str:
    """Render one field as a compact, human-readable line."""
    ftype = prop.get("type", "string")
    desc = prop.get("description", "")
    default = prop.get("default")

    # Tag: type + required / default / optional
    if is_required:
        tag = f"{ftype}, REQUIRED"
    elif default is not None:
        tag = f"{ftype}, default: {json.dumps(default)}"
    else:
        tag = f"{ftype}, optional"

    line = f"    {name} ({tag})"
    if desc:
        line += f" — {desc}"

    # Range constraints
    lo = prop.get("minimum")
    hi = prop.get("maximum")
    if lo is not None and hi is not None:
        line += f" Range: {lo}\u2013{hi}."
    elif lo is not None:
        line += f" Min: {lo}."
    elif hi is not None:
        line += f" Max: {hi}."

    # Enum / Literal values
    if "enum" in prop:
        line += f" Allowed values: {json.dumps(prop['enum'])}."

    return line


def _example_value(prop: dict[str, Any], field_name: str) -> Any:
    """Pick a representative value for an example JSON call."""
    if "enum" in prop:
        return prop["enum"][0]
    default = prop.get("default")
    if default is not None:
        return default
    ftype = prop.get("type", "string")
    if ftype == "string":
        return f"<{field_name}>"
    if ftype in _TYPE_EXAMPLES:
        lo = prop.get("minimum")
        return lo if lo is not None else _TYPE_EXAMPLES[ftype]
    return f"<{field_name}>"


def _tool_example_json(name: str, schema: dict[str, Any]) -> str:
    """Build a concrete example JSON call for a tool, including optional defaults."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    ex: dict[str, Any] = {"tool": name}
    for fname, fschema in props.items():
        if fname in required:
            ex[fname] = _example_value(fschema, fname)
        else:
            default = fschema.get("default")
            if default is not None:
                ex[fname] = default
    return json.dumps(ex)


# ---------------------------------------------------------------------------
# Internal helpers — text utilities
# ---------------------------------------------------------------------------

_BAD_OUTPUT_MAX = 300
_TRUNCATION_SUFFIX = " …[truncated]"


def _truncate(text: str, limit: int = _BAD_OUTPUT_MAX) -> str:
    """Truncate *text* if it exceeds *limit* characters."""
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_SUFFIX


# ---------------------------------------------------------------------------
# 1. Standard system prompt (chat-format adapters)
# ---------------------------------------------------------------------------

_SYSTEM_HEADER = """\
You are a tool-calling assistant. Read the user's request and respond with \
exactly one JSON object that calls a tool. Do not write anything else.

RESPONSE FORMAT:
{"tool": "<tool_name>", "arg1": value1, "arg2": value2}

RULES:
- Your entire response must be a single JSON object. No explanation, no markdown, no extra text.
- The "tool" field must exactly match one of the tool names listed below.
- Include all REQUIRED fields. Optional fields can be omitted.
"""


def system_prompt(registry: ToolRegistry) -> str:
    """Build the chat-format system message for *registry*.

    Each tool gets a one-line summary, a human-readable field list (not raw
    JSON Schema), and a concrete example call.  The raw Pydantic schema is
    flattened so $ref / allOf / anyOf never appear in the prompt text.
    """
    lines: list[str] = [_SYSTEM_HEADER.rstrip(), ""]
    lines.append("TOOLS:")
    lines.append("")

    for name in registry.list_tools():
        cls = registry._tools[name]  # noqa: SLF001 — prompts is a peer of registry
        raw_schema = cls.get_schema()
        schema = _flatten_schema(raw_schema)
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Tool summary from docstring (first paragraph only).
        doc = (cls.__doc__ or "").strip()
        summary = doc.split("\n\n")[0].replace("\n", " ").strip() if doc else name
        ns = cls.namespace

        lines.append(f"  {name} ({ns}): {summary}")
        lines.append("  Fields:")
        for fname, fschema in props.items():
            lines.append(_field_line(fname, fschema, fname in required))

        lines.append(f"  Example: {_tool_example_json(name, schema)}")
        lines.append("")

    lines.append("Respond with ONLY a JSON object. No other text.")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# 2. Message serialiser (for adapters that require a single flat string)
# ---------------------------------------------------------------------------

# Role labels used when flattening chat messages to a single string.
_ROLE_LABELS: dict[str, str] = {
    "system": "INSTRUCTIONS",
    "user": "USER",
    "assistant": "ASSISTANT",
}


def serialize_messages(messages: list[dict[str, str]]) -> str:
    """Flatten chat-format *messages* into a single prompt string.

    The engine builds ``messages`` the same way for every backend — this
    function just shapes that content for adapters whose inference API accepts
    a single string rather than a structured message list.

    The final ``ASSISTANT:`` line with no content primes the model to generate
    its JSON response directly.

    Args:
        messages: Conversation turns in chat format.  Each dict must
                  have ``role`` (system / user / assistant) and ``content``.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        label = _ROLE_LABELS.get(role, role.upper())
        parts.append(f"{label}:\n{content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 3. Syntax correction
# ---------------------------------------------------------------------------


def syntax_correction(raw: str) -> str:
    """Correction message sent back to the LLM after a response with no valid JSON.

    Kept short and direct — no markdown formatting that small models might
    try to reproduce.  The bad output is truncated to avoid copy-paste loops.
    """
    snippet = _truncate(raw, 200)
    return (
        "Your previous response was not valid JSON. "
        "I need a JSON object, not plain text.\n\n"
        f"You wrote: {snippet}\n\n"
        "Correct format:\n"
        '{"tool": "<tool_name>", "<arg>": <value>}\n\n'
        "Respond with ONLY a valid JSON object. Nothing else."
    )


# ---------------------------------------------------------------------------
# 4. Unknown-tool correction
# ---------------------------------------------------------------------------


def unknown_tool_correction(attempted: str, available: list[str]) -> str:
    """Correction message sent back after the model names a tool that doesn't exist.

    Lists available tools in a flat comma-separated format (not bulleted —
    small models may include the bullets in their response).  Includes a
    concrete example with the first available tool to anchor the format.
    """
    tool_list = ", ".join(available)
    lines = [
        f"The tool `{attempted}` does not exist.",
        "",
        f"Available tools: {tool_list}",
        "",
        "Pick one of the tools above and respond with ONLY a valid JSON object.",
    ]
    if available:
        lines.append(f'Example: {{"tool": "{available[0]}", ...}}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Schema correction
# ---------------------------------------------------------------------------


def schema_correction(tool_name: str, errors: list[dict[str, Any]]) -> str:
    """Correction message sent back after Pydantic field validation fails.

    Names the exact fields that are wrong, what the error was, and what
    the model provided.  Keeps backtick quoting for field names so they
    stand out in the text without heavy markdown.
    """
    lines = [
        f"Your JSON called `{tool_name}` but some fields have wrong values.",
        "",
        "Fix these fields:",
    ]
    for err in errors:
        loc = ".".join(str(p) for p in err.get("loc", [])) or "(root)"
        msg = err.get("msg", "")
        input_val = err.get("input")
        lines.append(
            f"  - `{loc}`: {msg} (you gave: {json.dumps(input_val, default=str)})"
        )
    lines.append("")
    lines.append(
        f"Respond with the corrected JSON for `{tool_name}`. "
        f"Fix only the bad fields, keep everything else."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Human review failure reason
# ---------------------------------------------------------------------------


def failure_reason(attempts: int) -> str:
    """Reason string for a HumanReviewRequest triggered by retry exhaustion."""
    return f"Engine exhausted {attempts} attempt(s) without a valid tool call."
