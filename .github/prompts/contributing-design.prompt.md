---
agent: ask
description: Coding style, design philosophy, and contribution standards for the keel codebase.
---

# Contributing to keel — Design DNA

This document captures the spirit behind keel's design. Before writing a single line, read it. Code that doesn't fit this grain will not be merged — not because the rules say so, but because keel has a coherent identity worth protecting.

---

## What keel is trying to do

keel solves one narrow, hard problem: making tool-calling work reliably with small, imperfect LLMs. Every design decision traces back to that. If your change doesn't serve that constraint, question whether it belongs here at all.

---

## The first principle: elegance over cleverness

Elegance is not the same as brevity, and it is not the same as cleverness. Elegant code is code that, once read, could not reasonably have been written any other way. Aim for that feeling. If a reader has to stare at something to understand it, it isn't elegant — rewrite it.

Clever tricks that save three lines but obscure intent are worse than three honest lines. This codebase is small and must stay readable to anyone with intermediate Python fluency.

---

## Lean and typed

**Add no more than what the problem demands.** If a feature is not needed to solve the stated problem, it doesn't exist yet. Resist the pull toward generality. One good abstraction beats three premature ones.

**Type everything.** Use `X | Y` union syntax, not `Optional[X]`. Use `ClassVar` when a field belongs to the class, not an instance. Use `Literal["a", "b"]` instead of a full `Enum` when the set is small and stable. Use `__slots__` on internal state containers. If it crosses a function boundary, it has a type annotation.

**Pydantic is the validation layer.** Use its field constraints (`ge=`, `le=`, `min_length=`) instead of writing runtime validation in business logic. The schema is the contract.

---

## Structure

### Single responsibility, strictly enforced

Each module owns exactly one concept:

- `registry.py` — storage and lookup only. No execution, no prompt generation.
- `dispatcher.py` — parse raw → validate. Nothing else.
- `engine.py` — the retry loop and conversation state. Not parsing, not storage.
- `prompts.py` — every string the LLM ever sees. Nowhere else.
- `middleware.py` — observer hooks. Not execution logic.

If you find yourself needing to reach sideways into another module's concerns, stop. The seam is in the wrong place. Restructure the seam, don't paper over it.

### Exceptions as typed structured data

Failures are not strings. Every distinct failure mode gets its own exception class that carries structured fields:

```python
class SchemaDispatchError(DispatchError):
    def __init__(self, tool_name: str, errors: list[dict[str, Any]]) -> None:
        self.tool_name = tool_name
        self.errors = errors
```

The engine catches by type and routes to the right correction prompt. This makes control flow explicit and the retry logic readable. Never raise `Exception("something went wrong")`.

### Module-level section dividers

Use the three-line comment block for visual structure inside files:

```python
# -----------------------------------------------------------------------
# Section name
# -----------------------------------------------------------------------
```

This is the file's table of contents. Use it consistently. The whole codebase does.

---

## The prompt module is sacred

`prompts.py` is the single source of truth for every natural-language string the model sees. This is not a convention — it's a hard rule. If you are adding new LLM-facing text anywhere other than `prompts.py`, you are doing it wrong.

The same applies to schema rendering. The flattening logic in `prompts.py` exists specifically to produce output that small models can follow: no `$ref` indirection, no raw Pydantic `$defs`, flat field descriptions. If you add a new tool field type, extend the flattening instead of letting raw schema leak into prompts.

---

## Circular imports: use the established patterns

Two patterns exist for modules that would otherwise create circular imports:

1. `TYPE_CHECKING` guard for forward references — import only for annotation purposes, not at runtime.
2. Local import inside a method body when a runtime import is genuinely needed in only one place.

Both patterns appear in the existing code. Use them. Do not restructure modules to avoid needing them if the restructuring otherwise makes things worse.

When you need to access a private attribute from a peer module under a recognized shared-module relationship, annotate the line with `# noqa: SLF001` and add a comment explaining why. Don't hide it.

---

## Testing

Three tiers, each with a distinct job:

**Unit tests** — isolate one class, validate every typed exception and every edge case. Keep setup local. Use factory functions (`_make_engine`, `_build_registry`) over fixtures when setup is complex and scenario-specific. Fixtures in `conftest.py` are for shared scaffolding only.

**Integration tests** — use `MockLLM` from `conftest.py` to test the engine + dispatcher + queue together. Verify that the right fields reach the right place, not just that no exception was raised.

**Chaos tests** — simulate broken LLM output. Inspect `llm.received_messages` to assert what correction content was actually sent. White-box the state machine. This is where you prove that error recovery works, not just that it doesn't crash.

Do not ship a feature without chaos coverage if the feature touches the retry loop or error handling. Untested recovery logic is broken recovery logic.

---

## What to avoid

**Don't add features for potential future use.** If there's no concrete test case for it, it shouldn't exist yet.

**Don't use inheritance where composition or a Protocol works.** The `MiddlewareHook` Protocol is intentionally structural — no import, no coupling. Follow that lead. ABCs exist but shouldn't multiply.

**Don't invent new logging formats.** The log format is `key=value  key2=value2` with double-space separation. Use it.

**Don't let LLM-facing strings drift into application logic.** Every formatted message, correction prompt, and system block belongs in `prompts.py`. Full stop.

**Don't catch broad exceptions silently.** The `MiddlewareChain` collects errors and re-raises them as an `ExceptionGroup` so nothing disappears. Follow that discipline.

**Don't write a `BaseTool` subclass that does real work in `execute()`** inside the framework itself. Execution is user territory. The framework's job ends at dispatch and retry.

---

## The final check

Before submitting, ask yourself: does this code read like the rest of keel? Would someone picking up this file for the first time understand what it does and why within two minutes? Is there anything in here that exists only because it was easier to add than to think harder?

If the answer to any of those is no, keep working.
