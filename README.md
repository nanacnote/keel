# Keel

A production-grade Python framework for **validated LLM tool use**. Add a new tool by writing one Pydantic class — the framework handles prompting, validation, and self-correction automatically.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

**1. Define a tool** — one Pydantic class, nothing else to touch:

```python
from keel.core.registry import BaseTool, ToolRegistry
from pydantic import Field

registry = ToolRegistry()

@register_tool(registry=registry)
class Search(BaseTool):
    """Search the web."""
    tool_name = "Search"
    query: str = Field(description="Search query")

    def execute(self):
        return f"results for: {self.query}"
```

**2. Implement an LLM adapter:**

```python
from keel.core.engine import LLMInterface

class MyLLM(LLMInterface):
    flat_input = True  # engine pre-serialises messages to a flat string before calling complete()
    def complete(self, messages):
        return '{"tool": "Search", "query": "hello"}'
```

**3. Wire up and run:**

```python
from keel.core.dispatcher import Dispatcher
from keel.core.engine import Engine

engine = Engine(
    llm=MyLLM(),
    registry=registry,
    dispatcher=Dispatcher(registry),
)
result = engine.run("search for hello")
```

**Optional — middleware hooks** run before and after every `tool.execute()`:

```python
from keel.core.middleware import MiddlewareChain, MiddlewareHook

class AuditLogger(MiddlewareHook):
    def pre_execute(self, tool): ...
    def post_execute(self, tool, result): ...

engine = Engine(..., middleware=MiddlewareChain([AuditLogger()]))
```

Common uses: structured logging, latency tracking, cost estimation, security sandboxing.

See [`examples/basic_usage.py`](examples/basic_usage.py) for a full multi-scenario demo.

## Architecture

The five modules you interact with directly:

```
framework/core/
├── registry.py      # BaseTool, ToolRegistry, @register_tool
├── dispatcher.py    # JSON extraction → validation pipeline
├── engine.py        # LLMInterface ABC + 3-level retry loop
├── middleware.py    # Pre/post execution hooks
└── human_review.py  # Escalation queue + HumanReviewTool
```

### Self-correction loop

On each LLM response the engine tries to dispatch it. Failures trigger a targeted correction message and retry:

| Error                    | Correction sent back to LLM                      |
| ------------------------ | ------------------------------------------------ |
| No JSON found            | Shows the raw malformed output                   |
| Unknown tool name        | Lists all registered tools                       |
| Pydantic validation fail | Lists each field, the problem, and the bad value |

Exhausted retries return a structured `FailureReport` instead of raising.

## Human review escalation

When the model can't confidently pick a tool, or all retries are exhausted, the framework can route the case to a human reviewer. Two paths trigger escalation:

- **Model-requested** — the model calls `HumanReviewTool` explicitly (it appears in the prompt like any other tool).
- **Failure-driven** — `Engine` exhausts `max_retries` and auto-forwards the `FailureReport`.

Implement a queue by subclassing `HumanReviewQueue` and pass it to the engine:

```python
from keel.core.human_review import HumanReviewQueue, HumanReviewRequest

class SlackQueue(HumanReviewQueue):
    def submit(self, request: HumanReviewRequest) -> None:
        # post to Slack, write to DB, open a ticket…
        pass

engine = Engine(
    llm=my_llm,
    registry=registry,
    dispatcher=dispatcher,
    human_review_queue=SlackQueue(),
)
```

`LoggingHumanReviewQueue` writes structured `WARNING` lines and is a safe default during development. Register `HumanReviewTool` in your registry so the model knows it can ask for help.

## Running tests

```bash
pytest tests/ -v
```

