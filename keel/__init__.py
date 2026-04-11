"""
LLM Tool Framework — production-grade validated tool use with self-correcting retry loop.

Quick start::

    from keel.core.registry import register_tool, BaseTool
    from keel.core.engine import Engine, LLMInterface
    from keel.core.dispatcher import Dispatcher
    from pydantic import Field

    @register_tool
    class Search(BaseTool):
        tool_name = "Search"
        query: str = Field(description="Search query string")

        def execute(self):
            return f"Results for: {self.query}"

    class MyLLM(LLMInterface):
        def complete(self, messages):
            return '{"tool": "Search", "query": "hello world"}'

    engine = Engine(llm=MyLLM(), registry=..., dispatcher=...)
    result = engine.run("search for hello world")
"""

try:
    from keel._version import __version__
except ImportError:
    __version__ = "0.0.0"

from keel.core.registry import BaseTool, ToolRegistry, register_tool
from keel.core.dispatcher import Dispatcher
from keel.core.engine import Engine, LLMInterface, FailureReport
from keel.core.middleware import MiddlewareHook, MiddlewareChain

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "register_tool",
    "Dispatcher",
    "Engine",
    "LLMInterface",
    "FailureReport",
    "MiddlewareHook",
    "MiddlewareChain",
]
