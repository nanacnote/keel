"""
Logging helpers for the LLM Tool Framework.

Provides a one-call setup so consumers get consistent, readable output
without writing boilerplate ``logging.basicConfig`` blocks themselves.
"""

from __future__ import annotations

import logging


def configure_debug_logging(level: int = logging.DEBUG) -> None:
    """Configure the root logger with the framework's standard format.

    Call this once at the top of any script or example that wants structured
    framework debug output::

        from keel.utils.logging import configure_debug_logging
        configure_debug_logging()

    Args:
        level: The logging level to apply to the root logger.
               Defaults to ``logging.DEBUG`` so all framework internals
               (attempt traces, raw LLM output, dispatch errors) are visible.
               Pass ``logging.INFO`` for quieter production-style output.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
