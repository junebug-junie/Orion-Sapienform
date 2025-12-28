"""Deprecated module.

Spark Introspector v2 moved to `worker.py` + V2 chassis.
This file is kept to avoid stale imports.
"""

from .worker import handle_candidate  # noqa: F401
