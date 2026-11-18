"""One exception type for "the model's answer was rejected", distinct from code bugs.

A durable-run work node that parses/validates an LLM answer raises ``ModelOutputRejected`` when
the ANSWER fails the contract (body over the cap, missing refs, invalid JSON, ...). A fresh sample
can pass, so the shared retry policy (``services/orion-durable-runs/app/admitted_graph.py``,
``failed_attempt``) treats it as a bounded, retryable attempt. A plain ``ValueError`` /
``TypeError`` raised anywhere else is still a code or input bug and fails the run at once.

Subclasses ``ValueError`` so every existing ``except ValueError`` caller keeps catching it.
"""
from __future__ import annotations


class ModelOutputRejected(ValueError):
    """The model's answer failed parsing or validation; a new sample may pass."""
