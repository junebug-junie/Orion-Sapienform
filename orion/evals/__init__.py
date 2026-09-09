"""Shared, deterministic eval scorers -- pure functions with unit tests,
imported by the per-service `evals/run_*.py` runners that own the live call.

The split is the Patch A rule from
docs/superpowers/specs/2026-09-08-orion-sense-of-self-design.md: the runner
owns the live endpoint (Hub owns chat), the scorer owns the number. A scorer
here must never touch the network or a database.
"""
