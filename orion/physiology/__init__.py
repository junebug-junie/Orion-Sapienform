"""Orion's body economy (design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md).

Only the pure price math lives here so far (``prices.py``, the seed of the Budget organ).
It does no I/O and is read by nothing live yet; ``scripts/analysis/replay_body_economy.py``
is its only caller.
"""
