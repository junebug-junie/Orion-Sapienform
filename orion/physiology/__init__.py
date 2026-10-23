"""Orion's body economy (design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md).

Pure, I/O-free modules. Hosts (orion-durable-runs) do the reads and pass plain values in.
``body.py`` is the Body organ (per-constraint readings). ``prices.py`` is the seed of the Budget
organ (price math); it is read by nothing live yet -- ``scripts/analysis/replay_body_economy.py``
is its only caller.
"""
