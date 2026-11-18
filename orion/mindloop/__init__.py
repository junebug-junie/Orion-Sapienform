"""A loop library for small resident minds in orion-mind (Orion Prime step 2C', patch 1).

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md section 10. One runtime, many
instances: the tick, the Juniper-first yield, the self-monitor gate on the thought, the bid that binds, the
model call (seat wait, no-model classification and backoff, starved route), the thought act opened only once
the model was reached, act publishing and per-instance state in Redis live here. An instance (Orion Prime is
#1, ``services/orion-mind/app/prime_loop.py``) supplies its identity (``LoopIdentity``), its state model, its
frame, its answer and what the answer does.

Pure plus an async harness: every I/O object (bus, model client, bid, release, clock, Juniper's turns) is
injected. Nothing here imports an instance (``orion.prime``); a test checks that.
"""
