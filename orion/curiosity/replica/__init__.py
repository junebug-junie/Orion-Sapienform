"""The curiosity replica: a small planner that writes the top of the offer order on its own arm.

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md. Pure code only (no I/O):

* ``candidates``: what Hub may offer this run, as the replica's candidate list (section 2).
* ``plan``: the deterministic plan check Hub runs before using a plan, and the plan row (section 4).
* ``frame``: what the replica reads before it plans, its system prompt and its answer's parse (sections 2-3).
* ``track_record``: how a plan is graded on its run, and the replica's own record (sections 1 and 6).

The instance itself (bid, model call, act) is ``services/orion-mind/app/replica_loop.py`` on the loop
library ``orion/mindloop/``; Hub's consumer is ``services/orion-hub/scripts/curiosity_investigation.py``.
"""
