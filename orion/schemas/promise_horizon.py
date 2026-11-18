"""The one 72 h clock for a promise Orion makes (#3595: there were three copies with no shared symbol).

Juniper fixed 72 h as the horizon for every promise on 2026-11-12 (intention ledger, answer 2) and accepted
72 h for intentions on 2026-11-13. Three readers use it, and all three import it from here:

* the promise ledger's act horizon (orion/grammar/promise_acts.py ``PROMISE_HORIZON_SEC``);
* Prime's promise concern life (orion/prime/state.py ``PROMISE_CONCERN_TTL``);
* the carry turn's horizon for a promise Prime takes on (orion/prime/intention_sources/turn_promise.py
  ``HORIZON_SEC``).

Not this clock: the 24 h turn-concern slot (orion/prime/handback.py ``holds_turn_slot`` uses ``CONCERN_TTL``),
which frees the one turn-concern slot 24 h after the last evidence while the promise is still held for 72 h.
"""

from __future__ import annotations

PROMISE_HORIZON_SEC: float = 72 * 3600.0

__all__ = ["PROMISE_HORIZON_SEC"]
