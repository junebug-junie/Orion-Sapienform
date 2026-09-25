"""Bind every Redis-backed situation store to this process's bus, in one call.

`build_situation_for_ctx` reads three stores that each hold a module-level
bus handle and need it bound once per process: conversation phase
(`session_turn_phase`), Juniper's latest affect read (`juniper_affect_state`),
and the identity-ask cooldown (`identity_ask_cooldown`). Two processes build
situation briefs -- orion-cortex-exec (legacy chat verbs) and orion-hub (every
unified turn, via `orion.hub.turn_orchestrator`) -- and each used to bind the
stores by hand. Hub bound only the affect store, so every unified turn read
its conversation phase as "unknown" (`session_turn_phase_read_bus_unbound`)
and never recorded the user's turn: the prompt said "Conversation phase:
unknown" however long the gap had been.

Both processes now call this, so a store cannot be bound in one and forgotten
in the other. `tests/test_state_buses.py` fails if a module in this package
grows a `bind_*_bus` that this function does not call.
"""

from __future__ import annotations

from orion.core.bus.async_service import OrionBusAsync

from .identity_ask_cooldown import bind_identity_ask_cooldown_bus
from .juniper_affect_state import bind_juniper_affect_state_bus
from .session_turn_phase import bind_session_turn_phase_bus


def bind_situation_state_buses(bus: OrionBusAsync) -> None:
    bind_session_turn_phase_bus(bus)
    bind_juniper_affect_state_bus(bus)
    bind_identity_ask_cooldown_bus(bus)
