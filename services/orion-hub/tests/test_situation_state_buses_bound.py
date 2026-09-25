"""Hub must bind every situation store its unified turns read.

Hub builds each Orion-mode turn's situation brief in its own process
(orion.hub.turn_orchestrator -> build_situation_for_ctx). Until 2026-09-25
its startup bound only the affect store, so the conversation-phase read was
unbound on every unified turn ("Conversation phase: unknown", and the user's
turn was never recorded). Source-level, same precedent as
test_memory_pg_pool.py's startup wiring check: running startup_event with a
live bus would drag in the rest of Hub's boot. That the helper covers every
store is gated in orion/situational/tests/test_state_buses.py.
"""

from __future__ import annotations

from pathlib import Path


def _startup_source() -> str:
    src = (Path(__file__).resolve().parents[1] / "scripts" / "main.py").read_text()
    return src.split("async def startup_event", 1)[1].split("\nasync def ", 1)[0]


def test_startup_binds_situation_stores_through_shared_helper() -> None:
    startup = _startup_source()
    assert "from orion.situational.state_buses import bind_situation_state_buses" in startup
    assert "bind_situation_state_buses(bus)" in startup


def test_startup_does_not_bind_situation_stores_piecemeal() -> None:
    """A single hand-picked bind is how the phase store got left out."""
    startup = _startup_source()
    for piecemeal in (
        "bind_session_turn_phase_bus(",
        "bind_juniper_affect_state_bus(",
        "bind_identity_ask_cooldown_bus(",
    ):
        assert piecemeal not in startup
