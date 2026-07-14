import pytest

import orion.substrate.attention_broadcast as ab
from orion.substrate.attention.common import stable_id


class _Node:
    def __init__(self, node_id, label, pressure):
        self.node_id = node_id
        self.label = label
        self.metadata = {"dynamic_pressure": pressure}
        self.signals = None


@pytest.fixture(autouse=True)
def _reset_broadcast_state():
    ab._recent_selected_counts.clear()
    ab._first_selected_at.clear()
    ab._dwell_ticks = 0
    ab._coalition_history.clear()
    ab._current_active_coalition = None
    ab._transition_history.clear()
    yield
    ab._recent_selected_counts.clear()
    ab._first_selected_at.clear()
    ab._dwell_ticks = 0
    ab._coalition_history.clear()
    ab._current_active_coalition = None
    ab._transition_history.clear()


def test_broadcast_history_tracks_selection(monkeypatch):
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    monkeypatch.setenv("ORION_ATTENTION_HABITUATION_ENABLED", "true")
    ab._recent_selected_counts.clear()
    ab._dwell_ticks = 0
    nodes = [_Node("n1", "runaway anomaly", 0.95)]
    for _ in range(3):
        frame = ab.build_substrate_attention_frame(nodes=nodes)
        ab.broadcast_projection_from_frame(frame)
    assert sum(ab._recent_selected_counts.values()) >= 1


def test_rumination_lock_breaks_on_produced_path(monkeypatch):
    """End-to-end: a repeatedly-selected stuck loop must lose the coalition on
    the REAL producer path (build_substrate_attention_frame +
    broadcast_projection_from_frame), with resonance derived by _current_history
    from selection counts — NOT hand-injected.
    """
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    monkeypatch.setenv("ORION_ATTENTION_HABITUATION_ENABLED", "true")
    ab._recent_selected_counts.clear()
    ab._dwell_ticks = 0

    stuck_node = _Node("n-stuck", "runaway anomaly loop", 0.95)
    fresh_node = _Node("n-fresh", "fresh competitor thread", 0.75)
    nodes = [stuck_node, fresh_node]

    stuck_loop_id = stable_id("open-loop", stuck_node.label.lower())
    fresh_loop_id = stable_id("open-loop", fresh_node.label.lower())

    selections: list[str | None] = []
    for _ in range(15):
        frame = ab.build_substrate_attention_frame(nodes=nodes)
        proj = ab.broadcast_projection_from_frame(frame)
        selections.append(proj.selected_open_loop_id)

    # The stuck loop must dominate initially (it is the strongest signal).
    assert selections[0] == stuck_loop_id, selections
    # The lock must BREAK on the produced path: the competitor is eventually
    # selected at least once, i.e. the stuck loop is no longer the exclusive
    # selection. This can only happen because _current_history() derived
    # resonance from the selection counts (Part A). No hand-injected resonance.
    assert fresh_loop_id in selections, (
        f"lock did not break on produced path: {selections}"
    )
    assert not all(s == stuck_loop_id for s in selections), selections
