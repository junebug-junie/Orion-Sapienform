import orion.substrate.attention_broadcast as ab


class _Node:
    def __init__(self, node_id, label, pressure):
        self.node_id = node_id
        self.label = label
        self.metadata = {"dynamic_pressure": pressure}
        self.signals = None


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
