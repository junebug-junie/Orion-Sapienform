"""Smoke for scripts/replay_substrate_propagation_decay.py on a pickled in-memory snapshot."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "orion" / "substrate" / "tests"))

import replay_substrate_propagation_decay as replay  # noqa: E402
from test_dynamics_propagation_decay import T0, _edge, _node, _store  # noqa: E402


def test_replay_reports_the_day_node_rise_from_a_saved_snapshot(tmp_path, capsys):
    store = _store([_node("cluster", observed_at=T0, activation=0.61),
                    _node("day-node", observed_at=T0.replace(hour=6, minute=0), salience=0.3)],
                   [_edge("cluster", "day-node")])
    snap = tmp_path / "snap.pkl"
    snap.write_bytes(pickle.dumps((T0 + timedelta(minutes=1), store)))
    assert replay.main(["--load", str(snap), "--ticks", "2", "--long-hours", "24", "--long-step-min", "60"]) == 0
    out = json.loads(capsys.readouterr().out)
    rises = {r["node_id"]: r for r in out["short"]["top20_rises"]}
    assert rises["day-node"]["new"] > 0.25 > 0.1 > rises["day-node"]["old"]
    assert out["long"]["dist_new"]["above_dormancy_0.08"] == 0
    assert datetime.fromisoformat(out["snapshot_at"]).tzinfo == timezone.utc
