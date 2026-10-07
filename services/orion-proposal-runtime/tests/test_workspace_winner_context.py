"""Proposal runtime: reads the workspace winner, and only then the world (attend-to-act loop)."""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from orion.autonomy.cabinet_heat import CABINET_NODE_ID
from orion.hardware_watch.rules import TempPoint
from orion.proposals.policy import load_proposal_policy

NOW = datetime.now(timezone.utc)


class _Store:
    def __init__(self, node=CABINET_NODE_ID, in_flight=(), occupancy=(1, 0)):
        self.node, self._in_flight, self._occ, self.world_reads = node, list(in_flight), occupancy, 0

    def load_broadcast_projection(self):
        return ({"generated_at": (NOW - timedelta(seconds=20)).isoformat(), "selected_action_type": "watch",
                 "selected_open_loop_id": "open-loop-cab", "dwell_ticks": 3, "attended_node_ids": [self.node]},
                "broadcast-1")

    def load_cabinet_points(self, since, until):
        self.world_reads += 1
        return [TempPoint(NOW - timedelta(seconds=30 * (40 - i)), 29.6 + 0.8 * i / 40) for i in range(41)]

    def load_background_occupancy(self):
        return self._occ

    def load_world_episodes_in_flight(self, template, now):
        return self._in_flight


def _worker(store, monkeypatch, health=None):
    import app.worker as wm

    w = object.__new__(wm.ProposalRuntimeWorker)
    w._store = store
    w._policy = load_proposal_policy(Path(__file__).resolve().parents[3] / "config/proposals/proposal_policy.v1.yaml")
    w._settings = SimpleNamespace(world_action_rise_threshold_c=0.5, hardware_watch_health_url="http://fake/health")
    body = health if health is not None else {"enabled": True, "last_tick_ok": True, "last_tick_at": NOW.isoformat(),
                                              "open_incidents": []}
    monkeypatch.setattr(wm, "urlopen", lambda url, timeout: io.BytesIO(json.dumps(body).encode()))
    return w


def test_bound_winner_gets_an_eligibility_snapshot(monkeypatch):
    ctx = _worker(_Store(), monkeypatch)._workspace_context(NOW)
    snap = ctx.eligibility["shed_background_gpu"]
    assert ctx.broadcast_log_id == "broadcast-1" and snap["eligible"], snap["refusals"]


def test_unbound_winner_reads_nothing_else(monkeypatch):
    store = _Store(node="node:substrate.chat")
    ctx = _worker(store, monkeypatch)._workspace_context(NOW)
    assert ctx.eligibility == {} and store.world_reads == 0


def test_active_reflex_and_unreachable_watcher_are_refusals(monkeypatch):
    # Thermal v2 (D8/C7): the reflex's own shed signal refuses; an alert-only incident no longer does.
    reflex = {"enabled": True, "last_tick_ok": True, "last_tick_at": NOW.isoformat(),
              "open_incidents": [{"incident_id": "abc", "rule": "cooling"}],
              "reflex_shed": {"active": True, "reason": "cabinet_hot"}}
    snap = _worker(_Store(), monkeypatch, health=reflex)._workspace_context(NOW).eligibility["shed_background_gpu"]
    assert "reflex_active:cabinet_hot" in snap["refusals"]
    alert_only = {**reflex, "reflex_shed": {"active": False, "reason": None}}
    snap = _worker(_Store(), monkeypatch, health=alert_only)._workspace_context(NOW).eligibility["shed_background_gpu"]
    assert not any(r.startswith(("reflex_active", "hardware_watch_incident")) for r in snap["refusals"])
    import app.worker as wm

    w = _worker(_Store(), monkeypatch)
    monkeypatch.setattr(wm, "urlopen", lambda url, timeout: (_ for _ in ()).throw(OSError("down")))
    snap = w._workspace_context(NOW).eligibility["shed_background_gpu"]
    assert "hardware_watch_unknown:unreachable" in snap["refusals"]


def test_ledger_unavailable_fails_closed(monkeypatch):
    store = _Store()
    store.load_world_episodes_in_flight = lambda template, now: (_ for _ in ()).throw(RuntimeError("no table"))
    snap = _worker(store, monkeypatch)._workspace_context(NOW).eligibility["shed_background_gpu"]
    assert "winner_loop_in_flight" in snap["refusals"]
