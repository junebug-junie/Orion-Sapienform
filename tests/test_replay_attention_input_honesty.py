"""The 2026-09-29 replay scripts reproduce the two defects they report on."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fade = _load("replay_candidate_a_staleness_fade")
gw = _load("replay_llm_inference_failure_window")

T0 = datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc)


def test_staleness_replay_moves_top1_off_a_stale_chat_reading(tmp_path) -> None:
    rows = []
    for i in range(3):
        gen = T0 + timedelta(minutes=i)
        rows.append(
            f"{gen.isoformat()}\tnode:substrate.chat\t0.8\tprecision-weighted ... (current error 0.8000, precision 10.00, n=300)\t"
            f"{(T0 - timedelta(hours=1)).isoformat()}"
        )
        rows.append(
            f"{gen.isoformat()}\tnode:substrate.execution\t0.3\tprecision-weighted ... (current error 0.3000, precision 10.00, n=300)\t"
            f"{gen.isoformat()}"
        )
    path = tmp_path / "frames.tsv"
    path.write_text("\n".join(rows) + "\n")
    r = fade.replay(fade.load(path))
    assert r["before"]["node:substrate.chat"] == 3 and r["before_stale"]["node:substrate.chat"] == 3
    assert r["after"]["node:substrate.execution"] == 3 and r["changed"] == 3


def _event(i: int, start: datetime, summary: str, role: str = "llm_inference_window_observed") -> dict:
    wid = start.strftime("%Y%m%dT%H%M%SZ")
    trace = f"llm_gateway.inference:gateway:{wid}"
    eid = f"{trace}:{i:02d}:{role}"
    end = (start + timedelta(seconds=60)).isoformat()
    return {
        "event_id": eid, "event_kind": "atom_emitted", "trace_id": trace, "emitted_at": end,
        "atom": {"atom_id": eid, "trace_id": trace, "atom_type": "observation", "semantic_role": role,
                 "layer": "inference", "summary": summary},
        "provenance": {"source_service": "orion-llm-gateway"},
    }


def test_gateway_replay_single_timeout_reads_full_failure_before_and_zero_after(tmp_path) -> None:
    lines = [
        _event(0, T0, "node=circe calls=1 served=0 upstream_failed=1 workers=circe-worker-agent classes=upstream_timeout:1"),
        _event(1, T0, "gateway=gateway calls=1 nodes=1 window_sec=60.0", role="llm_gateway_window_completed"),
        _event(0, T0 + timedelta(minutes=3), "node=circe calls=5 served=5 upstream_failed=0 workers=circe-worker-chat classes=served:5"),
    ]
    path = tmp_path / "gw.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    r = gw.replay(gw.load(path))
    old = gw.held_minutes(r["old"]["llm_node:circe"], r["start"], r["end"])
    new = gw.held_minutes(r["new"]["llm_node:circe"], r["start"], r["end"])
    assert max(v for v in old if v is not None) == 1.0
    assert max(v for v in new if v is not None) == 0.0
