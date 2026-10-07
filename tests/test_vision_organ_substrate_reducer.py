"""vision_organ lane: router window traces -> node:substrate.vision_organ readings."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.vision_organ_projection import (
    ORGAN_REPORTING,
    ORGAN_SILENT,
    ROLE_STREAM_WINDOW,
    ROLE_WINDOW_COMPLETED,
    STREAM_LIVE,
    STREAM_NEVER_SEEN,
    STREAM_STALE,
    VISION_ORGAN_SOURCE_SERVICE,
    VisionOrganProjectionV1,
)
from orion.substrate.vision_organ_loop.constants import VISION_ORGAN_NODE_ID
from orion.substrate.vision_organ_loop.pipeline import (
    empty_vision_organ_projection,
    process_vision_organ_grammar_events,
)
from orion.substrate.vision_organ_loop.reducer import (
    reduce_vision_organ_trace_events,
    silence_age_seconds,
    vision_organ_silence_receipt,
)

REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc)


def _stream(
    name: str,
    *,
    frames: int = 700,
    age: float | None = 0.1,
    uptime: float = 3600.0,
    dispatched: int = 12,
    ok: int = 12,
    failed: int = 0,
    classes: str = "none",
    configured: int = 1,
) -> str:
    a = "none" if age is None else f"{age:.1f}"
    return (
        f"stream={name} configured={configured} frames={frames} last_frame_age_sec={a} "
        f"uptime_sec={uptime:.1f} dispatched={dispatched} identity_dispatched=0 "
        f"replies_ok={ok} failed={failed} failure_classes={classes} skips=frame_sampled_out:600 "
        f"detect_replies={ok} objects={ok * 6} caption_requested=0 captions=0"
    )


def _window(
    end: datetime,
    streams: list[str],
    *,
    closing: bool = True,
    source: str = VISION_ORGAN_SOURCE_SERVICE,
) -> list[GrammarEventV1]:
    window_id = (end - timedelta(seconds=60)).strftime("%Y%m%dT%H%M%SZ")
    trace = f"vision.organ:vision-frame-router:{window_id}"
    prov = GrammarProvenanceV1(source_service=source, source_component="organ_window", source_trace_id=trace)
    out: list[GrammarEventV1] = []
    rows = [(ROLE_STREAM_WINDOW, s) for s in streams]
    if closing:
        rows.append((ROLE_WINDOW_COMPLETED, f"router=vision-frame-router streams={len(streams)}"))
    for i, (role, summary) in enumerate(rows):
        eid = f"{trace}:{i:02d}:{role}"
        out.append(
            GrammarEventV1(
                event_id=eid,
                event_kind="atom_emitted",
                trace_id=trace,
                emitted_at=end,
                observed_at=end,
                atom=GrammarAtomV1(
                    atom_id=eid, trace_id=trace, atom_type="observation", semantic_role=role, layer="perception",
                    summary=summary, text_value="x",
                ),
                provenance=prov,
            )
        )
    return out


def _empty() -> VisionOrganProjectionV1:
    return empty_vision_organ_projection(now=T0)


def _hints(receipt) -> dict:
    assert len(receipt.state_deltas) == 1
    return receipt.state_deltas[0].after["pressure_hints"]


def test_calm_cam0_live_while_carbon_dark_reads_calm_with_carbon_visible() -> None:
    """The live shape on 2026-10-02: cam0 delivering, carbon configured and silent
    since the router started. The organ reads calm (Orion can see), and carbon is
    named stale on the projection -- not hidden, not calm."""
    proj, receipt = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0"), _stream("carbon", frames=0, age=None, dispatched=0, ok=0)]),
        projection=_empty(),
        now=T0,
    )
    assert _hints(receipt) == {"vision_frame_staleness": 0.0, "vision_processing_failure_pressure": 0.0}
    assert proj.status == ORGAN_REPORTING
    assert proj.streams["cam0"].status == STREAM_LIVE
    assert proj.streams["carbon"].status == STREAM_NEVER_SEEN
    assert proj.streams["carbon"].frame_staleness == 1.0
    after = receipt.state_deltas[0].after
    assert after["node_id"] == VISION_ORGAN_NODE_ID
    assert after["streams"]["carbon"]["status"] == STREAM_NEVER_SEEN
    assert receipt.state_deltas[0].target_kind == "vision_organ"


def test_every_stream_dark_reads_full_staleness() -> None:
    _, receipt = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0", frames=0, age=300.0, dispatched=0, ok=0),
                            _stream("carbon", frames=0, age=None, dispatched=0, ok=0)]),
        projection=_empty(),
        now=T0,
    )
    hints = _hints(receipt)
    assert hints["vision_frame_staleness"] == 1.0
    # nothing dispatched anywhere in the span: failure is unmeasured, not calm
    assert "vision_processing_failure_pressure" not in hints


def test_stream_staleness_ramps_between_grace_and_saturation() -> None:
    proj, receipt = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0", frames=0, age=37.5)]), projection=_empty(), now=T0
    )
    assert proj.streams["cam0"].status == STREAM_STALE
    assert _hints(receipt)["vision_frame_staleness"] == 0.5


def test_never_seen_inside_startup_grace_is_not_an_alarm() -> None:
    _, receipt = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0", frames=0, age=None, uptime=10.0, dispatched=0, ok=0)]),
        projection=_empty(),
        now=T0,
    )
    assert _hints(receipt)["vision_frame_staleness"] == 0.0


def test_router_with_no_streams_is_blind_not_calm() -> None:
    _, receipt = reduce_vision_organ_trace_events(events=_window(T0, []), projection=_empty(), now=T0)
    assert _hints(receipt)["vision_frame_staleness"] == 1.0


def test_failure_share_is_rolling_and_floored() -> None:
    proj = _empty()
    # one timeout: below the 2-failure floor -> 0.0
    proj, r1 = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0", ok=11, failed=1, classes="timeout:1")]), projection=proj, now=T0
    )
    assert _hints(r1)["vision_processing_failure_pressure"] == 0.0
    # second timeout a minute later: 2 / 24 over the rolling span
    t1 = T0 + timedelta(seconds=60)
    proj, r2 = reduce_vision_organ_trace_events(
        events=_window(t1, [_stream("cam0", ok=11, failed=1, classes="timeout:1")]), projection=proj, now=t1
    )
    assert abs(_hints(r2)["vision_processing_failure_pressure"] - 2 / 24) < 1e-9
    # 11 minutes later the old failures have rolled out
    t2 = T0 + timedelta(seconds=60 + 660)
    proj, r3 = reduce_vision_organ_trace_events(
        events=_window(t2, [_stream("cam0")]), projection=proj, now=t2
    )
    assert _hints(r3)["vision_processing_failure_pressure"] == 0.0


def test_host_down_reads_full_failure() -> None:
    _, receipt = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0", ok=0, failed=12, classes="timeout:12")]), projection=_empty(), now=T0
    )
    assert _hints(receipt)["vision_processing_failure_pressure"] == 1.0
    assert receipt.state_deltas[0].after["failure_window"]["scope"] in {"organ", "cam0"}


def test_a_failing_stream_is_not_diluted_by_a_busy_healthy_one() -> None:
    _, receipt = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0", ok=120), _stream("carbon", ok=0, failed=10, classes="timeout:10")]),
        projection=_empty(),
        now=T0,
    )
    assert _hints(receipt)["vision_processing_failure_pressure"] == 1.0
    assert receipt.state_deltas[0].after["failure_window"]["scope"] == "carbon"


def test_replayed_window_does_not_double_count() -> None:
    events = _window(T0, [_stream("cam0", ok=10, failed=2, classes="timeout:2")])
    proj, r1 = reduce_vision_organ_trace_events(events=events, projection=_empty(), now=T0)
    proj, r2 = reduce_vision_organ_trace_events(events=events, projection=proj, now=T0)
    assert _hints(r1) == _hints(r2)
    assert sum(c.failed for c in proj.recent_windows) == 2


def test_split_window_waits_for_the_closing_atom() -> None:
    events = _window(T0, [_stream("cam0"), _stream("carbon", frames=0, age=None, dispatched=0, ok=0)])
    saved: list = []
    receipts: list = []
    state = {"p": _empty()}

    def load():
        return state["p"]

    def save(p):
        state["p"] = p
        saved.append(p)

    # first batch: only carbon's atom (would read 1.0 on its own)
    process_vision_organ_grammar_events(
        events=[events[1]], load_projection=load, save_projection=save, save_receipt=receipts.append, now=T0
    )
    assert receipts[-1].state_deltas == []
    process_vision_organ_grammar_events(
        events=[events[0], events[2]], load_projection=load, save_projection=save,
        save_receipt=receipts.append, now=T0,
    )
    assert _hints(receipts[-1])["vision_frame_staleness"] == 0.0
    assert set(state["p"].streams) == {"cam0", "carbon"}


def test_wrong_source_service_is_a_noop() -> None:
    proj = _empty()
    out, receipt = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0")], source="orion-vision-edge"), projection=proj, now=T0
    )
    assert receipt.state_deltas == [] and receipt.noop_event_ids
    assert out.last_window_id is None


def test_silence_path_writes_full_staleness_with_fresh_ids() -> None:
    proj, _ = reduce_vision_organ_trace_events(
        events=_window(T0, [_stream("cam0")]), projection=_empty(), now=T0
    )
    later = T0 + timedelta(seconds=400)
    age = silence_age_seconds(proj, now=later, process_started_at=T0 - timedelta(hours=1))
    assert age == 400.0
    p1, r1 = vision_organ_silence_receipt(proj, now=later, silent_for_sec=age)
    p2, r2 = vision_organ_silence_receipt(p1, now=later + timedelta(seconds=60), silent_for_sec=age + 60)
    assert p1.status == ORGAN_SILENT
    assert _hints(r1) == {"vision_frame_staleness": 1.0}
    assert r1.state_deltas[0].delta_id != r2.state_deltas[0].delta_id
    assert r1.receipt_id != r2.receipt_id
    # a window arriving afterwards returns the organ to reporting
    back = later + timedelta(seconds=120)
    p3, r3 = reduce_vision_organ_trace_events(events=_window(back, [_stream("cam0")]), projection=p2, now=back)
    assert p3.status == ORGAN_REPORTING
    assert _hints(r3)["vision_frame_staleness"] == 0.0


def test_never_heard_from_is_aged_from_process_start() -> None:
    assert silence_age_seconds(None, now=T0, process_started_at=T0 - timedelta(seconds=500)) == 500.0


def test_topology_feeds_capability_vision_from_the_organ_only() -> None:
    topo = yaml.safe_load((REPO / "config/field/orion_field_topology.v1.yaml").read_text())
    edges = [e for e in topo["edges"] if e.get("target_id") == "capability:vision"]
    assert [e["source_id"] for e in edges] == [VISION_ORGAN_NODE_ID]
    assert edges[0]["channel_map"] == {
        "vision_frame_staleness": "pressure",
        "vision_processing_failure_pressure": "reliability_pressure",
    }


def test_window_missing_a_stream_atom_gives_no_reading() -> None:
    """The closing atom says 2 streams; only carbon's arrived. A min over carbon
    alone would read 1.0 although cam0 was live -- skip the reading instead."""
    events = _window(T0, [_stream("cam0"), _stream("carbon", frames=0, age=None, dispatched=0, ok=0)])
    _, receipt = reduce_vision_organ_trace_events(events=[events[1], events[2]], projection=_empty(), now=T0)
    assert receipt.state_deltas == []
    assert any("1 of 2" in w for w in receipt.warnings)
