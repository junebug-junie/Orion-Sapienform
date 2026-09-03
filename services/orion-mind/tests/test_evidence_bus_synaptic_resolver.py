"""build_evidence_pack()'s resolver branch (2026-09-03).

Spec check 4: a fragment with no signal_kind (i.e. every fragment shape that
existed before this patch) must render byte-identically to the old
passthrough behavior.

conftest.py's autouse fixture re-binds `app` to this service before every
test in this directory.
"""

from __future__ import annotations


def _bus_synaptic_fragment(channel="orion:vision:edge:health", organ="vision-edge"):
    return {
        "id": f"bus_synaptic_publish:{organ}:{channel}",
        "source": "bus_synaptic_anomaly",
        "text": "",
        "meta": {"organ_id": organ, "channel": channel, "signal_kind": "publish_gap_zscore", "zscore": 7.1},
    }


def _snapshot_with_fragments(fragments):
    return {"facets": {"recall_bundle": {"fragments": fragments}}}


def test_fragments_without_signal_kind_render_byte_identically(monkeypatch):
    """Check 4: absent a signal_kind, behavior is unchanged."""
    import app.evidence as ev

    fragments = [
        {"source": "sql", "text": "hello world", "id": "a"},
        {"source": "rdf", "snippet": "earlier turn", "id": "b"},
    ]
    pack = ev.build_evidence_pack(_snapshot_with_fragments(fragments))
    recall_items = [it for it in pack.items if it.source_kind == "recall_fragment"]
    assert len(recall_items) == 2
    assert recall_items[0].text == "hello world"
    assert recall_items[1].text == "earlier turn"


def test_bus_synaptic_fragments_are_not_looped_individually(monkeypatch):
    import app.evidence as ev

    monkeypatch.setattr(ev, "render_bus_synaptic_digest_line", lambda *a, **k: None)
    fragments = [_bus_synaptic_fragment(), _bus_synaptic_fragment(channel="orion:cortex:trace", organ="cortex-exec")]

    pack = ev.build_evidence_pack(_snapshot_with_fragments(fragments))

    assert not any(it.source_kind in ("recall_fragment", "bus_synaptic_transport") for it in pack.items)


def test_bus_synaptic_multiple_fragments_collapse_to_one_item(monkeypatch):
    import app.evidence as ev

    monkeypatch.setattr(
        ev, "render_bus_synaptic_digest_line", lambda handled, **k: f"Transport: {len(handled)} edges resolved."
    )
    fragments = [
        _bus_synaptic_fragment(),
        _bus_synaptic_fragment(channel="c2", organ="o2"),
        {"source": "sql", "text": "an ordinary fragment", "id": "z"},
    ]

    pack = ev.build_evidence_pack(_snapshot_with_fragments(fragments))

    transport_items = [it for it in pack.items if it.source_kind == "bus_synaptic_transport"]
    assert len(transport_items) == 1
    assert transport_items[0].text == "Transport: 2 edges resolved."
    ordinary_items = [it for it in pack.items if it.source_kind == "recall_fragment"]
    assert len(ordinary_items) == 1


def test_causal_latency_fragments_still_pass_through_with_their_own_text(monkeypatch):
    """Non-goal: causal_latency_zscore fragments are untouched this pass --
    they still render their own pre-existing text like any ordinary
    fragment, never routed through the resolver."""
    import app.evidence as ev

    monkeypatch.setattr(
        ev, "render_bus_synaptic_digest_line", lambda handled, **k: None if not handled else "SHOULD_NOT_APPEAR"
    )
    causal_frag = {
        "id": "bus_synaptic_causal:a:b",
        "source": "bus_synaptic_anomaly",
        "text": "Bus synaptic snapshot: a -> b hop latency was unusual.",
        "meta": {"signal_kind": "causal_latency_zscore"},
    }
    pack = ev.build_evidence_pack(_snapshot_with_fragments([causal_frag]))

    texts = [it.text for it in pack.items]
    assert any("a -> b hop latency was unusual" in t for t in texts)
    assert not any("SHOULD_NOT_APPEAR" in t for t in texts)


def test_dsn_and_threshold_reach_the_resolver(monkeypatch):
    """engine.py's call site threads settings through -- confirm the kwargs
    actually reach render_bus_synaptic_digest_line rather than silently
    no-op'ing on defaults."""
    import app.evidence as ev

    seen = {}

    def _fake_render(handled, *, dsn, render_gate_threshold):
        seen["dsn"] = dsn
        seen["render_gate_threshold"] = render_gate_threshold
        return None

    monkeypatch.setattr(ev, "render_bus_synaptic_digest_line", _fake_render)

    ev.build_evidence_pack(
        _snapshot_with_fragments([_bus_synaptic_fragment()]),
        bus_synaptic_dsn="postgresql://from-settings",
        bus_synaptic_render_gate_threshold=0.15,
    )

    assert seen["dsn"] == "postgresql://from-settings"
    assert seen["render_gate_threshold"] == 0.15


def test_no_fragments_at_all_is_unaffected():
    import app.evidence as ev

    pack = ev.build_evidence_pack({"user_text": "hi"})
    assert not any(it.source_kind == "bus_synaptic_transport" for it in pack.items)
