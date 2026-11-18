"""The bus_synaptic transport line in the evidence pack is retired (2026-11,
PE suite rebuild step 5): node:substrate.bus_synaptic, the mesh-wide fraction
that gated it, is gone, and so are recall_signal_resolver.py and its settings.

conftest.py's autouse fixture re-binds `app` to this service before every
test in this directory.
"""

from __future__ import annotations

import importlib.util


def _snapshot_with_fragments(fragments):
    return {"facets": {"recall_bundle": {"fragments": fragments}}}


def test_resolver_module_and_settings_are_gone():
    import app.evidence as ev
    import app.settings as settings_mod

    assert importlib.util.find_spec("app.recall_signal_resolver") is None
    assert not hasattr(ev, "render_bus_synaptic_digest_line")
    fields = set(settings_mod.Settings.model_fields)
    assert "RECALL_TRANSPORT_PG_DSN" not in fields
    assert "RECALL_TRANSPORT_RENDER_GATE_THRESHOLD" not in fields


def test_recall_fragments_render_as_ordinary_items():
    import app.evidence as ev

    fragments = [
        {"source": "sql", "text": "hello world", "id": "a"},
        {
            "id": "bus_synaptic_causal:a:b",
            "source": "bus_synaptic_anomaly",
            "text": "Bus synaptic snapshot (not live traffic): a -> b hop latency was unusual",
            "meta": {"signal_kind": "causal_latency_zscore"},
        },
        # A stale text="" publish-gap fragment (recall no longer emits these)
        # is skipped like any empty fragment, never rendered as a line.
        {
            "id": "bus_synaptic_publish:x:c",
            "source": "bus_synaptic_anomaly",
            "text": "",
            "meta": {"signal_kind": "publish_gap_zscore"},
        },
    ]
    pack = ev.build_evidence_pack(_snapshot_with_fragments(fragments))
    texts = [it.text for it in pack.items if it.source_kind == "recall_fragment"]
    assert texts == [
        "hello world",
        "Bus synaptic snapshot (not live traffic): a -> b hop latency was unusual",
    ]
    assert not any(it.source_kind == "bus_synaptic_transport" for it in pack.items)
