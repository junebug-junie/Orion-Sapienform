from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

_guard = Path(__file__).resolve().parent / "_exec_import_guard.py"
_spec = importlib.util.spec_from_file_location("_exec_guard_boot_shared_spine", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.relational.beliefs import AnchorBeliefSliceV1, UnifiedRelationalBeliefSetV1


def _beliefs() -> UnifiedRelationalBeliefSetV1:
    concept = ConceptNodeV1(
        node_kind="concept",
        anchor_scope="orion",
        label="shared spine test concept",
        definition="A compact concept proving the shared stance spine produced projection data.",
        temporal=SubstrateTemporalWindowV1(observed_at=datetime.now(timezone.utc)),
        signals=SubstrateSignalBundleV1(salience=0.9, confidence=0.8),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred",
            source_kind="test",
            source_channel="unit",
            producer="test_chat_stance_shared_spine",
            tier_rank=3,
            evidence_refs=["ev-shared-spine"],
        ),
    )
    return UnifiedRelationalBeliefSetV1(
        anchors={"orion": AnchorBeliefSliceV1(anchor="orion", concepts=[concept])},
        cold_anchors=["orion"],
        degraded_producers=["producer-x"],
        lineage=["test:shared_spine"],
    )


def test_exec_app_import_installs_shared_chat_stance_spine(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED", raising=False)
    _guard_mod.ensure_orion_cortex_exec_app()

    import app
    from app import chat_stance
    from app import chat_stance_shared_spine

    assert app is not None
    assert getattr(chat_stance, "_CHAT_STANCE_SHARED_PROJECTION_SPINE") is True
    assert chat_stance._unified_beliefs_for_stance is chat_stance_shared_spine.shared_unified_beliefs_for_stance


def test_shared_chat_stance_spine_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED", "true")
    _guard_mod.ensure_orion_cortex_exec_app()

    import app
    from app import chat_stance
    from app import chat_stance_shared_spine

    assert app is not None
    assert chat_stance._unified_beliefs_for_stance is not chat_stance_shared_spine.shared_unified_beliefs_for_stance


def test_shared_spine_records_context_marker_and_projection(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED", raising=False)
    _guard_mod.ensure_orion_cortex_exec_app()

    from app import chat_stance_shared_spine

    monkeypatch.setattr(chat_stance_shared_spine, "unified_beliefs_for_chat_stance", lambda ctx, **kwargs: _beliefs())

    ctx = {"metadata": {}, "verb": "chat_general", "correlation_id": "corr-shared"}
    result = chat_stance_shared_spine.shared_unified_beliefs_for_stance(ctx)

    assert result is not None
    marker = ctx.get("chat_stance_shared_projection_spine")
    assert marker["enabled"] is True
    assert marker["beliefs_present"] is True
    assert marker["cold_anchors"] == ["orion"]
    assert marker["degraded_producers"] == ["producer-x"]
    assert marker["lineage"] == ["test:shared_spine"]
    assert ctx["metadata"]["chat_stance_shared_projection_spine"] == marker

    projection = ctx.get("chat_cognitive_projection")
    debug = ctx.get("chat_cognitive_projection_debug")
    assert projection["schema_version"] == "cognitive.projection.v1"
    assert projection["item_count"] == 1
    assert debug["present"] is True
    assert debug["item_count"] == 1
    assert debug["cold_anchors"] == ["orion"]
    assert debug["degraded_producers"] == ["producer-x"]


def test_shared_spine_records_absent_beliefs(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED", raising=False)
    _guard_mod.ensure_orion_cortex_exec_app()

    from app import chat_stance_shared_spine

    monkeypatch.setattr(chat_stance_shared_spine, "unified_beliefs_for_chat_stance", lambda ctx, **kwargs: None)

    ctx = {"metadata": {}, "verb": "chat_general", "correlation_id": "corr-empty"}
    result = chat_stance_shared_spine.shared_unified_beliefs_for_stance(ctx)

    assert result is None
    marker = ctx.get("chat_stance_shared_projection_spine")
    assert marker["enabled"] is True
    assert marker["beliefs_present"] is False
    assert ctx.get("chat_cognitive_projection") is None
    assert ctx.get("chat_cognitive_projection_debug") == {"present": False, "reason": "beliefs_absent"}
