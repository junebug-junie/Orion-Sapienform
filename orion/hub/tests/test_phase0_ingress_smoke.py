from __future__ import annotations

import pytest

from orion.hub.association import build_hub_association_bundle
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1


@pytest.mark.asyncio
async def test_ingress_observation_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[dict] = []

    def _fake_emit(**kwargs):
        emitted.append(kwargs)
        return type("Mol", (), {"molecule_id": "obs-1"})()

    monkeypatch.setattr("orion.mind.substrate_emit.emit_observation", _fake_emit)
    from orion.mind.substrate_emit import emit_observation

    mol = emit_observation(surface_text="hello", source_id="sess-1")
    assert mol.molecule_id == "obs-1"
    assert emitted[0]["surface_text"] == "hello"


def test_association_carries_repair_bundle_correlation() -> None:
    bundle = TurnAppraisalBundleV1(correlation_id="corr-ingress", paradigms={})
    assoc = build_hub_association_bundle(correlation_id="corr-ingress", repair_bundle=bundle)
    assert assoc.correlation_id == "corr-ingress"
    assert assoc.repair_bundle is not None
    assert assoc.repair_bundle.correlation_id == "corr-ingress"
