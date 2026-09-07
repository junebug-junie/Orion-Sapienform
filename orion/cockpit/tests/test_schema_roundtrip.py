from orion.schemas.cockpit_sighting import CockpitHopV1, COCKPIT_HOP_CHANNEL
from orion.schemas.registry import resolve


def test_cockpit_hop_roundtrip_and_registry():
    hop = CockpitHopV1(
        correlation_id="corr-1",
        seq=1,
        stage="stance_decision",
        visor_line="stance · proceed",
        status="ok",
        summary={"disposition": "proceed"},
        raw={"disposition": "proceed"},
        producer="orion-hub",
    )
    dumped = hop.model_dump(mode="json")
    again = CockpitHopV1.model_validate(dumped)
    assert again.seq == 1
    assert again.stage == "stance_decision"
    assert COCKPIT_HOP_CHANNEL == "orion:cockpit:hop"
    assert resolve("CockpitHopV1") is CockpitHopV1


def test_progress_stages_roundtrip():
    for stage in (
        "pre_turn_appraisal",
        "thought_rpc",
        "mind_enrichment",
        "situation",
        "harness_dispatch",
    ):
        hop = CockpitHopV1(
            correlation_id="corr-1",
            seq=0,
            stage=stage,  # type: ignore[arg-type]
            visor_line=f"{stage} · ok",
            status="started",
            summary={},
            raw={},
            producer="orion-hub",
        )
        assert CockpitHopV1.model_validate(hop.model_dump(mode="json")).stage == stage


def test_gap_status_allowed():
    hop = CockpitHopV1(
        correlation_id="corr-1",
        seq=2,
        stage="motor_boot",
        visor_line="gap · motor_boot not recorded (Slice B)",
        status="gap",
        summary={"deferred_to": "slice_b"},
        raw={},
        producer="orion-hub",
    )
    assert hop.status == "gap"
