from orion.schemas.collapse_mirror import attach_llm_uncertainty_to_collapse_payload

_UNC = {
    "schema_version": "v1",
    "available": True,
    "source": "llamacpp_native_completion",
    "mean_logprob": -0.74,
    "mean_top1_margin": 1.2,
    "unstable_span_count": 2,
}


def test_attach_llm_uncertainty_preserves_existing_telemetry() -> None:
    payload = {
        "state_snapshot": {
            "telemetry": {
                "change_type_meta": {"flow": 0.9},
                "gpu_util": 0.42,
            }
        }
    }
    attach_llm_uncertainty_to_collapse_payload(payload, _UNC)
    telemetry = payload["state_snapshot"]["telemetry"]
    assert telemetry["gpu_util"] == 0.42
    assert telemetry["change_type_meta"] == {"flow": 0.9}
    assert telemetry["llm_uncertainty"] == _UNC
    assert telemetry["llm_uncertainty_semantics"] == "language_surface_stability_not_truth"


def test_attach_llm_uncertainty_noop_when_missing() -> None:
    payload = {"state_snapshot": {"telemetry": {"gpu_util": 1.0}}}
    attach_llm_uncertainty_to_collapse_payload(payload, None)
    assert "llm_uncertainty" not in payload["state_snapshot"]["telemetry"]
