from orion.schemas.collapse_mirror import normalize_collapse_entry


def _base_entry(telemetry):
    return normalize_collapse_entry(
        {
            "observer": "orion",
            "trigger": "baseline",
            "observer_state": ["idle"],
            "type": "flow",
            "emergent_entity": "Test",
            "summary": "ok",
            "mantra": "keep",
            "state_snapshot": {"telemetry": telemetry},
        }
    )


def test_telemetry_key_normalization():
    entry = _base_entry({"gpu_ mem": 0.5, "gpu_ util": 0.9, "phi_ hint": {"valence": 0.2}})
    telemetry = entry.state_snapshot.telemetry
    assert "gpu_mem" in telemetry
    assert "gpu_util" in telemetry
    assert "phi_hint" in telemetry
    assert "gpu_ mem" not in telemetry
    assert "gpu_ util" not in telemetry
    assert "phi_ hint" not in telemetry


def test_phi_hint_canonical_band_shape():
    entry = _base_entry(
        {
            "phi_hint": {
                "valence_band": "neutral",
                "valence_dir": "positive",
                "energy_band": "high",
                "coherence_band": "low",
                "novelty_band": "medium",
            }
        }
    )
    phi_hint = entry.state_snapshot.telemetry["phi_hint"]
    assert phi_hint["schema_version"] == "v1"
    assert "bands" in phi_hint
    assert phi_hint["bands"]["valence_band"] == "neutral"


def test_phi_hint_canonical_numeric_shape():
    entry = _base_entry(
        {
            "phi_hint": {
                "valence": 0.2,
                "energy": 0.4,
                "coherence": 0.6,
                "novelty": 0.1,
            }
        }
    )
    phi_hint = entry.state_snapshot.telemetry["phi_hint"]
    assert phi_hint["schema_version"] == "v1"
    assert "numeric" in phi_hint
    assert phi_hint["numeric"]["valence"] == 0.2
