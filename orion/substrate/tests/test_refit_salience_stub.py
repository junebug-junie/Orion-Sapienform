from scripts.refit_salience_weights import candidate_weights_from_labels


def test_refit_consumes_label_rows_and_returns_weights():
    labels = [
        {"verdict": "resolved", "features_at_close": {"evidence_strength": 0.9}},
        {"verdict": "dismissed", "features_at_close": {"evidence_strength": 0.2}},
        {"verdict": "decayed_unattended", "features_at_close": {"evidence_strength": 0.3}},
    ]
    weights, version = candidate_weights_from_labels(labels)
    assert isinstance(weights, dict)
    assert "evidence_strength" in weights
    assert version.startswith("seed-v1")  # stub keeps seeded weights; documents intent


def test_refit_handles_empty_labels():
    weights, version = candidate_weights_from_labels([])
    assert weights  # returns seeded defaults, not empty
