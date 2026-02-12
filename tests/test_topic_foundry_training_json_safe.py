import sys

sys.path.insert(0, "services/orion-topic-foundry")

from app.services.training import _find_type_like_values, _json_safe


def test_run_meta_type_values_are_detected_and_sanitized():
    payload = {
        "stats": {"outlier_rate": 0.0},
        "vectorizer_params": {"dtype": int, "ngram_range": (1, 1)},
    }
    findings = _find_type_like_values(payload)
    paths = [path for path, _ in findings]
    assert "vectorizer_params.dtype" in paths

    sanitized = _json_safe(payload)
    assert sanitized["vectorizer_params"]["dtype"] == "int"
    assert sanitized["stats"]["outlier_rate"] == 0.0
