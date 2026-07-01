from orion.self_state.prediction import compute_overall_surprise


def test_compute_overall_surprise_is_max_of_dimension_errors():
    errors = {"execution_pressure": 0.12, "coherence": 0.45, "uncertainty": 0.03}
    assert compute_overall_surprise(errors) == 0.45


def test_compute_overall_surprise_empty_dict_is_zero():
    assert compute_overall_surprise({}) == 0.0


def test_compute_overall_surprise_handles_none_values():
    errors = {"execution_pressure": None, "coherence": 0.2}
    assert compute_overall_surprise(errors) == 0.2
