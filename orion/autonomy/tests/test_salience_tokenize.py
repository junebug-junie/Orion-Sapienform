from orion.autonomy.salience import tokenize_terms


def test_tokenize_terms_splits_and_lowercases():
    assert tokenize_terms("Hardware Compute GPU") == {"hardware", "compute", "gpu"}


def test_tokenize_terms_handles_underscores_as_nonword():
    assert tokenize_terms("hardware_compute_gpu") == {"hardware", "compute", "gpu"}


def test_tokenize_terms_empty():
    assert tokenize_terms("") == set()
