from orion.substrate.causal_geometry_engine import PairResult, _apply_fdr_correction


def _result(p_value: float | None) -> PairResult:
    r = PairResult("a", "b", n_lag0=10)
    r.p_value = p_value
    return r


def test_fdr_correction_is_more_permissive_than_flat_alpha_when_many_pairs_tested():
    # 20 pairs, all with p just above alpha=0.05 individually -- a flat
    # per-pair cutoff would reject every one of them, but BH's rank-dependent
    # threshold should let the smallest few through.
    results = [_result(0.001 * (i + 1)) for i in range(20)]
    _apply_fdr_correction(results, alpha=0.05)
    assert any(r.significant for r in results)
    # Significant results must be exactly the smallest-p prefix (BH is a
    # step-up procedure on sorted p-values).
    sig_ps = sorted(r.p_value for r in results if r.significant)
    all_ps = sorted(r.p_value for r in results)
    assert sig_ps == all_ps[: len(sig_ps)]


def test_fdr_correction_rejects_all_when_no_pair_clears_threshold():
    results = [_result(0.9), _result(0.8), _result(0.7)]
    _apply_fdr_correction(results, alpha=0.05)
    assert all(not r.significant for r in results)


def test_fdr_correction_skips_pairs_with_no_p_value():
    untested = _result(None)
    tested = _result(0.001)
    _apply_fdr_correction([untested, tested], alpha=0.05)
    assert untested.significant is False
    assert tested.significant is True


def test_fdr_correction_empty_input_does_not_raise():
    _apply_fdr_correction([], alpha=0.05)
