"""Whole-body structure (Layer 2): known-answer tests (#3290)."""
from __future__ import annotations

import numpy as np
import pytest

from orion.substrate import whole_body_structure as wbs

DAY = wbs.BUCKETS_PER_DAY


def test_tod_residuals_remove_the_daily_cycle_and_need_history():
    rng = np.random.default_rng(1)
    idx = np.arange(20 * DAY) + 1_000_000 * DAY  # any absolute start, aligned to a day
    clock = 5.0 * np.sin(2 * np.pi * (idx % DAY) / DAY)
    x = (clock + rng.normal(0, 0.5, len(idx)))[:, None]
    res = wbs.tod_residuals(x, idx)
    assert np.all(np.isnan(res[: wbs.PROFILE_MIN_DAYS * DAY]))
    tail = res[5 * DAY:, 0]
    assert np.all(np.isfinite(tail))
    # Hour-of-day means remove most of the cycle; what is left is noise plus within-hour slope.
    assert np.var(tail) < 0.1 * np.var(x[5 * DAY:, 0])


def test_tod_residual_profile_uses_only_earlier_days():
    idx = np.arange(10 * DAY)
    x = np.zeros((len(idx), 1))
    x[9 * DAY + 100, 0] = 50.0  # a spike on the last day
    res = wbs.tod_residuals(x, idx)
    # The spike is fully surprising: its own day's value never enters its profile.
    assert res[9 * DAY + 100, 0] == pytest.approx(50.0)
    # The same hour on that day reads the spike's hour as normal before it happened.
    assert res[9 * DAY + 99, 0] == pytest.approx(0.0)


def test_tod_residuals_keep_nan_and_reject_bad_shapes():
    idx = np.arange(6 * DAY)
    x = np.ones((len(idx), 2))
    x[5 * DAY + 3, 1] = np.nan
    res = wbs.tod_residuals(x, idx)
    assert np.isnan(res[5 * DAY + 3, 1]) and res[5 * DAY + 3, 0] == 0.0
    with pytest.raises(ValueError):
        wbs.tod_residuals(x, idx[:-1])


def test_select_channels_drops_sparse_and_constant():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 3))
    x[:20, 1] = np.nan  # 80% present
    x[:, 2] = 4.0
    assert list(wbs.select_channels(x)) == [0]


def test_total_correlation_known_values():
    assert wbs.total_correlation(np.eye(5)) == pytest.approx(0.0, abs=1e-12)
    r = np.array([[1.0, 0.6], [0.6, 1.0]])
    rs = 0.9 * 0.6
    assert wbs.total_correlation(r) == pytest.approx(-0.5 * np.log(1 - rs ** 2))


def test_o_information_sign_redundancy_vs_synergy():
    rng = np.random.default_rng(3)
    s = rng.normal(size=20000)
    redundant = np.stack([s + 0.3 * rng.normal(size=s.size) for _ in range(4)], axis=1)
    assert wbs.o_information(wbs.correlation(redundant)) > 0.3
    a, b = rng.normal(size=(2, 20000))
    synergy = np.stack([a, b, a + b], axis=1)  # a and b are independent; only together do they fix a+b
    assert wbs.o_information(wbs.correlation(synergy)) < -0.1


def test_participation_ratio_bounds():
    assert wbs.participation_ratio(np.eye(6)) == pytest.approx(6.0)
    assert wbs.participation_ratio(np.ones((6, 6))) == pytest.approx(1.0)


def test_cross_total_correlation_independent_vs_linked_groups():
    rng = np.random.default_rng(4)
    body = rng.normal(size=(5000, 3))
    mind_free = rng.normal(size=(5000, 3))
    mind_tied = body + 0.5 * rng.normal(size=(5000, 3))
    free = wbs.cross_total_correlation(wbs.correlation(np.hstack([body, mind_free])), [0, 1, 2], [3, 4, 5])
    tied = wbs.cross_total_correlation(wbs.correlation(np.hstack([body, mind_tied])), [0, 1, 2], [3, 4, 5])
    assert abs(free) < 0.01 and tied > 1.0


def test_match_modes_ignores_order_and_sign():
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    assert np.allclose(wbs.match_modes(q, -q[:, ::-1]), 1.0)
    unrelated, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    assert wbs.match_modes(q, unrelated).mean() < 0.9


def test_top_modes_sign_is_stable():
    r = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    shares, vecs = wbs.top_modes(r, 1)
    assert shares[0] == pytest.approx(1.8 / 3)
    assert vecs[np.argmax(np.abs(vecs[:, 0])), 0] > 0


def test_window_features_with_groups():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(300, 4))
    f = wbs.window_features(x, groups={"body": [0, 1], "mind": [2, 3]})
    assert f.body_mind is not None and abs(f.body_mind) < 0.05
    assert f.leading_vector.shape == (4,)
    assert wbs.window_features(x).body_mind is None


def test_shifted_null_breaks_coupling_and_keeps_gap():
    rng = np.random.default_rng(7)
    s = np.cumsum(rng.normal(size=3000)) * 0.05 + rng.normal(size=3000)
    x = np.stack([s + 0.2 * rng.normal(size=3000) for _ in range(6)], axis=1)
    observed = wbs.total_correlation(wbs.correlation(x[1000:1288]))
    nulls = []
    for _ in range(20):
        starts, gap = wbs.null_starts(len(x), 288, x.shape[1], rng=rng)
        diffs = np.abs(starts[:, None] - starts[None, :])[~np.eye(6, dtype=bool)]
        assert diffs.min() >= gap
        nulls.append(wbs.total_correlation(wbs.correlation(wbs.shifted_null_window(x, 288, starts))))
    z, p_real = wbs.z_and_p_real(observed, np.array(nulls))
    assert z > 5 and p_real == 1.0


def test_null_starts_refuse_no_room():
    with pytest.raises(ValueError):
        wbs.null_starts(300, 288, 20, rng=np.random.default_rng(0))


def test_z_and_p_real_direction():
    null = np.array([4.0, 5.0, 6.0])
    assert wbs.z_and_p_real(7.0, null) == (pytest.approx(2.0), 1.0)
    assert wbs.z_and_p_real(3.0, null, higher_is_structure=False)[1] == 1.0
    assert np.isnan(wbs.z_and_p_real(1.0, np.array([2.0, 2.0]))[0])


def test_structure_drift_zero_for_same_matrix():
    r = wbs.correlation(np.random.default_rng(8).normal(size=(100, 4)))
    assert wbs.structure_drift(r, r) == 0.0
    assert wbs.structure_drift(r, np.eye(4)) > 0.0


def test_correlation_handles_missing_and_constant():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(200, 3))
    x[5, 0] = np.nan
    x[:, 2] = 1.0
    r = wbs.correlation(x)
    assert np.all(np.isfinite(r))
    assert r[2, 0] == 0.0 and r[2, 2] == 1.0


def test_merge_copies_groups_plumbing_and_keeps_distinct_channels():
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=(2, 2000))
    x = np.stack([a, a + 0.01 * rng.normal(size=2000), b, 2 * a, b + 0.5 * rng.normal(size=2000)], axis=1)
    groups = wbs.merge_copies(x)
    assert groups == [[0, 1, 3], [2], [4]]
    merged = wbs.merged_series(x, groups)
    assert merged.shape == (2000, 3)
    assert abs(np.corrcoef(merged[:, 0], a)[0, 1]) > 0.99


def test_merging_copies_removes_fake_coupling():
    """Four copies of one independent process read as coupled; merged, they read as one channel."""
    rng = np.random.default_rng(11)
    base = rng.normal(size=(3000, 3))
    x = np.hstack([base, np.repeat(base[:, :1], 3, axis=1) + 1e-3 * rng.normal(size=(3000, 3))])
    assert wbs.total_correlation(wbs.correlation(x)) > 2.0
    merged = wbs.merged_series(x, wbs.merge_copies(x))
    assert merged.shape[1] == 3
    assert wbs.total_correlation(wbs.correlation(merged)) < 0.01


def test_merged_anti_copy_does_not_cancel():
    rng = np.random.default_rng(12)
    a = rng.normal(size=2000)
    x = np.stack([a, -a + 1e-3 * rng.normal(size=2000)], axis=1)
    groups = wbs.merge_copies(x)
    assert groups == [[0, 1]]
    merged = wbs.merged_series(x, groups)[:, 0]
    assert np.std(merged) > 0.9 and np.corrcoef(merged, a)[0, 1] > 0.99


def _two_mode_series(rng, n, flip_at=None):
    """Two fixed loading patterns over 12 channels; optionally swapped for new ones at ``flip_at``."""
    q, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    f = rng.normal(size=(n, 2)) * [3.0, 2.0]
    load = np.tile(q[:, :2], (n, 1, 1))
    if flip_at is not None:
        load[flip_at:] = q[:, 2:]
    return np.einsum("tc,tic->ti", f, load) + rng.normal(size=(n, 12))


def test_block_stability_finds_fixed_modes_above_chance():
    rng = np.random.default_rng(13)
    st = wbs.block_stability(_two_mode_series(rng, 6 * DAY), DAY, 2, rng=rng)
    assert st.pairs == 5
    assert st.stable_ranks == [1, 2]
    assert np.all(st.median_by_rank > st.chance_max_by_rank)


def test_block_stability_refuses_modes_that_change_every_block():
    rng = np.random.default_rng(14)
    x = np.vstack([_two_mode_series(rng, DAY) for _ in range(6)])  # new modes each day
    st = wbs.block_stability(x, DAY, 2, rng=rng)
    assert st.stable_ranks == []


def test_block_stability_needs_two_blocks():
    rng = np.random.default_rng(15)
    assert wbs.block_stability(rng.normal(size=(DAY + 10, 4)), DAY, 2, rng=rng) is None


def test_block_stability_chance_floor_refuses_few_channel_noise():
    """With 2 channels, both modes always match at >= 0.707, even in pure noise.

    The 0.7 bar alone passes it; only the chance floor stops it.
    """
    rng = np.random.default_rng(16)
    st = wbs.block_stability(rng.normal(size=(6 * DAY, 2)), DAY, 2, rng=rng)
    assert st.median_by_rank.min() >= wbs.STABLE_SIMILARITY
    assert st.stable_ranks == []


def test_block_stability_without_room_for_a_null_has_no_verdict():
    rng = np.random.default_rng(17)
    st = wbs.block_stability(rng.normal(size=(25, 20)), 10, 2, rng=rng)  # 15 rows of room, 20 channels
    assert st.pairs == 1 and st.chance_max_by_rank is None and st.stable_ranks is None
