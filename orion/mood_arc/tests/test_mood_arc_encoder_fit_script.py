from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1
from orion.schemas.telemetry.mood_arc import MoodArcEncoderManifestV1

REPO = Path(__file__).resolve().parents[3]
FIT_SCRIPT = REPO / "orion" / "mood_arc" / "fit_encoder.py"


def _run_fit(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _make_row(*, t: datetime, channels: dict[str, float], idx: int) -> FieldChannelCorpusRowV1:
    return FieldChannelCorpusRowV1(
        generated_at=t,
        tick_id=f"tick_{idx}",
        channels=channels,
    )


def _write_corpus(path: Path, rows: list[FieldChannelCorpusRowV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


def _synthetic_corpus_with_real_pattern(n_rows: int = 1500, seed: int = 0) -> list[FieldChannelCorpusRowV1]:
    """A deliberately-injected real periodic/trend pattern per channel, plus
    noise -- distinct from pure noise so the floor gate (real beats a
    within-window shuffle) is expected to genuinely pass. Also injects:
    - `redundant_coherence`, a near-duplicate of `coherence` (r > 0.9), to
      exercise prune_correlated_fields().
    - `flat_channel`, a constant value, to exercise select_fields()'s
      variance filter.
    - `recent_perturbation_count`, varying but excluded by NAME regardless
      of variance (select_fields()'s DEFAULT_EXCLUDE_CHANNELS)."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows: list[FieldChannelCorpusRowV1] = []
    for i in range(n_rows):
        t = start + timedelta(seconds=2 * i)
        phase = i * 0.05
        coherence = 0.6 + 0.05 * np.sin(phase) + rng.normal(0, 0.01)
        energy = 0.3 + 0.1 * np.sin(phase * 0.7) + rng.normal(0, 0.02)
        novelty = 0.5 + 0.3 * np.sin(phase * 1.3) + rng.normal(0, 0.03)
        redundant_coherence = float(coherence) + rng.normal(0, 0.001)
        channels = {
            "coherence": float(coherence),
            "energy": float(energy),
            "novelty": float(novelty),
            "redundant_coherence": float(redundant_coherence),
            "flat_channel": 0.42,
            "recent_perturbation_count": float(rng.integers(0, 5)),
        }
        rows.append(_make_row(t=t, channels=channels, idx=i))
    return rows


def test_fit_refuses_small_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "tiny.jsonl"
    corpus.write_text("")  # empty
    proc = _run_fit("train", "--corpus", str(corpus), "--out", str(tmp_path / "out"))
    assert proc.returncode != 0
    assert "min_rows" in proc.stdout.lower() or "min_rows" in proc.stderr.lower()


def test_filter_rows_by_min_generated_at_is_boundary_inclusive() -> None:
    from orion.mood_arc.fit_encoder import filter_rows_by_min_generated_at

    start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=timezone.utc)
    rows = [
        _make_row(t=start + timedelta(seconds=i), channels={"a": float(i)}, idx=i)
        for i in range(10)
    ]
    cutoff = start + timedelta(seconds=5)

    filtered = filter_rows_by_min_generated_at(rows, cutoff)
    assert [r.channels["a"] for r in filtered] == [5.0, 6.0, 7.0, 8.0, 9.0]
    # The row exactly AT the cutoff must be kept (>=, not >).
    assert filtered[0].generated_at == cutoff


def test_filter_rows_by_min_generated_at_none_is_a_no_op() -> None:
    from orion.mood_arc.fit_encoder import filter_rows_by_min_generated_at

    start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=timezone.utc)
    rows = [
        _make_row(t=start + timedelta(seconds=i), channels={"a": float(i)}, idx=i)
        for i in range(5)
    ]
    assert filter_rows_by_min_generated_at(rows, None) == rows


def test_cli_min_generated_at_excludes_pre_cutoff_rows(tmp_path: Path) -> None:
    """End-to-end CLI wiring: --min-generated-at must actually reach
    cmd_train and reduce the row count reported in the final result blob,
    not just be accepted as a no-op flag."""
    corpus = tmp_path / "field_channels.jsonl"
    out_dir = tmp_path / "filtered-candidate"
    rows = _synthetic_corpus_with_real_pattern()
    _write_corpus(corpus, rows)

    # Cutoff at the corpus's own midpoint timestamp -- roughly half the
    # rows should survive.
    midpoint = rows[len(rows) // 2].generated_at

    proc = _run_fit(
        "train",
        "--corpus", str(corpus),
        "--out", str(out_dir),
        "--min-generated-at", midpoint.isoformat().replace("+00:00", "Z"),
        "--hidden-dim", "32",
        "--latent-dim", "16",
        "--epochs", "5",
        "--min-hours", "0.1",
        "--min-rows", "100",
        "--purge-gap-windows", "6",
        "--bootstrap-n", "20",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert f"filtered to rows with generated_at >= {midpoint.isoformat()}" in proc.stdout

    result = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert result["rows"] == len(rows) - len(rows) // 2


def test_train_end_to_end_passes_floor_gate_on_synthetic_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "field_channels.jsonl"
    out_dir = tmp_path / "v1-candidate"
    _write_corpus(corpus, _synthetic_corpus_with_real_pattern())

    proc = _run_fit(
        "train",
        "--corpus", str(corpus),
        "--out", str(out_dir),
        "--hidden-dim", "32",
        "--latent-dim", "16",
        "--epochs", "150",
        "--lr", "0.003",
        "--min-hours", "0.5",
        "--min-rows", "100",
        "--purge-gap-windows", "6",
        "--bootstrap-n", "50",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "weights.npz").is_file()
    assert (out_dir / "probes.json").is_file()

    manifest = MoodArcEncoderManifestV1.model_validate_json((out_dir / "manifest.json").read_text())
    assert manifest.architecture == "mlp_shallow_v1"
    assert manifest.hidden_dim == 32
    assert manifest.latent_dim == 16
    assert manifest.window_size == 30
    assert manifest.stride == 15
    assert manifest.purge_gap_windows == 6

    # select_fields()/prune_correlated_fields() must have: excluded
    # recent_perturbation_count by name, dropped flat_channel for zero
    # variance, and pruned exactly one of coherence/redundant_coherence
    # (near-duplicate pair, r > 0.9).
    assert "recent_perturbation_count" not in manifest.channel_names
    assert "flat_channel" not in manifest.channel_names
    assert len({"coherence", "redundant_coherence"} & set(manifest.channel_names)) == 1
    assert "energy" in manifest.channel_names
    assert "novelty" in manifest.channel_names

    # cmd_train prints select_fields/prune_correlated_fields/channel
    # variance lines before the final JSON result blob -- slice from the
    # first "{" to isolate it.
    assert "select_fields: excluded 'recent_perturbation_count'" in proc.stdout
    assert "select_fields: dropped 'flat_channel'" in proc.stdout
    assert "prune_correlated_fields: dropping" in proc.stdout

    result = json.loads(proc.stdout[proc.stdout.index("{") :])
    gate = result["gate"]
    assert gate["floor_pass"] is True, f"expected floor gate to pass on real periodic pattern, got {gate}"
    assert gate["floor_ratio"] < 0.5
    assert result["channel_names"] == manifest.channel_names

    # New methodology fields populated and sane.
    assert manifest.ar1_surrogate_loss is not None
    assert manifest.ceiling_ratio is not None
    assert manifest.floor_ratio_ci_low is not None
    assert manifest.floor_ratio_ci_high is not None
    assert manifest.floor_ratio_ci_low <= manifest.floor_ratio_ci_high

    weights = np.load(out_dir / "weights.npz")
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        assert key in weights
    n_fields = len(manifest.channel_names)
    assert weights["W1"].shape[0] == manifest.window_size * n_fields

    # compute_window_probes() is unimplemented (open valence-replacement
    # question, not this task's scope) -- probes.json must be empty, not
    # fabricated, and cmd_train must have printed a clear skip reason
    # rather than silently swallowing the gap.
    probes = json.loads((out_dir / "probes.json").read_text())
    assert probes == {}
    assert "probes: skipped" in proc.stdout
    assert "no probe target is defined yet for field_channel_corpus.v1" in proc.stdout


def test_negative_control_untrained_weights_fail_floor_gate() -> None:
    """Required per the original spec's own acceptance-check list: a
    trivially-bad encoder (random, untrained weights) must FAIL the floor
    gate -- confirms the gate isn't a tautology that always passes.

    Uses the SAME structured (real periodic pattern) corpus as
    test_train_end_to_end_passes_floor_gate_on_synthetic_corpus, which
    passes the floor gate once trained -- if this test instead used pure
    i.i.d. noise windows, within-window shuffling would destroy nothing
    (there is no temporal structure to lose), and *any* model -- trained or
    not -- would score floor_ratio ~= 1.0 on it. That would only prove "the
    gate fails on structureless noise", not the actually-required claim
    that the gate distinguishes an untrained encoder from a real one on
    data that does have structure to learn (found by code review,
    2026-07-13)."""
    from orion.mood_arc.fit_encoder import (
        build_windows,
        init_weights,
        _per_window_losses,
        prune_correlated_fields,
        purged_temporal_split,
        select_fields,
        shuffle_within_windows,
        two_tier_gate,
    )

    rows = _synthetic_corpus_with_real_pattern()
    fields = prune_correlated_fields(rows, select_fields(rows))
    windows = build_windows(rows, fields=fields, window_size=30, stride=15, max_gap_sec=6.0)
    train_matrix, held_matrix = purged_temporal_split(windows, held_out_frac=0.15, purge_gap_windows=6)

    data_mean = np.mean(train_matrix, axis=0)
    # No training loop at all -- init_weights() output used as-is, the
    # trivially-bad encoder the spec's acceptance check requires.
    untrained_weights = init_weights(train_matrix.shape[1], hidden_dim=32, latent_dim=16, seed=99, data_mean=data_mean)

    real_losses = _per_window_losses(held_matrix, untrained_weights)
    shuffle_rng = np.random.default_rng(2)
    shuffled_held = shuffle_within_windows(held_matrix, window_size=30, n_fields=len(fields), rng=shuffle_rng)
    shuffle_losses = _per_window_losses(shuffled_held, untrained_weights)

    gate = two_tier_gate(float(np.mean(real_losses)), float(np.mean(shuffle_losses)), 1.0)
    assert gate["floor_pass"] is False, f"untrained weights should fail the floor gate, got {gate}"


def test_purged_temporal_split_boundary_and_exclusion_count() -> None:
    from orion.mood_arc.fit_encoder import purged_temporal_split

    n = 100
    # Distinct windows so we can identify exactly which ones survive.
    windows = [np.full(4, float(i)) for i in range(n)]

    train, held = purged_temporal_split(windows, held_out_frac=0.15, purge_gap_windows=6)

    held_n = max(1, int(round(n * 0.15)))
    held_start = n - held_n
    purge_start = held_start - 6

    assert held.shape[0] == held_n
    assert train.shape[0] == purge_start
    # Held-out set is the temporally-last slice.
    assert list(held[:, 0]) == list(range(held_start, n))
    # Train set is the temporally-first slice, stopping before the purge zone.
    assert list(train[:, 0]) == list(range(0, purge_start))
    # Exactly purge_gap_windows windows excluded from both sets.
    excluded_count = n - train.shape[0] - held.shape[0]
    assert excluded_count == 6


def test_purged_temporal_split_raises_clear_error_when_too_few_windows() -> None:
    from orion.mood_arc.fit_encoder import purged_temporal_split

    windows = [np.full(4, float(i)) for i in range(5)]  # far too few for held_out_frac + purge
    with pytest.raises(ValueError, match="purged_temporal_split"):
        purged_temporal_split(windows, held_out_frac=0.15, purge_gap_windows=6)


def test_purged_temporal_split_raises_on_empty_windows() -> None:
    from orion.mood_arc.fit_encoder import purged_temporal_split

    with pytest.raises(ValueError, match="no windows to split"):
        purged_temporal_split([], held_out_frac=0.15, purge_gap_windows=6)


def test_build_windows_breaks_on_real_recorded_gap() -> None:
    """Uses this session's own recorded max gap (8.09s) against the default
    max_gap_sec=6.0 to confirm a run correctly breaks there."""
    from orion.mood_arc.fit_encoder import build_windows

    fields = ("coherence", "energy", "novelty", "valence")
    start = datetime(2026, 7, 13, 6, 0, 0, tzinfo=timezone.utc)
    rows: list[FieldChannelCorpusRowV1] = []
    idx = 0
    # Run 1: 10 rows, 2s apart, coherence pinned to a distinguishable value.
    for i in range(10):
        rows.append(
            _make_row(
                t=start + timedelta(seconds=2 * i),
                channels={"coherence": 0.1, "energy": 0.1, "novelty": 0.1, "valence": 0.1},
                idx=idx,
            )
        )
        idx += 1
    gap_start = rows[-1].generated_at + timedelta(seconds=8.09)  # recorded real gap value
    # Run 2: 10 more rows, 2s apart, distinguishable coherence value.
    for i in range(10):
        rows.append(
            _make_row(
                t=gap_start + timedelta(seconds=2 * i),
                channels={"coherence": 0.9, "energy": 0.9, "novelty": 0.9, "valence": 0.9},
                idx=idx,
            )
        )
        idx += 1

    windows = build_windows(rows, fields=fields, window_size=5, stride=1, max_gap_sec=6.0)

    # 6 windows per 10-row run of window_size=5, stride=1 (10-5+1=6), two runs -> 12.
    assert len(windows) == 12
    for w in windows:
        coherence_values = w.reshape(5, len(fields))[:, 0]
        # No window may mix run-1 and run-2 values -- the gap must have
        # broken the run, so every window is entirely 0.1s or entirely 0.9s.
        assert np.allclose(coherence_values, 0.1) or np.allclose(coherence_values, 0.9), (
            f"window spans the 8.09s gap: {coherence_values}"
        )


def test_build_windows_variable_field_count_and_missing_channel_fill() -> None:
    """Row width is not fixed in field_channel_corpus.v1 -- a row missing a
    requested field must contribute 0.0 for it, not raise."""
    from orion.mood_arc.fit_encoder import build_windows

    fields = ("a", "b", "c")
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = [
        _make_row(t=start + timedelta(seconds=2 * i), channels={"a": float(i), "b": float(i) * 2}, idx=i)
        for i in range(5)
    ]  # "c" never present on any row

    windows = build_windows(rows, fields=fields, window_size=5, stride=1, max_gap_sec=6.0)
    assert len(windows) == 1
    w = windows[0].reshape(5, len(fields))
    assert list(w[:, 0]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert list(w[:, 1]) == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert list(w[:, 2]) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_select_fields_filters_low_variance_and_excludes_by_name() -> None:
    from orion.mood_arc.fit_encoder import select_fields

    rng = np.random.default_rng(1)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(200):
        rows.append(
            _make_row(
                t=start + timedelta(seconds=i),
                channels={
                    "real_signal": float(rng.normal(0, 1)),
                    "flat_signal": 0.5,  # zero variance -- must be dropped
                    "recent_perturbation_count": float(rng.integers(0, 3)),  # varying but excluded by name
                },
                idx=i,
            )
        )

    fields = select_fields(rows, exclude=frozenset({"recent_perturbation_count"}), variance_eps=1e-6)
    assert fields == ("real_signal",)


def test_select_fields_respects_custom_exclude() -> None:
    from orion.mood_arc.fit_encoder import select_fields

    rng = np.random.default_rng(1)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = [
        _make_row(
            t=start + timedelta(seconds=i),
            channels={"a": float(rng.normal(0, 1)), "b": float(rng.normal(0, 1))},
            idx=i,
        )
        for i in range(50)
    ]

    fields = select_fields(rows, exclude=frozenset({"a"}), variance_eps=1e-6)
    assert fields == ("b",)


def test_prune_correlated_fields_drops_lower_variance_member() -> None:
    from orion.mood_arc.fit_encoder import prune_correlated_fields

    rng = np.random.default_rng(1)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(300):
        base = float(rng.normal(0, 1))
        rows.append(
            _make_row(
                t=start + timedelta(seconds=i),
                channels={
                    "high_var": base * 10.0,  # same underlying signal, larger variance
                    "low_var": base * 1.0 + rng.normal(0, 0.001),  # near-duplicate, smaller variance
                    "independent": float(rng.normal(0, 1)),  # uncorrelated with the above
                },
                idx=i,
            )
        )

    kept = prune_correlated_fields(rows, ("high_var", "low_var", "independent"), corr_threshold=0.9)
    assert kept == ("high_var", "independent")


def test_prune_correlated_fields_collapses_a_three_way_correlated_chain() -> None:
    """Three mutually-correlated channels (all pairwise |r| > threshold,
    derived from one shared base signal at different scales/noise levels)
    must collapse to a SINGLE survivor -- the highest-variance member --
    regardless of which pair the greedy descending-|r| pass happens to
    consider first. Whichever pair is processed first drops its
    lower-variance member; the next pair involving the highest-variance
    channel then drops whatever remains, since it always has strictly more
    variance than either other member. This exercises the "already-dropped
    members are skipped, survivors are re-compared" interaction the plain
    2-member-pair tests don't reach."""
    from orion.mood_arc.fit_encoder import prune_correlated_fields

    rng = np.random.default_rng(4)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(300):
        base = float(rng.normal(0, 1))
        rows.append(
            _make_row(
                t=start + timedelta(seconds=i),
                channels={
                    "chain_biggest": base * 10.0 + rng.normal(0, 0.05),
                    "chain_medium": base * 5.0 + rng.normal(0, 0.05),
                    "chain_smallest": base * 2.0 + rng.normal(0, 0.05),
                    "unrelated": float(rng.normal(0, 1)),
                },
                idx=i,
            )
        )

    kept = prune_correlated_fields(
        rows, ("chain_biggest", "chain_medium", "chain_smallest", "unrelated"), corr_threshold=0.9
    )
    assert kept == ("chain_biggest", "unrelated")


def test_prune_correlated_fields_leaves_uncorrelated_channels_alone() -> None:
    from orion.mood_arc.fit_encoder import prune_correlated_fields

    rng = np.random.default_rng(2)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = [
        _make_row(
            t=start + timedelta(seconds=i),
            channels={"x": float(rng.normal(0, 1)), "y": float(rng.normal(0, 1))},
            idx=i,
        )
        for i in range(300)
    ]

    kept = prune_correlated_fields(rows, ("x", "y"), corr_threshold=0.9)
    assert kept == ("x", "y")


def test_fit_ar1_per_channel_dict_shaped() -> None:
    from orion.mood_arc.fit_encoder import fit_ar1_per_channel

    fields = ("coherence", "energy", "novelty", "valence")
    rng = np.random.default_rng(3)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    n = 400
    rows: list[FieldChannelCorpusRowV1] = []
    val = {f: 0.5 for f in fields}
    for i in range(n):
        t = start + timedelta(seconds=2 * i)
        channels = {}
        for f in fields:
            val[f] = 0.9 * val[f] + rng.normal(0, 0.05)
            channels[f] = val[f]
        rows.append(_make_row(t=t, channels=channels, idx=i))

    ar1_params = fit_ar1_per_channel(rows, fields=fields, max_gap_sec=6.0)
    assert set(ar1_params.keys()) == set(fields)
    for f in fields:
        a, b, innovation_std = ar1_params[f]
        # a should be close to the true AR(1) coefficient (0.9), loosely.
        assert 0.5 < a < 1.1
        assert innovation_std > 0.0


def test_generate_ar1_surrogate_windows_matches_variance_but_not_identical() -> None:
    from orion.mood_arc.fit_encoder import (
        fit_ar1_per_channel,
        generate_ar1_surrogate_windows,
    )

    fields = ("coherence", "energy", "novelty", "valence")
    rng = np.random.default_rng(3)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    n = 400
    rows: list[FieldChannelCorpusRowV1] = []
    val = {f: 0.5 for f in fields}
    for i in range(n):
        t = start + timedelta(seconds=2 * i)
        channels = {}
        for f in fields:
            val[f] = 0.9 * val[f] + rng.normal(0, 0.05)
            channels[f] = val[f]
        rows.append(_make_row(t=t, channels=channels, idx=i))

    ar1_params = fit_ar1_per_channel(rows, fields=fields, max_gap_sec=6.0)

    window_size = 30
    n_fields = len(fields)
    held_windows = np.stack(
        [
            np.asarray(
                [r.channels[f] for r in rows[i : i + window_size] for f in fields],
                dtype=np.float64,
            )
            for i in range(0, n - window_size, window_size)
        ]
    )

    surrogates = generate_ar1_surrogate_windows(
        held_windows, ar1_params, window_size=window_size, fields=fields, seed=7
    )

    assert surrogates.shape == held_windows.shape
    # Surrogates must not reproduce the exact real sequence.
    assert not np.array_equal(surrogates, held_windows)

    real_var = np.var(held_windows)
    surrogate_var = np.var(surrogates)
    # Similar order of magnitude variance (matched autocorrelation
    # structure), not wildly different -- loose bound, this is a sanity
    # check not a precise statistical equivalence test.
    assert surrogate_var > 0
    assert 0.1 < (surrogate_var / real_var) < 10.0


def test_compute_window_probes_raises_not_implemented() -> None:
    """This is a deliberate, documented gap (open valence-replacement
    question, not this task's scope) -- not a bug. Must raise, not
    fabricate a substitute probe."""
    from orion.mood_arc.fit_encoder import compute_window_probes

    with pytest.raises(NotImplementedError, match="field_channel_corpus.v1"):
        compute_window_probes(np.zeros((1, 12)), {"W2": np.zeros((4, 2))}, ("a", "b", "c"))
