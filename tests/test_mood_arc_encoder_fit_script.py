from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from orion.schemas.telemetry.mood_arc import MoodArcCorpusRowV1, MoodArcEncoderManifestV1

REPO = Path(__file__).resolve().parents[1]
FIT_SCRIPT = REPO / "scripts" / "fit_mood_arc_encoder.py"


def _run_fit(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _make_row(
    *,
    t: datetime,
    coherence: float,
    energy: float,
    novelty: float,
    valence: float,
    idx: int,
) -> MoodArcCorpusRowV1:
    return MoodArcCorpusRowV1(
        generated_at=t,
        self_state_id=f"tick_{idx}",
        coherence=coherence,
        energy=energy,
        novelty=novelty,
        valence=valence,
        valence_source="proxy",
        dominant_node="node:athena",
    )


def _synthetic_corpus_with_real_pattern(n_rows: int = 1500, seed: int = 0) -> list[MoodArcCorpusRowV1]:
    """A deliberately-injected real periodic/trend pattern per field, plus
    noise -- distinct from pure noise so the floor gate (real beats a
    within-window shuffle) is expected to genuinely pass."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows: list[MoodArcCorpusRowV1] = []
    for i in range(n_rows):
        t = start + timedelta(seconds=2 * i)
        phase = i * 0.05
        coherence = 0.6 + 0.05 * np.sin(phase) + rng.normal(0, 0.01)
        energy = 0.3 + 0.1 * np.sin(phase * 0.7) + rng.normal(0, 0.02)
        novelty = 0.5 + 0.3 * np.sin(phase * 1.3) + rng.normal(0, 0.03)
        valence = 0.2 * np.sin(phase * 0.9) + rng.normal(0, 0.02)
        rows.append(
            _make_row(
                t=t,
                coherence=float(coherence),
                energy=float(energy),
                novelty=float(novelty),
                valence=float(valence),
                idx=i,
            )
        )
    return rows


def _write_corpus(path: Path, rows: list[MoodArcCorpusRowV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


def test_fit_refuses_small_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "tiny.jsonl"
    corpus.write_text("")  # empty
    proc = _run_fit("train", "--corpus", str(corpus), "--out", str(tmp_path / "out"))
    assert proc.returncode != 0
    assert "min_rows" in proc.stdout.lower() or "min_rows" in proc.stderr.lower()


def test_train_end_to_end_passes_floor_gate_on_synthetic_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "mood_arc.jsonl"
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

    # cmd_train prints per-channel variance lines before the final JSON
    # result blob -- slice from the first "{" to isolate it.
    result = json.loads(proc.stdout[proc.stdout.index("{") :])
    gate = result["gate"]
    assert gate["floor_pass"] is True, f"expected floor gate to pass on real periodic pattern, got {gate}"
    assert gate["floor_ratio"] < 0.5

    # New methodology fields populated and sane.
    assert manifest.ar1_surrogate_loss is not None
    assert manifest.ceiling_ratio is not None
    assert manifest.floor_ratio_ci_low is not None
    assert manifest.floor_ratio_ci_high is not None
    assert manifest.floor_ratio_ci_low <= manifest.floor_ratio_ci_high

    weights = np.load(out_dir / "weights.npz")
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        assert key in weights

    probes = json.loads((out_dir / "probes.json").read_text())
    assert len(probes) == 16  # one entry per latent dim
    # At least one latent dim must show |r| > 0.4 against at least one
    # summary stat (per the original spec's own acceptance check).
    max_abs_r = max(abs(v) for dim_probes in probes.values() for v in dim_probes.values())
    assert max_abs_r > 0.4, f"expected at least one strong probe correlation, got max |r|={max_abs_r}"


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
    from scripts.fit_mood_arc_encoder import (
        build_windows,
        init_weights,
        _per_window_losses,
        purged_temporal_split,
        shuffle_within_windows,
        two_tier_gate,
    )

    rows = _synthetic_corpus_with_real_pattern()
    windows = build_windows(rows, window_size=30, stride=15, max_gap_sec=6.0)
    train_matrix, held_matrix = purged_temporal_split(windows, held_out_frac=0.15, purge_gap_windows=6)

    data_mean = np.mean(train_matrix, axis=0)
    # No training loop at all -- init_weights() output used as-is, the
    # trivially-bad encoder the spec's acceptance check requires.
    untrained_weights = init_weights(train_matrix.shape[1], hidden_dim=32, latent_dim=16, seed=99, data_mean=data_mean)

    real_losses = _per_window_losses(held_matrix, untrained_weights)
    shuffle_rng = np.random.default_rng(2)
    shuffled_held = shuffle_within_windows(held_matrix, window_size=30, n_fields=4, rng=shuffle_rng)
    shuffle_losses = _per_window_losses(shuffled_held, untrained_weights)

    gate = two_tier_gate(float(np.mean(real_losses)), float(np.mean(shuffle_losses)), 1.0)
    assert gate["floor_pass"] is False, f"untrained weights should fail the floor gate, got {gate}"


def test_purged_temporal_split_boundary_and_exclusion_count() -> None:
    from scripts.fit_mood_arc_encoder import purged_temporal_split

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
    from scripts.fit_mood_arc_encoder import purged_temporal_split

    windows = [np.full(4, float(i)) for i in range(5)]  # far too few for held_out_frac + purge
    with pytest.raises(ValueError, match="purged_temporal_split"):
        purged_temporal_split(windows, held_out_frac=0.15, purge_gap_windows=6)


def test_purged_temporal_split_raises_on_empty_windows() -> None:
    from scripts.fit_mood_arc_encoder import purged_temporal_split

    with pytest.raises(ValueError, match="no windows to split"):
        purged_temporal_split([], held_out_frac=0.15, purge_gap_windows=6)


def test_build_windows_breaks_on_real_recorded_gap() -> None:
    """Uses this session's own recorded max gap (8.09s) against the default
    max_gap_sec=6.0 to confirm a run correctly breaks there."""
    from scripts.fit_mood_arc_encoder import build_windows

    start = datetime(2026, 7, 13, 6, 0, 0, tzinfo=timezone.utc)
    rows: list[MoodArcCorpusRowV1] = []
    idx = 0
    # Run 1: 10 rows, 2s apart, coherence pinned to a distinguishable value.
    for i in range(10):
        rows.append(
            _make_row(
                t=start + timedelta(seconds=2 * i),
                coherence=0.1,
                energy=0.1,
                novelty=0.1,
                valence=0.1,
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
                coherence=0.9,
                energy=0.9,
                novelty=0.9,
                valence=0.9,
                idx=idx,
            )
        )
        idx += 1

    windows = build_windows(rows, window_size=5, stride=1, max_gap_sec=6.0)

    # 6 windows per 10-row run of window_size=5, stride=1 (10-5+1=6), two runs -> 12.
    assert len(windows) == 12
    for w in windows:
        coherence_values = w.reshape(5, 4)[:, 0]
        # No window may mix run-1 and run-2 values -- the gap must have
        # broken the run, so every window is entirely 0.1s or entirely 0.9s.
        assert np.allclose(coherence_values, 0.1) or np.allclose(coherence_values, 0.9), (
            f"window spans the 8.09s gap: {coherence_values}"
        )


def test_generate_ar1_surrogate_windows_matches_variance_but_not_identical() -> None:
    from scripts.fit_mood_arc_encoder import (
        FIELDS,
        fit_ar1_per_channel,
        generate_ar1_surrogate_windows,
    )

    rng = np.random.default_rng(3)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    n = 400
    rows: list[MoodArcCorpusRowV1] = []
    val = {f: 0.5 for f in FIELDS}
    for i in range(n):
        t = start + timedelta(seconds=2 * i)
        new_vals = {}
        for f in FIELDS:
            val[f] = 0.9 * val[f] + rng.normal(0, 0.05)
            new_vals[f] = val[f]
        rows.append(
            _make_row(
                t=t,
                coherence=new_vals["coherence"],
                energy=new_vals["energy"],
                novelty=new_vals["novelty"],
                valence=new_vals["valence"],
                idx=i,
            )
        )

    ar1_params = fit_ar1_per_channel(rows, max_gap_sec=6.0)

    window_size = 30
    n_fields = len(FIELDS)
    held_windows = np.stack(
        [
            np.asarray(
                [getattr(r, f) for r in rows[i : i + window_size] for f in FIELDS],
                dtype=np.float64,
            )
            for i in range(0, n - window_size, window_size)
        ]
    )

    surrogates = generate_ar1_surrogate_windows(
        held_windows, ar1_params, window_size=window_size, fields=FIELDS, seed=7
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
