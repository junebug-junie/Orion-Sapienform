#!/usr/bin/env python3
"""Offline mood-arc windowed felt-state-trajectory autoencoder: train subcommand.

Item 2 of docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md.
Trains a shallow MLP autoencoder over flattened windows of item 1's
mood_arc_corpus.v1 JSONL corpus (coherence/energy/novelty/valence per tick).
Artifacts: manifest.json, weights.npz, probes.json under --out. Dark
deployment -- disk-only, no bus publish, no cognition consumer.

This file mirrors scripts/fit_phi_encoder.py's harness shape (corpus gates,
Adam-free-but-analogous training loop, write_artifacts triad,
_percentile/_pearson reuse) but is NOT a copy: the phi script's
_init_weights/_forward/_losses/_apply_grads carry a vestigial phi-prediction
head with no meaning here, and use plain-SGD/zero-init, which a same-session
spike found to fail to converge on this problem (scored worse than a
trivial "repeat the window's own mean" baseline). This script's
init_weights/forward/recon_loss_and_grads are a dedicated, phi-head-free
rewrite using Adam and a mean-initialized decoder bias -- the fix.

A same-session spike also found the original spec's single shuffle-baseline
gate is too weak on its own: this corpus's real autocorrelation is largely
explained by a known, deliberate leaky-integrator decay mechanism
(BIOMETRICS_FIELD_DECAY_RATE=0.92,
services/orion-field-digester/app/digestion/decay.py), so an encoder can
pass the shuffle floor purely by learning that already-known filter. This
script adds a second, non-gating "ceiling" comparison against a
matched-autocorrelation AR(1) surrogate, plus a purged/embargoed temporal
train/held-out split (naive random window sampling leaks given ~10-15 tick
autocorrelation from 50%-overlapping windows) and a block-bootstrap
confidence interval on the floor ratio. None of this is in the original
written spec doc -- see MoodArcEncoderManifestV1's docstring for the
field-level rationale.

Only `train` is implemented here. `promote`/`infer` are later roadmap items
(3+), not built by this patch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/fit_mood_arc_encoder.py` puts scripts/ on
# sys.path[0], which shadows stdlib `platform` via scripts/platform/ and
# breaks pydantic (same issue documented in scripts/fit_phi_encoder.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from orion.schemas.telemetry.mood_arc import MoodArcCorpusRowV1, MoodArcEncoderManifestV1
from orion.schemas.telemetry.phi_encoder import CorpusStatsV1, TrainingStatsV1
from orion.telemetry.corpus_rotation import resolve_rotated_corpus_files

# Reused as-is from fit_phi_encoder.py -- these operate on plain np.ndarrays
# and are genuinely shape-agnostic (percentile-of-a-list, Pearson-of-two-
# vectors). Everything else in that file (_init_weights/_forward/_losses/
# _apply_grads) carries phi-specific baggage (a vestigial phi-prediction
# head, plain-SGD/zero-init) and is deliberately NOT reused -- see module
# docstring.
from scripts.fit_phi_encoder import _pearson, _percentile  # noqa: E402

ARCHITECTURE = "mlp_shallow_v1"  # same architecture family as the phi encoder
FIELDS: tuple[str, ...] = ("coherence", "energy", "novelty", "valence")

# Evidence-based defaults from this session's spike (NOT the original spec's
# hidden_dim=8/latent_dim=4, which failed to beat even a mean-repeat
# baseline -- see module docstring).
DEFAULT_WINDOW_SIZE = 30
DEFAULT_STRIDE = 15
DEFAULT_MAX_GAP_SEC = 6.0
DEFAULT_HIDDEN_DIM = 32
DEFAULT_LATENT_DIM = 16
DEFAULT_EPOCHS = 500
DEFAULT_LR = 0.003
DEFAULT_HELD_OUT_FRAC = 0.15
DEFAULT_PURGE_GAP_WINDOWS = 6
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 10
DEFAULT_BOOTSTRAP_N = 200
DEFAULT_MIN_HOURS = 6.0
DEFAULT_MIN_ROWS = 500
DEFAULT_BATCH_SIZE = 32

# Hard gate threshold, unchanged from the original spec: real held-out recon
# loss must beat the shuffle baseline by at least 2x.
FLOOR_MAX_RATIO = 0.5


def _git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except OSError:
        pass
    return "unknown"


def _load_jsonl(path: Path) -> list[MoodArcCorpusRowV1]:
    """Rotation-aware corpus load, mirroring fit_phi_encoder.py's own
    pattern -- InnerStateCorpusSink (the class backing this corpus too)
    rotates at CORPUS_SINK_MAX_BYTES, so reading only `path` would silently
    see just the post-rotation slice once the active file first rotates."""
    rows: list[MoodArcCorpusRowV1] = []
    for file in resolve_rotated_corpus_files(path):
        for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(MoodArcCorpusRowV1.model_validate_json(text))
            except Exception as exc:  # noqa: BLE001 -- corpus loader should skip bad lines with context
                raise ValueError(f"invalid JSONL at {file}:{line_no}: {exc}") from exc
    rows.sort(key=lambda r: r.generated_at)
    return rows


def _hours_span(rows: list[MoodArcCorpusRowV1]) -> float:
    if not rows:
        return 0.0
    start = min(r.generated_at for r in rows)
    end = max(r.generated_at for r in rows)
    return (end - start).total_seconds() / 3600.0


def check_corpus_gates(rows: list[MoodArcCorpusRowV1], *, min_rows: int, min_hours: float) -> None:
    """min_rows/min_hours are HARD gates (raise SystemExit), matching
    fit_phi_encoder.py's check_corpus_gates. Per-channel variance is
    reported separately (see report_channel_variance) but NOT hard-gated --
    unlike phi's SEEDV4_* variance constants (empirically tuned over time),
    there is no calibrated "too flat" threshold yet for this 4-channel
    signal. coherence/energy are known (this session) to have real but low
    variance in the current corpus; inventing a threshold here would be
    exactly the kind of ungrounded number CLAUDE.md's rules warn against."""
    if len(rows) < min_rows:
        raise SystemExit(f"corpus gate failed: min_rows={min_rows} got={len(rows)}")
    hours = _hours_span(rows)
    if hours < min_hours:
        raise SystemExit(f"corpus gate failed: min_hours={min_hours} got={hours:.3f}")


def report_channel_variance(rows: list[MoodArcCorpusRowV1]) -> dict[str, float]:
    """Reported, not gated -- see check_corpus_gates docstring."""
    if not rows:
        return {f: 0.0 for f in FIELDS}
    matrix = np.asarray([[getattr(r, f) for f in FIELDS] for r in rows], dtype=np.float64)
    variances = {f: float(np.var(matrix[:, i])) for i, f in enumerate(FIELDS)}
    for f, v in variances.items():
        print(f"channel variance: {f}={v:.6f}")
    return variances


def _build_windows_with_span(
    rows: list[MoodArcCorpusRowV1],
    *,
    window_size: int,
    stride: int,
    max_gap_sec: float,
) -> list[tuple[np.ndarray, datetime, datetime]]:
    """Internal: build_windows() plus each window's (start_ts, end_ts) --
    the generated_at of its first and last row. Neither timestamp is part
    of build_windows()'s public (spec-mandated) return shape, but cmd_train
    needs both: end_ts to report corpus stats, and start_ts to derive a
    leakage-safe cutoff for fit_ar1_per_channel's training-only rows that
    holds regardless of --purge-gap-windows (see cmd_train) rather than
    re-deriving the same run/stride math twice."""
    triples: list[tuple[np.ndarray, datetime, datetime]] = []
    if not rows:
        return triples

    runs: list[list[MoodArcCorpusRowV1]] = []
    current_run: list[MoodArcCorpusRowV1] = [rows[0]]
    for prev, row in zip(rows, rows[1:]):
        gap = (row.generated_at - prev.generated_at).total_seconds()
        if gap > max_gap_sec:
            runs.append(current_run)
            current_run = [row]
        else:
            current_run.append(row)
    runs.append(current_run)

    n_fields = len(FIELDS)
    for run in runs:
        if len(run) < window_size:
            continue
        for start in range(0, len(run) - window_size + 1, stride):
            chunk = run[start : start + window_size]
            vec = np.empty(window_size * n_fields, dtype=np.float64)
            for i, r in enumerate(chunk):
                base = i * n_fields
                vec[base + 0] = r.coherence
                vec[base + 1] = r.energy
                vec[base + 2] = r.novelty
                vec[base + 3] = r.valence
            triples.append((vec, chunk[0].generated_at, chunk[-1].generated_at))
    return triples


def build_windows(
    rows: list[MoodArcCorpusRowV1],
    *,
    window_size: int,
    stride: int,
    max_gap_sec: float,
) -> list[np.ndarray]:
    """Flatten contiguous runs of `rows` (strict timestamp order) into
    `(window_size*4,)` float vectors:
    `[c_0,e_0,n_0,v_0, c_1,e_1,n_1,v_1, ..., c_{W-1},e_{W-1},n_{W-1},v_{W-1}]`.
    A gap between consecutive rows' timestamps exceeding `max_gap_sec` breaks
    the run -- no window is built across it. Returns windows in timestamp
    order."""
    return [
        w
        for w, _, _ in _build_windows_with_span(
            rows, window_size=window_size, stride=stride, max_gap_sec=max_gap_sec
        )
    ]


def _purge_split_indices(n: int, held_out_frac: float, purge_gap_windows: int) -> tuple[int, int]:
    held_n = max(1, int(round(n * held_out_frac)))
    held_start = n - held_n
    purge_start = held_start - purge_gap_windows
    return purge_start, held_start


def purged_temporal_split(
    windows: list[np.ndarray],
    *,
    held_out_frac: float = DEFAULT_HELD_OUT_FRAC,
    purge_gap_windows: int = DEFAULT_PURGE_GAP_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    """windows must already be in strict temporal order (build_windows
    guarantees this). Held-out = the LAST held_out_frac windows by time --
    NOT a random sample, since random sampling over 50%-overlapping windows
    leaks (found this session). purge_gap_windows windows immediately before
    the held-out boundary are DROPPED entirely (neither train nor held-out)
    -- an embargo zone. Default purge_gap_windows=6 is a conservative ~180s
    buffer (stride=15 ticks=~30s per window-step * 6), well beyond the
    measured ~30-60s decorrelation horizon (ACF stays nonzero out to lag
    ~10-15 ticks this session) -- so even a held-out window immediately
    after the embargo zone is not meaningfully autocorrelated with the last
    training window.

    Returns (train_windows, held_out_windows) as stacked np.ndarray
    matrices. Raises ValueError (not a silent empty array) if there aren't
    enough windows to produce a nonempty train set and held-out set after
    purging."""
    n = len(windows)
    if n == 0:
        raise ValueError("purged_temporal_split: no windows to split")

    purge_start, held_start = _purge_split_indices(n, held_out_frac, purge_gap_windows)
    train_windows = windows[: max(0, purge_start)]
    held_windows = windows[held_start:]

    if not train_windows:
        raise ValueError(
            f"purged_temporal_split: no training windows remain after purge "
            f"(n={n}, held_out_frac={held_out_frac}, purge_gap_windows={purge_gap_windows}); "
            "reduce purge_gap_windows/held_out_frac or gather more corpus"
        )
    if not held_windows:
        raise ValueError(
            f"purged_temporal_split: no held-out windows (n={n}, held_out_frac={held_out_frac})"
        )
    return np.stack(train_windows, axis=0), np.stack(held_windows, axis=0)


def init_weights(d_in: int, hidden_dim: int, latent_dim: int, seed: int, data_mean: np.ndarray) -> dict[str, np.ndarray]:
    """W1,b1 (d_in->hidden, ReLU), W2,b2 (hidden->latent), W3,b3 (latent->d_in).

    b3 is initialized to data_mean (NOT zero) so training starts at the
    mean-baseline instead of having to learn it from scratch -- this was the
    root cause of this session's initial training failure (vanilla-SGD,
    zero-init scored worse than a trivial mean-repeat baseline because it
    never got anywhere close to it). W3 is scaled down (0.1x W1/W2's init
    scale) so its near-zero random component doesn't immediately fight the
    mean-start."""
    rng = np.random.default_rng(seed)
    scale = 0.05
    return {
        "W1": (rng.standard_normal((d_in, hidden_dim)) * scale).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": (rng.standard_normal((hidden_dim, latent_dim)) * scale).astype(np.float64),
        "b2": np.zeros(latent_dim, dtype=np.float64),
        "W3": (rng.standard_normal((latent_dim, d_in)) * scale * 0.1).astype(np.float64),
        "b3": np.asarray(data_mean, dtype=np.float64).copy(),
    }


def forward(x: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear -> ReLU -> linear bottleneck -> linear decoder. No phi head."""
    h_pre = x @ weights["W1"] + weights["b1"]
    h = np.maximum(0.0, h_pre)
    z = h @ weights["W2"] + weights["b2"]
    xhat = z @ weights["W3"] + weights["b3"]
    return h_pre, h, z, xhat


def recon_loss(x: np.ndarray, weights: dict[str, np.ndarray]) -> float:
    """Forward-only MSE reconstruction loss, no backward pass. Used
    everywhere a caller only wants the loss number (held-set eval, gate
    comparisons, probes) -- found by code review (2026-07-13) that
    reusing recon_loss_and_grads() for those call sites was computing a
    full, unused backward pass (6 outer-product gradient tensors) on every
    held-set evaluation, roughly doubling their cost for no reason."""
    _, _, _, xhat = forward(x, weights)
    recon = xhat - x
    return float(np.mean(recon**2))


def recon_loss_and_grads(x: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[float, dict[str, np.ndarray]]:
    """Pure MSE reconstruction loss + backprop, no phi term. Only used where
    gradients are actually consumed (the training step) -- see recon_loss()
    for the forward-only variant used everywhere else."""
    h_pre, h, z, xhat = forward(x, weights)
    recon = xhat - x
    loss = float(np.mean(recon**2))

    d_xhat = (2.0 / x.size) * recon
    d_W3 = np.outer(z, d_xhat)
    d_b3 = d_xhat

    d_z = d_xhat @ weights["W3"].T
    d_W2 = np.outer(h, d_z)
    d_b2 = d_z

    d_h = d_z @ weights["W2"].T
    relu_mask = (h_pre > 0.0).astype(np.float64)
    d_h_pre = d_h * relu_mask
    d_W1 = np.outer(x, d_h_pre)
    d_b1 = d_h_pre

    grads = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2, "W3": d_W3, "b3": d_b3}
    return loss, grads


class Adam:
    """Numpy-only Adam optimizer, no phi-specific baggage. Fixes the
    vanilla-SGD training failure this session's spike found -- SGD+zero-init
    never converged on this problem; Adam+mean-init converges in ~80-120
    epochs."""

    def __init__(
        self,
        weights: dict[str, np.ndarray],
        lr: float = DEFAULT_LR,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in weights.items()}

    def step(self, weights: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t
        for key in weights:
            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[key] / bias1
            v_hat = self.v[key] / bias2
            weights[key] = weights[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _per_window_losses(matrix: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([recon_loss(row, weights) for row in matrix], dtype=np.float64)


def train_autoencoder(
    train_matrix: np.ndarray,
    held_matrix: np.ndarray,
    *,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[dict[str, np.ndarray], TrainingStatsV1]:
    if train_matrix.shape[0] == 0:
        raise SystemExit("train: no training windows after purged temporal split")

    d_in = train_matrix.shape[1]
    data_mean = np.mean(train_matrix, axis=0)
    weights = init_weights(d_in, hidden_dim, latent_dim, seed, data_mean)
    optimizer = Adam(weights, lr=lr)
    rng = np.random.default_rng(seed)

    n_train = train_matrix.shape[0]
    best_held = float("inf")
    best_weights = {k: v.copy() for k, v in weights.items()}
    final_epoch = 0

    for epoch in range(1, epochs + 1):
        final_epoch = epoch
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            batch = train_matrix[idx]
            accum = {k: np.zeros_like(v) for k, v in weights.items()}
            for row in batch:
                _, grads = recon_loss_and_grads(row, weights)
                for k in accum:
                    accum[k] += grads[k]
            scale = 1.0 / max(1, len(batch))
            for k in accum:
                accum[k] *= scale
            optimizer.step(weights, accum)

        if held_matrix.shape[0] > 0:
            held_loss = float(np.mean(_per_window_losses(held_matrix, weights)))
        else:
            held_loss = float(np.mean(_per_window_losses(train_matrix, weights)))

        if held_loss < best_held:
            best_held = held_loss
            best_weights = {k: v.copy() for k, v in weights.items()}

    train_losses = _per_window_losses(train_matrix, best_weights)
    recon_errors = []
    for row in train_matrix:
        _, _, _, xhat = forward(row, best_weights)
        recon_errors.append(float(np.mean((row - xhat) ** 2)))

    stats = TrainingStatsV1(
        epochs=final_epoch,
        final_loss=float(np.mean(train_losses)) if train_losses.size else 0.0,
        held_out_loss=best_held,
        recon_error_p50=_percentile(recon_errors, 50),
        recon_error_p95=_percentile(recon_errors, 95),
    )
    return best_weights, stats


def shuffle_within_windows(X: np.ndarray, window_size: int, n_fields: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle tick order within each window independently, destroying
    temporal order but preserving the window's own marginal 4-dim
    distribution."""
    shuffled = X.copy()
    for i in range(shuffled.shape[0]):
        w = shuffled[i].reshape(window_size, n_fields)
        perm = rng.permutation(window_size)
        shuffled[i] = w[perm].reshape(-1)
    return shuffled


def fit_ar1_per_channel(
    train_rows: list[MoodArcCorpusRowV1],
    *,
    max_gap_sec: float | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Per raw field, fit x_t = a*x_{t-1} + b + noise via least squares on
    the TRAINING portion of the corpus only (not held-out -- this is a
    null-model fit, must not see held-out data any more than the real model
    does). Returns {field: (a, b, innovation_std)}.

    If max_gap_sec is given, lag-1 pairs whose timestamp gap exceeds it are
    dropped -- the same run-boundary discipline build_windows() uses, so an
    outage/restart boundary doesn't get treated as a real one-tick
    transition."""
    result: dict[str, tuple[float, float, float]] = {}
    for field in FIELDS:
        values: list[float] = []
        prev_row: MoodArcCorpusRowV1 | None = None
        pairs_prev: list[float] = []
        pairs_next: list[float] = []
        for row in train_rows:
            if prev_row is not None:
                gap = (row.generated_at - prev_row.generated_at).total_seconds()
                if max_gap_sec is None or gap <= max_gap_sec:
                    pairs_prev.append(getattr(prev_row, field))
                    pairs_next.append(getattr(row, field))
            prev_row = row
            values.append(getattr(row, field))

        if len(pairs_prev) < 3:
            mean_val = float(np.mean(values)) if values else 0.0
            result[field] = (0.0, mean_val, 1e-6)
            continue

        x_prev = np.asarray(pairs_prev, dtype=np.float64)
        x_next = np.asarray(pairs_next, dtype=np.float64)
        design = np.vstack([x_prev, np.ones_like(x_prev)]).T
        coeffs, *_ = np.linalg.lstsq(design, x_next, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
        resid = x_next - (a * x_prev + b)
        innovation_std = float(np.std(resid)) if resid.size > 1 else 1e-6
        result[field] = (a, b, max(innovation_std, 1e-6))
    return result


def generate_ar1_surrogate_windows(
    held_windows: np.ndarray,
    ar1_params: dict[str, tuple[float, float, float]],
    *,
    window_size: int,
    fields: tuple[str, ...] = FIELDS,
    seed: int,
) -> np.ndarray:
    """For each held-out window, simulate a synthetic replacement of the
    same shape: start from that window's own real first tick per field (so
    the starting point/marginal level matches), then roll forward
    window_size-1 steps using each field's fitted AR(1) coefficient/
    intercept plus i.i.d. Gaussian noise at the fitted innovation_std. This
    produces windows with the SAME autocorrelation structure as the known
    decay mechanism, but nothing beyond a single-lag linear process -- the
    "this is already explained by the known filter" null."""
    rng = np.random.default_rng(seed)
    n_fields = len(fields)
    surrogates = np.empty_like(held_windows)
    for i in range(held_windows.shape[0]):
        w = held_windows[i].reshape(window_size, n_fields)
        surro = np.empty_like(w)
        surro[0] = w[0]
        for f_idx, field in enumerate(fields):
            a, b, innovation_std = ar1_params[field]
            prev = float(w[0, f_idx])
            for t in range(1, window_size):
                noise = float(rng.normal(0.0, innovation_std))
                prev = a * prev + b + noise
                surro[t, f_idx] = prev
        surrogates[i] = surro.reshape(-1)
    return surrogates


def two_tier_gate(real_held_loss: float, shuffle_loss: float, ar1_surrogate_loss: float) -> dict:
    """floor_pass is the ONLY hard pass/fail (matches the original spec's
    gate, unchanged threshold: real held-out loss must beat the shuffle
    baseline by at least 2x). ceiling_ratio is reported but NOT hard-gated
    on an invented threshold -- we don't have a calibrated cutoff for it yet
    across multiple runs; it's recorded in the manifest for future
    calibration. This is intentional, not an oversight: inventing a ceiling
    threshold now, with a single training run's worth of evidence, would be
    exactly the kind of ungrounded number this project's own rules
    (CLAUDE.md) warn against -- the SEEDV4_* variance constants in
    fit_phi_encoder.py earned their thresholds from real tuning history;
    this hasn't yet."""
    floor_ratio = real_held_loss / max(shuffle_loss, 1e-12)
    ceiling_ratio = real_held_loss / max(ar1_surrogate_loss, 1e-12)
    return {
        "floor_ratio": floor_ratio,
        "floor_pass": floor_ratio < FLOOR_MAX_RATIO,
        "ceiling_ratio": ceiling_ratio,
    }


def block_bootstrap_ratio_ci(
    per_window_real_losses: np.ndarray,
    per_window_shuffle_losses: np.ndarray,
    *,
    block_size: int = DEFAULT_BOOTSTRAP_BLOCK_SIZE,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int,
) -> tuple[float, float]:
    """Resample CONTIGUOUS BLOCKS of held-out windows (size=block_size) with
    replacement -- not individual windows, since held-out windows are
    autocorrelated by construction (50% stride overlap plus the corpus's
    own real autocorrelation), so a naive i.i.d. bootstrap would overstate
    confidence. Returns the (2.5th, 97.5th) percentile of the resampled
    mean(real)/mean(shuffle) ratio distribution."""
    real = np.asarray(per_window_real_losses, dtype=np.float64)
    shuf = np.asarray(per_window_shuffle_losses, dtype=np.float64)
    n = real.size
    if n == 0:
        raise ValueError("block_bootstrap_ratio_ci: no held-out losses to bootstrap")

    rng = np.random.default_rng(seed)
    block_size = max(1, min(block_size, n))
    n_blocks_needed = int(np.ceil(n / block_size))
    ratios: list[float] = []
    for _ in range(n_boot):
        idx: list[int] = []
        for _ in range(n_blocks_needed):
            max_start = n - block_size
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            idx.extend(range(start, start + block_size))
        sample_idx = np.asarray(idx[:n])
        r_mean = float(np.mean(real[sample_idx]))
        s_mean = float(np.mean(shuf[sample_idx]))
        ratios.append(r_mean / max(s_mean, 1e-12))

    lo = float(np.percentile(ratios, 2.5))
    hi = float(np.percentile(ratios, 97.5))
    return lo, hi


def compute_window_probes(windows: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Per latent dim z_i, Pearson r against window summary stats:
    mean_valence, valence_range (max-min within window), sign_change_count
    (how many times valence crosses 0 within the window)."""
    if windows.shape[0] == 0:
        return {}
    latent_dim = int(weights["W2"].shape[1])
    n_fields = len(FIELDS)
    window_size = windows.shape[1] // n_fields
    valence_idx = FIELDS.index("valence")

    z_cols: list[list[float]] = [[] for _ in range(latent_dim)]
    mean_valence: list[float] = []
    valence_range: list[float] = []
    sign_change_count: list[float] = []

    for i in range(windows.shape[0]):
        x = windows[i]
        _, _, z, _ = forward(x, weights)
        for d in range(latent_dim):
            z_cols[d].append(float(z[d]))
        w = x.reshape(window_size, n_fields)
        series = w[:, valence_idx]
        mean_valence.append(float(np.mean(series)))
        valence_range.append(float(np.max(series) - np.min(series)))
        signs = np.sign(series)
        sign_change_count.append(float(np.sum(signs[1:] != signs[:-1])))

    summary_cols = {
        "mean_valence": np.asarray(mean_valence, dtype=np.float64),
        "valence_range": np.asarray(valence_range, dtype=np.float64),
        "sign_change_count": np.asarray(sign_change_count, dtype=np.float64),
    }
    probes: dict[str, dict[str, float]] = {}
    for d in range(latent_dim):
        z_arr = np.asarray(z_cols[d], dtype=np.float64)
        probes[f"z{d}"] = {
            name: round(_pearson(z_arr, vals), 4) for name, vals in summary_cols.items()
        }
    return probes


def write_artifacts(
    out_dir: Path,
    *,
    manifest: MoodArcEncoderManifestV1,
    weights: dict[str, np.ndarray],
    probes: dict[str, dict[str, float]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    np.savez(out_dir / "weights.npz", **weights)
    (out_dir / "probes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline mood-arc windowed autoencoder fit")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a candidate mood-arc encoder")
    train_p.add_argument("--corpus", type=Path, required=True, help="MoodArcCorpusRowV1 JSONL corpus")
    train_p.add_argument("--out", type=Path, required=True, help="Output directory for trained artifacts")
    train_p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    train_p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    train_p.add_argument("--max-gap-sec", type=float, default=DEFAULT_MAX_GAP_SEC)
    train_p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    train_p.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    train_p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train_p.add_argument("--lr", type=float, default=DEFAULT_LR)
    train_p.add_argument("--held-out-frac", type=float, default=DEFAULT_HELD_OUT_FRAC)
    train_p.add_argument("--purge-gap-windows", type=int, default=DEFAULT_PURGE_GAP_WINDOWS)
    train_p.add_argument("--bootstrap-block-size", type=int, default=DEFAULT_BOOTSTRAP_BLOCK_SIZE)
    train_p.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    train_p.add_argument("--min-hours", type=float, default=DEFAULT_MIN_HOURS)
    train_p.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument(
        "--encoder-version", type=str, default=None, help="manifest encoder_version (default: out dir name)"
    )
    return parser.parse_args(argv)


def cmd_train(args: argparse.Namespace) -> int:
    rows = _load_jsonl(args.corpus)
    check_corpus_gates(rows, min_rows=args.min_rows, min_hours=args.min_hours)
    variances = report_channel_variance(rows)

    triples = _build_windows_with_span(
        rows, window_size=args.window_size, stride=args.stride, max_gap_sec=args.max_gap_sec
    )
    if not triples:
        raise SystemExit(
            "train: no windows built from corpus (check window-size/stride/max-gap-sec "
            "against row count/gaps)"
        )
    windows = [w for w, _, _ in triples]
    start_ts = [s for _, s, _ in triples]
    end_ts = [e for _, _, e in triples]
    n_windows = len(windows)

    _purge_start, held_start = _purge_split_indices(n_windows, args.held_out_frac, args.purge_gap_windows)
    train_matrix, held_matrix = purged_temporal_split(
        windows, held_out_frac=args.held_out_frac, purge_gap_windows=args.purge_gap_windows
    )
    # AR(1) leakage-safe cutoff: rows strictly before the FIRST held-out
    # window's own start timestamp. Deliberately not derived from
    # purge_gap_windows/end_ts of the last training window -- found by code
    # review (2026-07-13) that with 50%-overlapping windows (window_size=30,
    # stride=15), a small --purge-gap-windows override (e.g. 0) could still
    # leave the last "training" window's end timestamp *after* the first
    # held-out window's start (they physically overlap in raw ticks), which
    # would leak held-out rows into fit_ar1_per_channel's training-only fit
    # regardless of how large purge_gap_windows claims to be. Using the
    # held-out boundary's own start_ts directly is correct for any
    # purge_gap_windows value, including 0.
    cutoff_ts = start_ts[held_start]
    train_rows_for_ar1 = [r for r in rows if r.generated_at < cutoff_ts]

    best_weights, training_stats = train_autoencoder(
        train_matrix,
        held_matrix,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    real_held_losses = _per_window_losses(held_matrix, best_weights)
    real_held_loss = float(np.mean(real_held_losses))

    shuffle_rng = np.random.default_rng(args.seed + 1)
    shuffled_held = shuffle_within_windows(held_matrix, args.window_size, len(FIELDS), shuffle_rng)
    shuffle_losses = _per_window_losses(shuffled_held, best_weights)
    shuffle_baseline_loss = float(np.mean(shuffle_losses))

    ar1_params = fit_ar1_per_channel(train_rows_for_ar1, max_gap_sec=args.max_gap_sec)
    surrogate_held = generate_ar1_surrogate_windows(
        held_matrix, ar1_params, window_size=args.window_size, fields=FIELDS, seed=args.seed + 2
    )
    ar1_losses = _per_window_losses(surrogate_held, best_weights)
    ar1_surrogate_loss = float(np.mean(ar1_losses))

    gate = two_tier_gate(real_held_loss, shuffle_baseline_loss, ar1_surrogate_loss)
    ci_low, ci_high = block_bootstrap_ratio_ci(
        real_held_losses,
        shuffle_losses,
        block_size=args.bootstrap_block_size,
        n_boot=args.bootstrap_n,
        seed=args.seed + 3,
    )

    probes = compute_window_probes(held_matrix, best_weights)

    times = [r.generated_at for r in rows]
    encoder_version = args.encoder_version or args.out.name
    manifest = MoodArcEncoderManifestV1(
        encoder_id=f"mood-arc-encoder:{encoder_version}",
        encoder_version=encoder_version,
        status="candidate",
        architecture=ARCHITECTURE,
        window_size=args.window_size,
        stride=args.stride,
        max_gap_sec=args.max_gap_sec,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        corpus=CorpusStatsV1(
            corpus_path=str(args.corpus),
            row_count=len(rows),
            excluded_degenerate=0,  # no per-row degenerate filter for this corpus (no phi_health field)
            time_range_start=min(times) if times else None,
            time_range_end=max(times) if times else None,
        ),
        training=training_stats,
        shuffle_baseline_loss=shuffle_baseline_loss,
        purge_gap_windows=args.purge_gap_windows,
        ar1_surrogate_loss=ar1_surrogate_loss,
        ceiling_ratio=gate["ceiling_ratio"],
        floor_ratio_ci_low=ci_low,
        floor_ratio_ci_high=ci_high,
        git_sha=_git_sha(),
        trained_at=datetime.now(timezone.utc),
    )
    write_artifacts(args.out, manifest=manifest, weights=best_weights, probes=probes)

    result = {
        "trained": str(args.out),
        "rows": len(rows),
        "windows_total": n_windows,
        "windows_train": int(train_matrix.shape[0]),
        "windows_held_out": int(held_matrix.shape[0]),
        "channel_variance": variances,
        "gate": {
            "floor_ratio": gate["floor_ratio"],
            "floor_pass": gate["floor_pass"],
            "ceiling_ratio": gate["ceiling_ratio"],
            "floor_ratio_ci_low": ci_low,
            "floor_ratio_ci_high": ci_high,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "train":
        return cmd_train(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
