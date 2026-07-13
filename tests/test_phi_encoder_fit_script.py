from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1
from orion.schemas.telemetry.phi_encoder import PhiEncoderManifestV1

REPO = Path(__file__).resolve().parents[1]
FIT_SCRIPT = REPO / "scripts" / "fit_phi_encoder.py"
PROMOTE_FIXTURE = REPO / "fixtures" / "phi_encoder_promote_gate.jsonl"

FELT_NAMES = (
    "coherence",
    "field_intensity",
    "agency_readiness",
    "execution_pressure",
    "reasoning_pressure",
    "resource_pressure",
    "reliability_pressure",
    "continuity_pressure",
    "social_pressure",
    "introspection_pressure",
)
COGNITIVE_NAMES = (
    "recall_gate_fired",
    "reasoning_present",
    "exec_step_fail_rate",
    "execution_friction",
)
INPUT_FEATURES = list(FELT_NAMES) + ["overall_intensity"] + list(COGNITIVE_NAMES)


def _run_fit(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _feature(name: str, raw: float, scaled: float | None = None) -> InnerFeatureV1:
    return InnerFeatureV1(
        name=name,
        raw_value=raw,
        scaled_value=raw if scaled is None else scaled,
        source=f"test.{name}",
    )


def _inner_row(
    *,
    t: datetime,
    scaled_by_name: dict[str, float] | None = None,
    phi_health: str = "ok",
    grammar_truth_degraded: bool = False,
    headline: float = 0.5,
) -> InnerStateFeaturesV1:
    scaled_by_name = scaled_by_name or {}
    features = []
    for name in INPUT_FEATURES:
        raw = float(np.sin(t.timestamp() + hash(name) % 7) * 0.4 + 0.5)
        scaled = scaled_by_name.get(name, raw * 0.5)
        features.append(_feature(name, raw, scaled))
    return InnerStateFeaturesV1(
        features_version="seed-v2",
        generated_at=t,
        features=features,
        headline=headline,
        phi_health=phi_health,
        grammar_truth_degraded=grammar_truth_degraded,
    )


def _write_corpus(path: Path, rows: list[InnerStateFeaturesV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.model_dump(mode="json"), separators=(",", ":")) + "\n")


def _synthetic_corpus(n_rows: int = 120, hours_span: float = 2.0) -> list[InnerStateFeaturesV1]:
    start = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    step = timedelta(hours=hours_span / max(1, n_rows - 1))
    rows: list[InnerStateFeaturesV1] = []
    for i in range(n_rows):
        t = start + step * i
        scaled = {name: float(np.sin(i * 0.17 + j * 0.31)) for j, name in enumerate(INPUT_FEATURES)}
        rows.append(_inner_row(t=t, scaled_by_name=scaled, headline=0.3 + (i % 10) * 0.05))
    return rows


def test_fit_refuses_small_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "tiny.jsonl"
    corpus.write_text("")  # empty
    proc = _run_fit("--corpus", str(corpus), "--out", str(tmp_path / "out"))
    assert proc.returncode != 0
    assert "min_rows" in proc.stdout.lower() or "min_rows" in proc.stderr.lower()


def test_fit_trains_on_synthetic_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "train.jsonl"
    out_dir = tmp_path / "encoder_v1"
    _write_corpus(corpus, _synthetic_corpus())
    proc = _run_fit(
        "--corpus",
        str(corpus),
        "--out",
        str(out_dir),
        "--features-version",
        "seed-v2",
        "--hidden-dim",
        "8",
        "--latent-dim",
        "4",
        "--min-rows",
        "100",
        "--min-hours",
        "1.0",
        "--epochs",
        "30",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "weights.npz").is_file()
    assert (out_dir / "probes.json").is_file()
    manifest = PhiEncoderManifestV1.model_validate_json((out_dir / "manifest.json").read_text())
    assert manifest.architecture == "mlp_shallow_v1"
    assert manifest.features_version == "seed-v2"
    assert manifest.input_features == INPUT_FEATURES
    weights = np.load(out_dir / "weights.npz")
    for key in ("W1", "b1", "W2", "b2", "W3", "b3", "w_phi", "b_phi"):
        assert key in weights


def test_eval_only_runs_promote_gate(tmp_path: Path) -> None:
    corpus = tmp_path / "train.jsonl"
    out_dir = tmp_path / "encoder_v1"
    _write_corpus(corpus, _synthetic_corpus())
    train = _run_fit(
        "--corpus",
        str(corpus),
        "--out",
        str(out_dir),
        "--features-version",
        "seed-v2",
        "--hidden-dim",
        "8",
        "--latent-dim",
        "4",
        "--min-rows",
        "100",
        "--min-hours",
        "1.0",
        "--epochs",
        "40",
    )
    assert train.returncode == 0, train.stderr or train.stdout
    proc = _run_fit("--eval-only", "--manifest", str(out_dir))
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "promote_gate" in (proc.stdout + proc.stderr).lower()


def test_promote_symlinks_active(tmp_path: Path) -> None:
    corpus = tmp_path / "train.jsonl"
    version_dir = tmp_path / "encoders" / "v20260708"
    encoders_root = tmp_path / "encoders"
    _write_corpus(corpus, _synthetic_corpus())
    train = _run_fit(
        "--corpus",
        str(corpus),
        "--out",
        str(version_dir),
        "--features-version",
        "seed-v2",
        "--hidden-dim",
        "8",
        "--latent-dim",
        "4",
        "--min-rows",
        "100",
        "--min-hours",
        "1.0",
        "--epochs",
        "20",
    )
    assert train.returncode == 0, train.stderr or train.stdout
    eval_proc = _run_fit("--eval-only", "--manifest", str(version_dir))
    assert eval_proc.returncode == 0, eval_proc.stderr or eval_proc.stdout
    promote = _run_fit(
        "--promote",
        "--version",
        "v20260708",
        "--encoders-root",
        str(encoders_root),
    )
    assert promote.returncode == 0, promote.stderr or promote.stdout
    active = encoders_root / "active"
    assert active.is_symlink()
    assert active.resolve() == version_dir.resolve()
    manifest = PhiEncoderManifestV1.model_validate_json((version_dir / "manifest.json").read_text())
    assert manifest.status == "active"


@pytest.mark.parametrize("path", [PROMOTE_FIXTURE])
def test_promote_fixture_exists(path: Path) -> None:
    assert path.is_file()
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 4


def test_input_features_seed_v3_excludes_flat_dims() -> None:
    from scripts.fit_phi_encoder import input_features_for_version

    names = input_features_for_version("seed-v3")
    assert "field_intensity" not in names
    assert "resource_pressure" not in names
    assert "introspection_pressure" not in names
    assert "reliability_pressure" not in names
    assert len(names) == 11


def test_variance_gate_seed_v3_needs_nine_of_eleven() -> None:
    from scripts.fit_phi_encoder import _variance_gate

    rng = np.random.default_rng(0)
    matrix = np.zeros((50, 11), dtype=np.float64)
    for col in range(9):
        matrix[:, col] = rng.standard_normal(50)
    ok, got, need = _variance_gate(matrix, fraction=0.8, eps=1e-6)
    assert need == 9
    assert got == 9
    assert ok is True

    flat = np.zeros((50, 11), dtype=np.float64)
    flat[:, :8] = rng.standard_normal((50, 8))
    ok2, got2, need2 = _variance_gate(flat, fraction=0.8, eps=1e-6)
    assert need2 == 9
    assert got2 == 8
    assert ok2 is False


SEEDV4_NAMES = [
    "agency_readiness",
    "execution_pressure",
    "reasoning_pressure",
    "overall_intensity",
    "recall_gate_fired",
    "reasoning_present",
    "execution_load",
    "reasoning_load",
]


def test_variance_gate_seedv4_needs_six_when_both_reasoning_dims_dead() -> None:
    from scripts.fit_phi_encoder import _variance_gate

    rng = np.random.default_rng(1)
    matrix = np.zeros((50, 8), dtype=np.float64)
    # Live: all 6 non-reasoning dims (indices 0-4, 6). Dead: reasoning_present
    # (5), reasoning_load (7) -- no thinking-capable model active.
    for col in (0, 1, 2, 3, 4, 6):
        matrix[:, col] = rng.standard_normal(50)
    ok, got, need = _variance_gate(matrix, fraction=0.8, eps=1e-6, feature_names=SEEDV4_NAMES)
    assert need == 6
    assert got == 6
    assert ok is True


def test_variance_gate_seedv4_needs_seven_when_one_reasoning_dim_live() -> None:
    from scripts.fit_phi_encoder import _variance_gate

    rng = np.random.default_rng(2)
    matrix = np.zeros((50, 8), dtype=np.float64)
    # Same 6 core dims live, PLUS reasoning_present (index 5) now live too --
    # got=7. Bar raises to need=7 the moment either reasoning dim shows signal.
    for col in (0, 1, 2, 3, 4, 5, 6):
        matrix[:, col] = rng.standard_normal(50)
    ok, got, need = _variance_gate(matrix, fraction=0.8, eps=1e-6, feature_names=SEEDV4_NAMES)
    assert need == 7
    assert got == 7
    assert ok is True


def test_variance_gate_seedv4_fails_below_six_even_with_both_reasoning_dead() -> None:
    from scripts.fit_phi_encoder import _variance_gate

    rng = np.random.default_rng(3)
    matrix = np.zeros((50, 8), dtype=np.float64)
    # Only 5 of the 6 core dims live -> below the relaxed 6/8 floor.
    for col in (0, 1, 2, 3, 4):
        matrix[:, col] = rng.standard_normal(50)
    ok, got, need = _variance_gate(matrix, fraction=0.8, eps=1e-6, feature_names=SEEDV4_NAMES)
    assert need == 6
    assert got == 5
    assert ok is False


def test_variance_gate_seedv4_needs_seven_when_both_reasoning_dims_live() -> None:
    from scripts.fit_phi_encoder import _variance_gate

    rng = np.random.default_rng(4)
    matrix = rng.standard_normal((50, 8))  # all 8 live
    ok, got, need = _variance_gate(matrix, fraction=0.8, eps=1e-6, feature_names=SEEDV4_NAMES)
    assert need == 7
    assert got == 8
    assert ok is True


def test_variance_gate_ignores_seedv4_policy_without_feature_names() -> None:
    """Without feature_names, behavior stays the plain fraction-based gate --
    backward compatible for legacy/seed-v3 callers that never pass names."""
    from scripts.fit_phi_encoder import _variance_gate

    rng = np.random.default_rng(5)
    matrix = np.zeros((50, 8), dtype=np.float64)
    for col in (0, 1, 2, 3, 4, 6):
        matrix[:, col] = rng.standard_normal(50)
    ok, got, need = _variance_gate(matrix, fraction=0.8, eps=1e-6)
    assert need == 7  # ceil(0.8 * 8), not the seed-v4-aware 6
    assert got == 6
    assert ok is False


def test_variance_gate_seedv3_feature_names_unaffected_by_seedv4_policy() -> None:
    """seed-v3's feature set doesn't contain reasoning_present/reasoning_load
    at all, so passing its own feature_names must fall through to the plain
    fraction-based gate untouched."""
    from scripts.fit_phi_encoder import _variance_gate, input_features_for_version

    names = input_features_for_version("seed-v3")
    rng = np.random.default_rng(6)
    matrix = np.zeros((50, len(names)), dtype=np.float64)
    for col in range(9):
        matrix[:, col] = rng.standard_normal(50)
    ok, got, need = _variance_gate(matrix, fraction=0.8, eps=1e-6, feature_names=names)
    assert need == int(np.ceil(0.8 * len(names)))
    assert got == 9


def test_load_jsonl_reads_across_rotated_corpus_files(tmp_path: Path) -> None:
    """2026-07-13, found by code review: InnerStateCorpusSink gained
    size-based rotation the same day this eval/training script's corpus
    loader existed. _load_jsonl(path) reading only the single active file
    would silently see just the post-rotation slice once a real corpus
    ever rotates -- no error, just fewer rows than an operator would
    assume. Fixed via orion.telemetry.corpus_rotation.resolve_rotated_
    corpus_files, the single shared source of truth for this pattern
    (also used by inner_state_sink.py's pruning and eval_phi_encoder_
    health.py's loader -- this test pins fit_phi_encoder.py's usage of it).
    """
    from scripts.fit_phi_encoder import _load_jsonl

    corpus_path = tmp_path / "inner_state.jsonl"
    rotated_path = tmp_path / "inner_state.jsonl.20260701T000000.000000Z"
    stray_path = tmp_path / "inner_state.jsonl.manual-backup"  # must be ignored, wrong pattern

    old_rows = _synthetic_corpus(n_rows=3, hours_span=1.0)
    new_rows = _synthetic_corpus(n_rows=2, hours_span=1.0)
    _write_corpus(rotated_path, old_rows)
    _write_corpus(corpus_path, new_rows)
    stray_path.write_text("not a real corpus row\n", encoding="utf-8")

    from orion.telemetry.corpus_rotation import resolve_rotated_corpus_files

    files = resolve_rotated_corpus_files(corpus_path)
    assert rotated_path in files
    assert corpus_path in files
    assert stray_path not in files
    # Rotated (older) file must come before the active file.
    assert files.index(rotated_path) < files.index(corpus_path)

    loaded = _load_jsonl(corpus_path)
    assert len(loaded) == len(old_rows) + len(new_rows)


def test_load_jsonl_single_file_unchanged_when_no_rotation_happened(tmp_path: Path) -> None:
    """The common case today (no rotation has ever fired) must behave
    identically to the pre-fix single-file read -- no rotated siblings on
    disk, resolve_rotated_corpus_files returns exactly the one active file."""
    from scripts.fit_phi_encoder import _load_jsonl
    from orion.telemetry.corpus_rotation import resolve_rotated_corpus_files

    corpus_path = tmp_path / "inner_state.jsonl"
    rows = _synthetic_corpus(n_rows=5, hours_span=1.0)
    _write_corpus(corpus_path, rows)

    assert resolve_rotated_corpus_files(corpus_path) == [corpus_path]
    assert len(_load_jsonl(corpus_path)) == len(rows)
