"""Gate tests for train/evals/eval_phi_encoder_health.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
EVAL_PATH = (
    REPO
    / "services"
    / "orion-spark-introspector"
    / "train"
    / "evals"
    / "eval_phi_encoder_health.py"
)


def _load_eval_mod():
    spec = importlib.util.spec_from_file_location("phi_encoder_health_eval", EVAL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def health():
    return _load_eval_mod()


def test_headline_fit_detects_identity_collapse(health) -> None:
    rng = np.random.default_rng(0)
    headline = rng.normal(0.65, 0.02, size=200)
    phi = headline.copy()  # exact copy
    fit = health.headline_fit_stats(phi, headline)
    assert fit.residual_std < 1e-9
    assert fit.near_identity_frac == 1.0
    reasons = health.collapse_reasons(
        fit, residual_std_max=1e-3, near_id_frac_max=0.5
    )
    assert reasons
    assert any("residual_std" in r for r in reasons)


def test_headline_fit_healthy_residual(health) -> None:
    rng = np.random.default_rng(1)
    headline = rng.normal(0.65, 0.02, size=200)
    phi = 0.7 * headline + 0.2 + rng.normal(0.0, 0.01, size=200)
    fit = health.headline_fit_stats(phi, headline)
    assert fit.residual_std > 1e-3
    assert fit.near_identity_frac < 0.5
    reasons = health.collapse_reasons(
        fit, residual_std_max=1e-3, near_id_frac_max=0.5
    )
    assert reasons == []


def test_constant_phi_is_not_identity_collapse(health) -> None:
    """Low residual_std alone must not flag collapse when slope≠1."""
    rng = np.random.default_rng(2)
    headline = rng.normal(0.65, 0.02, size=200)
    phi = np.full(200, 0.5) + rng.normal(0.0, 1e-4, size=200)
    fit = health.headline_fit_stats(phi, headline)
    assert fit.residual_std < 1e-3
    assert abs(fit.slope - 1.0) > 0.05
    reasons = health.collapse_reasons(
        fit, residual_std_max=1e-3, near_id_frac_max=0.5
    )
    assert reasons == []


def test_recon_stats(health) -> None:
    stats = health.recon_stats(np.asarray([0.01, 0.02, 0.04, 0.08]))
    assert stats.mean == pytest.approx(0.0375)
    assert stats.p50 == pytest.approx(0.03)
    assert stats.p95 > stats.p50


def test_discover_encoder_dirs_skips_active_symlink(health, tmp_path: Path) -> None:
    root = tmp_path / "encoders"
    root.mkdir()
    v1 = root / "v1"
    v1.mkdir()
    (v1 / "manifest.json").write_text("{}", encoding="utf-8")
    (v1 / "weights.npz").write_bytes(b"x")
    active = root / "active"
    active.symlink_to(v1)
    found = health.discover_encoder_dirs(root)
    assert found == [v1]
    assert health.resolve_active_target(root) == v1.resolve()


def test_build_report_fails_when_active_collapsed(health) -> None:
    collapsed = health.RunHealth(
        encoder_version="v-bad",
        dir_name="v-bad",
        is_active_symlink=True,
        status="active",
        features_version="seed-v4",
        architecture="mlp_shallow_v1",
        git_sha="abc",
        trained_at=None,
        promoted_at=None,
        parent_version=None,
        corpus_row_count_train=100,
        corpus_excluded_train=0,
        corpus_time_range_start=None,
        corpus_time_range_end=None,
        train_final_loss=0.1,
        train_held_out_loss=0.1,
        train_recon_p50=0.01,
        train_recon_p95=0.05,
        eval_rows=50,
        recon={"mean": 0.01, "p50": 0.01, "p95": 0.05},
        headline_fit={
            "residual_std": 1e-6,
            "near_identity_frac": 1.0,
            "phi_headline_corr": 1.0,
        },
        collapsed=True,
        collapse_reasons=["residual_std too low"],
    )
    healthy = health.RunHealth(
        encoder_version="v-good",
        dir_name="v-good",
        is_active_symlink=False,
        status="retired",
        features_version="seed-v4",
        architecture="mlp_shallow_v1",
        git_sha="def",
        trained_at=None,
        promoted_at=None,
        parent_version=None,
        corpus_row_count_train=200,
        corpus_excluded_train=0,
        corpus_time_range_start=None,
        corpus_time_range_end=None,
        train_final_loss=0.05,
        train_held_out_loss=0.05,
        train_recon_p50=0.005,
        train_recon_p95=0.02,
        eval_rows=50,
        recon={"mean": 0.005, "p50": 0.005, "p95": 0.02},
        headline_fit={
            "residual_std": 0.01,
            "near_identity_frac": 0.0,
            "phi_headline_corr": 0.8,
        },
        collapsed=False,
    )
    report = health.build_report(
        [collapsed, healthy],
        corpus_path=Path("/tmp/c.jsonl"),
        encoders_root=Path("/tmp/e"),
        residual_std_max=1e-3,
        near_id_frac_max=0.5,
        recon_p95_warn=0.25,
    )
    assert report["ok"] is False
    assert report["active_encoder_version"] == "v-bad"
    assert report["ranking_by_recon_p95_same_features_version"][0]["encoder_version"] == "v-good"
    text = health.format_text_report(report)
    assert "COLLAPSE" in text
    assert "v-bad" in text
    assert "v-good" in text


def _write_mini_encoder(run_dir: Path, *, version: str, status: str, features_version: str) -> None:
    """Write a tiny valid MLP matching PhiEncoderRuntime shape checks."""
    d_in, h, d_lat = 3, 4, 2
    features = ["agency_readiness", "execution_pressure", "overall_intensity"]
    rng = np.random.default_rng(abs(hash(version)) % (2**31))
    weights = {
        "W1": rng.normal(0, 0.1, size=(d_in, h)),
        "b1": np.zeros(h),
        "W2": rng.normal(0, 0.1, size=(h, d_lat)),
        "b2": np.zeros(d_lat),
        "W3": rng.normal(0, 0.1, size=(d_lat, d_in)),
        "b3": np.zeros(d_in),
        "w_phi": rng.normal(0, 0.1, size=(d_lat,)),
        "b_phi": np.asarray(0.0),
    }
    np.savez(run_dir / "weights.npz", **weights)
    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    manifest = {
        "encoder_id": f"phi-encoder:{version}",
        "encoder_version": version,
        "parent_version": None,
        "status": status,
        "architecture": "mlp_shallow_v1",
        "features_version": features_version,
        "input_features": features,
        "hidden_dim": h,
        "latent_dim": d_lat,
        "corpus": {
            "corpus_path": "/tmp/corpus.jsonl",
            "row_count": 10,
            "excluded_degenerate": 0,
            "time_range_start": now.isoformat(),
            "time_range_end": now.isoformat(),
        },
        "training": {
            "epochs": 1,
            "final_loss": 0.1,
            "held_out_loss": 0.1,
            "recon_error_p50": 0.01,
            "recon_error_p95": 0.05,
        },
        "probes": {},
        "git_sha": "deadbeef",
        "trained_at": now.isoformat(),
        "promoted_at": now.isoformat() if status == "active" else None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_corpus(path: Path, *, features_version: str, n: int = 40) -> None:
    from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1

    features_names = ["agency_readiness", "execution_pressure", "overall_intensity"]
    start = datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc)
    lines: list[str] = []
    for i in range(n):
        t = start.replace(minute=i % 60, second=i % 60)
        feats = [
            InnerFeatureV1(
                name=name,
                raw_value=0.4 + 0.01 * i,
                scaled_value=float(np.sin(i * 0.2 + j)),
                source=f"test.{name}",
            )
            for j, name in enumerate(features_names)
        ]
        row = InnerStateFeaturesV1(
            features_version=features_version,
            generated_at=t,
            features=feats,
            headline=0.6 + 0.001 * i,
            phi_health="ok",
            grammar_truth_degraded=False,
        )
        lines.append(row.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_evaluate_all_runs_end_to_end(health, tmp_path: Path) -> None:
    root = tmp_path / "encoders"
    root.mkdir()
    a = root / "v-a"
    b = root / "v-b"
    a.mkdir()
    b.mkdir()
    _write_mini_encoder(a, version="v-a", status="active", features_version="seed-v4")
    _write_mini_encoder(b, version="v-b", status="retired", features_version="seed-v4")
    (root / "active").symlink_to(a)

    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, features_version="seed-v4")

    PhiEncoderRuntime = health._load_phi_encoder_runtime()
    active_target = health.resolve_active_target(root)
    runs = [
        health.evaluate_run(
            run_dir=d,
            rows=health.load_corpus_rows(corpus),
            active_target=active_target,
            phi_runtime_cls=PhiEncoderRuntime,
            residual_std_max=1e-3,
            near_id_frac_max=0.5,
            recon_p95_warn=10.0,
        )
        for d in health.discover_encoder_dirs(root)
    ]
    assert len(runs) == 2
    assert {r.encoder_version for r in runs} == {"v-a", "v-b"}
    assert any(r.is_active_symlink for r in runs)
    assert all(not r.skipped for r in runs)
    assert all(r.eval_rows == 40 for r in runs)
    report = health.build_report(
        runs,
        corpus_path=corpus,
        encoders_root=root,
        residual_std_max=1e-3,
        near_id_frac_max=0.5,
        recon_p95_warn=10.0,
    )
    assert "runs" in report and len(report["runs"]) == 2
    assert report["active_encoder_version"] == "v-a"
    # Random tiny MLP should not identity-copy headline.
    assert report["ok"] is True
