"""Tests for cmd_promote / promote_encoder (2026-07-27).

Every real training run before this patch wrote its artifact only to a
--out path that has, in practice, always been scratch (/tmp/.../scratchpad/
...) -- gone the moment the scratch dir was cleaned up. Confirmed live: no
artifact survived from the 2026-07-18 production run that actually passed
its own gate. promote_encoder() is the missing durable-persistence step;
these tests focus on its one hard invariant -- fail-closed on an
unverified or failing candidate -- since that's the property most worth
protecting against a future regression.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from orion.mood_arc.fit_encoder import (
    load_artifacts,
    promote_encoder,
    resolve_active_encoder_dir,
    write_artifacts,
)
from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1
from orion.schemas.telemetry.mood_arc import MoodArcEncoderManifestV1
from orion.schemas.telemetry.phi_encoder import CorpusStatsV1, TrainingStatsV1

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
    return FieldChannelCorpusRowV1(generated_at=t, tick_id=f"tick_{idx}", channels=channels)


def _write_corpus(path: Path, rows: list[FieldChannelCorpusRowV1]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


def _normal_rows(n_rows: int = 1500, seed: int = 0) -> list[FieldChannelCorpusRowV1]:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows: list[FieldChannelCorpusRowV1] = []
    for i in range(n_rows):
        t = start + timedelta(seconds=2 * i)
        phase = i * 0.05
        channels = {
            "coherence": float(0.6 + 0.05 * np.sin(phase) + rng.normal(0, 0.01)),
            "energy": float(0.3 + 0.1 * np.sin(phase * 0.7) + rng.normal(0, 0.02)),
            "novelty": float(0.5 + 0.3 * np.sin(phase * 1.3) + rng.normal(0, 0.03)),
        }
        rows.append(_make_row(t=t, channels=channels, idx=i))
    return rows


def _train_small_encoder(tmp_path: Path, *, encoder_version: str = "v-test") -> Path:
    corpus = tmp_path / "train_corpus.jsonl"
    out_dir = tmp_path / "candidate"
    _write_corpus(corpus, _normal_rows())
    proc = _run_fit(
        "train",
        "--corpus", str(corpus),
        "--out", str(out_dir),
        "--encoder-version", encoder_version,
        "--hidden-dim", "32",
        "--latent-dim", "16",
        "--epochs", "150",
        "--min-hours", "0.5",
        "--min-rows", "100",
        "--purge-gap-windows", "6",
        "--bootstrap-n", "20",
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return out_dir


def _fake_manifest(*, encoder_version: str, floor_pass: bool | None) -> MoodArcEncoderManifestV1:
    now = datetime.now(timezone.utc)
    return MoodArcEncoderManifestV1(
        encoder_id=f"mood-arc-encoder:{encoder_version}",
        encoder_version=encoder_version,
        status="candidate",
        architecture="mlp_shallow_v1",
        window_size=30,
        stride=15,
        max_gap_sec=6.0,
        hidden_dim=32,
        latent_dim=16,
        channel_names=["coherence", "energy", "novelty"],
        corpus=CorpusStatsV1(
            corpus_path="fake.jsonl", row_count=100, excluded_degenerate=0,
            time_range_start=now, time_range_end=now,
        ),
        training=TrainingStatsV1(
            epochs=10, final_loss=0.01, held_out_loss=0.02,
            recon_error_p50=0.03, recon_error_p95=0.05,
        ),
        shuffle_baseline_loss=0.04,
        floor_ratio=0.3 if floor_pass else 0.9,
        floor_pass=floor_pass,
        git_sha="deadbeef",
        trained_at=now,
    )


def _fake_candidate_dir(tmp_path: Path, *, encoder_version: str, floor_pass: bool | None) -> Path:
    candidate_dir = tmp_path / f"candidate-{encoder_version}"
    manifest = _fake_manifest(encoder_version=encoder_version, floor_pass=floor_pass)
    write_artifacts(candidate_dir, manifest=manifest, weights={"w": np.zeros((2, 2))}, probes={})
    return candidate_dir


def test_promote_real_trained_candidate_end_to_end(tmp_path: Path) -> None:
    """The real CLI (train, via subprocess, same as the anomaly-detector
    tests) produces a candidate; promote_encoder must accept it if and only
    if it actually passed its own gate, and the promoted copy must be a
    genuine, independent copy (not the same files/directory)."""
    candidate_dir = _train_small_encoder(tmp_path, encoder_version="v-real-1")
    manifest, _ = load_artifacts(candidate_dir)
    models_root = tmp_path / "models_root"

    if not manifest.floor_pass:
        pytest.skip(
            "this small synthetic corpus's real training run did not pass its own "
            "gate this time (floor_ratio can vary run to run on a tiny fixture) -- "
            "not this test's concern, see test_promote_refuses_failing_candidate "
            "for the fail-closed behavior itself"
        )

    result = promote_encoder(candidate_dir, models_root=models_root)

    dest_dir = Path(result["promoted"])
    assert dest_dir.exists()
    assert dest_dir != candidate_dir
    promoted_manifest, promoted_weights = load_artifacts(dest_dir)
    assert promoted_manifest.status == "active"
    assert promoted_manifest.promoted_at is not None
    assert set(promoted_weights.keys()) == set(load_artifacts(candidate_dir)[1].keys())

    active_pointer = json.loads((models_root / "active.json").read_text(encoding="utf-8"))
    assert active_pointer["encoder_version"] == "v-real-1"
    assert active_pointer["path"] == str(dest_dir)


def test_promote_refuses_candidate_with_no_recorded_gate_result(tmp_path: Path) -> None:
    """floor_pass=None (a manifest written before this field existed, or one
    that never ran the gate) must never be silently treated as a pass."""
    candidate_dir = _fake_candidate_dir(tmp_path, encoder_version="v-none", floor_pass=None)
    with pytest.raises(ValueError, match="floor_pass"):
        promote_encoder(candidate_dir, models_root=tmp_path / "models_root")
    assert not (tmp_path / "models_root").exists()


def test_promote_refuses_failing_candidate(tmp_path: Path) -> None:
    candidate_dir = _fake_candidate_dir(tmp_path, encoder_version="v-fail", floor_pass=False)
    with pytest.raises(ValueError, match="floor_pass"):
        promote_encoder(candidate_dir, models_root=tmp_path / "models_root")
    assert not (tmp_path / "models_root").exists()


def test_promote_refuses_to_overwrite_existing_version(tmp_path: Path) -> None:
    models_root = tmp_path / "models_root"
    candidate_dir = _fake_candidate_dir(tmp_path, encoder_version="v-dup", floor_pass=True)
    promote_encoder(candidate_dir, models_root=models_root)
    with pytest.raises(ValueError, match="existing"):
        promote_encoder(candidate_dir, models_root=models_root)


def test_promote_retires_previous_active_version(tmp_path: Path) -> None:
    """Promoting a second version must flip the first version's own
    persisted manifest status to 'retired', not just repoint active.json --
    a consumer reading version v1's manifest directly (not through the
    pointer) must see that it's no longer current."""
    models_root = tmp_path / "models_root"
    v1_dir = _fake_candidate_dir(tmp_path, encoder_version="v1", floor_pass=True)
    result1 = promote_encoder(v1_dir, models_root=models_root)
    v1_dest = Path(result1["promoted"])
    assert MoodArcEncoderManifestV1.model_validate_json(
        (v1_dest / "manifest.json").read_text(encoding="utf-8")
    ).status == "active"

    v2_dir = _fake_candidate_dir(tmp_path, encoder_version="v2", floor_pass=True)
    result2 = promote_encoder(v2_dir, models_root=models_root)
    assert result2["previous_active"]["encoder_version"] == "v1"

    v1_manifest_after = MoodArcEncoderManifestV1.model_validate_json(
        (v1_dest / "manifest.json").read_text(encoding="utf-8")
    )
    assert v1_manifest_after.status == "retired"

    active_pointer = json.loads((models_root / "active.json").read_text(encoding="utf-8"))
    assert active_pointer["encoder_version"] == "v2"


def test_failed_promotion_leaves_previous_active_version_untouched(tmp_path: Path) -> None:
    """Regression test for a review-caught ordering bug: retiring the
    previous active version and flipping active.json used to happen BEFORE
    the new candidate's artifacts were copied. A failure during the copy
    step (simulated here by deleting the candidate's weights.npz after
    write_artifacts wrote it, so load_artifacts/shutil.copy2 hits a missing
    file) must leave the previously-promoted version fully, genuinely
    active -- not retired-with-nothing-to-replace-it."""
    models_root = tmp_path / "models_root"
    v1_dir = _fake_candidate_dir(tmp_path, encoder_version="v1", floor_pass=True)
    result1 = promote_encoder(v1_dir, models_root=models_root)
    v1_dest = Path(result1["promoted"])

    v2_dir = _fake_candidate_dir(tmp_path, encoder_version="v2", floor_pass=True)
    (v2_dir / "weights.npz").unlink()  # simulate a corrupt/missing source mid-promotion

    with pytest.raises(FileNotFoundError):
        promote_encoder(v2_dir, models_root=models_root)

    v1_manifest_after = MoodArcEncoderManifestV1.model_validate_json(
        (v1_dest / "manifest.json").read_text(encoding="utf-8")
    )
    assert v1_manifest_after.status == "active"

    active_pointer = json.loads((models_root / "active.json").read_text(encoding="utf-8"))
    assert active_pointer["encoder_version"] == "v1"
    assert active_pointer["path"] == str(v1_dest)


def test_resolve_active_encoder_dir_reads_promoted_path(tmp_path: Path) -> None:
    """The direct read-side mirror of promote_encoder(): resolving a
    models_root must return exactly the directory the most recent promotion
    wrote, tracking a second promotion too (not a cached/stale first read)."""
    models_root = tmp_path / "models_root"
    v1_dir = _fake_candidate_dir(tmp_path, encoder_version="v1", floor_pass=True)
    result1 = promote_encoder(v1_dir, models_root=models_root)
    assert resolve_active_encoder_dir(models_root) == Path(result1["promoted"])

    v2_dir = _fake_candidate_dir(tmp_path, encoder_version="v2", floor_pass=True)
    result2 = promote_encoder(v2_dir, models_root=models_root)
    assert resolve_active_encoder_dir(models_root) == Path(result2["promoted"])


def test_resolve_active_encoder_dir_raises_file_not_found_when_active_json_missing(
    tmp_path: Path,
) -> None:
    """Nothing has ever been promoted to this models_root -- fail loud, not
    a guessed/default directory."""
    models_root = tmp_path / "models_root"
    models_root.mkdir()
    with pytest.raises(FileNotFoundError, match="active.json"):
        resolve_active_encoder_dir(models_root)


def test_resolve_active_encoder_dir_raises_value_error_on_malformed_json(tmp_path: Path) -> None:
    models_root = tmp_path / "models_root"
    models_root.mkdir()
    (models_root / "active.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        resolve_active_encoder_dir(models_root)


def test_resolve_active_encoder_dir_raises_value_error_when_path_key_missing(tmp_path: Path) -> None:
    models_root = tmp_path / "models_root"
    models_root.mkdir()
    (models_root / "active.json").write_text(json.dumps({"encoder_version": "v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        resolve_active_encoder_dir(models_root)


def test_resolve_active_encoder_dir_raises_value_error_on_stale_pointer(tmp_path: Path) -> None:
    """active.json points at a real path, but that path's manifest.json is
    gone -- a stale pointer left over from a promoted directory that was
    later removed, not a directory that was ever genuinely active right now."""
    models_root = tmp_path / "models_root"
    models_root.mkdir()
    stale_dir = tmp_path / "vanished"
    stale_dir.mkdir()
    (models_root / "active.json").write_text(
        json.dumps({"encoder_id": "x", "encoder_version": "v1", "path": str(stale_dir)}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest.json"):
        resolve_active_encoder_dir(models_root)
