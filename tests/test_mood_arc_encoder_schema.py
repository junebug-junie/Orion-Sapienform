from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.telemetry.mood_arc import MoodArcCorpusRowV1, MoodArcEncoderManifestV1
from orion.schemas.telemetry.phi_encoder import CorpusStatsV1, TrainingStatsV1


def test_registry_resolves_mood_arc_schemas() -> None:
    assert resolve("MoodArcCorpusRowV1") is MoodArcCorpusRowV1
    assert resolve("MoodArcEncoderManifestV1") is MoodArcEncoderManifestV1
    assert SCHEMA_REGISTRY["MoodArcCorpusRowV1"].model is MoodArcCorpusRowV1
    assert SCHEMA_REGISTRY["MoodArcEncoderManifestV1"].model is MoodArcEncoderManifestV1
    assert SCHEMA_REGISTRY["MoodArcCorpusRowV1"].kind == "self.mood_arc_corpus.v1"
    assert SCHEMA_REGISTRY["MoodArcEncoderManifestV1"].kind == "self.mood_arc_encoder.manifest.v1"


def _manifest_kwargs() -> dict:
    return dict(
        encoder_id="mood-arc-test",
        encoder_version="v0-test",
        parent_version=None,
        status="candidate",
        architecture="mlp_shallow_v1",
        window_size=16,
        stride=4,
        max_gap_sec=30.0,
        hidden_dim=8,
        latent_dim=2,
        corpus=CorpusStatsV1(
            corpus_path="/tmp/mood_arc.jsonl",
            row_count=100,
            excluded_degenerate=0,
            time_range_start=datetime(2026, 7, 1, tzinfo=timezone.utc),
            time_range_end=datetime(2026, 7, 13, tzinfo=timezone.utc),
        ),
        training=TrainingStatsV1(
            epochs=10,
            final_loss=0.05,
            held_out_loss=0.06,
            recon_error_p50=0.04,
            recon_error_p95=0.09,
        ),
        shuffle_baseline_loss=0.5,
        git_sha="deadbeef",
        trained_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        promoted_at=None,
    )


def test_mood_arc_encoder_manifest_round_trips() -> None:
    manifest = MoodArcEncoderManifestV1(**_manifest_kwargs())
    restored = MoodArcEncoderManifestV1.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_mood_arc_encoder_manifest_forbids_unexpected_kwarg() -> None:
    kwargs = _manifest_kwargs()
    kwargs["not_a_real_field"] = "nope"
    with pytest.raises(ValidationError):
        MoodArcEncoderManifestV1(**kwargs)
