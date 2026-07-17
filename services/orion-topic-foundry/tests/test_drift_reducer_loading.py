from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import dump

from app.services.drift import _load_reducer
from app.settings import settings


def test_load_reducer_returns_none_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "topic_foundry_model_dir", str(tmp_path))
    assert _load_reducer("no-such-model", "v1") is None


def test_load_reducer_returns_none_when_version_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "topic_foundry_model_dir", str(tmp_path))
    assert _load_reducer("some-model", None) is None


def test_load_reducer_loads_persisted_reducer(monkeypatch, tmp_path):
    """Round-trips training.py's _write_artifacts persistence convention:
    <model_dir>/registry/<name>/versions/<version>/model/reducer.joblib."""
    monkeypatch.setattr(settings, "topic_foundry_model_dir", str(tmp_path))
    model_dir = Path(tmp_path) / "registry" / "some-model" / "versions" / "v1" / "model"
    model_dir.mkdir(parents=True)

    # A plain picklable object stands in for a fitted UMAP reducer here --
    # this test only proves the load path/round-trip, not UMAP's own
    # transform behavior (covered by test_training_umap_reduction.py).
    fake_reducer = {"n_components": 5, "marker": "fake-umap-reducer"}
    dump(fake_reducer, model_dir / "reducer.joblib")

    loaded = _load_reducer("some-model", "v1")
    assert loaded == fake_reducer
