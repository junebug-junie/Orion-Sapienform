"""Drift band parsing + wiring (daily/weekly/monthly checks per tick,
services/orion-topic-foundry/app/services/drift.py). See
docs/superpowers/... investigation: a single fixed 24h window was too noisy
on a sparse, bursty personal chat corpus, so the daemon now checks multiple
window sizes per tick, each stamped with its own window_label."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services.drift import _parse_drift_bands, drift_daemon_loop


class TestParseDriftBands:
    def test_parses_multiple_valid_bands(self):
        assert _parse_drift_bands("daily:24,weekly:168,monthly:720") == [
            ("daily", 24),
            ("weekly", 168),
            ("monthly", 720),
        ]

    def test_tolerates_surrounding_whitespace(self):
        assert _parse_drift_bands(" daily : 24 , weekly:168 ") == [("daily", 24), ("weekly", 168)]

    def test_empty_string_yields_no_bands(self):
        assert _parse_drift_bands("") == []

    def test_skips_malformed_entries_without_raising(self):
        # missing hours, non-numeric hours, missing label, zero hours -- all
        # skipped individually rather than failing the whole parse.
        assert _parse_drift_bands("daily:24,broken,weekly:notanumber,:168,monthly:0,quarterly:2160") == [
            ("daily", 24),
            ("quarterly", 2160),
        ]

    def test_trailing_comma_ignored(self):
        assert _parse_drift_bands("daily:24,") == [("daily", 24)]


class TestRunDriftCheckStampsLabel:
    def test_default_label_is_custom(self, monkeypatch):
        """The manual /drift/run path (DriftRunRequest.window_label unset)
        must keep working exactly as before this patch -- 'custom' is the
        explicit fallback, not silently None."""
        import app.services.drift as drift_mod

        calls = {}

        def fake_insert_drift_record(**kwargs):
            calls.update(kwargs)

        monkeypatch.setattr(drift_mod, "fetch_active_model_by_name", lambda name: {"model_id": "0" * 8 + "-0000-0000-0000-000000000000", "version": "v1"})
        monkeypatch.setattr(drift_mod, "fetch_latest_completed_run", lambda model_id: {"artifact_paths": {}})
        monkeypatch.setattr(drift_mod, "insert_drift_record", fake_insert_drift_record)
        monkeypatch.setattr(drift_mod, "_load_baseline_distribution", lambda run_row: ({"0": 1.0}, 0.0, 1.0))
        monkeypatch.setattr(drift_mod, "_compute_current_distribution", lambda *a, **k: ({"0": 1.0}, 0.0, 1.0))

        drift_mod.run_drift_check(
            model_name="m",
            window_days=None,
            window_hours=24,
            threshold_js=None,
            threshold_outlier=None,
        )
        assert calls["window_label"] == "custom"

    def test_explicit_label_is_stamped(self, monkeypatch):
        import app.services.drift as drift_mod

        calls = {}
        monkeypatch.setattr(drift_mod, "fetch_active_model_by_name", lambda name: {"model_id": "0" * 8 + "-0000-0000-0000-000000000000", "version": "v1"})
        monkeypatch.setattr(drift_mod, "fetch_latest_completed_run", lambda model_id: {"artifact_paths": {}})
        monkeypatch.setattr(drift_mod, "insert_drift_record", lambda **kwargs: calls.update(kwargs))
        monkeypatch.setattr(drift_mod, "_load_baseline_distribution", lambda run_row: ({"0": 1.0}, 0.0, 1.0))
        monkeypatch.setattr(drift_mod, "_compute_current_distribution", lambda *a, **k: ({"0": 1.0}, 0.0, 1.0))

        drift_mod.run_drift_check(
            model_name="m",
            window_days=None,
            window_hours=168,
            threshold_js=None,
            threshold_outlier=None,
            window_label="weekly",
        )
        assert calls["window_label"] == "weekly"


class TestDriftDaemonLoopChecksEveryBand:
    @pytest.mark.asyncio
    async def test_calls_run_drift_check_once_per_band_per_active_model(self, monkeypatch):
        import app.services.drift as drift_mod

        monkeypatch.setattr(drift_mod.settings, "topic_foundry_drift_bands", "daily:24,weekly:168")
        monkeypatch.setattr(
            "app.storage.repository.list_models",
            lambda: [{"name": "model-a", "stage": "active"}, {"name": "model-b", "stage": "development"}],
        )
        seen = []
        monkeypatch.setattr(drift_mod, "run_drift_check", lambda **kwargs: seen.append(kwargs))

        async def fake_sleep(_seconds):
            raise StopAsyncIteration  # break the infinite loop after one iteration

        monkeypatch.setattr(drift_mod, "_sleep", fake_sleep)

        with pytest.raises(StopAsyncIteration):
            await drift_daemon_loop()

        # only the active model, once per parsed band, each with its own label/hours
        assert [(c["model_name"], c["window_label"], c["window_hours"]) for c in seen] == [
            ("model-a", "daily", 24),
            ("model-a", "weekly", 168),
        ]

    @pytest.mark.asyncio
    async def test_malformed_bands_falls_back_to_single_daily_band_instead_of_checking_nothing(self, monkeypatch):
        """A typo'd/empty TOPIC_FOUNDRY_DRIFT_BANDS used to be impossible
        (window_hours was a typed int) -- must not silently run zero checks
        forever now that it's a free-text field."""
        import app.services.drift as drift_mod

        monkeypatch.setattr(drift_mod.settings, "topic_foundry_drift_bands", "totally broken, no colons")
        monkeypatch.setattr(drift_mod.settings, "topic_foundry_drift_window_hours", 24)
        monkeypatch.setattr(
            "app.storage.repository.list_models",
            lambda: [{"name": "model-a", "stage": "active"}],
        )
        seen = []
        monkeypatch.setattr(drift_mod, "run_drift_check", lambda **kwargs: seen.append(kwargs))

        async def fake_sleep(_seconds):
            raise StopAsyncIteration

        monkeypatch.setattr(drift_mod, "_sleep", fake_sleep)

        with pytest.raises(StopAsyncIteration):
            await drift_daemon_loop()

        assert [(c["model_name"], c["window_label"], c["window_hours"]) for c in seen] == [("model-a", "daily", 24)]
