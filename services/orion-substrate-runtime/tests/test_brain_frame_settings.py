from __future__ import annotations

import os
from unittest import mock


def test_brain_frame_settings_defaults_all_on():
    # Only POSTGRES_URI is required by Settings; provide it, clear brain keys.
    env = {"POSTGRES_URI": "postgresql://t:t@localhost/t"}
    to_clear = [
        "SUBSTRATE_BRAIN_FRAME_ENABLED",
        "BRAIN_FRAME_INTERVAL_SEC",
        "BRAIN_FRAME_RETENTION_HOURS",
        "BRAIN_FRAME_SAMPLE_NODES",
        "BRAIN_FRAME_SAMPLE_EDGES",
        "BRAIN_FRAME_FIRING_THRESHOLD",
        "BRAIN_FRAME_STARVING_THRESHOLD",
    ]
    with mock.patch.dict(os.environ, env, clear=False):
        for k in to_clear:
            os.environ.pop(k, None)
        from app.settings import Settings

        s = Settings()
        assert s.brain_frame_enabled is True
        assert s.brain_frame_interval_sec == 5.0
        assert s.brain_frame_retention_hours == 24
        assert s.brain_frame_sample_nodes == 40
        assert s.brain_frame_sample_edges == 60
        assert 0.0 < s.brain_frame_starving_threshold < s.brain_frame_firing_threshold <= 1.0
