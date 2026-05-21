from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.main import _trigger_world_pulse_run  # noqa: E402
from app.settings import Settings  # noqa: E402


def test_scheduler_uses_actions_world_pulse_run_dry_run_default_true() -> None:
    cfg = Settings()
    assert cfg.actions_world_pulse_run_dry_run is True

    resp = MagicMock()
    resp.ok = True

    with patch("app.main.settings", cfg), patch("app.main.requests.post", return_value=resp) as post:
        assert _trigger_world_pulse_run(date="2026-05-20", requested_by="scheduler") is True
        body = post.call_args.kwargs["json"]
        assert body["dry_run"] is True

    cfg_prod = Settings(ACTIONS_WORLD_PULSE_RUN_DRY_RUN=False)
    with patch("app.main.settings", cfg_prod), patch("app.main.requests.post", return_value=resp) as post:
        _trigger_world_pulse_run(date="2026-05-20", requested_by="scheduler")
        assert post.call_args.kwargs["json"]["dry_run"] is False
