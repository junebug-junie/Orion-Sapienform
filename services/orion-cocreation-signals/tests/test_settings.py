"""Unit tests for app/settings.py's validators."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.settings import Settings  # noqa: E402


def test_defaults_construct_cleanly() -> None:
    settings = Settings(_env_file=None)
    assert settings.COCREATION_SIGNALS_PR_FETCH_LIMIT == 200
    assert settings.COCREATION_SIGNALS_PR_LIFECYCLE_COLD_START_LOOKBACK_SEC == 3600.0


@pytest.mark.parametrize("value", [0, -1, -200])
def test_pr_fetch_limit_must_be_positive(value: int) -> None:
    """Regression guard (code review 2026-07-30): a fetch_limit of 0 would
    make orion/structural_mass/pr_lifecycle.py's `len(prs) >= fetch_limit`
    trivially true forever, making `possibly_truncated` fire spuriously on
    every real tick -- must fail fast at startup instead."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, COCREATION_SIGNALS_PR_FETCH_LIMIT=value)


@pytest.mark.parametrize(
    "field",
    [
        "COCREATION_SIGNALS_GIT_DELTA_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_PR_LIFECYCLE_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_GRAPH_DELTA_POLL_INTERVAL_SEC",
        "COCREATION_SIGNALS_PR_LIFECYCLE_COLD_START_LOOKBACK_SEC",
    ],
)
def test_intervals_and_lookback_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 0})
