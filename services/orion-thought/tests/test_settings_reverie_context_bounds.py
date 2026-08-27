"""Regression guard (review finding): ORION_REVERIE_CONTEXT_CHAR_LIMIT and
ORION_REVERIE_CONTEXT_MAX_AGE_SEC must reject non-positive values at
settings load rather than silently doing the wrong thing at read time --

- max_age_sec<=0 makes store.py's `created_at > now() - make_interval(secs
  => :max_age_sec)` clause permanently unsatisfiable (real rows are never
  in the future), silently degrading the context-seed to None forever --
  indistinguishable in logs/cockpit from the genuine "no data yet" case.
- char_limit<0 turns Python's `text[:max_chars]` into a negative-index
  slice that keeps almost the ENTIRE string instead of truncating it --
  the exact opposite of the cap this field exists for.

pydantic's `gt=0` on both Fields converts a config typo into a loud
ValidationError at settings construction, instead of a silent wrong
behavior discovered later by reading stale/oversized prompts.
"""
from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError


def test_reverie_context_bounds_default_to_positive_values(monkeypatch):
    monkeypatch.delenv("ORION_REVERIE_CONTEXT_CHAR_LIMIT", raising=False)
    monkeypatch.delenv("ORION_REVERIE_CONTEXT_MAX_AGE_SEC", raising=False)
    import app.settings as s

    importlib.reload(s)
    assert s.settings.reverie_context_char_limit == 240
    assert s.settings.reverie_context_max_age_sec == 900.0


def test_reverie_context_char_limit_rejects_non_positive(monkeypatch):
    # Import (and reload once with a clean env) BEFORE setting the bad value
    # -- a bare `import` is a no-op if the module is already cached from an
    # earlier test in this process, so the module must already exist in
    # sys.modules in a known-good state before the reload under test, or the
    # raises-block below could pass/fail depending on test execution order
    # instead of on the field validator actually working.
    import app.settings as s

    importlib.reload(s)

    monkeypatch.setenv("ORION_REVERIE_CONTEXT_CHAR_LIMIT", "-40")
    with pytest.raises(ValidationError):
        importlib.reload(s)


def test_reverie_context_max_age_sec_rejects_non_positive(monkeypatch):
    import app.settings as s

    importlib.reload(s)

    monkeypatch.setenv("ORION_REVERIE_CONTEXT_MAX_AGE_SEC", "0")
    with pytest.raises(ValidationError):
        importlib.reload(s)
