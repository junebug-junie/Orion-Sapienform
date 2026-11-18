"""Pinned rest per host (orion/physiology/rest_body.py, config/physiology/rest_body.yaml)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from orion.physiology.rest_body import REST_BODY_PATH, is_sponsored, load_rest_body, parse_rest_body


def _doc(**hosts):
    return {"version": 1, "hosts": hosts, "sponsored": [
        {"name": "dojo", "containers": "zelda-dojo-live-*", "host": "athena", "until": None},
        {"name": "old", "containers": "old-*", "host": "athena", "until": "2026-10-01"}]}


def test_the_committed_file_loads_with_the_proposal_defaults():
    cfg, err = load_rest_body(REST_BODY_PATH)
    assert err is None
    assert cfg.rest_w("circe") == (541.0, None) and cfg.rest_w("athena") == (340.0, None)
    assert cfg.hosts["hecate"].rest_w == 96.0
    assert cfg.sponsored_containers("athena", datetime(2026, 10, 30, tzinfo=timezone.utc)) == ["zelda-dojo-live-*"]


def test_raising_rest_above_the_approved_value_is_refused_for_that_host():
    """Raising rest hides activity: only a reviewed approved_rest_w change can raise it."""
    cfg = parse_rest_body(_doc(circe={"rest_w": 600, "approved_rest_w": 541},
                               athena={"rest_w": 300, "approved_rest_w": 340}))
    rest, why = cfg.rest_w("circe")
    assert rest is None and "rest_w 600 W > approved_rest_w 541 W" in why
    assert cfg.rest_w("athena") == (300.0, None)  # lowering is always allowed
    assert cfg.rest_w("hecate")[0] is None and "no rest for hecate" in cfg.rest_w("hecate")[1]


def test_sponsored_ends_on_its_date_and_matches_by_glob():
    cfg = parse_rest_body(_doc())
    at = datetime(2026, 10, 30, tzinfo=timezone.utc)
    assert cfg.sponsored_containers("athena", at) == ["zelda-dojo-live-*"]
    assert cfg.sponsored[1].until == date(2026, 10, 1)
    assert cfg.sponsored_containers("circe", at) == []
    assert is_sponsored("zelda-dojo-live-12", ["zelda-dojo-live-*"])
    assert not is_sponsored("orion-athena-sql-writer", ["zelda-dojo-live-*"])


@pytest.mark.parametrize("doc", [
    {"version": 2, "hosts": {}},
    {"version": 1, "hosts": {"circe": {"rest_w": -1, "approved_rest_w": 541}}},
    {"version": 1, "hosts": {"circe": {"rest_w": 541}}},
    {"version": 1, "hosts": {}, "sponsored": [{"name": "x"}]},
])
def test_malformed_documents_are_refused(doc):
    with pytest.raises(ValueError):
        parse_rest_body(doc)


def test_missing_file_is_a_reason_not_a_zero(tmp_path):
    cfg, err = load_rest_body(tmp_path / "nope.yaml")
    assert cfg is None and err == "missing:nope.yaml"
