"""Street rhythm surprise (app/street_rhythm.py): the score on the raw stated p, the refusals, the live pass
that builds prediction.street_rhythm_surprise.v1 events, the calm eval gate, and the fixture."""
from __future__ import annotations

import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import app.street_rhythm as sr
from orion.schemas.prediction import (PREDICTION_STREET_RHYTHM_SURPRISE_CHANNEL,
                                      PREDICTION_STREET_RHYTHM_SURPRISE_KIND, PredictionStreetRhythmSurpriseV1)

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
SERVICE = Path(__file__).resolve().parents[1]
FIXTURE = SERVICE / "evals" / "fixtures" / "street_rhythm_2026-11-25.json"


def _row(**over) -> dict:
    row = {
        "expectation_id": "e1", "stream_id": "walkway", "subject_key": "individual:abc",
        "window_start": NOW - timedelta(hours=3), "window_end": NOW - timedelta(minutes=20),
        "confidence": 0.7, "status": "met", "scored_at": NOW - timedelta(minutes=5),
        "last_seen_before": NOW - timedelta(days=1), "first_seen_before": NOW - timedelta(days=20),
    }
    row.update(over)
    return row


@pytest.mark.parametrize("p", [0.02, 0.3, 0.53, 0.7, 0.9, 0.98])
def test_excess_has_mean_zero_and_the_stated_variance_under_the_stated_p(p):
    hit, miss = sr.outcome_terms(p, True), sr.outcome_terms(p, False)
    assert p * hit.excess + (1 - p) * miss.excess == pytest.approx(0.0, abs=1e-12)
    assert p * hit.excess ** 2 + (1 - p) * miss.excess ** 2 == pytest.approx(hit.calm_var, rel=1e-9)
    assert hit.surprisal == pytest.approx(-math.log(p))
    assert miss.surprisal == pytest.approx(-math.log(1 - p))


def test_the_raw_stated_p_is_scored_only_clamped_never_recalibrated():
    assert sr.outcome_terms(1.0, False).p == sr.P_MAX
    assert sr.outcome_terms(1.0, False).surprisal == pytest.approx(-math.log(0.02))   # 3.91 nats, the cap
    assert sr.outcome_terms(0.0, True).p == sr.P_MIN
    assert sr.outcome_terms(0.67, True).p == 0.67


def test_a_confident_miss_is_the_surprise_and_a_coin_flip_carries_none():
    miss = sr.outcome_terms(0.9, False)
    assert miss.excess == pytest.approx(-math.log(0.1) - (-(0.9 * math.log(0.9) + 0.1 * math.log(0.1))))
    assert miss.excess > 1.9
    coin = sr.outcome_terms(0.5, False)
    assert coin.excess == pytest.approx(0.0, abs=1e-12) and coin.calm_var == pytest.approx(0.0, abs=1e-12)


def test_calibrated_outcomes_rest_at_zero_mean():
    rng = random.Random(3)
    ex = []
    for _ in range(40000):
        p = rng.choice([0.5, 0.55, 0.62, 0.75, 0.86, 0.95])
        ex.append(sr.outcome_terms(p, rng.random() < p).excess)
    assert statistics.fmean(ex) == pytest.approx(0.0, abs=0.01)


def test_unscorable_is_a_refusal_not_a_miss():
    assert sr.score_row(_row(status="unscorable")) == (None, "unscorable")
    assert "unscorable" in sr.REFUSAL_REASONS


def test_a_key_unseen_for_seven_days_is_not_scored_and_an_unknown_key_is_not_either():
    assert sr.score_row(_row(last_seen_before=NOW - timedelta(hours=3, days=7)))[1] == "stale_key"
    assert sr.score_row(_row(last_seen_before=NOW - timedelta(hours=3, days=6, minutes=59)))[1] == "scored"
    assert sr.score_row(_row(last_seen_before=None, first_seen_before=None))[1] == "unknown_key"


def test_a_label_subject_is_a_class_with_no_stale_rule():
    out, why = sr.score_row(_row(subject_key="label:mail truck", last_seen_before=None, first_seen_before=None))
    assert why == "scored" and out.subject_kind == "label" and out.key_age_days is None


@pytest.mark.parametrize("over,reason", [
    ({"stream_id": "doorbell"}, "not_walkway"),
    ({"status": "open"}, "not_closed"),
    ({"subject_key": "zone:door"}, "unknown_subject_kind"),
    ({"confidence": None}, "no_stated_p"),
])
def test_other_refusals(over, reason):
    assert sr.score_row(_row(**over)) == (None, reason)


def test_the_event_validates_against_the_registered_schema_and_carries_no_name():
    out, _ = sr.score_row(_row(status="missed", confidence=0.86))
    ev = PredictionStreetRhythmSurpriseV1.model_validate(out.event())
    assert ev.happened is False and ev.p == pytest.approx(0.86) and ev.excess > 0
    assert ev.key_age_days == pytest.approx(19.875)
    assert "subject_label" not in ev.model_dump()
    from orion.schemas.registry import resolve

    assert resolve("PredictionStreetRhythmSurpriseV1") is PredictionStreetRhythmSurpriseV1
    assert PREDICTION_STREET_RHYTHM_SURPRISE_KIND == "prediction.street_rhythm_surprise.v1"


def test_the_channel_is_registered_with_this_schema_producer_and_consumer():
    import yaml

    doc = yaml.safe_load((SERVICE.parents[1] / "orion" / "bus" / "channels.yaml").read_text())
    (ch,) = [c for c in doc["channels"] if c["name"] == PREDICTION_STREET_RHYTHM_SURPRISE_CHANNEL]
    assert ch["schema_id"] == "PredictionStreetRhythmSurpriseV1"
    assert ch["message_kind"] == PREDICTION_STREET_RHYTHM_SURPRISE_KIND
    assert ch["producer_services"] == ["orion-prediction"]
    assert ch["consumer_services"] == ["orion-substrate-runtime"]


class _Store:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def street_rhythm_outcomes(self, since, until=None):
        self.calls.append(since)
        return [r for r in self.rows if r["scored_at"] >= since]


def test_live_pass_queues_each_scored_outcome_once_and_counts_refusals():
    rows = [_row(expectation_id="a"), _row(expectation_id="b", status="unscorable"),
            _row(expectation_id="c", status="missed", confidence=0.9),
            _row(expectation_id="old", scored_at=NOW - timedelta(hours=7))]
    live = sr.StreetRhythmLive(_Store(rows), clock=lambda: NOW)
    assert live.tick() == 2
    events = live.take_pending_events()
    assert [e["expectation_id"] for e in events] == ["a", "c"]
    for e in events:
        PredictionStreetRhythmSurpriseV1.model_validate(e)
    assert live.counts == {"scored": 2, "unscorable": 1}
    assert live._store.calls == [NOW - sr.PUBLISH_MAX_AGE]   # a backlog older than the fold window is never read
    assert live.tick() == 0 and live.take_pending_events() == []   # no republish


def test_kill_switch_scores_but_queues_nothing():
    live = sr.StreetRhythmLive(_Store([_row()]), events_enabled=False, clock=lambda: NOW)
    assert live.tick() == 0 and live.take_pending_events() == []
    assert live.counts == {"scored": 1, "dropped_events_disabled": 1}


def test_store_query_is_read_only_and_reads_sightings_before_the_window():
    from app.store import STREET_RHYTHM_READ_TIMEOUT_MS, PostgresStore

    seen = {}

    class _S(PostgresStore):
        def __init__(self):
            pass

        def _read_only(self, sql, args, timeout_ms):
            seen.update(sql=sql, args=args, timeout=timeout_ms)
            return []

    _S().street_rhythm_outcomes(NOW - timedelta(minutes=30), NOW)
    sql = seen["sql"]
    assert "FROM vision_percept_expectation" in sql and "vision_individual_sighting" in sql
    assert "v.started_at < e.window_start" in sql and "substr(e.subject_key, 12)" in sql
    assert seen["timeout"] == STREET_RHYTHM_READ_TIMEOUT_MS
    assert seen["args"] == ["walkway", NOW - timedelta(minutes=30), NOW]
    assert len("individual:") + 1 == 12


# --- the calm eval (evals/run_street_rhythm_calm_eval.py) -----------------------------------------------------


def _eval():
    import importlib.util
    import sys

    path = SERVICE / "evals" / "run_street_rhythm_calm_eval.py"
    spec = importlib.util.spec_from_file_location("run_street_rhythm_calm_eval", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_eval_gate_numbers_are_the_design_rev4_tail_gate():
    ev = _eval()
    assert (ev.GATE_P99, ev.GATE_MAX, ev.GATE_CONFIDENT_MISSES, ev.CONFIDENT_P) == (1.2, 4.0, 2, 0.9)


def test_eval_seed_bar_matches_the_curiosity_bar():
    ev = _eval()
    import importlib.util

    path = SERVICE.parents[1] / "orion" / "substrate" / "curiosity_seed_bar.py"
    spec = importlib.util.spec_from_file_location("csb_for_test", path)
    csb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(csb)
    assert ev.SEED_BAR == pytest.approx(csb.CURIOSITY_SEED_BAR_BY_NODE_ID["node:prediction.street_rhythm"])


def test_eval_passes_on_the_committed_real_fixture(capsys):
    ev = _eval()
    assert ev.main(["--fixture", str(FIXTURE), "--check"]) == 0
    out = capsys.readouterr().out
    assert "GATE PASS" in out and "refusal rate" in out


def test_eval_fails_a_confident_miss_streak_and_a_biased_mean():
    ev = _eval()
    rows = []
    for k in range(150):
        t = NOW - timedelta(days=6) + timedelta(minutes=50 * k)
        rows.append(_row(expectation_id=f"x{k}", window_start=t - timedelta(hours=2), window_end=t,
                         scored_at=t + timedelta(minutes=10), confidence=0.95, status="missed",
                         last_seen_before=t - timedelta(days=1)))
    res = ev.run(rows, [], {}, NOW - timedelta(days=7), NOW)
    joined = "; ".join(res["gate_failures"])
    assert "does not cover 0" in joined and "p99" in joined and "misses at stated" in joined


def test_calm_crossing_rate_is_exact_for_one_outcome():
    ev = _eval()
    # one miss at stated 0.9 reads z = sqrt(0.9 / 0.1) = 3 -> value 1.0; it happens with calm probability 0.1
    assert ev.calm_crossing_rate([0.9], 0.6) == pytest.approx(0.1)
    assert ev.calm_crossing_rate([0.5], 0.6) == 0.0


def test_fixture_is_hashed_and_nameless():
    import json

    doc = json.loads(FIXTURE.read_text())
    keys = {r["subject_key"] for r in doc["rows"]}
    assert all(k.startswith(("individual:", "label:")) for k in keys)
    assert all(len(k.split(":", 1)[1]) == 16 for k in keys if k.startswith("individual:"))
    assert {k for k in keys if k.startswith("label:")} <= {"label:vehicle", "label:mail truck", "label:person"}
    assert all("subject_label" not in r for r in doc["rows"])
