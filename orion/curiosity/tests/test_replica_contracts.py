"""Curiosity replica, patch 2: the contracts hold together.

Design: docs/superpowers/specs/2026-11-04-curiosity-replica-design.md. Pure, no I/O.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orion.curiosity.kickoff_prompt import ORDER_SENTENCE, _priors_section
from orion.curiosity.value import (
    ARM_REPLICA_ORDER,
    ARM_UNCERTAINTY_ORDER,
    ARM_VALUE_ORDER,
    OFFER_ARMS,
)
from orion.curiosity.worldview import WorldviewSnapshot, select_priors
from orion.prime.state import (
    MAX_CONCERNS,
    PrimeConcernV1,
    PrimeConcernsViewV1,
    PrimeStateV1,
    concerns_view,
)
from orion.schemas import curiosity_replica as cr

REPO = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def _candidate(pid: str, **kw) -> dict:
    return {"prior_id": pid, "claim": f"claim {pid}", "confidence": 0.5, "times_tested": 0, **kw}


# --- schema shapes -----------------------------------------------------------------------------


def test_request_round_trips_and_refuses_duplicates_and_stale_candidates() -> None:
    req = cr.CuriosityOfferPlanRequestV1(
        run_id="r1", correlation_id="c1", deadline_at=NOW + timedelta(seconds=cr.PLAN_DEADLINE_SEC),
        candidates=[_candidate("a"), _candidate("b", last_tested_age_h=3.0, last_move=-0.2)],
        stale_ids=["s1"],
        pending_focus={"focus_id": "f1", "origin": "self", "question": "why?", "from_prime": True},
    )
    assert cr.CuriosityOfferPlanRequestV1.model_validate_json(req.model_dump_json()) == req
    with pytest.raises(ValidationError, match="distinct"):
        cr.CuriosityOfferPlanRequestV1(run_id="r", correlation_id="c", deadline_at=NOW,
                                       candidates=[_candidate("a"), _candidate("a")])
    with pytest.raises(ValidationError, match="stale"):
        cr.CuriosityOfferPlanRequestV1(run_id="r", correlation_id="c", deadline_at=NOW,
                                       candidates=[_candidate("a")], stale_ids=["a"])
    with pytest.raises(ValidationError):  # bus message: an unknown field is a contract break
        cr.CuriosityOfferPlanRequestV1(run_id="r", correlation_id="c", deadline_at=NOW, extra_field=1)
    with pytest.raises(ValidationError):  # naive time is not a deadline
        cr.CuriosityOfferPlanRequestV1(run_id="r", correlation_id="c", deadline_at=datetime(2026, 11, 4))
    with pytest.raises(ValidationError):
        cr.CuriosityOfferPlanRequestV1(
            run_id="r", correlation_id="c", deadline_at=NOW,
            candidates=[_candidate(f"p{i}") for i in range(cr.MAX_CANDIDATES + 1)])
    with pytest.raises(ValidationError):  # the frame's claim line is capped
        cr.CuriosityOfferCandidateV1(prior_id="a", claim="x" * (cr.CLAIM_MAX + 1))


def test_answer_parses_the_design_json_and_is_lenient_only_on_extra_keys() -> None:
    raw = (
        '{"reading": "the bus questions keep stalling", '
        '"picks": [{"prior_id": "a", "why": "never tested"}, {"prior_id": "b", "why": "moved last time"}, '
        '{"prior_id": "c", "why": "cheap to check"}], '
        '"focus_stance": {"stance": "challenge", "reason": "three runs already looked at this"}, '
        '"signal_stance": null, "abstain": false, "confidence": 0.7}'
    )
    ans = cr.CuriosityOfferPlanAnswerV1.model_validate_json(raw)
    assert [p.prior_id for p in ans.picks] == ["a", "b", "c"]
    assert ans.focus_stance.stance == "challenge"
    # Too many picks still parses: Hub's validation rejects it with a reason (invalid:<reason>).
    six = cr.CuriosityOfferPlanAnswerV1(picks=[{"prior_id": f"p{i}", "why": "w"} for i in range(6)])
    assert len(six.picks) == 6
    for bad in (
        {"abstain": True, "picks": [{"prior_id": "a", "why": "w"}]},
        {"picks": [{"prior_id": "a", "why": ""}]},
        {"picks": [{"prior_id": "a", "why": "x" * (cr.WHY_MAX + 1)}]},
        {"focus_stance": {"stance": "ignore", "reason": "r"}},
        {"focus_stance": {"stance": "refuse", "reason": ""}},
        {"signal_stance": {"signal_id": "", "stance": "support", "reason": "r"}},
    ):
        with pytest.raises(ValidationError):
            cr.CuriosityOfferPlanAnswerV1.model_validate(bad)
    assert cr.CuriosityOfferPlanAnswerV1(abstain=True).picks == []


@pytest.mark.parametrize("stance", ["support", "challenge", "refuse"])
def test_every_stance_needs_a_reason(stance: str) -> None:
    assert cr.CuriosityFocusStanceV1(stance=stance, reason="one sentence").stance == stance
    with pytest.raises(ValidationError):
        cr.CuriosityFocusStanceV1(stance=stance, reason="")


def test_result_only_ok_carries_picks() -> None:
    picks = [{"prior_id": "a", "why": "w"}]
    ok = cr.CuriosityOfferPlanResultV1(run_id="r", plan_id="p", status="ok", picks=picks, latency_ms=900)
    assert cr.CuriosityOfferPlanResultV1.model_validate_json(ok.model_dump_json()) == ok
    with pytest.raises(ValidationError, match="carries its picks"):
        cr.CuriosityOfferPlanResultV1(run_id="r", plan_id="p", status="ok", latency_ms=1)
    for status in ("abstain", "unfunded", "model_failed", "unparseable"):
        cr.CuriosityOfferPlanResultV1(run_id="r", plan_id="p", status=status, latency_ms=1)
        with pytest.raises(ValidationError, match="carries no picks"):
            cr.CuriosityOfferPlanResultV1(run_id="r", plan_id="p", status=status, picks=picks, latency_ms=1)
    # Hub's own verdicts are row statuses, never something orion-mind may send.
    for status in (cr.PLAN_STATUS_TIMEOUT, "invalid:duplicate"):
        with pytest.raises(ValidationError):
            cr.CuriosityOfferPlanResultV1(run_id="r", plan_id="p", status=status, latency_ms=1)


def test_registered_in_both_registries_and_resolvable() -> None:
    from orion.schemas.registry import SCHEMA_REGISTRY, resolve

    for name, model in (
        ("CuriosityOfferPlanRequestV1", cr.CuriosityOfferPlanRequestV1),
        ("CuriosityOfferPlanResultV1", cr.CuriosityOfferPlanResultV1),
        ("CuriosityOfferPlanAnswerV1", cr.CuriosityOfferPlanAnswerV1),
        ("PrimeConcernsViewV1", PrimeConcernsViewV1),
    ):
        assert resolve(name) is model
        assert SCHEMA_REGISTRY[name].model is model
    assert SCHEMA_REGISTRY["CuriosityOfferPlanRequestV1"].kind == cr.PLAN_REQUEST_KIND
    assert SCHEMA_REGISTRY["CuriosityOfferPlanResultV1"].kind == cr.PLAN_RESULT_KIND


def test_bus_channels_match_the_schema_module() -> None:
    doc = yaml.safe_load((REPO / "orion/bus/channels.yaml").read_text())
    by_name = {c["name"]: c for c in doc["channels"]}
    req = by_name[cr.PLAN_REQUEST_CHANNEL]
    res = by_name[cr.PLAN_RESULT_CHANNEL_PREFIX + "*"]
    assert (req["schema_id"], req["message_kind"]) == ("CuriosityOfferPlanRequestV1", cr.PLAN_REQUEST_KIND)
    assert (res["schema_id"], res["message_kind"]) == ("CuriosityOfferPlanResultV1", cr.PLAN_RESULT_KIND)
    assert req["producer_services"] == ["orion-hub"] and req["consumer_services"] == ["orion-mind"]
    assert res["producer_services"] == ["orion-mind"] and res["consumer_services"] == ["orion-hub"]
    assert req["single_consumer"] is True


# --- the plan table and Python agree on every status ---------------------------------------------


def test_plan_table_check_lists_exactly_the_python_statuses() -> None:
    sql = (REPO / "services/orion-sql-db/manual_migration_curiosity_replica_plan_v1.sql").read_text()
    check = re.search(r"CHECK \(status IN \(([^)]*)\)\s*OR status LIKE '([^']*)%'\)", sql)
    assert check, "the plan table's status CHECK moved; keep this test reading it"
    listed = set(re.findall(r"'([^']+)'", check.group(1)))
    assert listed == set(cr.REPLICA_PLAN_STATUSES) | {cr.PLAN_STATUS_TIMEOUT}
    assert check.group(2) == cr.PLAN_STATUS_INVALID_PREFIX
    for status in (*cr.REPLICA_PLAN_STATUSES, cr.PLAN_STATUS_TIMEOUT, "invalid:not_a_candidate"):
        assert cr.plan_row_status_allowed(status)
    for status in ("invalid:", "late", ""):
        assert not cr.plan_row_status_allowed(status)
    # No GRANT: the in-turn sandbox (orion_readonly) must never read the replica's reasons.
    assert "GRANT" not in sql.upper().replace("NO GRANT", "")


def test_migrations_are_on_the_post_pull_auto_apply_allowlist() -> None:
    from orion import sql_migration_drift as drift

    existing = {"curiosity_focus_request", "curiosity_offer_decisions"}
    for name in ("manual_migration_curiosity_replica_plan_v1.sql",
                 "manual_migration_curiosity_focus_request_refused_v1.sql"):
        text = (REPO / "services/orion-sql-db" / name).read_text()
        assert text.startswith("-- ORION-MIGRATION-REQUIRED-BY: orion-hub\n")
        assert drift.auto_apply_refusals(text, existing) == [], name


# --- the body-economy family -----------------------------------------------------------------------


def test_family_is_tier_2_lane_chat_90s_no_receptors() -> None:
    from orion.physiology.families import load_families

    fams, err = load_families(REPO / "config/physiology/families.yaml")
    assert err is None, err
    fam = fams.get(cr.REPLICA_FAMILY)
    assert fam is not None
    assert fam.tier == 2
    assert fam.deferrable_window_s == cr.PLAN_DEADLINE_SEC == 90
    assert list(fam.lanes) == ["chat"]
    assert not fam.receptors


# --- the order sentence is the same on every arm ----------------------------------------------------


def _rows(n: int = 14) -> list[dict]:
    return [
        {"prior_id": f"p{i:02d}", "claim": f"claim number {i}", "confidence": str(round(0.5 + i * 0.03, 2)),
         "status": "open", "times_tested": i % 3, "formed_from": "", "last_tested_at": ""}
        for i in range(n)
    ]


def _view(**kw) -> WorldviewSnapshot:
    live, stale, _ = select_priors(_rows(), sample=8, stale_after=3, rotate_seed="run-1", coverage_slots=2, **kw)
    return WorldviewSnapshot(live_priors=live, stale_priors=stale, live_total=14, closed_total=2)


def test_kickoff_text_is_identical_across_arms_except_the_prior_lines() -> None:
    arms = {
        ARM_UNCERTAINTY_ORDER: _view(),
        ARM_VALUE_ORDER: _view(expected_nats_for=lambda p: float(int(p.prior_id[1:]) % 5)),
        ARM_REPLICA_ORDER: _view(head_ids=["p09", "p04", "p12"]),
    }
    assert set(arms) == set(OFFER_ARMS)
    rendered = {arm: _priors_section(v, stale_after=3) for arm, v in arms.items()}
    orders = {arm: [line for line in lines if line.startswith("  - ")] for arm, lines in rendered.items()}
    assert len({tuple(o) for o in orders.values()}) == 3, "the three arms should show different orders"
    skeletons = {arm: [line for line in lines if not line.startswith("  - ")] for arm, lines in rendered.items()}
    assert skeletons[ARM_UNCERTAINTY_ORDER] == skeletons[ARM_VALUE_ORDER] == skeletons[ARM_REPLICA_ORDER]
    for lines in rendered.values():
        assert ORDER_SENTENCE in lines
    # The sentence names no rule and no planner, so it stays true on every arm.
    for word in ("uncertain", "unsure", "change your mind", "planner", "replica", "yield"):
        assert word not in ORDER_SENTENCE.lower()


def test_self_inquiry_line_renders_the_same_sentence() -> None:
    import orion.curiosity.self_inquiry_prompt as sip

    assert sip._priors_section is _priors_section


# --- Prime's published concerns view ------------------------------------------------------------------


def test_concerns_view_publishes_only_open_concerns_with_age() -> None:
    def concern(cid: str, hours: float, graded=None) -> PrimeConcernV1:
        opened = NOW - timedelta(hours=hours)
        return PrimeConcernV1(concern_id=cid, text=f"concern {cid}", arc_id="arc-1" if cid == "a" else None,
                              opened_at=opened, last_evidence_at=opened, graded=graded)

    state = PrimeStateV1(concerns=[concern("a", 2.0), concern("b", 5.5, graded="settled"), concern("c", 0.0)])
    view = concerns_view(state, now=NOW)
    assert [c.concern_id for c in view.concerns] == ["a", "c"]
    assert view.concerns[0].age_h == pytest.approx(2.0) and view.concerns[0].arc_id == "arc-1"
    assert view.as_of == NOW
    assert PrimeConcernsViewV1.model_validate_json(view.model_dump_json()) == view
    # The view is ids, words, age and arc only: nothing private rides along.
    assert set(view.concerns[0].model_dump()) == {"concern_id", "text", "age_h", "arc_id"}
    assert len(concerns_view(PrimeStateV1(), now=NOW).concerns) == 0
    assert MAX_CONCERNS >= 1
