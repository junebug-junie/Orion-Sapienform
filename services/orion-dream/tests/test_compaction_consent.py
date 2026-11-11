"""Attention patch 3d: dream compaction apply asks the want market's consent gate (its first consumer).

Lifecycle: a staged REM delta -> a consent want -> the market's pure core (real ``choice.decide`` with the
real families.yaml and consent.yaml, the decision passed through JSON as the HTTP answer is) -> the
applier's verdict. Stages: the shipped config (needs Juniper's standing consent: not_approved, rule named,
nothing written), a merged standing consent with apply off (disabled), apply on (applied, downscale only),
and every fail-closed branch (prune, no snapshot dir, market down, a verdict for another delta).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from orion.physiology.choice import ReservationBook, build_context, decide
from orion.physiology.consent import parse_consent
from orion.physiology.families import load_families
from orion.physiology.tests.consent_fixture import live_delta
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1
from orion.schemas.physiology_want import WantDecisionV1

from test_compaction_applier import FakeStore  # noqa: E402 (same tests dir, pytest prepend import mode)

REPO = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 11, 11, 8, 0, tzinfo=timezone.utc)
FAMS, _ = load_families(REPO / "config/physiology/families.yaml")
CONSENT_DOC = yaml.safe_load((REPO / "config/physiology/consent.yaml").read_text())
SHIPPED = parse_consent(CONSENT_DOC)
GRANTED = parse_consent({**CONSENT_DOC, "standing": {"dream-compaction-downscale": {
    "granted_by": "juniper", "granted_on": "2026-11-11", "pr": "fixture"}}})


def _state():
    p = {"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""}
    return PhysiologyStateV1(tick_id=f"physiology:{NOW.isoformat()}", generated_at=NOW, constraints=[
        R(constraint="heat", unit="c", level=28.0, budget=31.0, budget_kind="soft_line", price=p),
        R(constraint="lane:agent", unit="slot_s", level=0.0, budget=3600.0, budget_kind="capacity",
          detail={"slots": 1, "window_sec": 3600.0}, price=p)])


def market_bid(rules, calls=None):
    """The market's pure core, behind the client's interface: JSON in, JSON out, like POST /physiology/bid."""
    st = _state()

    def bid(want, timeout_sec=None):
        if calls is not None:
            calls.append(want)
        d = decide(want, st, build_context(FAMS, [], [], st, NOW), ReservationBook(), NOW, consent_rules=rules)
        return WantDecisionV1.model_validate_json(d.model_dump_json())

    return bid


def consent_for(rules, *, writable=True, calls=None):
    from app.compaction_consent import market_consent

    return market_consent(bid=market_bid(rules, calls), now_fn=lambda: NOW, writable_fn=lambda: writable)


def test_the_shipped_config_says_not_approved_with_the_rule_while_apply_is_off(caplog):
    """Acceptance check 7's dream line: `not_approved` with `rule=` on the consent verdict, apply off."""
    from app.compaction_applier import apply_compaction_delta

    caplog.set_level(logging.INFO)
    calls: list = []
    r = apply_compaction_delta(live_delta(), consent_for(SHIPPED, calls=calls), store=None, enabled=False)
    assert r.status == "not_approved" and r.applied is False
    assert (r.consent_verdict, r.consent_rule) == ("needs_consent", "dream.compaction.downscale_only")
    want = calls[0]
    assert want.family == "dream.compaction" and want.verb == "compact" and want.tier == 2
    assert want.consent_request.reversibility == "snapshot_restorable"
    assert set(want.consent_request.attests) == {"snapshot_dir_writable", "downscale_only"}
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any(m.startswith("compaction_apply not_approved") and "rule=dream.compaction.downscale_only" in m
               and "verdict=needs_consent" in m for m in msgs)
    assert any(m.startswith("compaction_consent ") and "approved=False" in m for m in msgs)


def test_lifecycle_consent_merged_then_apply_on(tmp_path, monkeypatch):
    from app import compaction_applier as ca
    from app.settings import settings

    monkeypatch.setattr(settings, "DREAM_COMPACTION_SNAPSHOT_DIR", str(tmp_path))
    delta = live_delta(downscale=1)
    # Juniper merges the standing consent; apply is still off: consented, nothing written.
    store = FakeStore()
    r = ca.apply_compaction_delta(delta, consent_for(GRANTED), store=store, enabled=False)
    assert r.status == "disabled" and r.consent_verdict == "allow" and store.calls == []
    # The REM ask path (no store) is still inert with apply on.
    assert ca.apply_compaction_delta(delta, consent_for(GRANTED), store=None, enabled=True).status == "no_store"
    # Apply on, a store wired: the snapshot first, downscale and gist cards, no prune.
    store = FakeStore()
    store.weights["edge-0"] = 0.9
    r = ca.apply_compaction_delta(delta, consent_for(GRANTED), store=store, enabled=True, downscale_only=True)
    assert r.status == "applied" and r.downscaled == 1 and r.cards_written == 3 and r.pruned == 0
    assert store.calls[0] == "snapshot" and r.consent_rule == "dream.compaction.downscale_only"


def test_a_prune_plan_asks_verb_prune_and_is_refused_even_with_the_standing_consent(tmp_path):
    from app.compaction_applier import apply_compaction_delta

    calls: list = []
    store = FakeStore()
    r = apply_compaction_delta(live_delta(prune=1), consent_for(GRANTED, calls=calls), store=store, enabled=True,
                               downscale_only=False)
    assert calls[0].verb == "prune" and calls[0].consent_request.reversibility == "irreversible"
    assert r.status == "not_approved" and r.consent_rule == "dream.compaction.prune" and store.calls == []


def test_a_delta_with_nothing_to_prune_is_downscale_only_whatever_the_flag():
    from app.compaction_consent import plan

    assert plan(live_delta(), False) == ("compact", "snapshot_restorable", True)
    assert plan(live_delta(prune=2), True) == ("compact", "snapshot_restorable", True)
    assert plan(live_delta(prune=2), False) == ("prune", "irreversible", False)


def test_an_unwritable_snapshot_dir_is_a_missing_attest():
    from app.compaction_applier import apply_compaction_delta

    r = apply_compaction_delta(live_delta(), consent_for(GRANTED, writable=False), store=FakeStore(), enabled=True)
    assert r.status == "not_approved" and r.consent_reason == "missing_attest:snapshot_dir_writable"


def test_snapshot_dir_writable_checks_the_real_directory(tmp_path):
    from app.compaction_consent import snapshot_dir_writable

    assert snapshot_dir_writable(str(tmp_path / "new")) is True
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert snapshot_dir_writable(str(blocker / "sub")) is False


def test_no_market_answer_is_not_approved():
    from app.compaction_applier import apply_compaction_delta
    from app.compaction_consent import market_consent

    r = apply_compaction_delta(live_delta(), market_consent(bid=lambda w, timeout_sec=None: None),
                               store=FakeStore(), enabled=True)
    assert r.status == "not_approved" and r.consent_reason.startswith("no_answer")

    def boom(w, timeout_sec=None):
        raise RuntimeError("down")

    assert apply_compaction_delta(live_delta(), market_consent(bid=boom), store=FakeStore(), enabled=True).status == "not_approved"


def test_the_real_client_with_the_bid_switch_off_is_not_approved():
    """conftest turns ORION_PHYSIOLOGY_BID_CLIENT_ENABLED off: the real client returns None, never a fund."""
    from app.compaction_applier import apply_compaction_delta
    from app.compaction_consent import market_consent

    r = apply_compaction_delta(live_delta(), market_consent(writable_fn=lambda: True), store=FakeStore(), enabled=True)
    assert r.status == "not_approved"


def test_an_allow_for_another_delta_is_not_approved():
    from app.compaction_applier import apply_compaction_delta
    from app.compaction_consent import consent_want, market_consent

    other = live_delta("compaction-delta:other", cards=1)
    real = market_bid(GRANTED)

    def wrong(want, timeout_sec=None):
        d = real(consent_want(other, True, writable=True))
        return d.model_copy(update={"want_id": want.want_id})

    r = apply_compaction_delta(live_delta(), market_consent(bid=wrong), store=FakeStore(), enabled=True)
    assert r.status == "not_approved" and r.consent_reason.startswith("verdict_mismatch")


def test_a_consented_want_deferred_for_hunger_still_goes_since_the_apply_has_no_hold():
    """dream.compaction has no act history, so a fresh want is sated. Only consent decides the one-shot apply."""
    from app.compaction_consent import consent_want, judge

    want = consent_want(live_delta(), True, writable=True)
    d = market_bid(GRANTED)(want)
    assert d.consent.verdict == "allow"
    assert judge(want, d, NOW).approved is True


def test_the_digest_binds_the_exact_delta():
    from app.compaction_consent import delta_digest

    a = live_delta()
    assert delta_digest(a) == delta_digest(a.model_copy())
    assert delta_digest(a) != delta_digest(live_delta(cards=2))


def test_the_rem_ask_never_hands_the_applier_a_store(monkeypatch):
    from app import compaction_applier as ca
    from app import compaction_consent as cc

    seen = {}

    def spy(delta, consent=None, *, store, **kw):
        seen["store"] = store
        return ca.CompactionApplyReceiptV1(delta_id=delta.delta_id, status="not_approved")

    monkeypatch.setattr(cc, "apply_compaction_delta", spy)
    r = asyncio.run(cc.ask_consent_without_store(live_delta(), consent=lambda d, o: False))
    assert seen == {"store": None} and r.status == "not_approved"


def test_the_rem_pass_asks_once_per_staged_delta_and_survives_a_failed_ask(monkeypatch):
    from app import rem_compaction as rc
    from app.settings import settings

    monkeypatch.setattr(settings, "ORION_DREAM_REM_ENABLED", True)
    req = [{"request_id": "r-1", "theme": "loop:ol-1", "op_hint": "consolidate", "evidence_refs": ["ol-1", "ol-2"]}]
    asked: list = []

    async def asker(delta):
        asked.append(delta.delta_id)

    async def boom(delta):
        raise RuntimeError("market down")

    async def run(ask):
        return await rc.run_rem_compaction_once(None, request_loader=lambda n: req, delta_persister=lambda d: True,
                                                consent_asker=ask)

    d = asyncio.run(run(asker))
    assert d is not None and asked == [d.delta_id]
    assert asyncio.run(run(boom)) is not None
    monkeypatch.setattr(settings, "DREAM_COMPACTION_CONSENT_ASK_ENABLED", False)
    asked.clear()
    asyncio.run(run(asker))
    assert asked == []


def test_the_window_constant_matches_families_yaml():
    assert FAMS.get("dream.compaction").deferrable_window_s == timedelta(days=1).total_seconds()
