"""Personality rev 4 Move 2 in the live market: the reader (tendencies, choices, runaway guard), the per-bid
term (arm, care, fail-open), and Market.bid carrying it into the decision trace. Fakes for Postgres, Redis
and vector-host."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app.personality_values import PersonalityValues, parse_choice_row  # noqa: E402
from app.physiology_market import Market  # noqa: E402
from app.run_bids import want_object_text  # noqa: E402
from orion.physiology import personality_values as pv  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1  # noqa: E402
from orion.schemas.physiology_want import WantV1  # noqa: E402

NOW = datetime.now(timezone.utc)
FAMS, _ = load_families(REPO_ROOT / "config/physiology/families.yaml")
DIM = 4
AXES = {"street": 0, "pipeline": 1, "people": 2}


def _vec_for(text: str) -> list[float]:
    v = [0.0] * DIM
    for word, i in AXES.items():
        if word in text:
            v[i] += 1.0
    if not any(v):
        v[3] = 1.0
    return v


class _Embedder:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    async def __call__(self, doc_id, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("vector-host down")
        return _vec_for(text)


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self):
        return self.rows


class _Pool:
    def __init__(self, *, tendencies=(), choices=(), acts=(), fail=False):
        self.tendencies, self.choices, self.acts, self.fail = list(tendencies), list(choices), list(acts), fail

    def connection(self):
        pool = self

        class _Conn:
            async def execute(self, sql, params=None):
                if pool.fail:
                    raise RuntimeError("db down")
                if "personality_disposition" in sql:
                    return _Cursor(pool.tendencies)
                if "personality_choice" in sql:
                    return _Cursor(pool.choices)
                if "grammar_acts" in sql:
                    return _Cursor(pool.acts)
                return _Cursor([])

        class _CM:
            async def __aenter__(self):
                return _Conn()

            async def __aexit__(self, *a):
                return False

        return _CM()


def _tendency_row(tid="t-street", axis="street", weight=10.0):
    c = [0.0] * DIM
    c[AXES[axis]] = 1.0
    return {"disposition_id": tid, "label": f"{axis} over pipeline, 9 of 10", "centroid": c, "weight": weight}


def _choice_row(i, *, at, chosen_text="street arrivals", alt_texts=("pipeline logs",), lane="outreach",
                declined=False):
    chosen = None if declined else {"key": f"k{i}", "text": chosen_text, "market_rank": 2}
    alts = [{"key": f"a{i}-{j}", "text": t, "market_rank": j + 1} for j, t in enumerate(alt_texts)]
    return {"choice_id": f"{lane}:{i}", "lane": lane, "occurred_at": at, "chosen": json.dumps(chosen) if chosen else None,
            "alternatives": json.dumps(alts), "declined_all": declined, "care_promoted": False, "act_id": None}


def _values(pool, embed=None, **kw):
    return PersonalityValues(enabled=True, pool_getter=lambda: pool, embed=embed or _Embedder(), **kw)


def _care_want_id(prefix="w"):
    return next(f"{prefix}{i}" for i in range(100) if pv.holdback_arm(f"{prefix}{i}") == pv.ARM_CARE)


def _holdback_want_id(prefix="w"):
    return next(f"{prefix}{i}" for i in range(100) if pv.holdback_arm(f"{prefix}{i}") == pv.ARM_HOLDBACK)


def test_parse_choice_row_reads_families_and_shown_texts():
    rec, shown = parse_choice_row(_choice_row(1, at=NOW))
    assert rec.chosen_family == "hub.outreach" and rec.real_choice
    assert shown.texts == ("street arrivals", "pipeline logs")
    rec2, _ = parse_choice_row(_choice_row(2, at=NOW, declined=True))
    assert rec2.chosen_family is None and rec2.declined_all


def test_refresh_reads_tendencies_and_choices_and_care_matches_the_text():
    pool = _Pool(tendencies=[_tendency_row()], choices=[_choice_row(1, at=NOW - timedelta(hours=1))])
    v = _values(pool)
    snap = asyncio.run(v.refresh(NOW))
    assert len(snap.tendencies) == 1 and len(snap.choices) == 1 and snap.shown_options == 2
    term = asyncio.run(v.term(_care_want_id(), "who parks on the street at night"))
    assert term.arm == "care" and term.care == pytest.approx(0.5) and term.tendency_id == "t-street"
    miss = asyncio.run(v.term(_care_want_id(), "pipeline throughput"))
    assert miss.care == 0.0 and miss.reason == pv.REASON_NO_MATCH


def test_care_fails_open_to_zero_with_a_reason():
    pool = _Pool(tendencies=[_tendency_row()])
    v = _values(pool, embed=_Embedder(fail=True))
    asyncio.run(v.refresh(NOW))
    t = asyncio.run(v.term(_care_want_id(), "street"))
    assert t.care == 0.0 and t.reason == pv.REASON_EMBED_FAILED and v.counts["embed_failed"] == 1
    assert asyncio.run(v.term(_care_want_id(), None)).reason == pv.REASON_NO_TEXT
    unread = _values(_Pool(fail=True))
    asyncio.run(unread.refresh(NOW))
    assert unread.snapshot is None and unread.choices() is None
    assert asyncio.run(unread.term(_care_want_id(), "street")).reason == pv.REASON_UNREAD


def test_holdback_want_gets_no_care_and_off_gets_no_term():
    v = _values(_Pool(tendencies=[_tendency_row()]))
    asyncio.run(v.refresh(NOW))
    t = asyncio.run(v.term(_holdback_want_id(), "street"))
    assert t.arm == "holdback" and t.care == 0.0 and t.endorsement_excluded == "holdback"
    off = PersonalityValues(enabled=False, pool_getter=lambda: _Pool(), embed=_Embedder())
    assert asyncio.run(off.term("w1", "street")) is None and off.choices() is None
    assert off.kick(NOW) is None


def test_the_runaway_guard_caps_a_tendency_shown_above_sixty_percent_for_seven_days(caplog):
    rows = []
    for d in range(1, 8):
        at = datetime.combine((NOW - timedelta(days=d)).date(), datetime.min.time(), tzinfo=timezone.utc) \
            + timedelta(hours=12)
        rows.append(_choice_row(d, at=at, chosen_text="street a", alt_texts=("street b", "street c", "pipeline")))
    v = _values(_Pool(tendencies=[_tendency_row()], choices=rows))
    with caplog.at_level("WARNING"):
        snap = asyncio.run(v.refresh(NOW))
    assert "t-street" in snap.caps and snap.caps["t-street"].mean_share == pytest.approx(0.75)
    assert snap.tendencies[0].cap == pytest.approx(0.8)
    assert "personality_values_tendency_capped tendency=t-street" in caplog.text
    t = asyncio.run(v.term(_care_want_id(), "street"))
    assert t.care == pytest.approx(0.5 * 0.8)


def _state():
    return PhysiologyStateV1(tick_id=f"physiology:{NOW.isoformat()}", generated_at=NOW, constraints=[
        R(constraint="lane:agent", unit="slot_s", level=600.0, budget=3600.0, budget_kind="capacity",
          detail={"slots": 1, "window_sec": 3600.0},
          price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""}),
    ])


class _Redis:
    async def get(self, key):
        return _state().model_dump_json()


def _acts():
    out = []
    for i in range(8):
        closed = NOW - timedelta(hours=2 + 3 * i)
        out.append({"act_id": f"a{i}", "actor": "durable.self_inquiry", "opened_at": closed - timedelta(hours=1),
                    "closed_at": closed, "outcome_value": 1.0, "outcome_source": "producer", "signal_id": None,
                    "cost_json": json.dumps({"seconds": 3600.0, "gpu_slot_s_by_lane": {"agent": 1200.0}})})
    return out


def _market(personality, pool):
    published = []

    async def publish(d):
        published.append(d)

    return Market(families=FAMS, families_error=None, pool_getter=lambda: pool, redis_getter=lambda: _Redis(),
                  publish=publish, personality=personality), published


def test_market_bid_records_arm_care_and_endorsement_in_the_decision_trace():
    pool = _Pool(tendencies=[_tendency_row()], acts=_acts())
    values = _values(pool)
    asyncio.run(values.refresh(NOW))
    m, published = _market(values, pool)
    want = WantV1(want_id=_care_want_id("si"), family="durable.self_inquiry", producer="unit", verb="ask", tier=2)
    d = asyncio.run(m.bid(want, NOW, object_text="street parking question"))
    assert "personality:arm=care" in d.warnings
    assert pv.trace_care(d.warnings) == pytest.approx(0.5)
    assert "personality:endorsement=none" in d.warnings
    assert published == [d] and m.health(NOW)["personality_values"]["care"] == 1


def test_market_with_personality_off_decides_exactly_as_before():
    pool = _Pool(acts=_acts())
    off = PersonalityValues(enabled=False, pool_getter=lambda: pool, embed=None)
    m_off, _ = _market(off, pool)
    m_none, _ = _market(None, pool)
    want = WantV1(want_id="w1", family="durable.self_inquiry", producer="unit", verb="ask", tier=2)
    a = asyncio.run(m_off.bid(want, NOW, object_text="street"))
    b = asyncio.run(m_none.bid(want, NOW))
    assert not any(w.startswith("personality:") for w in a.warnings)
    assert a.net == b.net and a.value_rank == b.value_rank and a.warnings == b.warnings


def test_want_object_text_reads_the_question_or_the_title_never_the_prompt():
    assert want_object_text({"brief": {"retrieval_query": "  why  do outreach\nmessages fail? ", "prompt": "x"}}) \
        == "why do outreach messages fail?"
    assert want_object_text({"brief": {"title": "fix(hub): a thing"}}) == "fix(hub): a thing"
    assert want_object_text({"brief": {"prompt": "long instructions"}}) is None
    assert want_object_text({}) is None


def test_a_failed_read_retries_once_per_ttl_not_on_every_bid():
    async def run():
        v = PersonalityValues(enabled=True, pool_getter=lambda: _Pool(fail=True), embed=_Embedder(), ttl_sec=600)
        t1 = v.kick(NOW)
        await t1
        assert v.kick(NOW + timedelta(seconds=5)) is None
        t2 = v.kick(NOW + timedelta(seconds=601))
        await t2
        return v.counts["refresh_failed"]

    assert asyncio.run(run()) == 2


def test_guard_embed_failures_keep_the_previous_caps():
    rows = []
    for d in range(1, 8):
        at = datetime.combine((NOW - timedelta(days=d)).date(), datetime.min.time(), tzinfo=timezone.utc) \
            + timedelta(hours=12)
        rows.append(_choice_row(d, at=at, chosen_text="street a", alt_texts=("street b", "street c", "pipeline")))
    emb = _Embedder()
    v = _values(_Pool(tendencies=[_tendency_row()], choices=rows), embed=emb)
    assert "t-street" in asyncio.run(v.refresh(NOW)).caps
    v._vec.clear()
    emb.fail = True
    snap = asyncio.run(v.refresh(NOW + timedelta(minutes=20)))
    assert snap.caps_kept and snap.guard_embed_failed == snap.shown_options and "t-street" in snap.caps


def _outreach_rows(n, said):
    return [_choice_row(100 + i, at=NOW - timedelta(days=1, minutes=i), declined=not said) for i in range(n)]


def test_market_care_arm_reads_source_c_and_holdback_does_not():
    acts = _acts() + [{"act_id": f"o{i}", "actor": "hub.outreach", "opened_at": NOW - timedelta(hours=5 + i),
                       "closed_at": NOW - timedelta(hours=4 + i), "outcome_value": None, "outcome_source": None,
                       "signal_id": None, "cost_json": "{}"} for i in range(3)]
    pool = _Pool(tendencies=[_tendency_row()], choices=_outreach_rows(30, said=False), acts=acts)
    values = _values(pool)
    asyncio.run(values.refresh(NOW))
    m, _ = _market(values, pool)
    m_off, _ = _market(None, pool)
    care = asyncio.run(m.bid(WantV1(want_id=_care_want_id("o"), family="hub.outreach", producer="unit",
                                    verb="say", tier=2), NOW))
    hold = asyncio.run(m.bid(WantV1(want_id=_holdback_want_id("o"), family="hub.outreach", producer="unit",
                                    verb="say", tier=2), NOW))
    off = asyncio.run(m_off.bid(WantV1(want_id="o-off", family="hub.outreach", producer="unit", verb="say",
                                       tier=2), NOW))
    assert "personality:endorsement=0.0000:n=30" in care.warnings
    assert care.family_weight < hold.family_weight
    assert hold.family_weight == off.family_weight


@pytest.mark.parametrize("hour", [0, 11, 13, 23])
def test_the_runaway_guard_window_covers_seven_complete_days_at_any_time_of_day(hour):
    """Regression: the guard read ``now - 7 days``, which cut the oldest complete day short after noon UTC,
    so no cap could ever form in the afternoon (CI went red at 16:36Z on 2026-11-13)."""
    now = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=hour, minutes=30)
    rows = []
    for d in range(1, 8):
        at = datetime.combine((now - timedelta(days=d)).date(), datetime.min.time(), tzinfo=timezone.utc) \
            + timedelta(hours=12)
        rows.append(_choice_row(d, at=at, chosen_text="street a", alt_texts=("street b", "street c", "pipeline")))
    v = _values(_Pool(tendencies=[_tendency_row()], choices=rows))
    assert "t-street" in asyncio.run(v.refresh(now)).caps
