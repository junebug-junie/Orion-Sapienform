"""Attention patch 3d: the market process runs the consent gate (decided item 8).

The gate answers BEFORE the body state or the act history is read, so a missing market ("no answer",
which binding producers admit) can never let an unconsented act through. Every verdict is one
``physiology_consent verdict=... rule=... want=...`` log line and rides the published decision to
want_ledger. A want without a consent request is decided and published exactly as before.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app.physiology_market import Market  # noqa: E402
from orion.physiology.consent import load_consent, parse_consent  # noqa: E402
from orion.schemas.physiology_want import ConsentRequestV1, WantDecisionV1, WantV1  # noqa: E402

from tests.test_physiology_bid import FAMS, NOW, _acts, _Pool, _Redis, _state, _want  # noqa: E402

RULES, ERR = load_consent(REPO_ROOT / "config/physiology/consent.yaml")


def _market(raw, pool, published, *, rules=RULES, families=FAMS):
    async def publish(decision):
        published.append(decision)

    return Market(families=families, families_error=None if families else "missing:families.yaml",
                  pool_getter=lambda: pool, redis_getter=lambda: _Redis(raw), publish=publish,
                  consent_rules=rules, consent_error=None if rules else "missing:consent.yaml")


def _consent_want(i=0, verb="compact", rev="snapshot_restorable"):
    wid = f"dream.compaction:compaction-delta:{i}"
    return WantV1(want_id=wid, family="dream.compaction", producer="orion-dream.compaction_applier.consent", verb=verb,
                  tier=2, consent_request=ConsentRequestV1(want_id=wid, family="dream.compaction", verb=verb,
                                                           payload_digest="sha256:00", reversibility=rev,
                                                           attests=["snapshot_dir_writable", "downscale_only"]))


def test_the_shipped_gate_answers_needs_consent_publishes_it_and_logs_the_rule(caplog):
    caplog.set_level(logging.INFO, logger="orion-durable-runs.physiology_market")
    published: list = []
    m = _market(_state().model_dump_json(), _Pool(acts=_acts()), published)
    d = asyncio.run(m.bid(_consent_want(), NOW))
    assert d.outcome == "deferred" and d.reason == "consent" and d.consent.verdict == "needs_consent"
    assert published == [d] and m.book.snapshot(NOW) == []
    # The ledger row is the full decision: want_ledger.decision->'consent' carries the verdict.
    row = json.loads(published[0].model_dump_json())
    assert row["consent"]["rule_id"] == "dream.compaction.downscale_only"
    assert row["want"]["consent_request"]["payload_digest"] == "sha256:00"
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("physiology_consent "))
    assert "verdict=needs_consent" in line and "rule=dream.compaction.downscale_only" in line
    assert f"want={_consent_want().want_id}" in line
    h = m.health(NOW)
    assert h["consent_needs_consent"] == 1 and h["consent_rules"] == 2 and h["consent_standing"] == 0


def test_the_gate_answers_before_any_read_so_a_dead_market_cannot_admit():
    """Body state absent, history unreadable, families config missing: each is "no answer" for an ordinary
    want, which a binding producer admits. A consent want is still refused, and nothing is read."""
    for raw, pool, fams in ((None, _Pool(acts=_acts()), FAMS), (_state().model_dump_json(), _Pool(fail=True), FAMS),
                            (_state().model_dump_json(), _Pool(acts=_acts()), None)):
        d = asyncio.run(_market(raw, pool, [], families=fams).bid(_consent_want(), NOW))
        assert d.reason == "consent", (raw, fams)
        assert pool.calls == 0


def test_missing_consent_rules_refuse_consent_wants_and_leave_every_other_want_alone():
    published: list = []
    m = _market(_state().model_dump_json(), _Pool(acts=_acts()), published, rules=None)
    d = asyncio.run(m.bid(_consent_want(), NOW))
    assert d.reason == "consent" and d.consent.verdict == "refuse" and d.consent.reason.startswith("rules_unavailable")
    ordinary = asyncio.run(m.bid(_want(), NOW))
    assert ordinary.outcome == "funded" and ordinary.consent is None
    assert "consent" not in ordinary.model_dump_json()


def test_an_ordinary_want_is_published_exactly_as_before():
    a, b = [], []
    asyncio.run(_market(_state().model_dump_json(), _Pool(acts=_acts()), a, rules=None).bid(_want(), NOW))
    asyncio.run(_market(_state().model_dump_json(), _Pool(acts=_acts()), b).bid(_want(), NOW))
    assert a[0].model_dump_json() == b[0].model_dump_json()


def test_with_a_standing_consent_the_want_is_priced_and_allowed_and_a_dead_market_still_carries_it():
    import yaml

    doc = yaml.safe_load((REPO_ROOT / "config/physiology/consent.yaml").read_text())
    doc["standing"] = {"dream-compaction-downscale": {"granted_by": "juniper", "granted_on": "2026-11-11"}}
    rules = parse_consent(doc)
    d = asyncio.run(_market(_state().model_dump_json(), _Pool(acts=_acts()), [], rules=rules).bid(_consent_want(), NOW))
    assert d.reason != "consent" and d.consent.verdict == "allow"
    assert d.consent.consent_ref == "standing:dream-compaction-downscale"
    dead = asyncio.run(_market(None, _Pool(acts=_acts()), [], rules=rules).bid(_consent_want(1), NOW))
    assert dead.reason == "physiology_unavailable" and dead.consent.verdict == "allow"
    WantDecisionV1.model_validate_json(dead.model_dump_json())


def test_prune_is_refused_even_with_a_standing_consent_for_downscale():
    import yaml

    doc = yaml.safe_load((REPO_ROOT / "config/physiology/consent.yaml").read_text())
    doc["standing"] = {"dream-compaction-downscale": {"granted_by": "juniper", "granted_on": "2026-11-11"}}
    m = _market(_state().model_dump_json(), _Pool(acts=_acts()), [], rules=parse_consent(doc))
    d = asyncio.run(m.bid(_consent_want(verb="prune", rev="irreversible"), NOW + timedelta(seconds=1)))
    assert d.reason == "consent" and d.consent.verdict == "refuse" and d.consent.rule_id == "dream.compaction.prune"


def test_main_loads_the_consent_path_from_settings():
    src = (SERVICE_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "load_consent(s.physiology_consent_path)" in src and "consent_rules=consent_rules" in src
    from app.settings import Settings

    assert Settings.model_fields["physiology_consent_path"].default == "/app/config/physiology/consent.yaml"
