"""Resonance shares obey the real existing outreach door; held notes do not block other readings."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from orion.schemas.reading_finding import ReadingFindingBundleV1
from orion.world_pulse_read.resonance_findings import finding_from_result
from orion.world_pulse_read.tests.test_resonance_runtime import receipt
from test_reading_finding_outreach import _setup


def bundle(decision="share", seed="read-1"):
    r = receipt(decision)
    f = finding_from_result(r, {(c.ref_kind, c.ref_id): "current record text" for c in r.judgment.citations})
    f.seed_id = seed
    return ReadingFindingBundleV1(seed_id=seed, learned="Learned about caches", findings=[f])


def test_disabled_resonance_does_not_stop_later_ordinary_reading(monkeypatch):
    loop, outreach, state, rows, composed, delivered = _setup(monkeypatch)
    normal = asyncio.run(loop._pending(loop._clock()))[0]
    loop.resonance_share_enabled = False
    loop._pending = AsyncMock(return_value=[bundle(), normal])
    result = asyncio.run(loop.tick())
    assert result["results"][0]["reason"] == "resonance_share_disabled"
    assert result["results"][1]["reason"] == "sent" and len(delivered) == 1


def test_mixed_bundle_still_sends_ordinary_finding_when_resonance_disabled(monkeypatch):
    loop, outreach, state, rows, composed, delivered = _setup(monkeypatch)
    normal = asyncio.run(loop._pending(loop._clock()))[0]
    normal.findings += bundle().findings
    loop.resonance_share_enabled = False
    result = asyncio.run(loop._tell(outreach, normal))
    assert result["reason"] == "sent"
    sent = [r for r in rows if r["reason"] == "sent"][0]
    assert "resonance" not in sent["finding_kinds"]


def test_same_local_day_does_not_reset_at_utc_midnight():
    from scripts.reading_finding_outreach import ReadingFindingOutreach
    class Conn:
        async def fetch(self, *args):
            return [{"decided_at": datetime(2026, 11, 10, 23, 30, tzinfo=timezone.utc),
                     "root": "root-1", "content_id": "earlier-other-citations"}]
    class Pool:
        @asynccontextmanager
        async def acquire(self):
            yield Conn()
    loop = ReadingFindingOutreach(enabled=True, pool_provider=Pool, outreach_provider=lambda: None,
        resonance_share_enabled=True, resonance_timezone="America/Denver", clock=lambda: datetime(2026,11,11,1,tzinfo=timezone.utc))
    assert asyncio.run(loop._resonance_gate(bundle().findings)) == "resonance_thread_already_shared"


def test_note_is_never_an_outreach_candidate(monkeypatch):
    from scripts.reading_finding_outreach import pending_bundles
    from scripts.reading_finding_outreach import reportable
    assert not reportable(bundle("note").findings[0])
    assert reportable(bundle("share").findings[0])
