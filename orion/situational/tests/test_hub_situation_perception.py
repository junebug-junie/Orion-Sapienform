"""Camera perception in Hub's unified-turn Situation block (2026-10-07).

Before this, `hub_settings_to_runtime_namespace` hardcoded perception to a
literal False, so Hub chat prompts always read "Room: haven't seen anything
recently" even while cortex-exec's prompts carried the live room narrative.
Now Hub reads its own ORION_SITUATION_PERCEPTION_ENABLED (default ON) into the
SAME shared builder cortex-exec runs.

What these tests pin:
- the flag on/off actually changes what reaches the prompt (and off never
  touches the database);
- a stale, absent, or erroring camera says "do not infer" and never carries
  the old scene text;
- the added text fits the live 7200 budget with every caution intact, and a
  tight cap still never slices a caution;
- the room read runs off the caller's event loop (Hub's).
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.situational import context as situation_mod
from orion.situational.context import (
    build_situation_for_ctx,
    hub_settings_to_runtime_namespace,
    settings_from_runtime,
)
from orion.situational.perception_reader import PresenceResolution, StreetSummary

# Real cam0 narratives from vision_events (2026-10-06/07). The second is the
# longest room narrative in the table (159 chars).
REAL_SCENE = (
    "Multiple chairs, doors, tables, and desks are visible in the scene. "
    "One item of clothing is also present."
)
LONGEST_REAL_SCENE = (
    "Three items are on the table, two chairs are present, two desks are visible, "
    "a door is present, a box is visible, clothing is present, and a person is visible."
)
OFF_LINE = "Room: haven't seen anything recently; do not infer."

# Every other provider off, so the fragment isolates perception.
_QUIET_HUB = dict(
    ORION_SITUATION_WEATHER_ENABLED=False,
    ORION_SITUATION_AFFECT_ENABLED=False,
    ORION_SITUATION_CURIOSITY_ENABLED=False,
    ORION_SITUATION_REVERIE_ENABLED=False,
    ORION_SITUATION_CABINET_ENABLED=False,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    situation_mod._SITUATION_CACHE.clear()

    async def _no_ask(*_a, **_k):
        return False

    monkeypatch.setattr(situation_mod, "try_claim_identity_ask", _no_ask)
    monkeypatch.setattr(
        situation_mod,
        "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: PresenceResolution(None, None, True),
    )
    monkeypatch.setattr(
        situation_mod,
        "fetch_street_summary",
        lambda sid, *, tz_name: StreetSummary(sid, [], True),
    )
    yield
    situation_mod._SITUATION_CACHE.clear()


def _hub_ns(**over):
    ns = hub_settings_to_runtime_namespace(SimpleNamespace(**{**_QUIET_HUB, **over}))
    ns.orion_situation_runtime_enabled = False
    return ns


def _fragment(ns, session: str = "s") -> tuple[dict, str]:
    brief, frag = asyncio.run(build_situation_for_ctx({"session_id": session}, ns))
    return brief, str(frag.get("compact_text") or "")


def _percept(age_seconds: float, scene: str = REAL_SCENE) -> dict:
    return {
        "scene_summary": scene,
        "observed_at": datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    }


# --- flag ---------------------------------------------------------------------


def test_hub_perception_defaults_on_with_cortex_exec_stream_defaults() -> None:
    cfg = settings_from_runtime(hub_settings_to_runtime_namespace(SimpleNamespace()))
    assert cfg.perception_enabled is True
    assert cfg.perception_max_age_seconds == 900
    # Same room cameras and street camera cortex-exec reads by default.
    assert cfg.perception_stream_ids == ["carbon", "cam0"]
    assert cfg.street_stream_ids == ["walkway"]


def test_hub_flag_false_is_the_kill_switch() -> None:
    cfg = settings_from_runtime(
        hub_settings_to_runtime_namespace(
            SimpleNamespace(ORION_SITUATION_PERCEPTION_ENABLED=False)
        )
    )
    assert cfg.perception_enabled is False


def test_hub_empty_street_ids_disables_only_the_street_line() -> None:
    cfg = settings_from_runtime(
        hub_settings_to_runtime_namespace(SimpleNamespace(ORION_SITUATION_STREET_STREAM_IDS=""))
    )
    assert cfg.perception_enabled is True
    assert cfg.street_stream_ids == []


def test_empty_street_ids_never_reads_the_street(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("street disabled: must not read")

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda **_: _percept(60))
    monkeypatch.setattr(situation_mod, "fetch_street_summary", _boom)
    _brief, text = _fragment(_hub_ns(ORION_SITUATION_STREET_STREAM_IDS=""))
    assert REAL_SCENE in text
    assert "Street" not in text


def test_flag_off_never_reads_the_camera_and_says_do_not_infer(monkeypatch) -> None:
    def _boom(**_k):
        raise AssertionError("perception disabled: must not read vision_events")

    def _boom_pos(*_a, **_k):
        raise AssertionError("perception disabled: must not read presence/street")

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", _boom)
    monkeypatch.setattr(situation_mod, "fetch_presence_resolved", _boom_pos)
    monkeypatch.setattr(situation_mod, "fetch_street_summary", _boom_pos)
    brief, text = _fragment(_hub_ns(ORION_SITUATION_PERCEPTION_ENABLED=False))
    assert OFF_LINE in text
    assert brief["perception"]["source"] == "disabled"


def test_flag_on_puts_the_live_room_narrative_in_the_hub_prompt(monkeypatch) -> None:
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda **_: _percept(480))
    brief, text = _fragment(_hub_ns())
    assert f"Room (seen 8 min ago): {REAL_SCENE}" in text
    assert OFF_LINE not in text
    assert brief["perception"]["source"] == "live"


# --- stale / absent camera ----------------------------------------------------


@pytest.mark.parametrize(
    "reader, source",
    [
        (lambda **_: _percept(901), "stale"),  # one second past the gate
        (lambda **_: None, "unavailable"),  # camera never wrote a row
        (lambda **_: {"scene_summary": "", "observed_at": None}, "unavailable"),
    ],
)
def test_stale_or_absent_camera_never_carries_scene_text(monkeypatch, reader, source) -> None:
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", reader)
    brief, text = _fragment(_hub_ns())
    assert OFF_LINE in text
    assert "chairs" not in text  # the old scene must not leak
    assert brief["perception"]["available"] is False
    assert brief["perception"]["source"] == source
    assert brief["perception"].get("scene_summary") in (None, "")


def test_stubbed_reader_raising_is_do_not_infer(monkeypatch) -> None:
    def _down(**_k):
        raise ConnectionError("db down")

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", _down)
    brief, text = _fragment(_hub_ns())
    assert OFF_LINE in text
    assert brief["perception"]["source"] == "error"


def test_real_reader_outage_is_do_not_infer_not_an_empty_room(monkeypatch) -> None:
    """The real reader swallows connection errors and returns None, so a
    production outage arrives as `unavailable` -- pin THAT path, through the
    real fetch_latest_percept, not only a stub that raises."""
    from orion.situational import perception_reader

    class _DeadEngine:
        def connect(self):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(perception_reader, "_get_engine", lambda: _DeadEngine())
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", perception_reader.fetch_latest_percept)
    brief, text = _fragment(_hub_ns())
    assert OFF_LINE in text
    assert "empty" not in text.lower()
    assert brief["perception"]["source"] == "unavailable"


def test_stale_presence_row_does_not_claim_someone_is_in_view(monkeypatch) -> None:
    """A fresh scene plus a presence row frozen hours ago must not render
    "Someone has been in view" as current."""
    old = {
        "state": "present",
        "since_sec": 30.0,
        "row_updated_at": datetime.now(timezone.utc) - timedelta(hours=3),
    }
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda **_: _percept(60))
    monkeypatch.setattr(
        situation_mod,
        "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: PresenceResolution("cam0", old, True),
    )
    _brief, text = _fragment(_hub_ns())
    assert "Someone has been in view" not in text
    assert REAL_SCENE in text


# --- budget -------------------------------------------------------------------


def _worst_case(monkeypatch) -> None:
    """Longest real scene, a fresh presence fragment, a four-line street
    summary, and the longest identity-ask caution, all at once."""
    fresh = {
        "state": "present",
        "since_sec": 3 * 3600.0,
        "row_updated_at": datetime.now(timezone.utc),
    }
    monkeypatch.setattr(
        situation_mod, "fetch_latest_percept", lambda **_: _percept(30, LONGEST_REAL_SCENE)
    )
    monkeypatch.setattr(
        situation_mod,
        "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: PresenceResolution("cam0", fresh, True),
    )
    street = [
        "On the walkway in the last 15 minutes: two people, a dog, and a bicycle.",
        "Walkway rhythm: the mail carrier came as expected around 14:10; "
        "the black dog did not come (usually around 07:40); a neighbor usually comes around 17:30; "
        "a cyclist is here, as expected.",
        "In the last hour I saw 3 things on the walkway I could not name; the latest at 15:02: "
        + "x" * 100
        + ".",
        "2 people are on the patio.",
    ]
    monkeypatch.setattr(
        situation_mod, "fetch_street_summary", lambda sid, *, tz_name: StreetSummary(sid, street, True)
    )

    async def _ask(*_a, **_k):
        return True

    monkeypatch.setattr(situation_mod, "try_claim_identity_ask", _ask)


def test_worst_case_perception_fits_the_live_7200_cap_with_every_caution(monkeypatch) -> None:
    _brief, off_text = _fragment(_hub_ns(ORION_SITUATION_PERCEPTION_ENABLED=False), "off")
    _worst_case(monkeypatch)
    brief, on_text = _fragment(_hub_ns(), "on")

    assert brief["perception"]["presence_identity_ask"] == "identity_unread"
    assert "Someone has been in view for" in on_text
    assert "…" not in on_text  # nothing truncated
    for caution in (
        "don't announce it unprompted.",
        "Situation context is grounding, not a requirement to mention.",
        "avoid contrived time/weather/location commentary.",
        "do not repeat it again this conversation once asked.",
    ):
        assert caution in on_text
    # Synthetic worst case measured at 1063 added chars (~15% of the live
    # budget); the real 2026-10-07 case added 77 (Street quiet, no ask).
    added = len(on_text) - len(off_text)
    assert 0 < added < 1500, added
    assert len(on_text) <= 7200


def test_tight_cap_shortens_facts_but_never_slices_a_caution(monkeypatch) -> None:
    _worst_case(monkeypatch)
    _brief, frag = asyncio.run(
        build_situation_for_ctx(
            {"session_id": "tight"}, _hub_ns(ORION_SITUATION_PROMPT_MAX_CHARS=600)
        )
    )
    text = frag["compact_text"]
    assert len(text) <= 600
    # Each caution is in the prompt whole or not at all -- never a fragment.
    lines = text.split("\n- ")
    for caution in frag["caution_lines"]:
        for line in lines:
            # No line may be a strict, cut-off prefix of a caution.
            assert not (line != caution and len(line) > 10 and caution.startswith(line)), line


# --- event loop ---------------------------------------------------------------


def test_room_read_runs_off_the_event_loop_thread(monkeypatch) -> None:
    seen: dict[str, int] = {}

    def _reader(**_k):
        seen["reader"] = threading.get_ident()
        return _percept(60)

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", _reader)

    async def _run() -> None:
        seen["loop"] = threading.get_ident()
        await build_situation_for_ctx({"session_id": "loop"}, _hub_ns())

    asyncio.run(_run())
    assert seen["reader"] != seen["loop"]


# --- identity ask only on Juniper's own turns ---------------------------------


def test_orion_authored_turn_never_spends_the_identity_ask(monkeypatch) -> None:
    """Hub builds briefs for endogenous outreach too (record_user_turn=False).
    Those must not claim the shared cooldown: no unprompted "is that you?",
    and the slot stays for Juniper's next real turn."""

    async def _must_not_claim(*_a, **_k):
        raise AssertionError("an Orion-authored turn must not claim the identity ask")

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda **_: _percept(60))
    monkeypatch.setattr(situation_mod, "try_claim_identity_ask", _must_not_claim)
    brief, frag = asyncio.run(
        build_situation_for_ctx({"session_id": "outreach", "record_user_turn": False}, _hub_ns())
    )
    assert brief["perception"]["presence_identity_ask"] is None
    assert "is that you" not in str(frag.get("compact_text"))
    assert REAL_SCENE in str(frag.get("compact_text"))


def test_juniper_turn_still_gets_the_identity_ask(monkeypatch) -> None:
    claims: list[str] = []

    async def _claim(scope, *, reason, ttl_seconds):
        claims.append(reason)
        return True

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda **_: _percept(60))
    monkeypatch.setattr(situation_mod, "try_claim_identity_ask", _claim)
    brief, _frag = asyncio.run(
        build_situation_for_ctx({"session_id": "juniper", "record_user_turn": True}, _hub_ns())
    )
    assert claims == ["no_visual_confirmation"]
    assert brief["perception"]["presence_identity_ask"] == "no_visual_confirmation"


# --- cache cannot outlive the staleness gate ----------------------------------


def test_cached_brief_is_rebuilt_once_its_percept_passes_the_gate(monkeypatch) -> None:
    reads: list[int] = []

    def _reader(**_k):
        reads.append(1)
        return _percept(899 if len(reads) == 1 else 2000)

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", _reader)
    ns = _hub_ns()
    _b, first = _fragment(ns, "cache")
    assert REAL_SCENE in first
    # Age the cache entry by 30 s: 899 + 30 > 900, so it must not be served.
    key, (built, brief, frag) = next(iter(situation_mod._SITUATION_CACHE.items()))
    situation_mod._SITUATION_CACHE[key] = (built - timedelta(seconds=30), brief, frag)
    _b, second = _fragment(ns, "cache")
    assert len(reads) == 2
    assert OFF_LINE in second


def test_fresh_cached_brief_is_still_served_from_cache(monkeypatch) -> None:
    reads: list[int] = []

    def _reader(**_k):
        reads.append(1)
        return _percept(60)

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", _reader)
    ns = _hub_ns()
    _fragment(ns, "cache2")
    _fragment(ns, "cache2")
    assert len(reads) == 1


# --- named presence + stale-read caution (2026-10-09) -------------------------
# A turn asking "can you see me?" was handed "Room (seen 7 min ago): Someone has
# been in view for 7 minutes. ..." while the camera had matched Juniper (0.70) a
# minute earlier, and answered "Right here, right now" off a 7-minute-old read.


def _live(monkeypatch, *, subject, percept_age, state="present", since=420.0) -> None:
    row = {
        "state": state,
        "since_sec": since,
        "subject": subject,
        "identity_confirmed": subject not in ("unknown", "none"),
        "row_updated_at": datetime.now(timezone.utc),
    }
    monkeypatch.setattr(
        situation_mod, "fetch_latest_percept", lambda **_: _percept(percept_age, REAL_SCENE)
    )
    monkeypatch.setattr(
        situation_mod,
        "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: PresenceResolution("cam0", row, True),
    )

    async def _ask(*_a, **_k):
        return True

    monkeypatch.setattr(situation_mod, "try_claim_identity_ask", _ask)


def test_matched_face_names_the_person_in_the_room_line(monkeypatch) -> None:
    _live(monkeypatch, subject="juniper", percept_age=30)
    brief, text = _fragment(_hub_ns(), "named")
    assert "Juniper has been in view for 7 minutes (matched by face)." in text
    assert "Someone has been in view" not in text
    assert brief["perception"]["presence_subject"] == "juniper"


def test_unmatched_presence_still_says_someone_and_never_a_name(monkeypatch) -> None:
    _live(monkeypatch, subject="unknown", percept_age=30)
    _brief, text = _fragment(_hub_ns(), "anon")
    assert "Someone has been in view for 7 minutes." in text
    assert "matched by face" not in text


def test_stale_read_gets_a_not_live_caution_and_fresh_read_does_not(monkeypatch) -> None:
    _live(monkeypatch, subject="juniper", percept_age=7 * 60)
    _brief, stale = _fragment(_hub_ns(), "stale")
    assert "Your last visual read of the room is 7 min ago, not live." in stale
    assert "Do not say you see anyone or anything right now" in stale

    _live(monkeypatch, subject="juniper", percept_age=20)
    _brief, fresh = _fragment(_hub_ns(), "fresh")
    assert "not live" not in fresh
