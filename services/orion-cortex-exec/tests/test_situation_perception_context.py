"""P4: the camera percept in the situation brief.

The two properties that matter more than the happy path: a stale percept must
never reach a prompt as a current observation, and the privacy contract in
PerceptionContextV1's docstring must be enforced by the schema rather than by
good intentions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orion.situational import context as situation_mod
from orion.situational.context import (
    SituationSettings,
    _build_perception_context,
    _build_prompt_fragment,
    settings_from_runtime,
)
from orion.situational.identity_ask_cooldown import (
    bind_identity_ask_cooldown_bus,
    reset_identity_ask_cooldown_bus_for_tests,
)
from orion.schemas.situation import PerceptionContextV1, SituationBriefV1


class _FakeCooldownRedis:
    """Matches real redis-py `SET ... NX` semantics: True the first time a
    key is set, None (falsy) while it already exists -- see
    identity_ask_cooldown.py's own atomic-claim contract."""

    def __init__(self, store: dict | None = None) -> None:
        self.store: dict = store if store is not None else {}
        self.set_calls: list = []

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.store:
            return None
        self.store[key] = value.encode("utf-8")
        return True


class _FakeCooldownBus:
    def __init__(self, redis: _FakeCooldownRedis) -> None:
        self.redis = redis


class _RaisingCooldownRedis:
    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        raise ConnectionError("redis unreachable")


@pytest.fixture(autouse=True)
def _reset_cooldown_bus():
    reset_identity_ask_cooldown_bus_for_tests()
    yield
    reset_identity_ask_cooldown_bus_for_tests()

NOW = datetime.now(timezone.utc)


def _cfg(**overrides) -> SituationSettings:
    cfg = settings_from_runtime(SimpleNamespace())
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _diag():
    from orion.schemas.situation import SituationDiagnosticsV1

    return SituationDiagnosticsV1()


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """Never touch a database from these tests."""
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda: None)
    # Every existing test that reaches the "ok" path now also calls
    # the resolved presence read -- without this default, they would attempt
    # a real Postgres connection the moment presence fusion was added.
    monkeypatch.setattr(
        situation_mod,
        "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: (None, None),
    )


# --- provider states -------------------------------------------------------


def test_disabled_by_default() -> None:
    # Camera-derived content about a private home is opt-in.
    assert settings_from_runtime(SimpleNamespace()).perception_enabled is False


def test_disabled_yields_unavailable_not_an_error() -> None:
    diag = _diag()
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=False), diag))
    assert ctx.available is False
    assert ctx.source == "disabled"
    assert diag.provider_status["perception"] == "disabled"


def test_no_percept_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda: None)
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.available is False
    assert ctx.source == "unavailable"
    assert ctx.scene_summary is None


def test_fresh_percept_is_available(monkeypatch) -> None:
    monkeypatch.setattr(
        situation_mod,
        "fetch_latest_percept",
        lambda: {"scene_summary": "Three chairs and a door are visible.", "observed_at": NOW},
    )
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.available is True
    assert ctx.source == "live"
    assert ctx.scene_summary == "Three chairs and a door are visible."
    assert ctx.observation_age_seconds is not None and ctx.observation_age_seconds < 60


def test_stale_percept_is_withheld_entirely(monkeypatch) -> None:
    """
    The core honesty gate. An old percept presented as current is a
    confabulation with a real referent, which is worse than saying nothing --
    so the summary is not merely flagged stale, it is not carried at all.
    """
    old = NOW - timedelta(hours=2)
    monkeypatch.setattr(
        situation_mod,
        "fetch_latest_percept",
        lambda: {"scene_summary": "A person is at the desk.", "observed_at": old},
    )
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True, perception_max_age_seconds=900), _diag()))
    assert ctx.available is False
    assert ctx.source == "stale"
    assert ctx.scene_summary is None, "a stale summary must not ride along in the payload"
    assert ctx.observation_age_seconds is not None and ctx.observation_age_seconds > 900


def test_age_boundary_is_inclusive_of_the_threshold(monkeypatch) -> None:
    at_limit = NOW - timedelta(seconds=900)
    monkeypatch.setattr(
        situation_mod,
        "fetch_latest_percept",
        lambda: {"scene_summary": "A door.", "observed_at": at_limit},
    )
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True, perception_max_age_seconds=900), _diag()))
    assert ctx.available is True, "exactly at the threshold is still fresh"


def test_reader_exception_fails_open(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr(situation_mod, "fetch_latest_percept", _boom)
    diag = _diag()
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), diag))
    assert ctx.available is False
    assert ctx.source == "error"
    assert "db gone" in diag.provider_errors["perception"]


def test_empty_narrative_is_not_a_percept(monkeypatch) -> None:
    monkeypatch.setattr(
        situation_mod, "fetch_latest_percept", lambda: {"scene_summary": "", "observed_at": NOW}
    )
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.available is False


# --- prompt rendering ------------------------------------------------------


def _brief(perception: PerceptionContextV1) -> SituationBriefV1:
    """Build a real brief via the production helpers, then swap in the percept.

    Hand-rolling the sub-contexts means guessing their required fields; using
    the real builders means this test breaks if their shape drifts, which is
    the more useful failure.
    """
    cfg = _cfg(perception_enabled=False)
    diag = _diag()
    time_ctx = situation_mod._build_time_context(cfg, diag)
    return SituationBriefV1(
        generated_at=NOW,
        time=time_ctx,
        # _build_conversation_phase is now async (Redis-backed session turn
        # state, see session_turn_phase.py) -- these tests only care about
        # perception rendering, so no bus is bound here and the call
        # fails open to phase="unknown" (an unbound-bus WARNING is expected
        # and harmless in this file's test output).
        conversation_phase=asyncio.run(situation_mod._build_conversation_phase({}, time_ctx, NOW)),
        place=situation_mod._build_place_context(cfg),
        perception=perception,
    )


def test_available_percept_renders_with_its_age() -> None:
    brief = _brief(
        PerceptionContextV1(
            available=True,
            source="live",
            scene_summary="Three chairs and a door are visible.",
            observation_age_seconds=120,
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "Room (seen 2 min ago): Three chairs and a door are visible." in text


def test_unavailable_renders_as_not_having_looked_not_as_an_empty_room() -> None:
    """
    "I haven't seen anything" and "there is nothing there" are different
    claims, and only the first is true when the camera is stale or off.
    """
    text = _build_prompt_fragment(_brief(PerceptionContextV1()), 4000).compact_text
    assert "haven't seen anything recently" in text
    for wrong in ("room is empty", "nothing in the room", "the room is quiet"):
        assert wrong not in text.lower()


def test_stale_percept_summary_never_reaches_the_prompt() -> None:
    stale = PerceptionContextV1(
        available=False, source="stale", observation_age_seconds=7200
    )
    text = _build_prompt_fragment(_brief(stale), 4000).compact_text
    assert "haven't seen anything recently" in text


# --- privacy contract ------------------------------------------------------


def test_schema_exposes_no_identity_or_raw_frame_fields() -> None:
    """
    The exposed-field list IS the privacy contract. `extra="forbid"` means a
    future caller cannot smuggle one of these in without changing this schema,
    which is the point.
    """
    fields = set(PerceptionContextV1.model_fields)
    for banned in (
        "entities",
        "faces",
        "identities",
        "objects",
        "boxes",
        "bounding_boxes",
        "frame_path",
        "image_path",
        "embedding",
        "detections",
    ):
        assert banned not in fields, f"{banned} must not be exposed to the prompt"


def test_percepts_are_session_only_by_default() -> None:
    assert PerceptionContextV1().privacy_mode == "session_only"


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(Exception):
        PerceptionContextV1(frame_path="/mnt/telemetry/vision/frames/x.jpg")


# --- presence fusion --------------------------------------------------------


def _with_percept(monkeypatch, text: str = "Three chairs and a door are visible."):
    monkeypatch.setattr(
        situation_mod, "fetch_latest_percept",
        lambda: {"scene_summary": text, "observed_at": NOW},
    )


def _with_presence(monkeypatch, row, *, stream_id: str = "cam0", age_sec: float = 1.0):
    """Stub the resolved presence read with a row of a REAL age.

    `row_updated_at` is not decoration in these tests: since 2026-08-29 the
    identity-ask decision is gated on how recently the presence row was
    written, because a camera that goes dark stops updating its row rather
    than writing "absent" into it. A fixture that omitted the timestamp would
    exercise a path no live row can take -- every row that comes back from
    `fetch_presence_resolved` carries one.
    """
    resolved = (
        None
        if row is None
        else {**row, "row_updated_at": datetime.now(timezone.utc) - timedelta(seconds=age_sec)}
    )
    monkeypatch.setattr(
        situation_mod,
        "fetch_presence_resolved",
        lambda stream_ids, *, max_age_seconds: (
            (stream_id, resolved) if resolved is not None else (None, None)
        ),
    )


def test_present_prefixes_a_duration_fragment(monkeypatch) -> None:
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, {"state": "present", "since_sec": 10800.0, "subject": "unknown"})
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.scene_summary.startswith("Someone has been in view for about 3 hours.")
    assert ctx.scene_summary.endswith("Three chairs and a door are visible.")
    assert ctx.presence_state == "present"
    assert ctx.presence_since_sec == 10800.0


def test_recent_uses_stepped_out_phrasing(monkeypatch) -> None:
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, {"state": "recent", "since_sec": 45.0, "subject": "unknown"})
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert "stepped out of view" in ctx.scene_summary
    assert ctx.presence_state == "recent"


def test_absent_adds_no_fragment_and_no_noise(monkeypatch) -> None:
    """An empty room is the default expectation, not news."""
    _with_percept(monkeypatch, "Two chairs and a table are visible.")
    _with_presence(monkeypatch, {"state": "absent", "since_sec": 3000.0, "subject": "none"})
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.scene_summary == "Two chairs and a table are visible."
    assert ctx.presence_state == "absent"


def test_no_presence_row_is_a_normal_no_op(monkeypatch) -> None:
    """A stream with no presence row yet (e.g. brand new) must not error."""
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, None)
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.scene_summary == "Three chairs and a door are visible."
    assert ctx.presence_state is None


def test_presence_read_failure_fails_open(monkeypatch) -> None:
    """Same fail-open contract as the percept reader itself.

    Changed 2026-08-29, and this is a real behavior fix rather than a test
    edit: the old code called fetch_presence bare, so a database blip
    PROPAGATED out of _build_perception_context and took the whole turn
    assembly with it -- the docstring said "fails open" while the assertion
    directly below it demanded a raise. The presence read now lives in
    _resolve_presence_and_identity_ask behind the same try/except every other
    provider in this module uses, so a broken database costs the presence
    enrichment and nothing else.
    """
    _with_percept(monkeypatch)

    def _boom(stream_ids, *, max_age_seconds):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(situation_mod, "fetch_presence_resolved", _boom)
    diag = _diag()
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), diag))
    assert ctx.available is True, "a presence failure must not cost the percept"
    assert ctx.presence_state is None
    assert ctx.presence_identity_ask is None, "no ask on a read we could not make"
    assert diag.provider_status["perception_presence"] == "error"


def test_presence_prose_never_enriches_a_stale_or_unavailable_percept(monkeypatch) -> None:
    """The presence PROSE stays gated on a valid percept -- a room not seen
    recently must not surface 'someone was there three hours ago' as if it
    were current.

    Narrowed 2026-08-29 from "fetch_presence must not be CALLED at all". The
    read now happens unconditionally, because the identity-ask decision needs
    it on exactly the paths where there is no valid percept: a closed laptop
    lid produces no percepts, and that is the case Juniper reported the ask
    failing in. What must stay gated is the narrative fragment and the
    structured presence fields, which is what this now asserts -- the
    distinction the old assertion conflated.
    """
    called = {"n": 0}

    def _spy(stream_ids, *, max_age_seconds):
        called["n"] += 1
        return ("cam0", {"state": "present", "since_sec": 10.0, "subject": "unknown"})

    monkeypatch.setattr(situation_mod, "fetch_presence_resolved", _spy)
    monkeypatch.setattr(situation_mod, "fetch_latest_percept", lambda: None)  # unavailable
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.available is False
    assert ctx.scene_summary is None, "no percept means no narrative, presence or not"
    assert ctx.presence_state is None, "structured presence stays gated on a valid percept"
    assert called["n"] == 1, "the read itself now happens on every non-disabled path"


# --- presence_identity_uncertain, 2026-08-26 --------------------------------
# Juniper's direct ask: confirmed -> silence (already true above, nothing new
# needed); genuinely uncertain -> surface it, but at most once per sit-down
# (identity_ask_cooldown.py); broken/not-running -> silence. These tests
# cover the cooldown wiring itself -- unit coverage for the cooldown module's
# own logic lives in test_identity_ask_cooldown.py.


def test_identity_uncertain_row_sets_the_context_field_and_marks_cooldown(monkeypatch) -> None:
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, {"state": "present", "since_sec": 5.0, "subject": "unknown", "identity_uncertain": True})
    redis = _FakeCooldownRedis()
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(redis))
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_uncertain is True
    assert len(redis.set_calls) == 1, "must mark the cooldown the first time this surfaces"


def test_identity_uncertain_suppressed_when_already_in_cooldown(monkeypatch) -> None:
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, {"state": "present", "since_sec": 5.0, "subject": "unknown", "identity_uncertain": True})
    redis = _FakeCooldownRedis()
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(redis))
    first = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert first.presence_identity_uncertain is True

    second = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert second.presence_identity_uncertain is False, "already asked -- next turn must stay quiet"
    # The atomic claim is attempted again (SET ... NX is the check, not a
    # separate read first) but does not win -- see _FakeCooldownRedis's own
    # NX semantics. Two attempts, one actual claim; the *result* (asserted
    # above) is what must not repeat, not the attempt count.
    assert len(redis.set_calls) == 2


def test_identity_uncertain_false_when_presence_row_lacks_the_field(monkeypatch) -> None:
    """Backward compatible with a presence row written before this field
    existed -- no cooldown bus needed at all since the field is absent."""
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, {"state": "present", "since_sec": 5.0, "subject": "unknown"})
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_uncertain is False


def test_identity_uncertain_false_when_the_field_is_explicitly_false(monkeypatch) -> None:
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, {"state": "present", "since_sec": 5.0, "subject": "juniper", "identity_uncertain": False})
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_uncertain is False


def test_identity_uncertain_cooldown_read_failure_fails_open_to_asking(monkeypatch) -> None:
    """Fail-open points TOWARD asking, not toward silence (see
    identity_ask_cooldown.py's module docstring): a Redis hiccup on the
    cooldown claim must not silently suppress a genuine identity mismatch --
    worst case is one redundant ask, not a feature that goes mute.
    try_claim_identity_ask itself is documented "never raises" (fail-open
    internally), so this also proves _build_perception_context does not
    need -- and per this file's own established convention (see
    fetch_presence above), should not add -- a redundant try/except of its
    own; turn assembly must not blow up either way."""
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, {"state": "present", "since_sec": 5.0, "subject": "unknown", "identity_uncertain": True})
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(_RaisingCooldownRedis()))
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_uncertain is True


# --- presence_identity_uncertain: prompt rendering ---------------------------


def test_identity_uncertain_caution_appears_in_prompt() -> None:
    brief = _brief(
        PerceptionContextV1(
            available=True,
            source="live",
            scene_summary="A person is visible.",
            observation_age_seconds=5,
            presence_identity_uncertain=True,
            presence_identity_ask="unmatched_face",
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "is that you" in text.lower()
    assert "Juniper" in text
    assert "currently in view" in text, "the unmatched-face wording describes a visible person"


def test_identity_uncertain_caution_absent_when_false() -> None:
    brief = _brief(
        PerceptionContextV1(
            available=True,
            source="live",
            scene_summary="A person is visible.",
            observation_age_seconds=5,
            presence_identity_uncertain=False,
            presence_identity_ask=None,
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "is that you" not in text.lower()


# --- duration formatting, hand-computed -------------------------------------


@pytest.mark.parametrize(
    "seconds,expected_substring",
    [
        (0.0, "0 seconds"),
        (45.0, "45 seconds"),
        (89.0, "89 seconds"),          # just under the 90s -> minutes cutover
        (90.0, "2 minutes"),           # 90s = 1.5min -> round(1.5) = 2 (banker's: half-to-even)
        (150.0, "2 minutes"),          # 150s = 2.5min -> round(2.5) = 2, NOT 3 (half-to-even
                                        # again; Python's round() is not "round half up", and a
                                        # first draft of this fixture assumed it was)
        (3599.0, "60 minutes"),        # just under 60min, still rendered in minutes
        (5400.0 - 1, "90 minutes"),    # 89min59s: under the 90min hour-cutover, minutes
        (5400.0, "about 2 hours"),     # exactly 90min = 1.5h: `hours < 1.5` EXCLUDES 1.5 itself,
                                        # so the cutover to "about an hour" sits strictly below
                                        # this value, not AT it -- a first draft of this fixture
                                        # assumed the boundary was inclusive and was wrong
        (7200.0, "about 2 hours"),     # 2h exactly
        (10800.0, "about 3 hours"),    # 3h -- the value used in the live test above
    ],
)
def test_coarse_duration_boundaries(seconds, expected_substring) -> None:
    # Promoted to orion.situational.perception_reader (2026-08-25) so
    # endogenous_outreach.py's presence-aware prompt block reads the same
    # formatting instead of a second, independently-drifting copy --
    # context.py now imports the public name rather than defining its own.
    from orion.situational.perception_reader import coarse_duration

    assert coarse_duration(seconds) == expected_substring


def test_negative_or_none_since_sec_produces_no_fragment() -> None:
    from orion.situational.perception_reader import presence_fragment

    assert presence_fragment("present", None) is None
    assert presence_fragment("present", -5.0) is None
    assert presence_fragment("absent", 100.0) is None
    assert presence_fragment(None, 100.0) is None


# --- the ask Juniper actually reported missing, 2026-08-29 -------------------
# "orion never bites when they can't recognize me (eg I close the camera lid).
# they are supposed to ask who dis"
#
# Every test below fails against the pre-2026-08-29 code, and that is the
# point: the old trigger was `present_now AND identity_confidence ==
# "uncertain"`, which requires a face to have been DETECTED and not matched.
# A closed lid emits no frames at all, so no face, so no "uncertain" -- the
# one situation the feature was asked for was the one it structurally could
# not reach.


def test_closed_lid_asks_who_this_is(monkeypatch) -> None:
    """THE regression test. No presence row readable at all -- the webcam is
    off, so orion-vision-window has nothing to write about."""
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, None)
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(_FakeCooldownRedis()))
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_ask == "no_visual_confirmation"
    assert ctx.presence_identity_uncertain is False, (
        "no face was detected, so the unmatched-face observation stays False -- "
        "the ask does not borrow a fact it does not have"
    )


def test_camera_that_went_dark_mid_session_asks(monkeypatch) -> None:
    """The lid closed a while ago: the row still SAYS 'present' because
    nothing has overwritten it, but it stopped being updated. Row content
    alone cannot tell this from a live sighting -- only its age can."""
    _with_percept(monkeypatch)
    _with_presence(
        monkeypatch,
        {"state": "present", "since_sec": 300.0, "subject": "juniper", "identity_confirmed": True},
        age_sec=3600.0,
    )
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(_FakeCooldownRedis()))
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_ask == "no_visual_confirmation", (
        "an hour-old 'confirmed' row is not a confirmation of who is here now"
    )


def test_ask_survives_the_stale_percept_early_return(monkeypatch) -> None:
    """The structural fix. A dark camera produces no fresh percepts either, so
    this function returns source='stale' well before it used to reach the
    identity block. The ask has to outlive that early return or the feature is
    unreachable in exactly the case it exists for."""
    monkeypatch.setattr(
        situation_mod,
        "fetch_latest_percept",
        lambda: {
            "scene_summary": "An empty room.",
            "observed_at": NOW - timedelta(seconds=100000),
        },
    )
    _with_presence(monkeypatch, None)
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(_FakeCooldownRedis()))
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.source == "stale"
    assert ctx.available is False
    assert ctx.scene_summary is None, "the stale-summary privacy rule still holds"
    assert ctx.presence_identity_ask == "no_visual_confirmation"


def test_fresh_confirmed_read_stays_silent(monkeypatch) -> None:
    """The one case that must never ask: Orion can see Juniper right now."""
    _with_percept(monkeypatch)
    _with_presence(
        monkeypatch,
        {"state": "present", "since_sec": 30.0, "subject": "juniper", "identity_confirmed": True},
    )
    redis = _FakeCooldownRedis()
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(redis))
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_ask is None
    assert ctx.presence_identity_uncertain is False
    assert redis.set_calls == [], "no cooldown burned on a turn with nothing to ask"


def test_nobody_in_frame_asks_who_is_talking(monkeypatch) -> None:
    """Camera alive and reporting, but the chair is empty -- someone is
    nonetheless mid-conversation with Orion, since this only runs while a turn
    is being assembled."""
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, {"state": "absent", "since_sec": 900.0, "subject": "none"})
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(_FakeCooldownRedis()))
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_ask == "no_visual_confirmation"


def test_unmatched_face_still_reports_its_own_reason(monkeypatch) -> None:
    """The original 2026-08-26 signal is preserved, not swallowed by the
    broader one -- a detected stranger is a different fact from a dark
    camera, and they carry different cooldowns and different prompt text."""
    _with_percept(monkeypatch)
    _with_presence(
        monkeypatch,
        {"state": "present", "since_sec": 5.0, "subject": "unknown", "identity_uncertain": True},
    )
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(_FakeCooldownRedis()))
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert ctx.presence_identity_ask == "unmatched_face"
    assert ctx.presence_identity_uncertain is True


def test_the_two_reasons_hold_independent_cooldowns(monkeypatch) -> None:
    """A shared key would let the common reason starve the rare one: hours of
    lid-closed chat would claim the slot, and then a genuine stranger walking
    into frame would go unremarked."""
    _with_percept(monkeypatch)
    redis = _FakeCooldownRedis()
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(redis))

    _with_presence(monkeypatch, None)
    first = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert first.presence_identity_ask == "no_visual_confirmation"

    _with_presence(
        monkeypatch,
        {"state": "present", "since_sec": 5.0, "subject": "unknown", "identity_uncertain": True},
    )
    second = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert second.presence_identity_ask == "unmatched_face", (
        "the unconfirmed cooldown must not suppress a detected mismatch"
    )
    assert len({key for key, *_ in redis.set_calls}) == 2, "two distinct cooldown keys"


def test_unconfirmed_ask_is_rate_limited_like_the_other_one(monkeypatch) -> None:
    _with_percept(monkeypatch)
    _with_presence(monkeypatch, None)
    bind_identity_ask_cooldown_bus(_FakeCooldownBus(_FakeCooldownRedis()))
    first = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    second = asyncio.run(_build_perception_context(_cfg(perception_enabled=True), _diag()))
    assert first.presence_identity_ask == "no_visual_confirmation"
    assert second.presence_identity_ask is None, "asked once, then quiet"


def test_unconfirmed_cooldown_is_much_longer_than_the_mismatch_one() -> None:
    """Not a knob check -- a mis-set ratio here is the difference between one
    question per sitting and nine across an evening with the lid shut."""
    cfg = settings_from_runtime(SimpleNamespace())
    assert cfg.identity_ask_cooldown_seconds == 1200
    assert cfg.identity_ask_unconfirmed_cooldown_seconds == 21600
    assert cfg.identity_ask_unconfirmed_cooldown_seconds >= 8 * cfg.identity_ask_cooldown_seconds


def test_unconfirmed_prompt_never_claims_someone_is_visible() -> None:
    """The old wording said 'the person currently in view'. On this path there
    is no one in view -- handing the model that sentence is how a clarifying
    question becomes a confabulated one."""
    brief = _brief(
        PerceptionContextV1(
            available=False,
            source="stale",
            presence_identity_ask="no_visual_confirmation",
        )
    )
    text = _build_prompt_fragment(brief, 4000).compact_text
    assert "is that you" in text.lower()
    assert "currently in view" not in text
    assert "cannot see" in text or "can't see" in text


def test_perception_disabled_never_asks(monkeypatch) -> None:
    """Perception off means Orion does not discuss its cameras at all --
    including to say it cannot see."""
    _with_presence(monkeypatch, None)
    ctx = asyncio.run(_build_perception_context(_cfg(perception_enabled=False), _diag()))
    assert ctx.presence_identity_ask is None
