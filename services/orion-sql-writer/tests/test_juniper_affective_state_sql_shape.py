"""Contract tests for persisting juniper_affective_state events.

This is the first real consumer of ``orion:substrate:juniper_affective_state``.
The channel was ``consumer_services: []`` from 2026-08-11, and because
``OrionBusAsync.publish()`` is Redis pub/sub rather than a stream, that meant
every event was dropped on publish -- confirmed live, ``PUBSUB NUMSUB``
returned 0. Three days of real readings exist nowhere. These tests pin the
wiring that stops that, and the column shape those samples get read back
through.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SERVICE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.models import JuniperAffectiveStateSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402

from orion.schemas.affective_state import JuniperAffectiveStateV1  # noqa: E402

_SINCE = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
_UNTIL = _SINCE + timedelta(seconds=900)


def _event(**overrides) -> JuniperAffectiveStateV1:
    payload = dict(
        observed_at=_UNTIL,
        window_since=_SINCE,
        window_until=_UNTIL,
        message_count=12,
        word_count=1840,
        swear_count=3,
        swear_frequency=3 / 1840,
    )
    payload.update(overrides)
    return JuniperAffectiveStateV1(**payload)


# --------------------------------------------------------------------------
# Routing / subscription wiring
# --------------------------------------------------------------------------


def test_message_kind_routes_to_the_sql_model() -> None:
    assert DEFAULT_ROUTE_MAP["juniper.affective_state.v1"] == "JuniperAffectiveStateSQL"


def test_route_map_survives_an_env_override_of_other_routes() -> None:
    """`route_map` merges the env JSON over the code defaults, so a live .env
    carrying an older, shorter override still picks this route up. This is the
    forgiving half of the config -- and it is why the live .env's route map was
    left alone while its subscribe list had to be hand-edited."""
    settings = Settings(SQL_WRITER_ROUTE_MAP_JSON='{"some.other.kind.v1":"SomethingElseSQL"}')
    assert settings.route_map["juniper.affective_state.v1"] == "JuniperAffectiveStateSQL"
    assert settings.route_map["some.other.kind.v1"] == "SomethingElseSQL"


def test_channel_is_in_the_default_subscribe_list() -> None:
    """The unforgiving half: `effective_subscribe_channels` REPLACES the code
    default with whatever the env sets -- it does not merge. A live .env that
    predates this patch therefore leaves the writer permanently unsubscribed
    and the rows silently never appear. Confirmed live 2026-08-12 with
    dev_economics, which is why the live .env is edited by hand in the same
    patch.

    Asserts on the declared FIELD DEFAULT, not on `Settings()`. Instantiating
    Settings reads a cwd-relative .env which already contains this channel, so
    an instance-based assertion passes even with the code default deleted --
    precisely the regression this test is named for."""
    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    assert "orion:substrate:juniper_affective_state" in default_channels


def test_env_example_lists_the_channel() -> None:
    """`.env_example` is the operator contract and what
    scripts/sync_local_env_from_example.py propagates into a real .env."""
    subscribe_line = next(
        line
        for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert "orion:substrate:juniper_affective_state" in subscribe_line


# --------------------------------------------------------------------------
# Column shape
# --------------------------------------------------------------------------


def test_every_payload_field_has_a_column_or_is_deliberately_dropped() -> None:
    columns = set(JuniperAffectiveStateSQL.__table__.columns.keys())
    payload_fields = set(JuniperAffectiveStateV1.model_fields.keys())
    deliberately_dropped = {"schema_version"}  # constant discriminator, not data
    assert payload_fields - deliberately_dropped <= columns


def test_swear_frequency_is_nullable_because_empty_is_not_calm() -> None:
    """NULL means word_count == 0 -- Juniper typed nothing in this window.
    That is a different fact from a calm window (messages existed, none were
    swears), and 78% of real 15-minute windows are the former. A NOT NULL
    column with a 0.0 default would merge the two and drag every downstream
    average toward zero."""
    assert JuniperAffectiveStateSQL.__table__.columns["swear_frequency"].nullable is True


def test_raw_counts_are_persisted_not_only_the_rate() -> None:
    """The whole reason this table is worth having.

    swear_frequency is computed over a fixed 15-minute window, but real typing
    is bursty: only ~6.7% of windows carry a non-zero rate, and a short window
    turns one swear into a spike (a real window reads 0.077 off 13 words).
    Storing the counts lets any consumer re-derive the rate over an hour, a
    day, or a session by SUMMING -- which is how aggregate_scores() weights
    correctly across messages of very different lengths. A stored ratio cannot
    be re-aggregated without reintroducing the short-message bias, so dropping
    these columns would permanently bake in the wrong window."""
    columns = JuniperAffectiveStateSQL.__table__.columns
    for name in ("swear_count", "word_count", "message_count"):
        assert name in columns, name
        assert columns[name].nullable is False, name


def test_window_bounds_are_indexed_for_reaggregation() -> None:
    """A range scan over window bounds is the first query anyone
    re-aggregating this will run."""
    columns = JuniperAffectiveStateSQL.__table__.columns
    assert columns["window_since"].index is True
    assert columns["window_until"].index is True


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


def test_event_id_is_derived_from_both_window_bounds() -> None:
    assert _event().event_id == (
        f"juniper-affective-state:{_SINCE.isoformat()}:{_UNTIL.isoformat()}"
    )


def test_republishing_the_same_window_reuses_the_same_key() -> None:
    """Derived rather than uuid4 so a redelivered event upserts onto the same
    row. Duplicates would inflate the sample count and corrupt the very
    distribution this table is kept for."""
    assert _event(swear_count=1).event_id == _event(swear_count=9).event_id


def test_distinct_windows_get_distinct_keys() -> None:
    later = _UNTIL + timedelta(seconds=900)
    assert _event().event_id != _event(window_since=_UNTIL, window_until=later).event_id


def test_restart_overlap_window_does_not_collapse_onto_the_same_row() -> None:
    """Why the key uses BOTH bounds, not window_since alone.

    Windows tile contiguously tick-to-tick, so window_since looks unique --
    until a container restart, which seeds `last_until = now -
    cold_start_lookback_sec` and produces a window overlapping ones already
    published. Those are real observations over different spans; keying on
    window_since alone would silently overwrite the shorter with the longer
    and lose a reading."""
    normal = _event()
    after_restart = _event(window_until=_UNTIL + timedelta(seconds=3600))
    assert normal.window_since == after_restart.window_since
    assert normal.event_id != after_restart.event_id


def test_explicit_event_id_is_not_overwritten() -> None:
    """The validator fills a blank; clobbering a real wire value would rewrite
    a replayed historical event under a new key and duplicate the row it was
    meant to update."""
    assert _event(event_id="juniper-affective-state:legacy").event_id == (
        "juniper-affective-state:legacy"
    )


def test_event_id_is_the_primary_key() -> None:
    pk = [c.name for c in JuniperAffectiveStateSQL.__table__.primary_key.columns]
    assert pk == ["event_id"]


# --------------------------------------------------------------------------
# Summability (the reason this table exists)
# --------------------------------------------------------------------------


def test_cold_start_is_persisted_and_indexed() -> None:
    """`WHERE cold_start IS FALSE` is the qualifier on every correct aggregate
    over this table, so it must be both stored and indexed.

    Without it the table's whole justification is false: a consumer summing
    counts to re-derive the rate over an hour silently double-counts. Caught
    by review on the first two rows this table ever held -- a 913s window
    entirely inside a 3600s one, SUM returning 7669 words against a true
    5913 (+34.7%)."""
    columns = JuniperAffectiveStateSQL.__table__.columns
    assert "cold_start" in columns
    assert columns["cold_start"].nullable is False
    assert columns["cold_start"].index is True


def test_cold_start_defaults_false_so_old_events_are_not_misread() -> None:
    """An event published before this field existed carries no `cold_start`.
    Defaulting True would mark every historical row unsummable; defaulting
    False treats it as an ordinary tiling window, which is what those events
    were -- the overlap only ever comes from a restart tick."""
    assert _event().cold_start is False


def test_a_restart_window_is_flagged_and_a_normal_one_is_not() -> None:
    """The real shape, taken from the live rows that exposed this.

    Both windows are legitimate observations and both are kept -- the
    cold-start row is the only record of a span crossing a restart. The flag
    is what lets a consumer sum the tiling rows and use the cold-start row
    only to fill a genuine downtime gap."""
    normal = _event()
    restart = _event(
        window_since=_SINCE - timedelta(seconds=2700),
        window_until=_UNTIL,
        cold_start=True,
    )
    assert normal.cold_start is False
    assert restart.cold_start is True
    # ...and the overlap the flag exists to warn about is real: the normal
    # window sits entirely inside the restart window.
    assert restart.window_since < normal.window_since
    assert restart.window_until >= normal.window_until
    # They stay distinct rows -- flagging is not deduplication.
    assert normal.event_id != restart.event_id
