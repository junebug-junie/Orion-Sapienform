from __future__ import annotations

from orion.substrate.link_accepted_seeds import (
    ACCEPTED_READING_LINK_TIMES_SQL,
    ACCEPTED_UNSEEDED_LINKS_SQL,
    READING_PROPOSAL_ACTORS,
    accepted_links_params,
    accepted_reading_link_events_params,
)
from orion.substrate.neighborhood import ACCEPTED_ASSERTION_STATES


def test_reading_event_query_shares_accepted_link_scope() -> None:
    seed = accepted_links_params(lookback_hours=168.0, limit=10)
    events = accepted_reading_link_events_params(lookback_hours=168.0)
    assert events["actors"] == seed["actors"] == list(READING_PROPOSAL_ACTORS)
    assert events["accepted_states"] == seed["accepted_states"] == list(ACCEPTED_ASSERTION_STATES)
    assert "event_kind = 'materialization'" in ACCEPTED_READING_LINK_TIMES_SQL
    assert "outcome = 'applied'" in ACCEPTED_READING_LINK_TIMES_SQL
    assert "m.payload->'edge_ids'" in ACCEPTED_READING_LINK_TIMES_SQL
    assert "m.recorded_at <= now()" in ACCEPTED_READING_LINK_TIMES_SQL
    assert "p.actor = ANY(:actors)" in ACCEPTED_READING_LINK_TIMES_SQL
    assert "l.state = ANY(:accepted_states)" in ACCEPTED_READING_LINK_TIMES_SQL


def test_reading_event_query_does_not_use_seed_idempotency_filter() -> None:
    assert "NOT EXISTS" in ACCEPTED_UNSEEDED_LINKS_SQL
    assert "NOT EXISTS" not in ACCEPTED_READING_LINK_TIMES_SQL
