from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import drive_history_reflection_synthesis as dhrs  # noqa: E402


# ===========================================================================
# Reducer -- pure, synthetic-input/known-output (mirrors
# orion/spark/concept_induction/tests/test_drive_tension.py's bar).
# ===========================================================================


def _ev(day: str, hour: int, *, dominant: str | None, artifact_id: str | None = None, active=(), pressures=None) -> dhrs.DriveAuditEvent:
    return dhrs.DriveAuditEvent(
        artifact_id=artifact_id or f"drive-audit-{day}-{hour}",
        created_at=datetime.fromisoformat(f"{day}T{hour:02d}:00:00+00:00"),
        dominant_drive=dominant,
        summary=f"summary for {day} {hour}h",
        active_drives=tuple(active),
        drive_pressures=pressures or {},
    )


class TestReduceDriveHistory:
    def test_empty_events_is_insufficient(self):
        agg = dhrs.reduce_drive_history([], subject="orion", since=datetime.now(timezone.utc))

        assert agg.total_events == 0
        assert agg.distinct_days == 0
        assert agg.sufficient is False
        assert "0 event" in agg.insufficiency_reason
        assert agg.top_dominant_drive is None
        assert agg.cited_event_ids == ()

    def test_below_min_events_is_insufficient(self):
        events = [_ev("2026-06-14", h, dominant="continuity") for h in range(3)]

        agg = dhrs.reduce_drive_history(events, subject="orion", since=datetime.now(timezone.utc))

        assert agg.total_events == 3
        assert agg.sufficient is False
        assert "need >= 5" in agg.insufficiency_reason

    def test_below_min_distinct_days_is_insufficient_even_with_many_events(self):
        # 10 ticks, all in one session/day -- event COUNT alone is satisfied,
        # but this must still refuse: a burst within one sitting is not a
        # "long-horizon pattern".
        events = [_ev("2026-06-14", h, dominant="continuity") for h in range(10)]

        agg = dhrs.reduce_drive_history(events, subject="orion", since=datetime.now(timezone.utc))

        assert agg.total_events == 10
        assert agg.distinct_days == 1
        assert agg.sufficient is False
        assert "distinct day" in agg.insufficiency_reason

    def test_sufficient_history_known_counts_and_shares(self):
        events = [
            _ev("2026-06-14", 9, dominant="continuity"),
            _ev("2026-06-14", 10, dominant="continuity"),
            _ev("2026-06-14", 11, dominant="autonomy"),
            _ev("2026-06-15", 9, dominant="continuity"),
            _ev("2026-06-15", 10, dominant="continuity"),
            _ev("2026-06-15", 11, dominant="continuity"),
        ]
        since = datetime(2026, 6, 14, tzinfo=timezone.utc)

        agg = dhrs.reduce_drive_history(events, subject="orion", since=since)

        assert agg.total_events == 6
        assert agg.distinct_days == 2
        assert agg.sufficient is True
        assert agg.insufficiency_reason is None
        assert agg.dominant_drive_counts == {"continuity": 5, "autonomy": 1}
        assert agg.top_dominant_drive == "continuity"
        assert agg.top_dominant_drive_count == 5
        assert abs(agg.top_dominant_drive_share - (5 / 6)) < 1e-9
        assert agg.day_counts == {"2026-06-14": 3, "2026-06-15": 3}
        assert agg.day_dominant_drive == {"2026-06-14": "continuity", "2026-06-15": "continuity"}
        assert agg.until == events[-1].created_at

    def test_tie_break_is_deterministic_alphabetical(self):
        events = [
            _ev("2026-06-14", 9, dominant="continuity"),
            _ev("2026-06-15", 9, dominant="autonomy"),
        ]
        events = events * 3  # can't dup artifact_id, build distinct instead below
        events = [
            _ev("2026-06-14", 9, dominant="continuity", artifact_id="a1"),
            _ev("2026-06-15", 9, dominant="autonomy", artifact_id="a2"),
        ]

        agg = dhrs.reduce_drive_history(
            events, subject="orion", since=datetime.now(timezone.utc), min_events=2, min_distinct_days=2
        )

        # Tied 1-1 -- alphabetically first ("autonomy" < "continuity") wins deterministically.
        assert agg.top_dominant_drive == "autonomy"

    def test_cited_event_ids_include_first_and_last(self):
        events = [
            _ev("2026-06-14", h, dominant="continuity", artifact_id=f"a{h}") for h in range(8)
        ] + [_ev("2026-06-15", 0, dominant="continuity", artifact_id="a_last")]

        agg = dhrs.reduce_drive_history(events, subject="orion", since=datetime.now(timezone.utc), max_cited_events=4)

        assert agg.cited_event_ids[0] == "a0"
        assert agg.cited_event_ids[-1] == "a_last"
        assert len(agg.cited_event_ids) <= 4

    def test_active_drive_frequency_and_mean_pressure(self):
        events = [
            _ev("2026-06-14", 9, dominant="continuity", active=("continuity", "autonomy"), pressures={"continuity": 0.8, "autonomy": 0.4}),
            _ev("2026-06-15", 9, dominant="continuity", active=("continuity",), pressures={"continuity": 0.6}),
        ]

        agg = dhrs.reduce_drive_history(events, subject="orion", since=datetime.now(timezone.utc), min_events=2, min_distinct_days=2)

        assert agg.active_drive_frequency == {"continuity": 2, "autonomy": 1}
        assert abs(agg.mean_pressure_by_drive["continuity"] - 0.7) < 1e-9
        assert abs(agg.mean_pressure_by_drive["autonomy"] - 0.4) < 1e-9

    def test_events_with_no_dominant_drive_are_not_fabricated(self):
        events = [
            _ev("2026-06-14", 9, dominant=None, artifact_id="a1"),
            _ev("2026-06-15", 9, dominant=None, artifact_id="a2"),
        ]

        agg = dhrs.reduce_drive_history(events, subject="orion", since=datetime.now(timezone.utc), min_events=2, min_distinct_days=2)

        assert agg.dominant_drive_counts == {}
        assert agg.top_dominant_drive is None
        assert agg.top_dominant_drive_share == 0.0


# ===========================================================================
# Fact sheet -- deterministic rendering of an aggregation.
# ===========================================================================


class TestBuildFactSheet:
    def test_fact_sheet_contains_real_tokens_only(self):
        events = [
            _ev("2026-06-14", 9, dominant="continuity", artifact_id="a1"),
            _ev("2026-06-14", 10, dominant="continuity", artifact_id="a2"),
            _ev("2026-06-15", 9, dominant="continuity", artifact_id="a3"),
        ]
        since = datetime(2026, 6, 14, tzinfo=timezone.utc)
        agg = dhrs.reduce_drive_history(events, subject="orion", since=since, min_events=3, min_distinct_days=2)
        events_by_id = {e.artifact_id: e for e in events}

        facts = dhrs.build_fact_sheet(agg, events_by_id)

        assert facts[0].index == 1
        assert "continuity" in facts[0].text
        assert "3" in facts[0].tokens  # top_dominant_drive_count
        # day facts follow, sorted ascending
        day_facts = [f for f in facts if f.text.startswith("On ")]
        assert [f.tokens[0] for f in day_facts] == ["2026-06-14", "2026-06-15"]
        # event facts carry a real artifact_id back-reference
        event_facts = [f for f in facts if f.artifact_id]
        assert {f.artifact_id for f in event_facts} <= {"a1", "a2", "a3"}

    def test_event_fact_tokens_are_naturally_reproducible(self):
        # Individual-event fact tokens must be a naturalized (no seconds/tz-
        # offset) timestamp + dominant drive -- not a full ISO-8601 string or
        # the opaque hashed artifact_id, which a short reflective-voice
        # sentence is very unlikely to ever reproduce verbatim.
        events = [
            _ev("2026-06-14", 9, dominant="continuity", artifact_id="a1"),
            _ev("2026-06-15", 9, dominant="continuity", artifact_id="a2"),
        ]
        since = datetime(2026, 6, 14, tzinfo=timezone.utc)
        agg = dhrs.reduce_drive_history(events, subject="orion", since=since, min_events=2, min_distinct_days=2)
        events_by_id = {e.artifact_id: e for e in events}

        facts = dhrs.build_fact_sheet(agg, events_by_id)
        event_facts = [f for f in facts if f.artifact_id]

        assert event_facts
        for f in event_facts:
            assert len(f.tokens) == 2
            ts_token, drive_token = f.tokens
            assert "T" not in ts_token  # not a raw ISO-8601 timestamp
            assert "+00:00" not in ts_token
            assert f.artifact_id not in f.tokens  # opaque hash is not a grounding token
            assert drive_token == "continuity"

    def test_day_facts_are_capped_independent_of_window_size(self):
        # A wide --since-days window with daily activity must not inflate the
        # fact sheet/prompt without bound -- day-facts are capped at
        # MAX_DAY_FACTS regardless of how many distinct days are present.
        events = [
            _ev(f"2026-0{1 + (d // 28)}-{1 + (d % 28):02d}", 9, dominant="continuity", artifact_id=f"a{d}")
            for d in range(20)
        ]
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        agg = dhrs.reduce_drive_history(events, subject="orion", since=since, min_events=5, min_distinct_days=2)
        assert agg.distinct_days == 20
        events_by_id = {e.artifact_id: e for e in events}

        facts = dhrs.build_fact_sheet(agg, events_by_id)

        day_facts = [f for f in facts if f.text.startswith("On ")]
        assert len(day_facts) == dhrs.MAX_DAY_FACTS
        # keeps the MOST RECENT days, not the earliest
        assert day_facts[-1].tokens[0] == sorted(agg.day_dominant_drive)[-1]


class TestIsUndefinedTableError:
    # Local copy of measure_autonomy_gate.py's helper -- same contract.
    def test_pgcode_42p01_is_undefined_table(self):
        exc = RuntimeError("boom")
        exc.pgcode = "42P01"
        assert dhrs.is_undefined_table_error(exc) is True

    def test_class_name_undefined_table_matches(self):
        exc = type("UndefinedTable", (Exception,), {})("relation missing")
        assert dhrs.is_undefined_table_error(exc) is True

    def test_message_sniff_fallback(self):
        assert dhrs.is_undefined_table_error(RuntimeError('relation "drive_audits" does not exist')) is True

    def test_other_errors_are_not_undefined_table(self):
        assert dhrs.is_undefined_table_error(RuntimeError("connection refused")) is False


# ===========================================================================
# Narrative validation -- the mandatory anti-hallucination guardrail.
# ===========================================================================


def _facts_fixture() -> list[dhrs.GroundedFact]:
    return [
        dhrs.GroundedFact(index=1, text="'continuity' dominant 5/6 (83%)", tokens=("continuity", "83%", "5")),
        dhrs.GroundedFact(index=2, text="On 2026-06-14 continuity", tokens=("2026-06-14", "continuity")),
        dhrs.GroundedFact(
            index=3,
            text="Audit a1 at 2026-06-14 09:00 UTC: dominant_drive=continuity",
            tokens=("2026-06-14 09:00", "continuity"),
            artifact_id="a1",
        ),
    ]


class TestParseAndValidateNarrative:
    def test_valid_grounded_narrative_is_accepted(self):
        facts = _facts_fixture()
        content = json.dumps(
            {
                # ALL tokens of fact 1 (continuity, 83%, 5) and fact 3
                # (2026-06-14 09:00, continuity) appear verbatim.
                "narrative": (
                    "Since 2026-06-14 09:00, continuity has stood out, appearing in 5 of this "
                    "window's audited ticks (83%)."
                ),
                "cited_fact_numbers": [1, 3],
            }
        )

        result = dhrs.parse_and_validate_narrative(content, facts)

        assert result.cited_fact_numbers == (1, 3)
        assert "continuity" in result.narrative

    def test_case_insensitive_token_match_is_accepted(self):
        # An LLM naturally capitalizing a drive name at a sentence start must
        # not be falsely rejected over case alone.
        facts = _facts_fixture()
        content = json.dumps(
            {
                "narrative": (
                    "Continuity dominated since 2026-06-14 09:00, present in 5 of this window's ticks (83%)."
                ),
                "cited_fact_numbers": [1, 3],
            }
        )

        result = dhrs.parse_and_validate_narrative(content, facts)

        assert result.cited_fact_numbers == (1, 3)

    def test_partial_token_match_is_rejected(self):
        # Reproducing only the drive name from fact 1 (not its percentage or
        # count) must NOT count as a grounded citation -- this is the core
        # hallucination defense: matching one low-entropy token is not proof
        # the LLM actually reproduced the fact's specific numbers.
        facts = _facts_fixture()
        content = json.dumps(
            {
                "narrative": (
                    "continuity dominated recently, at 2026-06-14 09:00 in particular, roughly most of the time."
                ),
                "cited_fact_numbers": [1, 3],
            }
        )
        with pytest.raises(dhrs.NarrativeUngrounded):
            dhrs.parse_and_validate_narrative(content, facts)

    def test_aggregate_only_citations_without_individual_event_are_rejected(self):
        # Citing only fact 1 (top-dominant, aggregate) and fact 2 (a day-level
        # aggregate) -- neither carries an artifact_id -- must be rejected:
        # at least one citation must be a real per-tick DriveAuditV1 artifact.
        facts = _facts_fixture()
        content = json.dumps(
            {
                "narrative": "continuity dominated 5 of these ticks (83%) on 2026-06-14 in particular.",
                "cited_fact_numbers": [1, 2],
            }
        )
        with pytest.raises(dhrs.NarrativeUngrounded):
            dhrs.parse_and_validate_narrative(content, facts)

    def test_malformed_json_is_rejected(self):
        with pytest.raises(dhrs.NarrativeUngrounded):
            dhrs.parse_and_validate_narrative("not json at all", _facts_fixture())

    def test_empty_narrative_is_rejected(self):
        content = json.dumps({"narrative": "  ", "cited_fact_numbers": [1, 3]})
        with pytest.raises(dhrs.NarrativeUngrounded):
            dhrs.parse_and_validate_narrative(content, _facts_fixture())

    def test_missing_citations_is_rejected(self):
        content = json.dumps({"narrative": "Something vague happened.", "cited_fact_numbers": []})
        with pytest.raises(dhrs.NarrativeUngrounded):
            dhrs.parse_and_validate_narrative(content, _facts_fixture())

    def test_too_few_valid_citations_is_rejected(self):
        # Only fact 1 is a real index; fact 99 does not exist and is dropped,
        # leaving 1 valid citation -- below MIN_CITED_FACTS=2.
        content = json.dumps(
            {"narrative": "continuity dominant recently.", "cited_fact_numbers": [1, 99]}
        )
        with pytest.raises(dhrs.NarrativeUngrounded):
            dhrs.parse_and_validate_narrative(content, _facts_fixture())

    def test_citation_without_real_token_in_text_is_rejected(self):
        # Cites facts 1 and 3 but never actually quotes any of their real
        # tokens in the narrative -- this is the core hallucination defense.
        content = json.dumps(
            {
                "narrative": "Orion seems to enjoy exploring new ideas lately, generally speaking.",
                "cited_fact_numbers": [1, 3],
            }
        )
        with pytest.raises(dhrs.NarrativeUngrounded):
            dhrs.parse_and_validate_narrative(content, _facts_fixture())


# ===========================================================================
# Row -> event conversion -- pure (the I/O half of the fetch, psycopg2, stays
# untested; everything convertible is exercised here without a live DB).
# Row column order follows _DRIVE_AUDIT_EVENTS_SQL:
#   (artifact_id, created_at, dominant_drive, summary, drive_pressures, active_drives)
# ===========================================================================


def _row(
    artifact_id="a1",
    created_at=None,
    dominant="continuity",
    summary="continuity pressure concentrates",
    pressures=None,
    actives=None,
):
    return (
        artifact_id,
        created_at if created_at is not None else datetime(2026, 6, 14, 9, tzinfo=timezone.utc),
        dominant,
        summary,
        pressures if pressures is not None else {"continuity": 0.7},
        actives if actives is not None else ["continuity"],
    )


class TestDriveAuditRowToEvent:
    def test_happy_row_is_fully_populated(self):
        ev = dhrs.drive_audit_row_to_event(_row())

        assert ev is not None
        assert ev.artifact_id == "a1"
        assert ev.created_at == datetime(2026, 6, 14, 9, tzinfo=timezone.utc)
        assert ev.dominant_drive == "continuity"
        assert ev.summary == "continuity pressure concentrates"
        assert ev.drive_pressures == {"continuity": 0.7}
        assert ev.active_drives == ("continuity",)

    def test_none_summary_and_dominant_are_kept_as_none(self):
        ev = dhrs.drive_audit_row_to_event(_row(dominant=None, summary=None))

        assert ev is not None
        assert ev.dominant_drive is None
        assert ev.summary is None

    def test_missing_artifact_id_skips_row(self):
        assert dhrs.drive_audit_row_to_event(_row(artifact_id=None)) is None
        assert dhrs.drive_audit_row_to_event(_row(artifact_id="   ")) is None

    def test_missing_or_garbage_created_at_skips_row(self):
        assert dhrs.drive_audit_row_to_event(_row(created_at="not-a-timestamp")) is None
        row = list(_row())
        row[1] = None
        assert dhrs.drive_audit_row_to_event(tuple(row)) is None

    def test_string_created_at_is_parsed_and_naive_datetime_coerced_utc(self):
        ev = dhrs.drive_audit_row_to_event(_row(created_at="2026-06-14T09:00:00+00:00"))
        assert ev is not None
        assert ev.created_at == datetime(2026, 6, 14, 9, tzinfo=timezone.utc)

        ev_naive = dhrs.drive_audit_row_to_event(_row(created_at=datetime(2026, 6, 14, 9)))
        assert ev_naive is not None
        assert ev_naive.created_at.tzinfo is not None
        assert ev_naive.created_at == datetime(2026, 6, 14, 9, tzinfo=timezone.utc)

    def test_malformed_jsonb_degrades_to_empty_not_raise(self):
        # Wrong-typed JSONB columns (a bare string / int instead of dict /
        # list) degrade to {} / () -- the tick itself is still real and kept.
        ev = dhrs.drive_audit_row_to_event(_row(pressures="garbage", actives=42))

        assert ev is not None
        assert ev.drive_pressures == {}
        assert ev.active_drives == ()

    def test_jsonb_returned_as_text_is_parsed_not_silently_emptied(self):
        # If a driver ever hands JSONB back as text (psycopg2 normally returns
        # dict/list), the value is parsed rather than silently emptied --
        # review finding: silent {} here would report success while zeroing
        # the full-window statistics.
        ev = dhrs.drive_audit_row_to_event(
            _row(pressures='{"continuity": 0.7}', actives='["continuity", "novelty"]')
        )

        assert ev is not None
        assert ev.drive_pressures == {"continuity": 0.7}
        assert ev.active_drives == ("continuity", "novelty")

    def test_negative_and_garbage_pressure_entries_are_skipped(self):
        ev = dhrs.drive_audit_row_to_event(
            _row(
                pressures={
                    "continuity": 0.7,
                    "autonomy": -0.2,  # negative: not a real pressure
                    "novelty": "high",  # non-numeric garbage
                    "social": None,
                    "": 0.5,  # empty key
                    "curiosity": float("nan"),  # non-finite
                }
            )
        )

        assert ev is not None
        assert ev.drive_pressures == {"continuity": 0.7}

    def test_non_string_active_drive_entries_are_skipped_and_deduped(self):
        ev = dhrs.drive_audit_row_to_event(_row(actives=["continuity", 3, None, "", "continuity", "autonomy"]))

        assert ev is not None
        assert ev.active_drives == ("continuity", "autonomy")

    def test_wrong_shaped_row_is_skipped(self):
        assert dhrs.drive_audit_row_to_event(None) is None
        assert dhrs.drive_audit_row_to_event("not a row") is None
        assert dhrs.drive_audit_row_to_event(("a1", datetime.now(timezone.utc))) is None  # wrong length


class TestParseDriveAuditRows:
    def test_bad_rows_are_skipped_good_rows_kept(self):
        rows = [
            _row(artifact_id="a1"),
            _row(artifact_id=None),  # skipped: no identity
            _row(artifact_id="a2", pressures="garbage"),  # kept, pressures degrade
        ]

        events = dhrs.parse_drive_audit_rows(rows)

        assert [e.artifact_id for e in events] == ["a1", "a2"]
        assert events[1].drive_pressures == {}

    def test_populated_jsonb_flows_into_reducer_aggregations(self):
        # The old graph fetch left drive_pressures/active_drives empty (a
        # disclosed SPARQL-transfer-cost scope reduction), so
        # mean_pressure_by_drive/active_drive_frequency reflected nothing.
        # Postgres rows arrive fully populated -- prove the full path
        # row -> event -> reduce now yields real full-window statistics.
        rows = [
            _row(
                artifact_id="a1",
                created_at=datetime(2026, 6, 14, 9, tzinfo=timezone.utc),
                pressures={"continuity": 0.8, "autonomy": 0.4},
                actives=["continuity", "autonomy"],
            ),
            _row(
                artifact_id="a2",
                created_at=datetime(2026, 6, 15, 9, tzinfo=timezone.utc),
                pressures={"continuity": 0.6},
                actives=["continuity"],
            ),
        ]

        events = dhrs.parse_drive_audit_rows(rows)
        agg = dhrs.reduce_drive_history(
            events, subject="orion", since=datetime(2026, 6, 14, tzinfo=timezone.utc), min_events=2, min_distinct_days=2
        )

        assert agg.sufficient is True
        assert abs(agg.mean_pressure_by_drive["continuity"] - 0.7) < 1e-9
        assert abs(agg.mean_pressure_by_drive["autonomy"] - 0.4) < 1e-9
        assert agg.active_drive_frequency == {"continuity": 2, "autonomy": 1}


# ===========================================================================
# Fetch layer degrade posture -- psycopg2 mocked at its boundary. The fetch
# must NEVER raise and NEVER fall back to another source: missing table /
# no connection / query failure all degrade to an empty event list.
# ===========================================================================


class TestFetchDriveHistoryEventsDegrades:
    def _since(self):
        return datetime(2026, 6, 14, tzinfo=timezone.utc)

    def test_happy_path_parses_rows_and_passes_params(self):
        psycopg2 = pytest.importorskip("psycopg2")
        cur = MagicMock()
        cur.fetchall.return_value = [_row()]
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur

        with patch.object(psycopg2, "connect", return_value=conn) as connect_mock:
            events = dhrs.fetch_drive_history_events(
                "postgresql://fake/db", since=self._since(), subject="orion", max_events=50
            )

        assert len(events) == 1
        assert events[0].artifact_id == "a1"
        assert events[0].drive_pressures == {"continuity": 0.7}
        connect_mock.assert_called_once_with("postgresql://fake/db")
        sql, params = cur.execute.call_args[0]
        assert "drive_audits" in sql
        assert "COALESCE(observed_at, created_at)" in sql
        assert params == ("orion", self._since(), 50)
        conn.close.assert_called_once()

    def test_connection_failure_returns_empty_never_raises(self):
        psycopg2 = pytest.importorskip("psycopg2")
        with patch.object(psycopg2, "connect", side_effect=RuntimeError("db down")):
            events = dhrs.fetch_drive_history_events("postgresql://fake/db", since=self._since(), subject="orion")

        assert events == []

    def test_missing_table_returns_empty_never_raises(self):
        psycopg2 = pytest.importorskip("psycopg2")
        exc = RuntimeError('relation "drive_audits" does not exist')
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.execute.side_effect = exc

        with patch.object(psycopg2, "connect", return_value=conn):
            events = dhrs.fetch_drive_history_events("postgresql://fake/db", since=self._since(), subject="orion")

        assert events == []
        conn.close.assert_called_once()

    def test_query_failure_returns_empty_never_raises(self):
        psycopg2 = pytest.importorskip("psycopg2")
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError("timeout")

        with patch.object(psycopg2, "connect", return_value=conn):
            events = dhrs.fetch_drive_history_events("postgresql://fake/db", since=self._since(), subject="orion")

        assert events == []
        conn.close.assert_called_once()


# ===========================================================================
# End-to-end orchestration (_run_once) -- event fetch + bus mocked at their
# boundaries; Postgres write path mocked with a FakeConn double, same style
# as tests/test_concept_relation_digest.py.
# ===========================================================================


class FakeFetcher:
    """Stands in for fetch_drive_history_events (same call signature)."""

    def __init__(self, events):
        self.events = list(events)
        self.calls: list[dict] = []

    def __call__(self, postgres_uri, *, since, subject, max_events=dhrs.DEFAULT_MAX_EVENTS):
        self.calls.append(
            {"postgres_uri": postgres_uri, "since": since, "subject": subject, "max_events": max_events}
        )
        return list(self.events)


class _NullAsyncCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class FakeConn:
    """Minimal in-process double for the asyncpg.Connection calls
    repository.py::insert_crystallization() makes -- mirrors
    tests/test_concept_relation_digest.py's FakeConn exactly."""

    def __init__(self, existing_sources: list[dict] | None = None) -> None:
        self.crystallizations: dict[str, dict] = {}
        self.sources: list[dict] = list(existing_sources or [])
        self.closed = False

    def transaction(self):
        return _NullAsyncCtx()

    async def close(self) -> None:
        self.closed = True

    async def fetchrow(self, sql: str, *args):
        if "INSERT INTO memory_crystallizations" in sql:
            crystallization_id = args[0] if args and args[0] else str(uuid4())
            self.crystallizations[crystallization_id] = {"args": args}
            return {"crystallization_id": crystallization_id}
        raise AssertionError(f"FakeConn.fetchrow: unexpected query: {sql}")

    async def fetchval(self, sql: str, *args):
        if "SELECT crystallization_id" in sql and "memory_crystallization_sources" in sql:
            source_kind, source_id = args
            for row in self.sources:
                row_args = row["args"]
                if row_args[1] == source_kind and row_args[2] == source_id:
                    return row_args[0]
            return None
        raise AssertionError(f"FakeConn.fetchval: unexpected query: {sql}")

    async def execute(self, sql: str, *args):
        if "INSERT INTO memory_crystallization_sources" in sql:
            self.sources.append({"args": args})
            return "INSERT 0 1"
        raise AssertionError(f"FakeConn.execute: unexpected query: {sql}")


def _sufficient_history_events(*, with_detail: bool = False) -> list[dhrs.DriveAuditEvent]:
    # with_detail=True populates JSONB-derived pressures/active-drives on the
    # earliest event (a1, always in the cited sample) -- what a real Postgres
    # fetch returns for every row.
    a1_kwargs = (
        {"active": ("continuity",), "pressures": {"continuity": 0.71}} if with_detail else {}
    )
    return [
        _ev("2026-06-14", 9, dominant="continuity", artifact_id="a1", **a1_kwargs),
        _ev("2026-06-14", 10, dominant="continuity", artifact_id="a2"),
        _ev("2026-06-14", 11, dominant="autonomy", artifact_id="a3"),
        _ev("2026-06-15", 9, dominant="continuity", artifact_id="a4"),
        _ev("2026-06-15", 10, dominant="continuity", artifact_id="a5"),
        _ev("2026-06-15", 11, dominant="continuity", artifact_id="a6"),
    ]


def _grounded_llm_content() -> str:
    # Fact 1 is the top-dominant-drive fact: continuity dominant in 5/6 ticks
    # (83%). With 2 distinct days, facts 2-3 are the per-day facts and facts
    # 4+ are individual cited events (starting with the earliest, at
    # 2026-06-14 09:00, artifact_id "a1"). ALL tokens of both cited facts
    # must appear verbatim, and at least one cited fact (4) carries a real
    # artifact_id so evidence-building covers both the aggregation and a real
    # per-tick artifact.
    return json.dumps(
        {
            "narrative": (
                "Since 2026-06-14 09:00, continuity has stood out as Orion's dominant drive, "
                "appearing in 5 of this window's audited ticks (83%)."
            ),
            "cited_fact_numbers": [1, 4],
        }
    )


def _make_bus(reply_content: str | None = None, *, raise_exc: Exception | None = None) -> MagicMock:
    bus = MagicMock()
    if raise_exc is not None:
        bus.rpc_request = AsyncMock(side_effect=raise_exc)
    else:
        bus.rpc_request = AsyncMock(return_value={"data": b"whatever"})
        decoded = MagicMock()
        decoded.ok = True
        decoded.envelope.payload = {"content": reply_content or ""}
        bus.codec.decode = MagicMock(return_value=decoded)
    return bus


@pytest.mark.asyncio
async def test_sufficient_history_creates_grounded_crystallization():
    fetcher = FakeFetcher(_sufficient_history_events())
    bus = _make_bus(_grounded_llm_content())
    conn = FakeConn()

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        result = await dhrs._run_once(
            fetch_events=fetcher,
            bus=bus,
            postgres_uri="postgresql://fake/db",
            subject="orion",
            since=datetime(2026, 6, 14, tzinfo=timezone.utc),
        )

    assert result.status == "created"
    assert result.crystallization_id is not None
    assert result.narrative is not None
    assert "continuity" in result.narrative
    assert len(conn.crystallizations) == 1
    stored_args = next(iter(conn.crystallizations.values()))["args"]
    assert stored_args[1] == "reflection"
    assert stored_args[3] == result.narrative  # summary column
    # Evidence includes the aggregation ref plus real cited raw events.
    assert len(conn.sources) >= 2
    source_kinds = {row["args"][1] for row in conn.sources}
    assert source_kinds == {"rdf_memory_graph"}
    bus.rpc_request.assert_awaited_once()
    # The fetch got the SAME DSN the write path uses -- one DSN concept.
    assert fetcher.calls == [
        {
            "postgres_uri": "postgresql://fake/db",
            "since": datetime(2026, 6, 14, tzinfo=timezone.utc),
            "subject": "orion",
            "max_events": dhrs.DEFAULT_MAX_EVENTS,
        }
    ]


@pytest.mark.asyncio
async def test_fetched_jsonb_detail_populates_aggregation_and_evidence_note():
    # Events now arrive from Postgres with drive_pressures/active_drives
    # already populated (no separate enrichment pass) -- the single reduce
    # must surface them in mean_pressure_by_drive/active_drive_frequency,
    # and the per-event evidence note must carry them too.
    fetcher = FakeFetcher(_sufficient_history_events(with_detail=True))
    bus = _make_bus(_grounded_llm_content())
    conn = FakeConn()

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        result = await dhrs._run_once(
            fetch_events=fetcher,
            bus=bus,
            postgres_uri="postgresql://fake/db",
            subject="orion",
            since=datetime(2026, 6, 14, tzinfo=timezone.utc),
        )

    assert result.status == "created"
    assert result.aggregation.mean_pressure_by_drive == {"continuity": 0.71}
    assert result.aggregation.active_drive_frequency == {"continuity": 1}
    assert result.aggregation.total_events == 6
    assert result.aggregation.distinct_days == 2

    raw_event_source = next(row for row in conn.sources if row["args"][2] == "a1")
    note = raw_event_source["args"][5]
    assert "pressures=" in note
    assert "0.71" in note
    assert "active_drives=" in note


@pytest.mark.asyncio
async def test_insufficient_history_refuses_cleanly_no_llm_no_write():
    # Only 2 events, both the same day -- fails both thresholds.
    fetcher = FakeFetcher(
        [
            _ev("2026-06-14", 9, dominant="continuity", artifact_id="a1"),
            _ev("2026-06-14", 10, dominant="continuity", artifact_id="a2"),
        ]
    )
    bus = _make_bus("")

    with patch("asyncpg.connect", new=AsyncMock()) as connect_mock:
        result = await dhrs._run_once(
            fetch_events=fetcher,
            bus=bus,
            postgres_uri="postgresql://fake/db",
            subject="orion",
            since=datetime(2026, 6, 14, tzinfo=timezone.utc),
        )

    assert result.status == "insufficient_history"
    assert result.error is not None
    assert result.crystallization_id is None
    bus.rpc_request.assert_not_called()
    connect_mock.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_degrade_to_empty_reports_insufficient_history_no_llm_no_write():
    # The fetch layer NEVER raises: a down Postgres / missing table degrades
    # to an empty event list (see fetch_drive_history_events), which must
    # surface as an honest insufficient_history refusal -- no LLM call, no
    # write, no fabricated pattern, and no fallback source.
    fetcher = FakeFetcher([])
    bus = _make_bus("")

    with patch("asyncpg.connect", new=AsyncMock()) as connect_mock:
        result = await dhrs._run_once(
            fetch_events=fetcher,
            bus=bus,
            postgres_uri="postgresql://fake/db",
            subject="orion",
            since=datetime(2026, 6, 14, tzinfo=timezone.utc),
        )

    assert result.status == "insufficient_history"
    assert "0 event" in result.error
    assert result.crystallization_id is None
    bus.rpc_request.assert_not_called()
    connect_mock.assert_not_called()


@pytest.mark.asyncio
async def test_llm_call_failure_writes_nothing():
    # A real (empty) FakeConn -- the dedup guard now runs a fetchval SELECT
    # before the LLM call even for a sufficient/non-duplicate window, so
    # asyncpg.connect IS called once for that check; the assertion that
    # matters is that no crystallization/source row is ever written.
    fetcher = FakeFetcher(_sufficient_history_events())
    bus = _make_bus(raise_exc=RuntimeError("gateway timeout"))
    conn = FakeConn()

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        result = await dhrs._run_once(
            fetch_events=fetcher,
            bus=bus,
            postgres_uri="postgresql://fake/db",
            subject="orion",
            since=datetime(2026, 6, 14, tzinfo=timezone.utc),
        )

    assert result.status == "llm_call_failed"
    assert "gateway timeout" in result.error
    assert result.crystallization_id is None
    assert conn.crystallizations == {}
    assert conn.sources == []


@pytest.mark.asyncio
async def test_llm_output_ungrounded_writes_nothing():
    fetcher = FakeFetcher(_sufficient_history_events())
    # Cites nothing real -- generic filler, no verbatim tokens.
    ungrounded_content = json.dumps(
        {"narrative": "Orion has been exploring many interesting things lately.", "cited_fact_numbers": [1, 2]}
    )
    bus = _make_bus(ungrounded_content)
    conn = FakeConn()

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        result = await dhrs._run_once(
            fetch_events=fetcher,
            bus=bus,
            postgres_uri="postgresql://fake/db",
            subject="orion",
            since=datetime(2026, 6, 14, tzinfo=timezone.utc),
        )

    assert result.status == "llm_output_ungrounded"
    assert result.crystallization_id is None
    assert conn.crystallizations == {}
    assert conn.sources == []


@pytest.mark.asyncio
async def test_dedup_guard_skips_when_already_synthesized_same_window():
    since = datetime(2026, 6, 14, tzinfo=timezone.utc)
    dedup_source_id = f"drive_history_aggregation:orion:{since.date().isoformat()}"
    existing = [{"args": ("existing-crys-id", "rdf_memory_graph", dedup_source_id, None, 1.0, "note")}]
    conn = FakeConn(existing_sources=existing)
    fetcher = FakeFetcher(_sufficient_history_events())
    bus = _make_bus("")  # must never be called

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        result = await dhrs._run_once(
            fetch_events=fetcher,
            bus=bus,
            postgres_uri="postgresql://fake/db",
            subject="orion",
            since=since,
        )

    assert result.status == "already_synthesized_today"
    assert result.crystallization_id == "existing-crys-id"
    bus.rpc_request.assert_not_called()
    assert conn.crystallizations == {}  # no new row inserted


@pytest.mark.asyncio
async def test_dedup_guard_failure_does_not_block_synthesis():
    # A dedup-check failure (e.g. transient DB error) must degrade to
    # "proceed" rather than blocking an otherwise-valid, already-reduced run.
    fetcher = FakeFetcher(_sufficient_history_events())
    bus = _make_bus(_grounded_llm_content())

    call_count = {"n": 0}
    real_conn = FakeConn()

    async def _flaky_connect(uri):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient connect failure")
        return real_conn

    with patch("asyncpg.connect", new=_flaky_connect):
        result = await dhrs._run_once(
            fetch_events=fetcher,
            bus=bus,
            postgres_uri="postgresql://fake/db",
            subject="orion",
            since=datetime(2026, 6, 14, tzinfo=timezone.utc),
        )

    assert result.status == "created"
    assert result.crystallization_id is not None


def test_main_reports_missing_postgres_uri(capsys):
    exit_code = dhrs.main(["--postgres-uri", ""])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "POSTGRES_URI" in captured.err
