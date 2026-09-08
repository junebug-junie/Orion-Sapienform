"""The self-inquiry line of the curiosity loop (orion/curiosity/self_inquiry.py).

What these pin down:

- the line has its OWN budget: its runs never touch the investigation
  line's cooldown or counter, and vice versa, and its cap survives a restart
- it needs the graph, the role, AND the outcome-table grants -- each missing
  piece is its own block reason, checked deterministically before a turn
- a run's `:SelfDefinition` is mirrored to self_concept_history ONLY when it
  has text and evidence; absent or evidence-less definitions are refused by
  cause, never inferred from prose
- the durable finish event carries the definition and Hub mirrors it there
- the journal entry is distinguishable from an investigation entry
- the prompt carries the standing question, this run's id, and no hardcoded
  duration (same rule as the investigation prompt)
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import pytest

# sys.path is arranged by tests/conftest.py (Hub root first, so `scripts` is
# Hub's package, not the repo-root one). Do not insert paths here.

from orion.core.bus.bus_schemas import ServiceRef
from orion.curiosity.journal import build_investigation_journal_entry, MaterialCounts
from orion.curiosity.self_inquiry import (
    LATEST_SELF_DEFINITION_CYPHER,
    LINE_SELF_INQUIRY,
    LIVE_SELF_PRIORS_CYPHER,
    SELF_CONCEPT_ID,
    SELF_DEFINITION_PRODUCER,
    SELF_INQUIRY_PG_TABLES,
    STANDING_QUESTION,
    LedgerRow,
    SelfDefinition,
    build_self_definition,
    build_self_definition_history_write,
    self_definition_for_run_cypher,
    self_definition_from_detail,
    self_definition_to_detail,
    worldview_evidence_ref,
)
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.worldview import LIVE_PRIORS_CYPHER
from orion.schemas.durable_run import DurableRunStateV1

from test_curiosity_investigation import (
    _FakeBus,
    _FakeConn,
    _FakePool,
    _FakeReader,
    _graph_loop,
    _loop,
    _outcome_rows,
)
from scripts.curiosity_investigation import (
    _COOLDOWN_KEY,
    _DAILY_COUNT_KEY_PREFIX,
    _SELF_COOLDOWN_KEY,
    _SELF_DAILY_COUNT_KEY_PREFIX,
    JOURNAL_WRITE_CHANNEL,
    SELF_CONCEPT_HISTORY_WRITE_CHANNEL,
)

SOURCE = ServiceRef(name="orion-hub", version="test", node="athena")
RUN = "abcdef123456"


# --- pure module ----------------------------------------------------------


def test_cypher_builders_refuse_a_non_hex_run_id() -> None:
    with pytest.raises(ValueError):
        self_definition_for_run_cypher("'; MATCH (n) DETACH DELETE n //")


def test_self_priors_cypher_is_the_live_priors_cypher_plus_one_where() -> None:
    assert "p.line = 'self'" in LIVE_SELF_PRIORS_CYPHER
    assert "p.line" not in LIVE_PRIORS_CYPHER
    # Same fields, same limit: the two lines cannot drift on what a prior is.
    assert LIVE_SELF_PRIORS_CYPHER.split("RETURN")[1] == LIVE_PRIORS_CYPHER.split("RETURN")[1]


def test_build_self_definition_reads_list_and_json_string_evidence() -> None:
    as_list = build_self_definition({"run_id": RUN, "text": "I am", "evidence": ["a", "b", "a"]})
    as_json = build_self_definition({"run_id": RUN, "text": "I am", "evidence": '["a", "b"]'})
    as_none = build_self_definition({"run_id": RUN, "text": "I am", "evidence": None})
    assert as_list is not None and as_list.evidence == ["a", "b"]
    assert as_json is not None and as_json.evidence == ["a", "b"]
    assert as_none is not None and as_none.evidence == [] and not as_none.is_substantive


def test_build_self_definition_drops_rows_with_no_text_or_run_id() -> None:
    assert build_self_definition({"run_id": RUN, "text": "  "}) is None
    assert build_self_definition({"run_id": "", "text": "I am"}) is None


def test_history_write_refuses_absent_and_evidence_less_definitions() -> None:
    assert build_self_definition_history_write(None, version=1) is None
    assert build_self_definition_history_write(SelfDefinition(RUN, "I am", []), version=1) is None


def test_history_write_appends_the_worldview_ref_and_names_the_producer() -> None:
    row = build_self_definition_history_write(SelfDefinition(RUN, "I am a mesh", ["README.md"]), version=4)
    assert row is not None
    assert row.concept_id == SELF_CONCEPT_ID
    assert row.produced_by == SELF_DEFINITION_PRODUCER
    assert row.version == 4
    assert row.evidence_refs == ["README.md", worldview_evidence_ref(RUN)]
    assert row.entry_id == f"self-definition:{RUN}", "keyed on the run so a re-fire upserts"


def test_detail_round_trip() -> None:
    d = SelfDefinition(RUN, "I am", ["x"], revises="000000aaaaaa", written_at=5)
    assert self_definition_from_detail({"self_definition": self_definition_to_detail(d)}) == d
    assert self_definition_from_detail({"self_definition": None}) is None
    assert self_definition_from_detail(None) is None


def test_journal_entry_for_the_self_line_is_distinguishable() -> None:
    counts = MaterialCounts(approved_total=0, approved_by_kind={}, crystallization_count=0, relation_total=0, relation_count=0)
    entry = build_investigation_journal_entry(
        material=counts, body_text="I looked.", correlation_id="c", run_id=RUN, line=LINE_SELF_INQUIRY
    )
    plain = build_investigation_journal_entry(material=counts, body_text="I looked.", correlation_id="c", run_id=RUN)
    assert entry.title == "Self-inquiry" and plain.title == "Curiosity"
    assert entry.entry_id != plain.entry_id
    # Same source_ref on purpose: the atlas page joins journal bodies on it.
    assert entry.source_ref == plain.source_ref == f"curiosity:{RUN}"
    assert entry.source_kind == plain.source_kind == "self_study"


# --- the prompt -----------------------------------------------------------


def _prompt(**over) -> str:
    kwargs = dict(
        run_id=RUN,
        ledger=[LedgerRow("dreams", 17, "2026-09-06T08:27:32+00:00")],
        granted_tables=SELF_INQUIRY_PG_TABLES,
        repo_root="/repo",
    )
    kwargs.update(over)
    return build_self_inquiry_prompt(**kwargs)


def test_prompt_carries_the_standing_question_and_this_run_id() -> None:
    text = _prompt()
    assert STANDING_QUESTION in text
    assert f'run_id: "{RUN}"' in text
    assert "<RUN_ID>" not in text
    assert f'MERGE (s:SelfDefinition {{run_id: "{RUN}"}})' in text
    assert "CREATE (:SelfDefinition" not in text, "one node per run, rewritten as Orion goes"
    assert "by your second hop at the latest" in text
    assert 'p.line = "self"' in text
    assert "/repo/" in text
    assert "dreams" in text and "17 rows" in text


def test_prompt_lists_the_granted_tables_only_when_told_they_are_granted() -> None:
    with_grants = _prompt()
    without = _prompt(granted_tables=())
    assert "harness_turn_trace" in with_grants
    assert "harness_turn_trace" not in without.split("HOW TO REACH")[1]
    assert "four tables" in without and "four tables" not in with_grants


def test_prompt_shows_the_previous_definition_as_orions_own_to_revise() -> None:
    latest = SelfDefinition("000000aaaaaa", "I am a distributed thing.", ["README.md"], revises="")
    text = _prompt(latest=latest, definition_count=2)
    assert "WHAT YOU LAST WROTE ABOUT YOURSELF (2 so far; run 000000aaaaaa)" in text
    assert "I am a distributed thing." in text
    assert "yours to revise" in text
    first = _prompt(latest=None, definition_count=0)
    assert "This would be the first" in first


def test_prompt_never_states_a_hardcoded_duration() -> None:
    """Same rule as the investigation prompt: the deadline is read from the
    sandbox env, never written as a literal that can drift."""
    text = _prompt()
    assert not re.search(r"\b\d{3,4}\s*(s|sec|seconds)\b", text), text


def test_prompt_drops_write_sections_when_the_graph_is_unreadable() -> None:
    from orion.curiosity.worldview import WorldviewSnapshot

    text = _prompt(view=WorldviewSnapshot(unavailable_reason="ConnectionError"))
    assert "SelfDefinition" not in text.split("HOW TO REACH")[1]
    assert STANDING_QUESTION in text


# --- the loop: budget ------------------------------------------------------


def _self_loop(bus, *, reader=None, conn=None, **over):
    """A loop with the self line on and a fake reader that answers the
    self-definition read for whatever run id the loop generates."""
    reader = reader if reader is not None else _FakeReader()
    kwargs = dict(self_inquiry_enabled=True, self_inquiry_daily_cap=3, self_inquiry_min_cooldown_sec=0.0)
    kwargs.update(over)
    return _graph_loop(bus, reader=reader, conn=conn, **kwargs)


class _GrantConn(_FakeConn):
    """`_FakeConn` plus the two self-inquiry queries: the grant check (returns
    the MISSING tables) and the ledger counts."""

    def __init__(self, *, missing=(), **kw) -> None:
        super().__init__(**kw)
        self.missing = list(missing)
        self.version_lookups = 0

    async def fetch(self, sql, *args):
        if "has_table_privilege" in sql:
            return [{"table_name": t} for t in self.missing]
        return await super().fetch(sql, *args)

    async def fetchrow(self, sql, *args):
        if "count(*) AS n" in sql:
            return {"n": 17, "last": datetime(2026, 9, 6, 8, 27, tzinfo=timezone.utc)}
        return None

    async def fetchval(self, sql, *args):
        if "MAX(version)" in sql:
            self.version_lookups += 1
            return 3
        return await super().fetchval(sql, *args)


def _definition_rows(run_id: str, *, evidence=("README.md#Project Overview",)):
    return [{"run_id": run_id, "text": "I am a mesh of services.", "evidence": list(evidence), "revises": "", "written_at": 1}]


class _DefinitionReader(_FakeReader):
    """Answers the per-run SelfDefinition read for ANY run id the loop mints,
    since the test cannot know it in advance."""

    def __init__(self, *, evidence=("README.md#Project Overview",), write_definition=True, **kw) -> None:
        super().__init__(**kw)
        self.evidence = evidence
        self.write_definition = write_definition

    def query(self, cypher: str):
        self.queries.append(cypher)
        m = re.search(r"\(s:SelfDefinition\) WHERE s\.run_id = '([0-9a-f]+)'", cypher)
        if m and self.write_definition:
            return _definition_rows(m.group(1), evidence=self.evidence)
        return super().query(cypher)


def _mirrors(bus) -> list:
    return [(c, e) for c, e in bus.published if c == SELF_CONCEPT_HISTORY_WRITE_CHANNEL]


def _journal(bus) -> list:
    return [(c, e) for c, e in bus.published if c == JOURNAL_WRITE_CHANNEL]


def test_a_self_inquiry_run_uses_its_own_keys_and_never_the_investigation_budget() -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=False)
    assert asyncio.run(loop.tick_self_inquiry()) is None
    keys = bus.redis.values
    assert _SELF_COOLDOWN_KEY in keys
    assert any(k.startswith(_SELF_DAILY_COUNT_KEY_PREFIX) for k in keys)
    assert _COOLDOWN_KEY not in keys
    assert not any(k.startswith(_DAILY_COUNT_KEY_PREFIX) for k in keys)


def test_an_investigation_run_does_not_consume_the_self_budget() -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=False)
    # The investigation line only: force skips the self line by design.
    assert asyncio.run(loop.tick(force=True)) is None
    assert _COOLDOWN_KEY in bus.redis.values
    assert _SELF_COOLDOWN_KEY not in bus.redis.values


def test_the_self_cap_survives_a_restart() -> None:
    bus = _FakeBus()
    today = datetime.now(timezone.utc).astimezone(timezone.utc).date().isoformat()
    bus.redis.values[f"{_SELF_DAILY_COUNT_KEY_PREFIX}{today}"] = "3"
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=False)
    assert asyncio.run(loop.tick_self_inquiry()) == "daily_cap"
    assert _mirrors(bus) == []


def test_force_overrides_the_self_cap_and_still_counts() -> None:
    bus = _FakeBus()
    today = datetime.now(timezone.utc).date().isoformat()
    bus.redis.values[f"{_SELF_DAILY_COUNT_KEY_PREFIX}{today}"] = "3"
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=False)
    assert asyncio.run(loop.tick_self_inquiry(force=True)) is None
    assert bus.redis.values[f"{_SELF_DAILY_COUNT_KEY_PREFIX}{today}"] == "4"


def test_a_scheduled_tick_gives_the_self_line_first_refusal() -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=False)
    assert asyncio.run(loop.tick()) is None
    assert _SELF_COOLDOWN_KEY in bus.redis.values
    assert _COOLDOWN_KEY not in bus.redis.values
    # Second tick: the self line is on cooldown-by-count? No -- cooldown floor
    # is 0 here and cap is 3, so it runs again. Cap it to see the fall-through.
    loop.self_inquiry_daily_cap = 1
    assert asyncio.run(loop.tick()) is None
    assert _COOLDOWN_KEY in bus.redis.values, "with the self cap spent, the investigation line ran"


def test_disabled_self_line_never_touches_a_scheduled_tick() -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=False, self_inquiry_enabled=False)
    assert asyncio.run(loop.tick()) is None
    assert _SELF_COOLDOWN_KEY not in bus.redis.values
    assert asyncio.run(loop.tick_self_inquiry()) == "disabled"


# --- the loop: gates ---------------------------------------------------------


def test_self_inquiry_needs_a_graph() -> None:
    bus = _FakeBus()
    loop = _loop(bus, conn=_GrantConn(), self_inquiry_enabled=True, self_inquiry_min_cooldown_sec=0.0, kickoff_via_cortex=False)
    assert loop.graph_enabled is False
    assert asyncio.run(loop.tick_self_inquiry()) == "graph_required"


def test_missing_grants_block_the_run_and_name_the_tables(caplog) -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(missing=["dreams", "harness_turn_trace"]), kickoff_via_cortex=False)
    with caplog.at_level("WARNING"):
        assert asyncio.run(loop.tick_self_inquiry()) == "pg_grants_missing"
    assert "tables=dreams,harness_turn_trace" in caplog.text
    assert _SELF_COOLDOWN_KEY not in bus.redis.values, "a blocked run spends no slot"


class _BrokenGrantConn(_GrantConn):
    async def fetch(self, sql, *args):
        if "has_table_privilege" in sql:
            raise RuntimeError('role "" does not exist')
        return await super().fetch(sql, *args)


def test_a_failed_grant_check_blocks_rather_than_passing(caplog) -> None:
    """Review finding 2026-09-08: `has_table_privilege` raises for a missing
    table or an empty role, and the exception used to read as 'all granted'."""
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_BrokenGrantConn(), kickoff_via_cortex=False)
    with caplog.at_level("WARNING"):
        assert asyncio.run(loop.tick_self_inquiry()) == "grant_check_failed"
    assert "nothing is assumed granted" in caplog.text
    assert _SELF_COOLDOWN_KEY not in bus.redis.values
    assert _mirrors(bus) == [] and _journal(bus) == []


def test_the_grant_query_treats_a_missing_table_as_missing_not_as_an_error() -> None:
    from orion.curiosity.self_inquiry import SELF_INQUIRY_GRANTS_SQL

    assert "to_regclass" in SELF_INQUIRY_GRANTS_SQL
    # CASE, not OR: SQL does not guarantee OR short-circuits, and
    # has_table_privilege raises on a relation that does not exist.
    assert SELF_INQUIRY_GRANTS_SQL.index("CASE WHEN to_regclass") < SELF_INQUIRY_GRANTS_SQL.index("has_table_privilege")


def test_no_pool_yet_reads_as_stores_not_ready_for_the_self_line() -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), kickoff_via_cortex=False, pool_provider=lambda: None)
    assert asyncio.run(loop.tick_self_inquiry()) == "stores_not_ready"


# --- the mirror -----------------------------------------------------------


def test_a_definition_with_evidence_is_mirrored_with_the_next_version() -> None:
    bus = _FakeBus()
    conn = _GrantConn()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=conn, kickoff_via_cortex=False)
    assert asyncio.run(loop.tick_self_inquiry()) is None
    mirrored = _mirrors(bus)
    assert len(mirrored) == 1
    payload = mirrored[0][1].payload
    assert payload["concept_id"] == SELF_CONCEPT_ID
    assert payload["produced_by"] == SELF_DEFINITION_PRODUCER
    assert payload["version"] == 3, "MAX(version)+1 from the real table, not a guess"
    assert payload["content"] == "I am a mesh of services."
    assert payload["evidence_refs"][0] == "README.md#Project Overview"
    assert payload["evidence_refs"][-1].startswith("worldview:SelfDefinition:")
    assert mirrored[0][1].kind == "self_concept.history.write.v1"
    # And the journal entry is the self line's.
    assert len(_journal(bus)) == 1
    assert _journal(bus)[0][1].payload["title"] == "Self-inquiry"


def test_a_definition_without_evidence_is_refused_by_cause(caplog) -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(evidence=()), conn=_GrantConn(), kickoff_via_cortex=False)
    with caplog.at_level("INFO"):
        assert asyncio.run(loop.tick_self_inquiry()) is None
    assert _mirrors(bus) == []
    assert "reason=no_evidence" in caplog.text
    assert len(_journal(bus)) == 1, "the journal still records what Orion wrote"


def test_a_run_that_wrote_no_definition_mirrors_nothing(caplog) -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(write_definition=False), conn=_GrantConn(), kickoff_via_cortex=False)
    with caplog.at_level("INFO"):
        assert asyncio.run(loop.tick_self_inquiry()) is None
    assert _mirrors(bus) == []
    assert "reason=absent" in caplog.text


def test_the_self_run_reads_only_self_priors_and_the_latest_definition() -> None:
    bus = _FakeBus()
    reader = _DefinitionReader()
    loop = _self_loop(bus, reader=reader, conn=_GrantConn(), kickoff_via_cortex=False)
    assert asyncio.run(loop.tick_self_inquiry()) is None
    assert any("p.line = 'self'" in q for q in reader.queries)
    assert LATEST_SELF_DEFINITION_CYPHER in reader.queries
    from orion.curiosity.self_inquiry import SELF_COUNTS_CYPHER

    assert SELF_COUNTS_CYPHER in reader.queries, "the prompt's live_total is the self line's"
    assert not any(q == LIVE_PRIORS_CYPHER for q in reader.queries)


# --- durable finish ------------------------------------------------------------


def _finish_event(run_id: str, detail: dict) -> DurableRunStateV1:
    return DurableRunStateV1(
        run_id=run_id,
        workflow="curiosity.investigate",
        thread_id=run_id,
        node="finish",
        status="completed",
        correlation_id="7dcc3944-29bb-5d8f-915f-90f4e6968d47",
        detail=detail,
    )


def _deliver(loop, event: DurableRunStateV1) -> None:
    from orion.core.bus.bus_schemas import BaseEnvelope

    env = BaseEnvelope(kind="durable.run.state.v1", source=SOURCE, payload=event.model_dump(mode="json"))
    raw = loop._bus.codec.encode(env) if hasattr(loop._bus, "codec") else None
    if raw is None:
        # `_handle_run_state` only needs a decodable message; build one with
        # the real codec so the test exercises the real validation path.
        from orion.core.bus.codec import OrionCodec

        codec = OrionCodec()
        loop._bus.codec = codec
        raw = codec.encode(env)
    asyncio.run(loop._handle_run_state({"data": raw}))


def test_the_durable_finish_event_mirrors_a_self_inquiry_definition() -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=True)
    _deliver(
        loop,
        _finish_event(
            RUN,
            {
                "line": "self_inquiry",
                "self_definition": self_definition_to_detail(SelfDefinition(RUN, "I am a mesh.", ["README.md"])),
                "reach_out": False,
            },
        ),
    )
    mirrored = _mirrors(bus)
    assert len(mirrored) == 1
    assert mirrored[0][1].payload["content"] == "I am a mesh."


def test_the_durable_finish_event_for_an_investigation_run_mirrors_nothing() -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn(), kickoff_via_cortex=True)
    _deliver(
        loop,
        _finish_event(
            RUN,
            {"line": "investigate", "self_definition": self_definition_to_detail(SelfDefinition(RUN, "I am.", ["x"])), "reach_out": False},
        ),
    )
    assert _mirrors(bus) == []


def test_the_durable_brief_for_a_self_run_names_its_line() -> None:
    from orion.curiosity.study_material import StudyMaterial

    bus = _FakeBus()
    loop = _self_loop(bus, reader=_DefinitionReader(), conn=_GrantConn())
    brief = loop._run_brief(prompt="q", material=StudyMaterial(generated_at=datetime.now(timezone.utc)), line=LINE_SELF_INQUIRY)
    assert brief.line == "self_inquiry"
    assert brief.source_tag == "curiosity_self_inquiry"
    plain = loop._run_brief(prompt="q", material=StudyMaterial(generated_at=datetime.now(timezone.utc)))
    assert plain.line == "investigate" and plain.source_tag == "curiosity_investigation"


def test_paced_self_cooldown_derives_from_the_self_cap_not_the_investigation_cap() -> None:
    bus = _FakeBus()
    loop = _self_loop(
        bus, reader=_DefinitionReader(), conn=_GrantConn(),
        window_start_hour=8, window_end_hour=22, daily_cap=6, self_inquiry_daily_cap=3,
        min_cooldown_sec=0.0, self_inquiry_min_cooldown_sec=0.0,
    )
    assert loop.effective_cooldown_sec == 14 * 3600 / 6
    assert loop.effective_self_inquiry_cooldown_sec == 14 * 3600 / 3
