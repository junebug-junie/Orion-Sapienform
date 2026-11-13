"""Orion's own coined concepts are shown back in the kickoff prompt, so a run reuses one.

Before this, the prompt only said `CREATE (:Concept ...)` and never showed an
existing concept: live 2026-11-13, 9 of 35 concepts had 2+ links. Priors are
shown back and got 341 revisions. Spec: docs/superpowers/specs/
2026-11-13-core-memory-pages-and-concept-growth-design.md, Phase 2, source 1.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.study_material import assemble_study_material
from orion.curiosity.worldview import (
    CONCEPT_COUNT_CYPHER,
    CONCEPTS_CYPHER,
    CONCEPT_FRESH_SLOTS,
    CONCEPTS_SHOWN,
    ConceptRecord,
    WorldviewReader,
    WorldviewSnapshot,
    WorldviewUnavailable,
    build_concepts,
    read_snapshot,
)

RUN_ID = "a1b2c3d4e5f6"


class _Reader(WorldviewReader):
    def __init__(self, answers=None, raises=False):
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.raises = raises
        self.queries: list[str] = []

    def query(self, cypher):
        self.queries.append(cypher)
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: nope")
        return self.answers.get(cypher, [])


def _row(cid, deg=0, name=None, note="", own=None, run_at=None, run_id="abc123def456"):
    return {
        "concept_id": cid, "name": name or f"name of {cid}", "note": note,
        "run_id": run_id, "concept_written_at": own,
        "concept_degree": deg, "concept_run_written_at": run_at,
    }


def _prompt(view):
    material = assemble_study_material(
        now=datetime(2026, 11, 13, tzinfo=timezone.utc),
        approved_counts=[], approved_rows=[],
        relation_counts=[], relation_rows=[], relation_resolvable=0,
    )
    return build_kickoff_prompt(material, view=view, run_id=RUN_ID)


# --- cypher shape -------------------------------------------------------------


def test_concepts_cypher_reads_only_and_counts_degree_before_the_outcome_join():
    q = CONCEPTS_CYPHER
    for word in ("CREATE", "MERGE", "SET ", "DELETE"):
        assert word not in q
    # Degree aggregated before the TurnOutcome join, or two outcomes double it.
    assert q.index("count(r) AS concept_degree") < q.index("OPTIONAL MATCH (t:TurnOutcome)")
    assert "(c)-[r]-()" in q  # undirected: an inbound edge is a link too
    # No server-side ORDER BY over hand-written timestamps (mixed ISO/int).
    assert "ORDER BY" not in q
    assert "LIMIT" in q
    returned = q.split("RETURN", 1)[1]
    for col in ("c.concept_id AS concept_id", "c.name AS name", "c.note AS note",
                "c.written_at AS concept_written_at", ", concept_degree,",
                "t.written_at AS concept_run_written_at"):
        assert col in returned


def test_read_snapshot_reads_concepts_through_the_same_reader():
    reader = _Reader({
        CONCEPTS_CYPHER: [_row("c-1", deg=2, note="a note")],
        CONCEPT_COUNT_CYPHER: [{"n": 1}],
    })
    view = read_snapshot(reader, sample=8, stale_after=3)
    assert CONCEPTS_CYPHER in reader.queries
    assert [c.concept_id for c in view.concepts] == ["c-1"]
    assert view.concepts[0].degree == 2


def test_unreachable_graph_is_unavailable_not_empty():
    view = read_snapshot(_Reader(raises=True), sample=8, stale_after=3)
    assert view.is_unavailable
    assert view.concepts == []


# --- build_concepts: ordering, grouping, cap ----------------------------------


def test_most_linked_first_then_newest():
    rows = [
        _row("old-1link", deg=1, own=1000),
        _row("new-1link", deg=1, run_at="2026-11-12T00:00:00Z"),
        _row("three", deg=3),
        _row("none", deg=0, own=9_999_999_999_999),
    ]
    assert [c.concept_id for c in build_concepts(rows)] == [
        "three", "new-1link", "old-1link", "none",
    ]


def test_own_written_at_wins_over_the_run_timestamp():
    [c] = build_concepts([_row("c", own="1000", run_at="5000")])
    assert c.newest_ms == 1000


def test_two_outcomes_for_one_run_show_one_concept_with_the_newest_stamp():
    rows = [_row("c", deg=2, run_at=100), _row("c", deg=2, run_at=300)]
    [c] = build_concepts(rows)
    assert c.degree == 2 and c.newest_ms == 300


def test_forked_concept_shows_once_as_the_most_linked_copy():
    rows = [_row("c", deg=1, note="thin"), _row("c", deg=4, note="rich")]
    [c] = build_concepts(rows)
    assert c.degree == 4 and c.note == "rich"


def test_a_concept_with_no_id_is_dropped():
    assert build_concepts([_row(None, deg=5), _row("", deg=5)]) == []


def test_cap():
    rows = [_row(f"c{i:02d}", deg=i) for i in range(40)]
    out = build_concepts(rows)
    assert len(out) == CONCEPTS_SHOWN == 15
    assert out[0].concept_id == "c39"


def test_a_fresh_one_link_concept_is_shown_even_when_many_are_better_linked():
    # Ranked by links alone, the well-linked set would hold every slot forever.
    rows = [_row(f"old{i:02d}", deg=3, own=1000 + i) for i in range(20)]
    rows.append(_row("fresh", deg=1, own=10_000))
    out = [c.concept_id for c in build_concepts(rows)]
    assert len(out) == CONCEPTS_SHOWN
    assert "fresh" in out
    assert out[: CONCEPTS_SHOWN - CONCEPT_FRESH_SLOTS] == [
        f"old{i:02d}" for i in range(19, 19 - (CONCEPTS_SHOWN - CONCEPT_FRESH_SLOTS), -1)
    ]
    assert out[CONCEPTS_SHOWN - CONCEPT_FRESH_SLOTS] == "fresh"


def test_fewer_concepts_than_the_cap_are_all_shown():
    rows = [_row(f"c{i}", deg=0) for i in range(4)]
    assert len(build_concepts(rows)) == 4


def test_a_hand_written_infinite_timestamp_does_not_crash_the_read():
    [c] = build_concepts([_row("c", own="inf", run_at="1e400")])
    assert c.newest_ms is None
    reader = _Reader({CONCEPTS_CYPHER: [_row("c", own="inf")]})
    view = read_snapshot(reader, sample=8, stale_after=3)
    assert not view.is_unavailable
    assert [x.concept_id for x in view.concepts] == ["c"]


def test_read_concepts_false_skips_the_query():
    reader = _Reader({CONCEPTS_CYPHER: [_row("c")]})
    view = read_snapshot(reader, sample=8, stale_after=3, read_concepts=False)
    assert CONCEPTS_CYPHER not in reader.queries
    assert view.concepts == []


def test_degree_as_string_still_reads_as_a_number():
    # FalkorDB with decode_responses=True can hand numbers back as strings.
    [c] = build_concepts([_row("c", deg="3")])
    assert c.degree == 3


# --- rendering ----------------------------------------------------------------


def test_prompt_shows_concepts_with_full_ids_and_clipped_notes():
    long_note = "x" * 900
    view = WorldviewSnapshot(
        concept_total=35,
        concepts=[
            ConceptRecord("concept:served-brain-drift", "served-brain drift", "", 2),
            ConceptRecord("frame-inside-frame-20261105", "The frame", long_note, 1),
        ],
    )
    text = _prompt(view)
    assert "CONCEPTS YOU HAVE ALREADY COINED -- 35 in your graph, 2 shown" in text
    assert "concept_id: concept:served-brain-drift" in text
    assert "concept_id: frame-inside-frame-20261105" in text
    assert "[2 links] served-brain drift" in text
    assert "x" * 900 not in text
    assert "x" * 200 + "…" in text


def test_concepts_appear_after_priors_and_before_the_write_instructions():
    view = WorldviewSnapshot(concept_total=1, concepts=[ConceptRecord("c-1", "one")])
    text = _prompt(view)
    assert text.index("YOUR OWN GRAPH IS EMPTY") < text.index("CONCEPTS YOU HAVE ALREADY COINED")
    assert text.index("CONCEPTS YOU HAVE ALREADY COINED") < text.index("LOOK AT YOUR OWN CONCEPTS FIRST")


def test_write_instruction_reuses_before_it_creates():
    text = _prompt(WorldviewSnapshot(concepts=[ConceptRecord("c-1", "one")], concept_total=1))
    assert "CREATE (:Concept" not in text
    match = text.index(
        'MATCH (c:Concept {concept_id: "<an id from your concept list>"}) RETURN c.name, c.note'
    )
    merge = text.index('MERGE (c:Concept {concept_id: "<something unique>"})')
    assert match < merge
    assert "MERGE (f)-[e:ABOUT]->(c)" in text
    assert "MERGE (p)-[e:ABOUT]->(c)" in text
    # The id-rules paragraph must not read as "concept ids never go in a MATCH".
    assert "it DOES go in a MATCH" in text
    assert "`crystallization_id:` under a concept" not in text
    # A MERGE that bound an existing concept is not a failed write.
    assert "Do not coin a new id to make the count move" in text
    # The "connect it in the same breath" rule is kept.
    assert "THEN CONNECT IT, in the same breath" in text


def test_empty_graph_says_no_concepts_yet():
    text = _prompt(WorldviewSnapshot())
    assert "YOU HAVE NOT COINED A CONCEPT YET" in text
    assert "CONCEPTS YOU HAVE ALREADY COINED" not in text


def test_concepts_that_exist_but_cannot_be_read_are_not_called_none():
    text = _prompt(WorldviewSnapshot(concept_total=3, concepts=[]))
    assert "YOUR GRAPH HOLDS 3 CONCEPTS THAT COULD NOT BE READ BACK by id" in text
    assert "YOU HAVE NOT COINED A CONCEPT YET" not in text


def test_unavailable_graph_renders_unavailable_never_no_concepts():
    text = _prompt(WorldviewSnapshot(unavailable_reason="ConnectionError: nope"))
    assert "YOUR OWN CONCEPTS COULD NOT BE READ THIS RUN (ConnectionError: nope)" in text
    assert "YOU HAVE NOT COINED A CONCEPT YET" not in text
    assert "CONCEPTS YOU HAVE ALREADY COINED" not in text


def test_no_graph_configured_says_nothing_about_concepts():
    material = assemble_study_material(
        now=datetime(2026, 11, 13, tzinfo=timezone.utc),
        approved_counts=[], approved_rows=[],
        relation_counts=[], relation_rows=[], relation_resolvable=0,
    )
    text = build_kickoff_prompt(material, graph_enabled=False, run_id=RUN_ID)
    assert "COINED A CONCEPT" not in text
    assert "CONCEPTS YOU HAVE ALREADY COINED" not in text
    assert "YOUR OWN CONCEPTS COULD NOT BE READ" not in text


def test_read_only_prompt_lists_concepts_without_asking_for_writes():
    material = assemble_study_material(
        now=datetime(2026, 11, 13, tzinfo=timezone.utc),
        approved_counts=[], approved_rows=[],
        relation_counts=[], relation_rows=[], relation_resolvable=0,
    )
    view = WorldviewSnapshot(concept_total=1, concepts=[ConceptRecord("c-1", "one")])
    text = build_kickoff_prompt(material, view=view, run_id="")
    assert "concept_id: c-1" in text
    assert "reuse its concept_id" not in text
