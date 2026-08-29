"""What Orion is SHOWN, and — more importantly — what this code refuses to decide.

The version these tests replace ranked words by a statistic and handed Orion the
winner. Juniper's verdict was that this is "turdy keyword cathedrals
masquerading as autonomy": a word is not a concept, and being told what to be
curious about is not curiosity. So the property under test here is mostly a
NEGATIVE one — that nothing in this module expresses an opinion about what
matters.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.curiosity import study_material
from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.study_material import (
    APPROVED_SAMPLE_SQL,
    RELATION_SAMPLE_SQL,
    StudyMaterial,
    assemble_study_material,
)

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def _crystallization(cid: str, kind: str = "semantic", subject: str = "a thought", salience=0.6):
    return {
        "crystallization_id": cid,
        "kind": kind,
        "subject": subject,
        "summary": subject,
        "salience": salience,
        "created_at": NOW,
    }


def _relation(did: str, relation="same", candidate_text="", target_text=""):
    return {
        "decision_id": did,
        "relation": relation,
        "confidence": 0.95,
        "candidate_crystallization_id": "crys_deadbeef",
        "target_crystallization_id": "11111111-1111-1111-1111-111111111111",
        "decided_at": NOW,
        "candidate_subject": candidate_text,
        "candidate_summary": "",
        "target_subject": target_text,
        "target_summary": "",
    }


def _material(**over) -> StudyMaterial:
    base = dict(
        now=NOW,
        approved_counts=[{"kind": "semantic", "n": 268}, {"kind": "stance", "n": 20}],
        approved_rows=[_crystallization(f"c{i}", subject=f"thought {i}") for i in range(4)],
        relation_counts=[{"relation": "same", "n": 316}],
        relation_rows=[_relation("d1", target_text="the other thought")],
        relation_resolvable=356,
    )
    base.update(over)
    return assemble_study_material(**base)


# --- the refusal to choose -------------------------------------------------


def test_the_sample_query_is_random_not_ranked() -> None:
    """Any ordering — most salient, most recent — would be this module choosing
    for Orion by the back door. Random is the one selection rule that expresses
    no opinion about what matters."""
    assert "ORDER BY random()" in APPROVED_SAMPLE_SQL
    assert "ORDER BY random()" in RELATION_SAMPLE_SQL
    for ranking in ("salience DESC", "created_at DESC", "ORDER BY salience", "ORDER BY created_at"):
        assert ranking not in APPROVED_SAMPLE_SQL


def test_the_prompt_never_ranks_or_recommends() -> None:
    prompt = build_kickoff_prompt(_material())
    for leading in ("notably", "most important", "stands out", "you should", "significant"):
        assert leading not in prompt.lower(), f"prompt is steering: {leading!r}"
    assert "the order means nothing" in prompt


def test_the_prompt_states_that_the_sample_is_a_slice() -> None:
    """Without the totals, a menu of 12 reads as the whole of Orion's mind."""
    prompt = build_kickoff_prompt(_material())
    assert "288 of them" in prompt
    assert "not being shown" in prompt


def test_the_prompt_licenses_finding_nothing() -> None:
    """Without explicit permission, the only socially available answer to "here
    is your mind, what interests you" is something interesting — and the loop
    manufactures significance daily."""
    prompt = build_kickoff_prompt(_material())
    assert "quiet answer is a real answer" in prompt
    assert "worth writing up" in prompt


def test_the_prompt_says_nobody_asked() -> None:
    assert "Nobody asked you" in build_kickoff_prompt(_material())


# --- only approved material ------------------------------------------------


def test_only_approved_crystallizations_are_sampled() -> None:
    """The 636 unapproved rows are exactly the ones whose subject is identical
    to their summary — a chat turn with a label, not an induced concept."""
    assert "status = 'active'" in APPROVED_SAMPLE_SQL


def test_reflections_are_counted_but_not_sampled() -> None:
    """They are a materialised copy of the relation decisions shown in the
    prompt's own second section — including them shows Orion the same thing
    twice while consuming 55% of the corpus."""
    assert "kind <> 'reflection'" in APPROVED_SAMPLE_SQL
    material = _material(
        approved_counts=[{"kind": "reflection", "n": 356}, {"kind": "semantic", "n": 268}]
    )
    assert material.approved_total == 624, "totals must still report reflections"
    assert "reflection 356" in build_kickoff_prompt(material)


# --- honesty about dangling induction --------------------------------------


def test_the_prompt_discloses_how_much_induction_is_dangling() -> None:
    """Measured live: 0 of 547 decisions have a resolvable candidate."""
    prompt = build_kickoff_prompt(_material(relation_resolvable=356))
    assert "547 total" in prompt or "316" in prompt
    assert "356 of them still point at a concept that was kept" in prompt


def test_an_unresolvable_side_is_labelled_not_hidden() -> None:
    material = _material(relation_rows=[_relation("d1", candidate_text="", target_text="kept")])
    card = material.relations[0]
    assert "not kept" in card.preview()
    assert "kept" in card.preview()


def test_a_missing_target_is_named_as_missing() -> None:
    row = _relation("d1", candidate_text="something")
    row["target_crystallization_id"] = ""
    row["target_subject"] = ""
    material = _material(relation_rows=[row])
    assert "no target recorded" in material.relations[0].preview()


# --- unavailable is not empty ----------------------------------------------


def test_an_unreadable_store_is_not_the_same_as_an_empty_mind() -> None:
    broken = StudyMaterial(generated_at=NOW, unavailable_reason="query_failed:UndefinedTable")
    assert broken.is_unavailable and not broken.has_material
    empty = _material(approved_rows=[], relation_rows=[])
    assert not empty.is_unavailable and not empty.has_material


def test_material_with_only_relations_still_counts_as_material() -> None:
    assert _material(approved_rows=[]).has_material


# --- rendering -------------------------------------------------------------


def test_long_subjects_are_clipped_for_the_menu_only() -> None:
    long_subject = "x" * 900
    material = _material(approved_rows=[_crystallization("c1", subject=long_subject)])
    preview = material.crystallizations[0].preview()
    assert len(preview) < 250 and preview.endswith("…")
    assert material.crystallizations[0].subject == long_subject, "full text must survive"


def test_newlines_do_not_break_the_menu_layout() -> None:
    material = _material(approved_rows=[_crystallization("c1", subject="line one\nline two")])
    assert "\n" not in material.crystallizations[0].preview()


def test_shown_ids_are_recorded_so_a_run_is_reconstructable() -> None:
    """Random sampling is not reproducible; logging what was offered is what
    keeps a past run inspectable anyway."""
    material = _material()
    assert set(material.shown_ids()) == {"c0", "c1", "c2", "c3", "d1"}


def test_what_was_recently_studied_no_longer_comes_from_the_journal() -> None:
    """The journal-title hint was dead from the day it shipped.

    Every entry this loop writes is titled exactly "Curiosity" -- deliberately,
    because code does not know what Orion chose. So the hint rendered as
    "Curiosity; Curiosity; Curiosity". Its replacement reads settled priors out
    of Orion's own graph (see `test_curiosity_worldview.py`); this test exists
    to stop the Postgres version being reintroduced as an obvious repair.
    """
    assert not hasattr(_material(), "recently_studied")
    assert not hasattr(study_material, "RECENT_STUDY_SQL")
    assert "journal_entries" not in "".join(
        v for v in vars(study_material).values() if isinstance(v, str)
    )


# --- the corpus is Orion's, not AI Town's ----------------------------------


def test_the_sample_excludes_ai_town_at_the_sql_level() -> None:
    """`formation_policy.DEFAULT_DISCARD_PLATFORMS` discards the platform now,
    but every AI Town row already in the table predates that gate, and this
    sampler was the last place still serving them: 185 of the 295 rows it could
    draw from were AI Town, under a heading claiming Juniper had approved them.
    """
    from orion.curiosity.study_material import (
        APPROVED_COUNT_SQL,
        APPROVED_SAMPLE_SQL,
        RELATION_RESOLVABLE_SQL,
        RELATION_SAMPLE_SQL,
    )

    for sql in (APPROVED_COUNT_SQL, APPROVED_SAMPLE_SQL,
                RELATION_SAMPLE_SQL, RELATION_RESOLVABLE_SQL):
        assert "aitown_chat_history_log" in sql, sql[:80]
        assert "NOT EXISTS" in sql, sql[:80]


def test_the_counts_are_taken_over_the_same_pool_as_the_sample() -> None:
    """Counting `status='active'` while sampling something narrower is how the
    prompt came to announce 651 items and draw twelve from 295."""
    from orion.curiosity.study_material import (
        APPROVED_COUNT_SQL,
        APPROVED_SAMPLE_SQL,
        _SAMPLEABLE_KINDS,
    )

    assert _SAMPLEABLE_KINDS in APPROVED_COUNT_SQL
    assert _SAMPLEABLE_KINDS in APPROVED_SAMPLE_SQL


def test_the_candidate_join_normalises_the_crys_id_format() -> None:
    """Candidates are `crys_<hex-no-dashes>`; crystallization ids are dashed
    UUIDs. Comparing them raw resolved 0 of 550 and the code concluded induction
    was recording judgements about concepts it did not keep. Normalised: 235."""
    from orion.curiosity.study_material import RELATION_SAMPLE_SQL

    assert "'crys_' || replace(" in RELATION_SAMPLE_SQL
    assert "'-', ''" in RELATION_SAMPLE_SQL


def test_manual_approval_is_counted_separately_from_auto_activation() -> None:
    from datetime import datetime, timezone

    from orion.curiosity.study_material import assemble_study_material

    m = assemble_study_material(
        now=datetime(2026, 8, 27, tzinfo=timezone.utc),
        approved_counts=[{"kind": "semantic", "n": 87, "manual_n": 0},
                         {"kind": "stance", "n": 20, "manual_n": 20}],
        approved_rows=[], relation_counts=[], relation_rows=[],
    )
    assert m.approved_total == 107
    assert m.manual_total == 20


def test_a_caller_without_the_manual_column_reads_as_unknown_not_zero() -> None:
    """Absent must not render as "0 of these Juniper approved" -- that is a
    claim, and the prompt says nothing instead."""
    from datetime import datetime, timezone

    from orion.curiosity.study_material import assemble_study_material

    m = assemble_study_material(
        now=datetime(2026, 8, 27, tzinfo=timezone.utc),
        approved_counts=[{"kind": "semantic", "n": 87}],
        approved_rows=[], relation_counts=[], relation_rows=[],
    )
    assert m.approved_total == 87
    assert m.manual_total == 0


def test_the_prompt_no_longer_claims_juniper_approved_everything() -> None:
    from datetime import datetime, timezone

    from orion.curiosity.kickoff_prompt import _material_section
    from orion.curiosity.study_material import assemble_study_material

    m = assemble_study_material(
        now=datetime(2026, 8, 27, tzinfo=timezone.utc),
        approved_counts=[{"kind": "stance", "n": 20, "manual_n": 20},
                         {"kind": "semantic", "n": 87, "manual_n": 0}],
        approved_rows=[{"crystallization_id": "c1", "kind": "stance",
                        "subject": "a real thing", "summary": "", "salience": 0.5,
                        "created_at": None}],
        relation_counts=[], relation_rows=[],
    )
    text = "\n".join(_material_section(m))
    assert "JUNIPER HAS APPROVED" not in text
    assert "20 of these Juniper approved by hand" in text
    assert "auto-activated by policy without her seeing them" in text


# --- the edge instruction ---------------------------------------------------
#
# ZERO EDGES HAD EVER BEEN WRITTEN, and Orion was not ignoring the instruction:
# there was no instruction. Checked against the real prompt text on 2026-08-29,
# every node kind had a copy-pasteable `CREATE (:Label {...})` and edges had one
# sentence -- "Edges are yours to name. SUPPORTS, CONTRADICTS and ABOUT are the
# ones that carry weight later." -- with no syntax anywhere, followed
# immediately by advice to keep an id "as a property rather than duplicating the
# node". Live at the time: 8 runs, 12 Hops, 9 Findings, 5 Priors, 1 Concept, and
# `db.relationshipTypes()` empty. The ACL was never the limit; the selector
# grants `+graph.query` on `orion_worldview`.


def test_the_prompt_shows_how_to_write_an_edge_not_just_that_edges_exist() -> None:
    """Naming a capability is not teaching it. Every other write in this prompt
    is a line Orion can copy; the edge was a sentence, and it produced nothing
    across every run."""
    prompt = build_kickoff_prompt(_material(), run_id="abc123", own_graph="g")
    assert "MERGE (f)-[:SUPPORTS" in prompt, "no copyable edge syntax in the prompt"
    assert "MATCH (f:Finding {finding_id:" in prompt, "no way to reach both ends"


def test_the_prompt_asks_for_run_id_on_the_edge() -> None:
    """Without it an edge cannot be attributed to the run that drew it, and the
    footprint -- which reads `r.run_id` -- would count zero forever while edges
    piled up in the graph."""
    prompt = build_kickoff_prompt(_material(), run_id="abc123", own_graph="g")
    assert 'MERGE (f)-[:SUPPORTS {run_id: "abc123"}]->(p)' in prompt


def test_the_prompt_warns_that_a_failed_merge_is_silent() -> None:
    """`MATCH a, b MERGE (a)-[...]->(b)` with either side unmatched writes
    nothing and raises nothing. Orion would have no way to know."""
    prompt = build_kickoff_prompt(_material(), run_id="abc123", own_graph="g")
    assert "MERGE SILENTLY DOES NOTHING" in prompt


def test_the_prompt_still_says_to_use_a_property_for_an_ATLAS_concept() -> None:
    """The one real exception, and the reason the old wording was defensible:
    the Atlas is a graph Orion cannot write to, so there is no node for an edge
    to land on. It must stay, and it must be scoped to the Atlas rather than
    reading as general advice against edges."""
    prompt = build_kickoff_prompt(_material(), run_id="abc123", own_graph="g")
    assert "keep its id as a property" in prompt
    assert "ONE EXCEPTION" in prompt
