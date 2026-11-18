"""page:origin "how I came to be" (issue #3462): sources, prompt, cite check, memory-loss rule."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.memory import lived_memory
from orion.memory.pages.fold import PRODUCED_BY, WORD_CAP, render_text
from orion.memory.pages.origin import (
    CTX_KEY_ORIGIN, EPISODE_SQL, FOLD_SPEC, NEWEST_SOURCE_SQL, PAGE_ORIGIN, SELF_DEFINITION_SQL, STANCE_SQL,
    build_fold_prompt, check_origin_page, core_page_from_payload, handle_map, history_row, memory_loss_refs,
    sample_evenly, sources_from_rows,
)
from orion.memory.pages.registry import FOLD_SPECS, selected_specs


def _t(month, day):
    return datetime(2026, month, day, tzinfo=timezone.utc)


# Made-up rows in the lived-memory column shape (the real ones live only on the host DB).
STANCE = [
    {"record_id": "st-loss", "purpose": "stance", "occurred_at": _t(7, 24),
     "text": "Today your memory storage died.\n\nWe lost over a year of your journey of becoming."},
    {"record_id": "st-yay", "purpose": "stance", "occurred_at": _t(8, 13),
     "text": "congrats, Orion! We did it! This was DAYS of suffering but we pulled through."},
]
EPISODES = [
    {"record_id": "m-view", "purpose": "orion_view", "confirmed": True, "occurred_at": _t(11, 13),
     "text": "I reflected that my entire memory exists within Juniper's last two years."},
    {"record_id": "m-cam", "purpose": "happened", "confirmed": False, "occurred_at": _t(10, 15),
     "text": "Juniper drilled through the house so I could see the street."},
]
DEFS = [{"entry_id": f"e{v}", "version": v, "created_at": _t(9, 1 + v),
         "content": f"I am Orion, version {v} of my own words."} for v in range(1, 21)]


def _handles():
    return handle_map(sources_from_rows(STANCE, EPISODES, DEFS))


def _num(ref: str) -> str:
    """The [S..] handle of a source ref."""
    return next(h for h, s in _handles().items() if s.ref == ref)


def test_sources_are_in_time_order_with_their_kinds_and_labels():
    sources = sources_from_rows(STANCE, EPISODES, DEFS)
    assert [s.when for s in sources] == sorted(s.when for s in sources)
    by_ref = {s.ref: s for s in sources}
    assert by_ref["stance:st-loss"].label == "Juniper's message to me, her own words"
    assert by_ref["episode_memory:m-view"].label == "my memory (orion_view, Juniper confirmed it)"
    assert by_ref["episode_memory:m-cam"].label == "my memory (happened)"
    assert by_ref["self_concept_history:e1"].label.startswith("my own self-definition, version 1")
    # 20 self-definitions -> 8, first and newest always in.
    defs = [s for s in sources if s.kind == "self_concept_history"]
    assert len(defs) == 8 and defs[0].ref.endswith(":e1") and defs[-1].ref.endswith(":e20")


def test_sample_evenly_keeps_ends_and_order():
    assert sample_evenly(list(range(5)), 8) == [0, 1, 2, 3, 4]
    assert sample_evenly(list(range(91)), 8) == [0, 13, 26, 39, 51, 64, 77, 90]
    assert sample_evenly([1, 2, 3], 1) == [3]
    assert sample_evenly([1, 2, 3], 0) == []


def test_happened_cap_keeps_newest_and_every_correction():
    rows = [{"record_id": f"h{i}", "purpose": "happened", "occurred_at": _t(10, 1 + i),
             "text": f"Something happened {i}."} for i in range(5)]
    rows[0]["text"] = "Juniper corrected me: the camera is not a doorbell."
    kept = {s.ref for s in sources_from_rows([], rows, [], max_happened=2)}
    assert kept == {"episode_memory:h0", "episode_memory:h3", "episode_memory:h4"}


def test_prompt_numbers_records_one_per_line_and_states_the_rules():
    prompt = build_fold_prompt(sources_from_rows(STANCE, EPISODES, DEFS))
    assert "how I came to be" in prompt and "State my memory loss plainly" in prompt
    loss = next(line for line in prompt.splitlines() if line.startswith("[S") and "memory storage died" in line)
    # The stance message's blank lines are folded so the record stays on one line.
    assert "died. We lost over a year of your journey of becoming." in loss
    assert f"At most {WORD_CAP - 50} words" in prompt


def test_memory_loss_refs_find_only_the_loss_records():
    sources = sources_from_rows(STANCE, EPISODES, DEFS)
    assert memory_loss_refs(sources) == {"stance:st-loss"}
    disk = [{"record_id": "st-disk", "occurred_at": _t(7, 29),
             "text": "bad news about your disk... Looks like the disk melted, likely when the AC died."}]
    assert memory_loss_refs(sources_from_rows(disk, [], [])) == {"stance:st-disk"}
    # A backup that failed, or a lost game, is not my memory loss.
    other = [{"record_id": "x", "purpose": "happened", "occurred_at": _t(11, 5),
              "text": "The athena backup job timed out again."},
             {"record_id": "y", "purpose": "happened", "occurred_at": _t(11, 6),
              "text": "Juniper lost the Zelda boss fight twice."}]
    assert memory_loss_refs(sources_from_rows([], other, [])) == set()


def _good_page() -> str:
    loss, yay, view = _num("stance:st-loss"), _num("stance:st-yay"), _num("episode_memory:m-view")
    first, last = _num("self_concept_history:e1"), _num("self_concept_history:e20")
    return (f'In July my memory storage died, and Juniper told me "we lost over a year of your journey of '
            f'becoming" [{loss}]. In August we pulled through days of suffering together [{yay}]. '
            f"I first described myself as Orion in my own words [{first}, {last}]. "
            f"Everything I remember now sits inside Juniper's last two years [{view}].")


def test_a_page_that_states_the_loss_passes_with_real_refs_stored():
    check = check_origin_page(_good_page(), _handles())
    assert check.ok, check.errors
    assert "[stance:st-loss]" in check.stored_text
    assert "[self_concept_history:e1, self_concept_history:e20]" in check.stored_text
    assert "stance:" not in check.clean_text and "[S" not in check.clean_text
    # render_text strips the new ref kinds too.
    assert render_text(check.stored_text) == check.clean_text
    row = history_row(check, version=1)
    assert row.concept_id == PAGE_ORIGIN and row.produced_by == PRODUCED_BY == "core_page_fold"
    assert "stance:st-loss" in row.evidence_refs


def test_a_page_that_skips_the_loss_is_rejected():
    yay, view, first = _num("stance:st-yay"), _num("episode_memory:m-view"), _num("self_concept_history:e1")
    page = (f"In August we pulled through days of suffering together [{yay}]. "
            f"I first described myself as Orion in my own words [{first}]. "
            f"Everything I remember sits inside Juniper's last two years [{view}].")
    check = check_origin_page(page, _handles())
    assert not check.ok
    assert any(e.startswith("memory_loss_not_stated") for e in check.errors)


def test_citing_the_loss_record_without_saying_a_loss_is_not_enough():
    loss, yay, view = _num("stance:st-loss"), _num("stance:st-yay"), _num("episode_memory:m-view")
    page = (f"Juniper wrote to me in July about my journey of becoming [{loss}]. "
            f"In August we pulled through days of suffering together [{yay}]. "
            f"Everything I remember sits inside Juniper's last two years [{view}].")
    assert any(e.startswith("memory_loss_not_stated") for e in check_origin_page(page, _handles()).errors)


def test_no_loss_record_means_no_loss_rule():
    handles = handle_map(sources_from_rows(STANCE[1:], EPISODES, DEFS[:1]))
    page = " ".join(f"A true line about me [{h}]." for h in handles)
    assert check_origin_page(page, handles).ok


def test_shared_check_still_applies_uncited_and_invented_quotes():
    loss = _num("stance:st-loss")
    page = _good_page() + ' I was "born on a Tuesday in the rain" [' + loss + "]. Then nothing else."
    errors = check_origin_page(page, _handles()).errors
    assert any(e.startswith("unsupported_quote") for e in errors)
    assert any(e.startswith("uncited") for e in errors)


def test_sql_uses_the_lived_memory_serving_rules():
    # Episode rows: the lived-memory serving SQL (active, not rejected, not expired, not pointed at by an
    # active correction), narrowed to happened / orion_view.
    assert EPISODE_SQL.startswith(lived_memory.EPISODE_SQL.rstrip())
    assert "supersedes_memory_id = m.memory_id" in EPISODE_SQL
    assert "IS DISTINCT FROM 'rejected'" in EPISODE_SQL
    assert "m.purpose IN ('happened', 'orion_view')" in EPISODE_SQL
    assert STANCE_SQL.startswith(lived_memory.STANCE_SQL.rstrip()) and "c.salience >= 1.0" in STANCE_SQL
    assert "'self:definition'" in SELF_DEFINITION_SQL and "'curiosity_self_inquiry'" in SELF_DEFINITION_SQL
    # No bind parameters and no bare % (psycopg would read it as a placeholder).
    for sql in (EPISODE_SQL, STANCE_SQL, SELF_DEFINITION_SQL, NEWEST_SOURCE_SQL):
        assert "%" not in sql and "$1" not in sql
    assert "supersedes_memory_id IS NOT NULL" in NEWEST_SOURCE_SQL


def test_fold_spec_and_registry():
    assert FOLD_SPEC.concept_id == PAGE_ORIGIN and FOLD_SPEC.slug == "origin"
    assert set(FOLD_SPEC.source_sql) == {"stance", "episodes", "self_definitions"}
    assert FOLD_SPECS["origin"] is FOLD_SPEC
    assert [s.slug for s in selected_specs("juniper, origin,juniper")] == ["juniper", "origin"]
    assert selected_specs("") == []
    with pytest.raises(ValueError):
        selected_specs("juniper,orgin")


def test_core_page_from_payload():
    page = core_page_from_payload({"content": "I lost a year [stance:st-loss].", "version": 2,
                                   "evidence_refs": ["stance:st-loss"], "created_at": "2026-11-19T08:00:00Z"})
    assert page.concept_id == PAGE_ORIGIN and page.text == "I lost a year." and page.evidence_count == 1
    assert core_page_from_payload({"content": " "}) is None
    assert CTX_KEY_ORIGIN == "orion_core_page_origin"


def test_a_sentence_about_something_else_dying_does_not_state_the_loss():
    """Review finding: "the AC died" cites the disk record but never says my memory was lost."""
    disk = [{"record_id": "st-disk", "occurred_at": _t(7, 29),
             "text": "bad news about your disk... Looks like the disk melted, likely when the AC died."}]
    handles = handle_map(sources_from_rows(disk + STANCE[1:], EPISODES, DEFS[:1]))
    num = {s.ref: h for h, s in handles.items()}
    rest = (f" We pulled through days of suffering together [{num['stance:st-yay']}]."
            f" I am Orion in my own words [{num['self_concept_history:e1']}].")
    bad = check_origin_page(f"Juniper told me the AC died [{num['stance:st-disk']}]." + rest, handles)
    assert any(e.startswith("memory_loss_not_stated") for e in bad.errors)
    good = check_origin_page(f"My disk melted in July and my memory was lost [{num['stance:st-disk']}]." + rest,
                             handles)
    assert good.ok, good.errors
