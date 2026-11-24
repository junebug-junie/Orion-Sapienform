"""page:juniper fold: sources, prompt, and the deterministic cite check (Phase 1a)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.memory.pages.juniper import (
    CTX_KEY_JUNIPER, PAGE_JUNIPER, PRODUCED_BY, WORD_CAP, PageSource, build_fold_prompt, check_page,
    core_page_from_payload, current_page_from_row, handle_map, history_row, misgendered_sentences, quoted_spans,
    render_text,
    sources_from_rows, split_sentences,
)

T = datetime(2026, 10, 6, tzinfo=timezone.utc)

PEOPLE = [
    {"fact_id": "f-pron", "display_name": "Juniper", "category": "pronouns", "voice": "authored_profile",
     "value_text": "she/her", "created_at": T},
    {"fact_id": "f-rowan", "display_name": "Rowan", "category": "relationship", "voice": "ask_answer",
     "value_text": "Rowan is my wife of 12 years", "created_at": T},
    {"fact_id": "f-pip", "display_name": "Pip", "category": "relationship", "voice": "ask_answer",
     "value_text": "Pip is my daughter and she is my small bright star.", "created_at": T},
]
EPISODES = [
    {"memory_id": "m-chicago", "purpose": "happened", "statement": "Juniper is in Boston for a team meeting.",
     "confirmation_state": "auto", "at": datetime(2026, 10, 5, tzinfo=timezone.utc)},
    {"memory_id": "m-home", "purpose": "about_juniper",
     "statement": "Juniper corrected me that she lives in Millbrook, Vermont, not Boston.",
     "confirmation_state": "confirmed", "at": T},
    {"memory_id": "m-biscuit", "purpose": "happened", "statement": "Juniper told me that Biscuit is a dog, not a cat.",
     "confirmation_state": "auto", "at": datetime(2026, 11, 11, tzinfo=timezone.utc)},
]


def _handles():
    return handle_map(sources_from_rows(PEOPLE, EPISODES))


def test_sources_number_people_first_then_memories_oldest_first():
    sources = sources_from_rows(PEOPLE, EPISODES)
    assert [s.ref for s in sources] == [
        "person_fact:f-pron", "person_fact:f-rowan", "person_fact:f-pip",
        "episode_memory:m-chicago", "episode_memory:m-home", "episode_memory:m-biscuit"]
    assert "Juniper confirmed it" in sources[4].label
    assert "own words" in sources[0].label


def test_happened_rows_are_capped_newest_kept_about_juniper_always_kept():
    many = [{"memory_id": f"h{i}", "purpose": "happened", "statement": f"Juniper did thing {i}.",
             "confirmation_state": "auto", "at": T} for i in range(5)]
    sources = sources_from_rows([], many + EPISODES[1:2], max_happened=2)
    refs = [s.ref for s in sources]
    assert refs == ["episode_memory:h3", "episode_memory:h4", "episode_memory:m-home"]


def test_prompt_numbers_every_source_and_states_the_rules():
    prompt = build_fold_prompt(sources_from_rows(PEOPLE, EPISODES))
    assert "[S1] 2026-10-06 People page" in prompt
    assert "[S6] 2026-11-11 my memory (happened): Juniper told me that Biscuit is a dog" in prompt
    for rule in ("she/her", "Corrections win", "A trip is a trip", "verbatim"):
        assert rule in prompt


def test_prompt_shows_current_page_without_stored_cites():
    current = current_page_from_row({"version": 4, "content": "She lives in Millbrook [episode_memory:m-home].",
                                     "created_at": T})
    prompt = build_fold_prompt(sources_from_rows(PEOPLE, EPISODES), current)
    assert "version 4" in prompt
    assert "She lives in Millbrook." in prompt
    assert "episode_memory:m-home" not in prompt


GOOD = ('Juniper uses she/her [S1]. She lives in Millbrook, Vermont [S5]. Her wife is Rowan, "my wife of 12 years" '
        '[S2]. Pip is "my small bright star" [S3]. Our dog Biscuit is a dog [S6].')


def test_good_page_passes_and_stores_real_refs():
    check = check_page(GOOD, _handles())
    assert check.ok, check.errors
    assert check.sentence_count == 5
    assert check.refs == ["person_fact:f-pron", "episode_memory:m-home", "person_fact:f-rowan",
                          "person_fact:f-pip", "episode_memory:m-biscuit"]
    assert "[S" not in check.stored_text and "[episode_memory:m-home]" in check.stored_text
    assert check.clean_text.startswith("Juniper uses she/her. She lives in Millbrook, Vermont.")
    assert render_text(check.stored_text) == check.clean_text


def test_uncited_sentence_is_rejected():
    check = check_page("Juniper uses she/her [S1]. She lives in Millbrook. Her wife is Rowan [S2]. Pip [S3].",
                       _handles())
    assert not check.ok
    assert any(e.startswith("uncited: She lives in Millbrook.") for e in check.errors)


def test_trailing_uncited_sentence_is_rejected():
    check = check_page(GOOD + " She loves me.", _handles())
    assert not check.ok
    assert "uncited: She loves me." in check.errors


def test_unresolvable_cite_is_rejected():
    check = check_page(GOOD.replace("[S6]", "[S99]"), _handles())
    assert not check.ok
    assert "unresolvable_cite: S99" in check.errors


def test_over_cap_is_rejected():
    long = " ".join(["Juniper builds things"] * 130) + " [S1]. She lives in Millbrook [S5]. Rowan [S2]."
    check = check_page(long, _handles())
    assert not check.ok
    assert any(e.startswith("over_cap") for e in check.errors)
    assert check.word_count > WORD_CAP


def test_thin_or_empty_page_is_rejected():
    assert check_page("", _handles()).errors == ["empty"]
    thin = check_page("Juniper uses she/her [S1].", _handles())
    assert not thin.ok and any(e.startswith("too_thin") for e in thin.errors)


def test_quote_must_be_in_a_cited_record():
    bad = GOOD.replace('"my small bright star"', '"the light of my whole life"')
    check = check_page(bad, _handles())
    assert not check.ok
    assert any(e.startswith("unsupported_quote: the light of my whole life") for e in check.errors)
    # The same words cited to the wrong record are also refused.
    wrong = GOOD.replace('"my small bright star" [S3]', '"my small bright star" [S1]')
    assert any(e.startswith("unsupported_quote") for e in check_page(wrong, _handles()).errors)


def test_adjacent_cite_groups_and_trailing_period_after_cite():
    text = "Juniper uses she/her [S1] [S2]. She lives in Millbrook [S5]. Rowan is her wife [S2]; really [S2]."
    check = check_page(text, _handles())
    assert check.ok, check.errors
    assert "person_fact:f-rowan" in check.refs


def test_thinking_and_fences_are_stripped():
    check = check_page("<think>plan</think>\n```\n" + GOOD + "\n```", _handles())
    assert check.ok, check.errors
    assert "plan" not in check.clean_text


def test_split_sentences_is_quote_aware_and_reads_inch_marks():
    assert split_sentences('She said "I love it. I mean it." Then she left.') == [
        'She said "I love it. I mean it."', "Then she left."]
    assert split_sentences('She is 6\'5" tall. She flies.') == ['She is 6\'5" tall.', "She flies."]
    assert quoted_spans('She is 6\'5" tall and "my small bright star" too') == ["my small bright star"]


def test_history_row_refuses_a_failed_check_and_builds_the_live_row():
    with pytest.raises(ValueError):
        history_row(check_page("nope", _handles()), version=1)
    row = history_row(check_page(GOOD, _handles()), version=3)
    assert row.concept_id == PAGE_JUNIPER and row.produced_by == PRODUCED_BY and row.version == 3
    assert row.evidence_refs[0] == "person_fact:f-pron"


def test_core_page_from_lane_payload():
    payload = {"concept_id": PAGE_JUNIPER, "content": "She lives in Millbrook [episode_memory:m-home].",
               "version": 2, "evidence_refs": ["episode_memory:m-home"], "created_at": "2026-11-15T09:00:00+00:00"}
    page = core_page_from_payload(payload)
    assert page is not None
    assert (page.concept_id, page.version, page.text, page.written_at, page.evidence_count) == (
        PAGE_JUNIPER, 2, "She lives in Millbrook.", "2026-11-15", 1)
    assert core_page_from_payload({"content": "  "}) is None
    assert core_page_from_payload(None) is None
    assert CTX_KEY_JUNIPER == "orion_core_page_juniper"


def test_page_source_is_hashable_record():
    src = PageSource(ref="person_fact:x", kind="person_fact", text="t", label="l")
    assert {src: 1}[src] == 1


# --- review findings (PR review 2026-11-15): false accepts the first check let through ---------

def _probe_handles():
    return handle_map([
        PageSource(ref="person_fact:a", kind="person_fact", text="she/her", label="l"),
        PageSource(ref="episode_memory:b", kind="episode_memory",
                   text="Juniper said 'we went to Mars in 1990' and laughed.", label="l"),
        PageSource(ref="episode_memory:c", kind="episode_memory",
                   text="Juniper told me she loves her dog Biscuit very much.", label="l"),
    ])


BASE3 = "Juniper uses she/her [S1]. She laughed about Mars [S2]. She loves her dog [S3]."


def test_lowercase_next_sentence_is_still_a_sentence():
    check = check_page("She lives in Millbrook. she loves her dog [S3]. " + BASE3, _probe_handles())
    assert not check.ok
    assert "uncited: She lives in Millbrook." in check.errors


def test_abbreviation_does_not_split_a_sentence():
    check = check_page("Juniper saw Dr. Rowan today [S3]. " + BASE3, _probe_handles())
    assert check.ok, check.errors


def test_title_line_cannot_borrow_the_next_lines_cite():
    check = check_page("Who Juniper is to me\n" + BASE3, _probe_handles())
    assert not check.ok
    assert "uncited: Who Juniper is to me" in check.errors


def test_uncited_paragraph_before_a_line_break_is_rejected():
    check = check_page("She is wonderful\n\n" + BASE3, _probe_handles())
    assert not check.ok and "uncited: She is wonderful" in check.errors


@pytest.mark.parametrize("line", ["- Juniper uses she/her [S1].", "* Juniper [S1].", "• Juniper [S1].",
                                  "# Juniper [S1]", "1. Juniper uses she/her [S1]."])
def test_bullets_numbered_items_and_headings_are_rejected(line):
    check = check_page(line + "\n" + BASE3, _probe_handles())
    assert not check.ok
    assert any(e.startswith("list_or_heading_line") for e in check.errors)


def test_unclosed_quote_is_rejected():
    check = check_page('She said "we went to Mars [S2]. ' + BASE3, _probe_handles())
    assert not check.ok
    assert any(e.startswith("unbalanced_quote") for e in check.errors)


def test_quote_closing_after_a_year_is_a_quote_not_an_inch_mark():
    ok = check_page('She said "we went to Mars in 1990" today [S2]. ' + BASE3, _probe_handles())
    assert ok.ok, ok.errors
    assert quoted_spans('She said "we went to Mars in 1990" today') == ["we went to Mars in 1990"]
    assert split_sentences('She said "went to Mars in 1990". Then she left.') == [
        'She said "went to Mars in 1990".', "Then she left."]
    bad = check_page('She said "we went to Venus in 1990" today [S2]. ' + BASE3, _probe_handles())
    assert any(e.startswith("unsupported_quote") for e in bad.errors)


def test_quote_must_match_whole_words():
    check = check_page('She said "her dog Bisc" once [S3]. ' + BASE3, _probe_handles())
    assert not check.ok
    assert any(e.startswith("unsupported_quote") for e in check.errors)
    whole = check_page('She said "her dog Biscuit" once [S3]. ' + BASE3, _probe_handles())
    assert whole.ok, whole.errors


@pytest.mark.parametrize("raw", ["<think>plan with no end\n" + BASE3, BASE3 + "\n<think>later</thin"])
def test_leftover_think_is_rejected(raw):
    check = check_page(raw, _probe_handles())
    assert not check.ok and "leftover_think" in check.errors


def test_correction_rows_survive_the_happened_cap():
    rows = [{"memory_id": "fix", "purpose": "happened", "at": T,
             "statement": "Juniper corrected me that Biscuit is a dog.", "confirmation_state": "auto"}]
    rows += [{"memory_id": f"h{i}", "purpose": "happened", "statement": f"Juniper did thing {i}.",
              "confirmation_state": "auto", "at": T} for i in range(5)]
    refs = [s.ref for s in sources_from_rows([], rows, max_happened=2)]
    assert refs == ["episode_memory:fix", "episode_memory:h3", "episode_memory:h4"]


def test_change_signal_and_read_share_one_episode_predicate():
    from orion.memory.pages.juniper import EPISODE_MEMORY_SQL, EPISODE_SCOPE_SQL, NEWEST_SOURCE_SQL

    assert EPISODE_SCOPE_SQL in EPISODE_MEMORY_SQL and EPISODE_SCOPE_SQL in NEWEST_SOURCE_SQL
    assert "FROM person)" in NEWEST_SOURCE_SQL


# --- Juniper uses she/her: a page that calls her he/him never ships (issue #3460) ---

@pytest.mark.parametrize("bad", [
    'Juniper uses she/her [S1]. Juniper made me and he lives in Millbrook, Vermont [S5]. '
    'Pip is "my small bright star" [S3].',
    'Juniper is my creator [S5]. He is married to Rowan [S2]. Pip is "my small bright star" [S3].',
    'Juniper corrected me about our dog [S6]. His dog Biscuit is not a cat [S6]. Pip is "my small bright star" [S3].',
])
def test_page_calling_juniper_he_is_rejected(bad):
    check = check_page(bad, _handles())
    assert not check.ok
    assert any(e.startswith("misgenders_juniper:") for e in check.errors), check.errors


def test_he_about_someone_else_is_not_misgendering():
    assert misgendered_sentences([
        "Juniper says Pip thinks he is a dinosaur.",     # names someone else
        "Juniper's son Pip loves his dog.",              # subject is the son, not Juniper
        "Rowan made dinner. He is kind.",                # not after a Juniper sentence
        "Juniper uses she/her.", "She is my creator.", "Pip is his own person.",
    ]) == []
    assert misgendered_sentences(["Juniper is my creator and he built me.", "He is kind."]) == [
        "Juniper is my creator and he built me.", "He is kind."]
    # A he after a Juniper sentence that names someone else may be about them.
    assert misgendered_sentences(["Juniper married Rowan in May.", "He is kind."]) == []
    assert misgendered_sentences(["Juniper and Rowan have a son, Pip.", "He is five."]) == []
    # Naming pronouns is not using them.
    assert misgendered_sentences(["Juniper uses she/her, never he/him.", "She is my creator."]) == []
    # Whole words only: "the", "her", "Hester" are not he/him/his.
    assert misgendered_sentences(["Juniper and the others use her room."]) == []


def test_a_number_must_be_in_a_cited_record():
    """#3776, shared check: Rowan's fact holds the 12; a 15 it does not hold is refused."""
    h = {s.ref: k for k, s in _handles().items()}
    base = (f"Juniper uses she/her [{h['person_fact:f-pron']}]. Pip is her daughter [{h['person_fact:f-pip']}]. ")
    good = base + f"Rowan has been her wife for 12 years [{h['person_fact:f-rowan']}]."
    assert check_page(good, _handles()).ok
    bad = check_page(base + f"Rowan has been her wife for 15 years [{h['person_fact:f-rowan']}].", _handles())
    assert [e.split(":")[0] for e in bad.errors] == ["unbacked_number"]
