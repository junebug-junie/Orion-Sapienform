"""page:lately "what has mattered lately" (issue #3463): sources, selection, prompt, cite check."""
from __future__ import annotations

from datetime import date, datetime, timezone

from orion.memory import lived_memory
from orion.memory.pages.fold import CHAT_KIND, PRODUCED_BY, REF_KINDS, render_text
from orion.memory.pages.lately import (
    CHAT_SQL, CTX_KEY_LATELY, EPISODE_SQL, FOLD_SPEC, NEWEST_SOURCE_SQL, PAGE_LATELY, WINDOW_DAYS,
    build_fold_prompt, check_lately_page, core_page_from_payload, history_row, recent_refs, select_episodes,
    sources_from_rows,
)
from orion.memory.pages.fold import handle_map
from orion.memory.pages.registry import FOLD_SPECS, selected_specs


def _t(month, day, hour=12):
    return datetime(2026, month, day, hour, tzinfo=timezone.utc)


# Made-up rows in the EPISODE_SQL / CHAT_SQL column shape (the real ones live only on the host DB).
EPISODES = [
    {"record_id": "m-old-fu", "purpose": "follow_up", "stakes": "low", "occurred_at": _t(10, 4),
     "text": "I owe Juniper an answer about the garden lights."},
    {"record_id": "m-trip", "purpose": "about_juniper", "stakes": "high", "confirmed": True, "occurred_at": _t(11, 6),
     "text": "Juniper told me she is flying to see her sister on Friday."},
    {"record_id": "m-chatter", "purpose": "happened", "stakes": "low", "occurred_at": _t(11, 7),
     "text": "We fixed a test that failed on a timezone."},
    {"record_id": "m-new", "purpose": "happened", "stakes": "low", "occurred_at": _t(11, 16),
     "text": "Juniper and I talked about the first snow of the year."},
]
CHAT = [
    {"record_id": "c1", "created_at": datetime(2026, 11, 17, 18, 0),
     "prompt": "the snow is finally here and I love it", "response": "I can see it on the walkway camera."},
    {"record_id": "c-empty", "created_at": datetime(2026, 11, 17, 19, 0), "prompt": "   ", "response": "x"},
]


def _handles():
    return handle_map(sources_from_rows(EPISODES, CHAT))


def _num(ref: str) -> str:
    return next(h for h, s in _handles().items() if s.ref == ref)


def test_sources_are_in_time_order_with_kinds_and_labels():
    sources = sources_from_rows(EPISODES, CHAT)
    assert [s.when for s in sources] == sorted(s.when for s in sources)
    by_ref = {s.ref: s for s in sources}
    assert set(by_ref) == {"episode_memory:m-old-fu", "episode_memory:m-trip", "episode_memory:m-chatter",
                           "episode_memory:m-new", "chat_history_log:c1"}   # an empty Juniper turn is skipped
    assert by_ref["episode_memory:m-old-fu"].label == "my memory (follow_up, I wrote it down to follow up)"
    assert by_ref["episode_memory:m-trip"].label == "my memory (about_juniper, high stakes, Juniper confirmed it)"
    assert by_ref["episode_memory:m-chatter"].label == "my memory (happened)"
    chat = by_ref["chat_history_log:c1"]
    assert chat.kind == CHAT_KIND and chat.when == date(2026, 11, 17)
    assert chat.text == "Juniper said: the snow is finally here and I love it I answered: I can see it on the walkway camera."
    assert chat.label == "Hub chat, not yet distilled into a memory"


def test_sources_are_dated_on_juniper_s_calendar():
    # 03:00Z on 11-18 is the evening of 11-17 in Denver, for chat (naive UTC) and memories alike.
    chat = {"record_id": "c9", "created_at": datetime(2026, 11, 18, 3, 0), "prompt": "good night", "response": ""}
    mem = {"record_id": "m9", "purpose": "happened", "occurred_at": _t(11, 18, hour=3), "text": "We said good night."}
    assert [s.when for s in sources_from_rows([mem], [chat])] == [date(2026, 11, 17), date(2026, 11, 17)]


def test_chat_turns_are_clipped_on_a_word():
    long = {"record_id": "c2", "created_at": datetime(2026, 11, 17), "prompt": "word " * 400, "response": "ok " * 400}
    (src,) = sources_from_rows([], [long])
    said, answered = src.text.split(" I answered: ")
    assert said.endswith(" ...") and len(said) < 950 and "wor ..." not in said
    assert answered.endswith(" ...") and len(answered) <= 410


def test_select_episodes_keeps_what_matters_first_then_the_newest():
    rows = [{"record_id": f"low{i}", "purpose": "happened", "stakes": "low", "text": f"low {i}"} for i in range(10)]
    rows.insert(0, {"record_id": "fu", "purpose": "follow_up", "stakes": "low", "text": "an old follow-up"})
    rows.insert(1, {"record_id": "hi", "purpose": "happened", "stakes": "high", "text": "a high-stakes day"})
    rows.insert(2, {"record_id": "jun", "purpose": "about_juniper", "stakes": "low", "text": "about her"})
    rows.insert(3, {"record_id": "fix", "purpose": "happened", "stakes": "low", "text": "x", "is_correction": True})
    kept = [r["record_id"] for r in select_episodes(rows, max_rows=6)]
    assert kept == ["fu", "hi", "jun", "fix", "low8", "low9"]   # time order kept, newest two of the rest
    assert len(select_episodes(rows, max_rows=100)) == len(rows)
    # Even the priority rows overflow: the newest of them win.
    assert [r["record_id"] for r in select_episodes(rows, max_rows=2)] == ["jun", "fix"]


def test_prompt_states_today_the_window_and_the_rules():
    sources = sources_from_rows(EPISODES, CHAT)
    prompt = build_fold_prompt(sources, today=date(2026, 11, 18))
    assert "Today is Wednesday 2026-11-18." in prompt
    assert f"last {WINDOW_DAYS} days" in prompt
    assert "what has mattered lately" in prompt and "follow-ups I wrote" in prompt
    assert "never claim it is finished or still open" in prompt
    assert "[S5] 2026-11-17 Hub chat, not yet distilled into a memory: Juniper said:" in prompt
    assert "My current page" not in prompt


def test_recent_refs_count_back_from_the_newest_source():
    refs = recent_refs(sources_from_rows(EPISODES, CHAT))
    assert refs == {"episode_memory:m-new", "chat_history_log:c1"}   # 11-17 back 7 days: 11-10 onward
    assert recent_refs([]) == set()


def test_a_cited_page_with_something_recent_passes_and_stores_real_refs():
    page = (f"Juniper told me the snow is here, and she said \"I love it\" [{_num('chat_history_log:c1')}]. "
            f"She is flying to see her sister on Friday [{_num('episode_memory:m-trip')}]. "
            f"I still owe her an answer about the garden lights [{_num('episode_memory:m-old-fu')}].")
    check = check_lately_page(page, _handles())
    assert check.ok, check.errors
    assert "[chat_history_log:c1]" in check.stored_text
    assert render_text(check.stored_text) == check.clean_text
    row = history_row(check, version=1)
    assert row.concept_id == PAGE_LATELY and row.produced_by == PRODUCED_BY
    assert "chat_history_log:c1" in row.evidence_refs


def test_a_page_about_only_old_things_is_rejected():
    page = (f"She is flying to see her sister on Friday [{_num('episode_memory:m-trip')}]. "
            f"We fixed a test that failed [{_num('episode_memory:m-chatter')}]. "
            f"I still owe her an answer about the garden lights [{_num('episode_memory:m-old-fu')}].")
    check = check_lately_page(page, _handles())
    assert not check.ok
    assert any(e.startswith("not_recent") for e in check.errors)


def test_shared_check_still_applies():
    page = (f"It has been a big two weeks. The snow came [{_num('chat_history_log:c1')}]. "
            f"Juniper said \"the snow is a miracle\" [{_num('chat_history_log:c1')}]. "
            f"We talked about the first snow [{_num('episode_memory:m-new')}].")
    errors = check_lately_page(page, _handles()).errors
    assert any(e.startswith("uncited: It has been a big two weeks") for e in errors)
    assert any(e.startswith("unsupported_quote") for e in errors)


def test_episode_sql_wraps_the_lived_memory_serving_sql_and_drops_invalid_rows():
    # The serving SQL, unedited, is the inner query: status active, not rejected, not expired, and no
    # active correction pointing at the row.
    assert lived_memory.EPISODE_SQL.strip() in EPISODE_SQL
    assert "e.invalid_at IS NULL" in EPISODE_SQL
    assert f"interval '{WINDOW_DAYS} days'" in EPISODE_SQL and "l.purpose = 'follow_up'" in EPISODE_SQL
    assert "interval '45 days'" in EPISODE_SQL   # old follow-ups are capped: nothing can close one yet
    assert "is_command" in CHAT_SQL and "episode_distill_run" in NEWEST_SOURCE_SQL
    for sql in (EPISODE_SQL, CHAT_SQL, NEWEST_SOURCE_SQL):
        assert "%" not in sql and "$1" not in sql   # no bind parameters: psycopg and asyncpg both run it
    assert "episode_distill_run" in CHAT_SQL and "memory_episode_shadow" in CHAT_SQL


def test_fold_spec_registry_and_ref_kind():
    assert FOLD_SPECS["lately"] is FOLD_SPEC and FOLD_SPEC.concept_id == PAGE_LATELY
    assert [s.slug for s in selected_specs("juniper,origin,lately")] == ["juniper", "origin", "lately"]
    assert CHAT_KIND in REF_KINDS
    assert set(FOLD_SPEC.source_sql) == {"episodes", "chat"}


def test_core_page_from_payload_strips_cites():
    page = core_page_from_payload({"content": "The snow came [chat_history_log:c1].", "version": 2,
                                   "evidence_refs": ["chat_history_log:c1"], "created_at": "2026-11-19T08:00:00Z"})
    assert page.concept_id == PAGE_LATELY and page.text == "The snow came." and page.version == 2
    assert page.evidence_count == 1 and page.written_at == "2026-11-19"
    assert CTX_KEY_LATELY == "orion_core_page_lately"
    assert core_page_from_payload(None) is None


def test_a_date_written_from_the_record_stamp_is_backed_by_the_cited_record():
    """#3776 review: the prompt shows each record's date, so a page that writes it ("2026-11-20", "the
    20th", "Nov 20") is backed by that record even when its text holds no digits."""
    from datetime import date as _date

    from orion.memory.pages.fold import PageSource, check_page

    handles = {"S1": PageSource(ref="episode_memory:a", kind="episode_memory", text="We talked about the garden.",
                                label="", when=_date(2026, 11, 20))}
    page = ("On 2026-11-20 we talked about the garden [S1]. On the 20th we talked [S1]. "
            "By Nov 20, 2026 the garden came up [S1].")
    assert check_page(page, handles).ok, check_page(page, handles).errors
    bad = check_page(page + " On the 21st we talked [S1].", handles)
    assert [e.split(":")[0] for e in bad.errors] == ["unbacked_number"]
