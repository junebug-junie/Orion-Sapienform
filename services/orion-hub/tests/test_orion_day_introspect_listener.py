"""Hub orion_day responder: rereading a letter, its claim check and citations, empty vs unknown.

The fake pool answers `orion/orion_day/store.py`'s own SQL with rows built from a real
`OrionDayLetterV1` (the fixture day plus planted claims), so the responder runs the real
letter_parts splitting, citation resolution and claim check underneath.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pytest

import scripts.orion_day_introspect_listener as ol
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.semantic_index import INDEX_LAG_MARGIN, IndexPass, SearchConfig
from orion.introspect.tests.fake_chroma import FakeChroma
from orion.introspect.transport import REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.orion_day import store
from orion.orion_day.letter_parts import split_carry, split_note
from orion.orion_day.tests import fixtures as fx
from orion.schemas.introspect import IntrospectRequestV1, IntrospectResultV1, IntrospectToolBindingV1

BINDING = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=False,
)
SEARCH = SearchConfig(chroma_url="http://c", embed_url="http://embed.test/embedding",
                      collection=ol.SEARCH_COLLECTION, min_similarity=0.65)
DAY = fx.LETTER_DATE.isoformat()
MCP_BUDGET = 12000


def _letters():
    first = fx.reread_letter()
    older_material = first.material.model_copy(update={"letter_date": date(2026, 9, 28), "chat_compactor": None})
    older = fx.reread_letter(letter_date=date(2026, 9, 28), material=older_material,
                             created_at=first.created_at - timedelta(days=1))
    return {first.letter_date: first, older.letter_date: older}


class _Conn:
    def __init__(self, letters, fail=False):
        self.letters, self.fail = letters, fail
        self.calls: list[str] = []
        self.readonly_flags: list[bool] = []

    @asynccontextmanager
    async def transaction(self, readonly=False):
        self.readonly_flags.append(readonly)
        yield

    def _row(self, letter):
        return letter.model_dump(mode="json") | {"letter_date": letter.letter_date}

    async def fetchrow(self, sql, *args):
        self.calls.append(sql)
        if self.fail:
            raise RuntimeError("password=hunter2 host=db")
        if sql == store.SELECT_LETTER_SQL:
            letter = self.letters.get(args[0])
            return self._row(letter) if letter else None
        if sql == store.SELECT_LATEST_LETTER_SQL:
            return self._row(self.letters[max(self.letters)]) if self.letters else None
        raise AssertionError(sql)

    async def fetch(self, sql, *args):
        self.calls.append(sql)
        assert sql == store.SELECT_LETTER_TEXTS_SQL
        return [{"letter_date": d, "note_md": l.note_md, "carry_forward_md": l.carry_forward_md,
                 "created_at": l.created_at} for d, l in sorted(self.letters.items())]

    async def fetchval(self, sql, *args):
        self.calls.append(sql)
        assert sql == store.LETTER_CREATED_SINCE_SQL
        return any(l.created_at >= args[0] for l in self.letters.values())


class _Pool:
    def __init__(self, letters=None, fail=False):
        self.conn = _Conn(_letters() if letters is None else letters, fail=fail)

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _listener(pool=None, search=None):
    pool = pool if pool is not None else _Pool()
    listener = ol.OrionDayIntrospectListener(
        pool_provider=lambda: pool, source_ref=ServiceRef(name="orion-hub"), search=search,
    )
    listener.bus = Bus()
    return listener, pool


def _ask(listener, args, *, reply_to=None, operation="orion_day", kind=REQUEST_KIND) -> IntrospectResultV1 | None:
    corr = uuid4()
    env = BaseEnvelope(
        kind=kind, correlation_id=corr, source=ServiceRef(name="orion-harness-governor"),
        reply_to=reply_to if reply_to is not None else f"{RESULT_PREFIX}{corr}",
        payload={"operation": operation, "binding": BINDING.model_dump(mode="json"), "args": args},
    )
    before = len(listener.bus.published)
    asyncio.run(listener.handle(env))
    if len(listener.bus.published) == before:
        return None
    channel, out = listener.bus.published[-1]
    assert channel == f"{RESULT_PREFIX}{corr}" and out.kind == RESULT_KIND
    result = IntrospectResultV1.model_validate(out.payload)
    assert len(json.dumps(out.payload, ensure_ascii=False)) <= MCP_BUDGET
    return result


def _paragraph(n):
    return next(p for p in split_note(fx.REREAD_NOTE) if p.kind == "paragraph" and p.index == n)


def _carry(n):
    return next(p for p in split_carry(fx.REREAD_CARRY) if p.kind == "carry" and p.index == n)


# --- trust -------------------------------------------------------------------------------------


def test_only_a_trusted_reply_subject_is_answered():
    listener, _ = _listener()
    assert _ask(listener, {}, reply_to="orion:somewhere:else") is None
    assert _ask(listener, {}, kind="dream.tool.request.v1") is None


def test_invalid_arguments_are_unknown_not_empty():
    listener, _ = _listener()
    out = _ask(listener, {"index": 3})
    assert not out.ok and out.error.startswith("invalid orion_day request")
    out = _ask(listener, {}, operation="curiosity")
    assert not out.ok and "not answered here" in out.error


# --- acceptance 1: outline and exact parts -----------------------------------------------------


def test_list_is_the_most_recent_letters_outline_with_counts(caplog):
    listener, pool = _listener()
    with caplog.at_level(logging.INFO, logger="orion-hub.orion_day_introspect"):
        out = _ask(listener, {})
    assert out.ok and out.total_available == 3
    note, carry, sections = out.items
    assert note.id == f"{DAY} note" and note.extra["paragraphs"] == 5 and note.extra["unnumbered"] == 3
    assert note.text.splitlines()[0].startswith("¶1 The dream organ woke at 06:35:27Z")
    assert [line.split(" ", 1)[0] for line in note.text.splitlines()] == ["¶1", "¶2", "¶3", "¶4", "¶5"]
    assert carry.id == f"{DAY} carry" and carry.extra["items"] == 3
    assert carry.extra["citations"] == 4 and carry.extra["citations_unresolved"] == 1
    assert carry.extra["unresolved_refs"] == ["reading_journal:nope-not-here"]
    assert carry.text.splitlines()[0].startswith("carry 1: - **The dream gap.**")
    assert sections.kind == "orion_day_record" and sections.epistemic_status == "record"
    counts = sections.extra["section_counts"]
    assert counts["curiosity"] == 4 and counts["dreams"] == 2 and counts["conversations"] == 1
    assert counts["reveries"] == 302  # 301 thoughts + 1 visual reverie
    assert all(ro for ro in pool.conn.readonly_flags) and pool.conn.readonly_flags
    assert "orion_day_introspect_answered" in caplog.text and "part=list" in caplog.text and "items=3" in caplog.text


def test_note_index_returns_exactly_that_paragraph():
    listener, _ = _listener()
    out = _ask(listener, {"letter_date": DAY, "part": "note", "index": 3})
    assert out.ok and out.total_available == 1
    [item] = out.items
    assert item.id == f"{DAY} ¶3" and item.text == _paragraph(3).text and not item.truncated
    assert item.kind == "orion_day_letter_part" and item.epistemic_status == "unsettled"
    assert item.extra["part"] == "note" and item.extra["index"] == 3


def test_note_without_index_returns_the_first_parts_up_to_limit_honestly():
    listener, _ = _listener()
    out = _ask(listener, {"part": "note", "limit": 2})
    assert out.ok and out.total_available == 5
    assert [i.id for i in out.items] == [f"{DAY} ¶1", f"{DAY} ¶2"]
    assert [i.text for i in out.items] == [_paragraph(1).text, _paragraph(2).text]


# --- acceptance 2: carry citations -------------------------------------------------------------


def test_carry_citation_resolves_to_the_runs_excerpt_and_a_planted_id_is_unresolved():
    listener, _ = _listener()
    out = _ask(listener, {"part": "carry_forward", "index": 1})
    [item] = out.items
    assert item.text == _carry(1).text
    [cite] = item.extra["citations"]
    assert cite["ref"] == f"curiosity:{fx.REREAD_RUN_ID}" and cite["resolved"] is True
    assert cite["excerpt"].startswith("Dream organ gap: The dream organ was silent for 260.6 hours")
    out = _ask(listener, {"part": "carry_forward", "index": 2})
    cites = {c["ref"]: c for c in out.items[0].extra["citations"]}
    assert cites["reading_journal:8c62d21d"]["resolved"] is True
    assert cites["reading_journal:nope-not-here"] == {"ref": "reading_journal:nope-not-here", "resolved": False,
                                                      "excerpt": None}
    assert out.items[0].extra["citations_unresolved"] == ["reading_journal:nope-not-here"]


# --- acceptance 3: claim check -----------------------------------------------------------------


def test_claim_check_names_the_record_holding_each_token_and_flags_planted_ones():
    listener, _ = _listener()
    one = _ask(listener, {"part": "note", "index": 1}).items[0]
    checks = {c["token"]: c for c in one.extra["claim_check"]}
    run_ref = f"curiosity:{fx.REREAD_RUN_ID}"
    for token in ("260.6", "06:35:27Z", "01:02:09Z"):
        assert checks[token]["found_in"] == [run_ref], token
    assert one.extra["claims_found"] == 3 and one.extra["claims_not_found"] == 0
    two = _ask(listener, {"part": "note", "index": 2}).items[0]
    checks = {c["token"]: c for c in two.extra["claim_check"]}
    assert checks["999.4"]["found_in"] == [] and checks["2557"]["found_in"] == []
    assert checks["0.72"]["found_in"] == [run_ref]
    assert two.extra["claims_not_found"] == 2


def test_a_token_found_in_many_records_lists_five_and_counts_the_rest():
    listener, _ = _listener()
    item = _ask(listener, {"part": "note", "index": 3}).items[0]
    [check] = item.extra["claim_check"]
    assert check["token"] == "the coalition keeps circling"
    assert len(check["found_in"]) == 5 and check["found_in_more"] == 296


# --- acceptance 4: unknown vs empty ------------------------------------------------------------


def test_unknown_date_is_a_not_found_tool_error_never_empty(caplog):
    listener, _ = _listener()
    with caplog.at_level(logging.INFO, logger="orion-hub.orion_day_introspect"):
        out = _ask(listener, {"letter_date": "2030-01-01"})
    assert not out.ok and out.items == []
    assert "unknown letter_date 2030-01-01" in out.error and "not found" in out.error
    assert "orion_day_introspect_not_found" in caplog.text


def test_no_letter_at_all_is_not_found():
    listener, _ = _listener(_Pool(letters={}))
    out = _ask(listener, {})
    assert not out.ok and "not found" in out.error


@pytest.mark.parametrize("args, phrase", [
    ({"part": "note", "index": 9}, "¶9: that letter has 5 numbered note paragraphs"),
    ({"part": "carry_forward", "index": 4}, "carry 4: that letter has 3 numbered carry items"),
])
def test_index_out_of_range_is_a_not_found_tool_error(args, phrase):
    listener, _ = _listener()
    out = _ask(listener, args)
    assert not out.ok and phrase in out.error and "unknown part" in out.error


def test_existing_letter_with_an_empty_section_is_ok_and_empty():
    listener, _ = _listener()
    out = _ask(listener, {"letter_date": "2026-09-28", "part": "section", "section": "conversations"})
    assert out.ok and out.items == [] and out.total_available == 0


def test_section_returns_its_records_with_their_refs_and_truncates_honestly():
    listener, _ = _listener()
    out = _ask(listener, {"part": "section", "section": "curiosity", "limit": 2})
    assert out.ok and out.total_available == 4 and len(out.items) == 2
    first = out.items[0]
    assert first.id == f"curiosity:{fx.REREAD_RUN_ID}" and first.kind == "orion_day_record"
    assert first.text.startswith("Dream organ gap: The dream organ was silent")
    assert first.epistemic_status == "unsettled" and first.extra["section"] == "curiosity"
    failed = _ask(listener, {"part": "section", "section": "curiosity"}).items[-1]
    assert failed.id.startswith("curiosity_failed:") and failed.epistemic_status == "record"
    dreams = _ask(listener, {"part": "section", "section": "dreams"})
    assert [i.id for i in dreams.items] == ["dream:20", "dream_offered:1"]
    assert dreams.items[0].occurred_at == fx.DREAM_NARRATIVES[0]["occurred_at"]


def test_postgres_failure_or_no_pool_is_unknown_and_leaks_nothing(caplog):
    listener, _ = _listener(_Pool(fail=True))
    with caplog.at_level(logging.WARNING):
        out = _ask(listener, {})
    assert not out.ok and out.error == ol.QUERY_UNAVAILABLE
    assert "hunter2" not in caplog.text and "hunter2" not in out.error
    listener = ol.OrionDayIntrospectListener(pool_provider=lambda: None, source_ref=ServiceRef(name="h"), search=None)
    listener.bus = Bus()
    assert _ask(listener, {}).error == ol.QUERY_UNAVAILABLE


def test_worst_case_results_fit_the_mcp_budget():
    quoted = 'quoted words with \\ backslashes and \t tabs ' * 4
    long_note = "\n\n".join(
        f'Paragraph {i}: "{quoted}" ' + " ".join(f"{n}.{i}" for n in range(100, 160))
        for i in range(1, 8)
    )
    long_carry = "\n".join(
        f"- item {i} " + " ".join(f"[curiosity:{fx.REREAD_RUN_ID}] [reading:missing{n:04d}]" for n in range(20))
        for i in range(1, 8)
    )
    letter = fx.reread_letter().model_copy(update={"note_md": long_note, "carry_forward_md": long_carry})
    listener, _ = _listener(_Pool(letters={letter.letter_date: letter}))
    for args in ({}, {"part": "note"}, {"part": "carry_forward"}, {"part": "note", "index": 1},
                 {"part": "carry_forward", "index": 1}, {"part": "section", "section": "reveries"},
                 {"part": "section", "section": "readings"}):
        out = _ask(listener, args)  # _ask asserts the serialized payload fits MCP_BUDGET
        assert out.ok, args
    one = _ask(listener, {"part": "note", "index": 1}).items[0]
    assert one.extra["claim_check_truncated"] is True and one.extra["claims_not_found"] > len(one.extra["claim_check"])


# --- search ------------------------------------------------------------------------------------


def _search_listener(chroma, letters=None):
    listener, pool = _listener(_Pool(letters=letters), search=SEARCH)
    listener.client_factory = chroma.client
    return listener, pool


def _index_all(listener, chroma):
    """Run index passes, storing each pass's upserts, until one confirms the index complete."""
    for _ in range(10):
        result = asyncio.run(listener.index_once())
        chroma.apply(listener.bus)
        listener.bus.published.clear()
        if result == IndexPass(indexed=0, pending=0):
            return
    raise AssertionError("index never caught up")


def test_index_docs_are_numbered_parts_keyed_by_ref():
    rows = asyncio.run(store.fetch_letter_texts(_Conn(_letters())))
    from scripts.orion_day_introspect import index_docs_from_rows

    docs = index_docs_from_rows(rows)
    ids = [d[0] for d in docs]
    assert f"{DAY} ¶1" in ids and f"{DAY} carry 3" in ids and "2026-09-28 ¶5" in ids
    assert len(ids) == len(set(ids)) == 2 * (5 + 3)
    meta = dict((d[0], d[2]) for d in docs)[f"{DAY} carry 2"]
    assert meta["letter_date"] == DAY and meta["part"] == "carry_forward"


def test_search_rereads_hits_from_the_letter_in_ranked_order():
    chroma = FakeChroma(ol.SEARCH_COLLECTION)
    listener, _ = _search_listener(chroma)
    _index_all(listener, chroma)
    chroma.scores.update({f"{DAY} ¶1": 0.9, f"{DAY} carry 1": 0.8, "2026-09-28 ¶1": 0.7, f"{DAY} ¶2": 0.3})
    out = _ask(listener, {"query": "the dream organ gap"})
    assert [i.id for i in out.items] == [f"{DAY} ¶1", f"{DAY} carry 1", "2026-09-28 ¶1"]
    assert out.items[0].text == _paragraph(1).text and out.items[0].extra["similarity"] == 0.9
    assert out.items[0].extra["claim_check"] and out.items[1].extra["citations"]
    out = _ask(listener, {"query": "the dream organ gap", "letter_date": DAY, "part": "note"})
    assert [i.id for i in out.items] == [f"{DAY} ¶1"]
    assert chroma.queries[-1]["where"] == {"$and": [{"letter_date": DAY}, {"part": "note"}]}


def test_search_narrowed_to_a_missing_letter_is_not_found():
    chroma = FakeChroma(ol.SEARCH_COLLECTION)
    listener, _ = _search_listener(chroma)
    _index_all(listener, chroma)
    out = _ask(listener, {"query": "anything", "letter_date": "2030-01-01"})
    assert not out.ok and "not found" in out.error


def test_empty_search_is_unknown_until_the_index_has_caught_up():
    chroma = FakeChroma(ol.SEARCH_COLLECTION)
    listener, _ = _search_listener(chroma)
    asyncio.run(listener.index_once())  # publishes upserts; proves nothing yet
    assert listener.index_complete_as_of is None
    chroma.apply(listener.bus)
    out = _ask(listener, {"query": "cookie recipes"})  # every score 0.0, below the floor
    assert not out.ok and out.error == ol.SEARCH_UNAVAILABLE
    listener.bus.published.clear()
    _index_all(listener, chroma)
    assert listener.index_complete_as_of is not None
    out = _ask(listener, {"query": "cookie recipes"})
    assert out.ok and out.items == [] and out.total_available == 0


def test_a_hit_that_cannot_be_read_back_is_unknown():
    chroma = FakeChroma(ol.SEARCH_COLLECTION)
    chroma.put("2026-09-27 ¶1", {"letter_date": "2026-09-27", "part": "note"}, score=0.9)
    chroma.put(f"{DAY} ¶42", {"letter_date": DAY, "part": "note"}, score=0.9)
    listener, _ = _search_listener(chroma)
    listener.index_complete_as_of = datetime.now(timezone.utc)
    out = _ask(listener, {"query": "the dream organ gap"})
    assert not out.ok and out.error == ol.QUERY_UNAVAILABLE


def test_search_not_configured_or_failing_is_unknown():
    listener, _ = _listener(search=None)
    assert _ask(listener, {"query": "x"}).error == ol.SEARCH_UNAVAILABLE
    chroma = FakeChroma(ol.SEARCH_COLLECTION)  # collection never created
    listener, _ = _search_listener(chroma)
    assert _ask(listener, {"query": "x"}).error == ol.SEARCH_UNAVAILABLE


def test_index_once_without_a_pool_skips_and_complete_mark_carries_the_lag_margin():
    listener = ol.OrionDayIntrospectListener(pool_provider=lambda: None, source_ref=ServiceRef(name="h"), search=SEARCH)
    assert asyncio.run(listener.index_once()) is None
    chroma = FakeChroma(ol.SEARCH_COLLECTION)
    listener, _ = _search_listener(chroma)
    _index_all(listener, chroma)
    assert listener.index_complete_as_of is not None
    assert listener.index_complete_as_of <= datetime.now(timezone.utc) - INDEX_LAG_MARGIN
