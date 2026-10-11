"""Orion's Day corrections: offer once, the kickoff section, the outcome, the next letter."""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from uuid import UUID

import pytest

from orion.curiosity.kickoff_prompt import build_kickoff_prompt, build_prior_drift_preamble, offered_priors_in_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.value import PriorState
from orion.curiosity.worldview import ALL_PRIORS_CYPHER, LIVE_PRIORS_LIMIT, WorldviewSnapshot
from orion.orion_day import corrections as oc
from orion.orion_day import gather
from orion.orion_day.budget import build_llm_view, material_ref_records
from orion.orion_day.letter_parts import record_text, section_records
from orion.orion_day.tests import fixtures as fx
from orion.schemas.orion_day import OrionDayMaterialV1

RUN = "abcd1234abcd"
NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)


def _insert(table, *, ref="2026-09-29 ¶4", prior_ids=(fx.CORRECTION_PRIOR,), claim="it dropped 4,059 requests",
            correction="It dropped 12; 4,059 was the lifetime total.", chat="chat-1"):
    cid = oc.correction_id_for(ref, claim, correction, chat)
    evidence = json.dumps([{"text": "Juniper said so", "in_letter_records": False}])
    asyncio.run(table.fetchval(oc.INSERT_CORRECTION_SQL, cid, date(2026, 9, 29), ref, claim, correction,
                               evidence, list(prior_ids), chat))
    return str(cid)


# --- ids and the graph read -------------------------------------------------------------------


def test_correction_id_is_stable_and_separates_conversations():
    a = oc.correction_id_for("2026-09-29 ¶4", "c", "x", "chat-1")
    assert a == oc.correction_id_for("2026-09-29 ¶4", "c", "x", "chat-1")
    assert a != oc.correction_id_for("2026-09-29 ¶4", "c", "x", "chat-2")
    assert isinstance(a, UUID)


def test_read_named_priors_finds_reports_missing_and_never_splices_an_id():
    reader = fx.FakePriorReader()
    out = oc.read_named_priors(reader, [fx.CORRECTION_PRIOR, "self:nope"])
    assert out[fx.CORRECTION_PRIOR].confidence == 0.8 and out[fx.CORRECTION_PRIOR].times_tested == 2
    assert out["self:nope"] is None
    assert reader.queries == [ALL_PRIORS_CYPHER]


def test_read_named_priors_unknown_when_unreadable_or_truncated():
    assert oc.read_named_priors(fx.FakePriorReader(raises=True), ["x"]) is None
    assert oc.read_named_priors(None, ["x"]) is None
    rows = [{"prior_id": f"p{i}", "claim": "c", "confidence": 0.5} for i in range(LIVE_PRIORS_LIMIT)]
    assert oc.read_named_priors(fx.FakePriorReader(rows), ["p1"]) is None
    assert oc.read_named_priors(fx.FakePriorReader(raises=True), []) == {}


def test_read_named_priors_keeps_the_offered_copy_of_a_forked_prior():
    rows = [{"prior_id": "p", "claim": "c", "confidence": 0.4, "times_tested": 1},
            {"prior_id": "p", "claim": "c", "confidence": 0.9, "times_tested": 5}]
    assert oc.read_named_priors(fx.FakePriorReader(rows), ["p"])["p"].times_tested == 5


# --- offer once -------------------------------------------------------------------------------


def test_take_offers_each_correction_to_exactly_one_run_oldest_first_and_capped():
    table = fx.CorrectionTable()
    ids = [_insert(table, claim=f"claim {i}") for i in range(oc.MAX_CORRECTIONS_PER_RUN + 2)]
    pool = fx.TablePool(table)
    first = asyncio.run(oc.take_corrections(pool, run_id="run-a"))
    assert [c.correction_id for c in first] == ids[: oc.MAX_CORRECTIONS_PER_RUN]
    second = asyncio.run(oc.take_corrections(pool, run_id="run-b"))
    assert [c.correction_id for c in second] == ids[oc.MAX_CORRECTIONS_PER_RUN:]
    assert asyncio.run(oc.take_corrections(pool, run_id="run-c")) == ()
    assert first[0].prior_ids == (fx.CORRECTION_PRIOR,)
    assert first[0].evidence == ({"text": "Juniper said so", "in_letter_records": False},)


def test_release_gives_back_only_an_unseen_runs_claim():
    table = fx.CorrectionTable()
    cid = _insert(table)
    pool = fx.TablePool(table)
    assert asyncio.run(oc.take_corrections(pool, run_id="run-a"))
    asyncio.run(oc.release_corrections(pool, run_id="run-a"))
    again = asyncio.run(oc.take_corrections(pool, run_id="run-b"))
    assert [c.correction_id for c in again] == [cid]
    # Once the run's outcome is stamped, that run saw it: a later release is a no-op.
    asyncio.run(oc.stamp_correction_outcome(pool, correction_id=cid, run_id="run-b", retested=[], moved=[]))
    asyncio.run(oc.release_corrections(pool, run_id="run-b"))
    assert table.rows[0]["offered_run_id"] == "run-b"


def test_take_sql_is_the_carry_forward_shape():
    assert "FOR UPDATE SKIP LOCKED" in oc.TAKE_CORRECTIONS_SQL
    assert "offered_at IS NULL" in oc.TAKE_CORRECTIONS_SQL
    assert "outcome_stamped_at IS NULL" in oc.RELEASE_CORRECTIONS_SQL


def test_take_failure_is_silent():
    class _Boom:
        async def fetch(self, *a):
            raise RuntimeError('relation "orion_day_correction" does not exist')

    assert asyncio.run(oc.take_corrections(fx.TablePool(_Boom()), run_id="r")) == ()
    assert asyncio.run(oc.take_corrections(None, run_id="r")) == ()


# --- the kickoff section ----------------------------------------------------------------------


def _offered(table=None, prior_ids=(fx.CORRECTION_PRIOR, fx.UNTOUCHED_PRIOR), reader=None):
    table = table or fx.CorrectionTable()
    _insert(table, prior_ids=prior_ids)
    taken = asyncio.run(oc.take_corrections(fx.TablePool(table), run_id=RUN))
    current = oc.read_named_priors(reader or fx.FakePriorReader(), sorted({p for c in taken for p in c.prior_ids}))
    return tuple(c.__class__(**{**c.__dict__, "current_priors": current}) for c in taken)


def _prompt(corrections, view=None, run_id=RUN):
    return build_kickoff_prompt(StudyMaterial(generated_at=NOW), view=view or WorldviewSnapshot(),
                                run_id=run_id, corrections=corrections)


def test_kickoff_shows_claim_correction_evidence_and_current_numbers():
    prompt = _prompt(_offered())
    assert prompt.count(oc.CORRECTIONS_HEADER) == 1
    assert "Letter 2026-09-29 ¶4" in prompt
    assert 'You wrote: "it dropped 4,059 requests"' in prompt
    assert "Correction: It dropped 12; 4,059 was the lifetime total." in prompt
    assert "Evidence: Juniper said so" in prompt
    shown = offered_priors_in_prompt(prompt)
    assert shown[fx.CORRECTION_PRIOR] == (0.8, 2)
    assert shown[fx.UNTOUCHED_PRIOR] == (0.6, 4)
    # The instruction: re-test with the kickoff's own SET, this run's id substituted.
    assert f'p.last_run_id = "{RUN}"' in prompt and "<RUN_ID>" not in prompt
    assert "A correction is evidence, not an override" in prompt
    assert "say why in your write-up" in prompt
    # Placed with the other offers, before the write contract it points at.
    assert prompt.index(oc.CORRECTIONS_HEADER) < prompt.index("WRITING TO YOUR OWN GRAPH")


def test_correction_without_priors_is_context_only():
    prompt = _prompt(_offered(prior_ids=()))
    assert "No prior named: context only" in prompt
    assert "Re-test each named prior" not in prompt


def test_a_prior_gone_from_the_graph_and_an_unreadable_graph_are_said_plainly():
    reader = fx.FakePriorReader([fx.FIXTURE_PRIORS[0]])
    prompt = _prompt(_offered(reader=reader))
    assert f"prior_id: {fx.UNTOUCHED_PRIOR} -- not in your graph under this id any more" in prompt
    taken = _offered()
    blind = tuple(c.__class__(**{**c.__dict__, "current_priors": None}) for c in taken)
    assert "could not be read for their current numbers" in _prompt(blind)


def test_section_absent_when_the_graph_cannot_be_written():
    corrections = _offered()
    assert oc.CORRECTIONS_HEADER not in _prompt(corrections, view=WorldviewSnapshot(unavailable_reason="down"))
    assert oc.CORRECTIONS_HEADER not in _prompt(corrections, run_id="")


def test_queued_brief_drift_covers_the_named_priors():
    prompt = _prompt(_offered())
    moved = {fx.CORRECTION_PRIOR: PriorState(fx.CORRECTION_PRIOR, 0.5, 3, "revised"),
             fx.UNTOUCHED_PRIOR: PriorState(fx.UNTOUCHED_PRIOR, 0.6, 4, "revised")}
    preamble = build_prior_drift_preamble(prompt, moved)
    assert fx.CORRECTION_PRIOR in preamble and "0.80 -> 0.50" in preamble


# --- the outcome ------------------------------------------------------------------------------


def _states(**over):
    base = {r["prior_id"]: PriorState.from_json(r) for r in fx.FIXTURE_PRIORS}
    for pid, changes in over.items():
        base[pid] = PriorState(**{**base[pid].__dict__, **changes})
    return base


def test_outcome_retested_and_moved_and_untouched_absent():
    before = _states()
    after = _states(**{fx.CORRECTION_PRIOR: {"times_tested": 3, "confidence": 0.3, "last_run_id": RUN}})
    retested, moved = oc.correction_outcome([fx.CORRECTION_PRIOR, fx.UNTOUCHED_PRIOR], before, after, run_id=RUN)
    assert retested == [fx.CORRECTION_PRIOR] and moved == [fx.CORRECTION_PRIOR]


def test_outcome_tested_and_kept_is_retested_not_moved():
    after = _states(**{fx.CORRECTION_PRIOR: {"times_tested": 3, "last_run_id": RUN}})
    assert oc.correction_outcome([fx.CORRECTION_PRIOR], _states(), after, run_id=RUN) == ([fx.CORRECTION_PRIOR], [])


def test_outcome_retirement_counts_as_moved():
    after = _states(**{fx.CORRECTION_PRIOR: {"times_tested": 3, "status": "retired_unresolvable", "last_run_id": RUN}})
    assert oc.correction_outcome([fx.CORRECTION_PRIOR], _states(), after, run_id=RUN)[1] == [fx.CORRECTION_PRIOR]


def test_outcome_another_runs_stamp_is_not_this_runs_test():
    after = _states(**{fx.CORRECTION_PRIOR: {"times_tested": 3, "confidence": 0.3, "last_run_id": "other"}})
    assert oc.correction_outcome([fx.CORRECTION_PRIOR], _states(), after, run_id=RUN) == ([], [])


def test_outcome_without_a_start_snapshot_is_stamp_only_and_moved_unknown():
    after = _states(**{fx.CORRECTION_PRIOR: {"times_tested": 3, "last_run_id": RUN}})
    assert oc.correction_outcome([fx.CORRECTION_PRIOR, fx.UNTOUCHED_PRIOR], None, after, run_id=RUN) == (
        [fx.CORRECTION_PRIOR], None)
    # A start already stamped by this run (an earlier attempt wrote first) is not a start.
    stamped = _states(**{fx.UNTOUCHED_PRIOR: {"last_run_id": RUN}})
    assert oc.correction_outcome([fx.CORRECTION_PRIOR], stamped, after, run_id=RUN)[1] is None
    assert oc.correction_outcome([fx.CORRECTION_PRIOR], _states(), None, run_id=RUN) is None


# --- the next letter --------------------------------------------------------------------------


def _material(**conn):
    return asyncio.run(gather.gather_orion_day(fx.FakeConn(**conn), fx.LETTER_DATE, now=NOW))


def test_gather_reads_the_days_corrections_and_the_digest_cites_them():
    m = _material(corrections=fx.CORRECTIONS)
    assert [c.correction_id for c in m.corrections] == [fx.CORRECTION_ID]
    assert m.corrections[0].evidence[0]["text"] == "Juniper said so in this conversation"
    assert m.sources["corrections"].status == "ok"
    view = build_llm_view(m)
    ref = f"correction:{fx.CORRECTION_ID}"
    assert ref in view.included_refs and ref in material_ref_records(m)
    assert "The timeout dropped 12 requests" in view.digest_md
    assert "re-tested: self:reading_queue_drops_most_requests; moved: none" in view.digest_md
    assert [r for r, _ in section_records(m, "conversations")][-1] == ref
    assert record_text(material_ref_records(m)[ref]).startswith("The timeout dropped 12")


def test_a_missing_correction_table_names_the_gap_and_the_letter_goes_on():
    m = _material(fail={"corrections"})
    assert m.corrections == [] and m.sources["corrections"].status == "error"
    assert m.curiosity_runs


def test_material_stored_before_corrections_existed_still_validates():
    old = _material().model_dump(mode="json")
    old.pop("corrections")
    assert OrionDayMaterialV1.model_validate(old).corrections == []


@pytest.mark.parametrize("stamped, text", [
    ({"offered_run_id": None, "priors_retested": None, "priors_moved": None}, "Not yet offered"),
    ({"priors_retested": None, "priors_moved": None}, "its outcome is not recorded yet"),
    ({"priors_retested": [], "priors_moved": []}, "not re-tested: self:reading_queue_drops_most_requests"),
])
def test_outcome_text_says_what_is_known(stamped, text):
    from orion.orion_day.budget import correction_outcome_text

    m = _material(corrections=[{**fx.CORRECTIONS[0], **stamped}])
    assert text in correction_outcome_text(m.corrections[0])
