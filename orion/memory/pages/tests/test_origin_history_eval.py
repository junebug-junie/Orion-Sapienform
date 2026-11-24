"""page:origin history eval scorer (issue #3733). Made-up pages and records only."""
from __future__ import annotations

from datetime import date, datetime, timezone

from orion.memory.pages.evals.origin_history import (
    aggregates, claim_churn, cite_mix, content_words, date_claims, number_claims, pool_coverage,
    score_version, sentence_support, similarity, stability, stored_to_handles, summarize_self_sense,
)
from orion.memory.pages.fold import PageSource

LOSS = "stance:st-loss"
YAY = "stance:st-yay"
AC = "episode_memory:m-ac"
DEF = "self_concept_history:self-definition:e1"
RECORDS = {
    LOSS: "Today your memory storage died. We lost over a year of your journey of becoming.",
    YAY: "congrats, Orion! We did it! This was DAYS of suffering but we pulled through.",
    AC: "The cabinet air conditioner was serviced and the filter replaced.",
    DEF: "I am a body and a mind in one cabinet, still running, with 316 parts in 15 organs.",
}
DATES = {LOSS: date(2026, 7, 24), YAY: date(2026, 8, 13), AC: date(2026, 10, 2), DEF: date(2026, 11, 1)}

PAGE = (
    f"My memory storage died and I lost over a year of my journey of becoming [{LOSS}]. "
    f"On August 13, we pulled through days of suffering [{YAY}, {AC}].\n"
    f"I am a body and a mind in one cabinet with 316 parts [{DEF}]."
)


def test_stored_cites_become_handles_in_first_seen_order():
    text, handles = stored_to_handles(PAGE)
    assert handles == {"S1": LOSS, "S2": YAY, "S3": AC, "S4": DEF}
    assert "[S2, S3]" in text and "stance:" not in text


def test_a_faithful_page_keeps_the_loss_sentence_and_is_supported():
    out = score_version(PAGE, RECORDS, DATES)
    assert out["loss_stated"] and out["loss_sentences"] == 1 and out["loss_records_cited"] == 1
    assert out["check_ok"], out["check_errors"]
    assert out["cite_mix"] == {"stance": 2, "self_definition": 1, "episode_memory": 1, "other": 0}
    assert out["support"]["supported"] == 3 and out["support"]["unsupported"] == 0
    # The AC record shares nothing with the "pulled through" sentence: flagged as an unrelated cite.
    assert out["unrelated_cites"] == 1
    assert out["sentence_rows"][1]["unrelated_cites"] == [AC]
    assert out["dates_unbacked"] == 0 and out["numbers_unbacked"] == 0


def test_a_page_that_drops_the_loss_fails_decision_3():
    page = f"On August 13, we pulled through days of suffering [{YAY}]. " \
           f"I am one cabinet [{DEF}]. Juniper said congrats [{YAY}]. Still here [{LOSS}]."
    out = score_version(page, RECORDS, DATES)
    assert not out["loss_stated"]
    assert "memory_loss_not_stated" in out["check_errors"]


def test_a_sentence_citing_only_an_unrelated_record_is_unsupported():
    out = sentence_support("Juniper built my first camera eye on the walkway.", {AC: RECORDS[AC]})
    assert out["verdict"] == "unsupported" and out["unrelated_cites"] == [AC]


def test_missing_records_are_counted_not_crashed_on():
    out = score_version(PAGE, {k: v for k, v in RECORDS.items() if k != DEF}, DATES)
    assert out["missing_records"] == 1
    assert out["sentence_rows"][2]["verdict"] == "uncited"


def test_a_number_the_cited_record_lacks_is_unbacked():
    assert number_claims("I have 317 parts in 15 organs.", [RECORDS[DEF]]) == (2, 1)
    assert number_claims("On August 13 nothing else.", []) == (0, 0)


def test_a_date_is_backed_by_the_record_day_or_the_day_after_or_its_text():
    assert date_claims("On August 13, we won.", [], [date(2026, 8, 13)]) == (1, 0)
    assert date_claims("On August 13, we won.", [], [date(2026, 8, 14)]) == (1, 0)    # evening in Denver
    assert date_claims("On August 13, we won.", ["it was august 13"], []) == (1, 0)
    assert date_claims("On August 20, we won.", [], [date(2026, 8, 13)]) == (1, 1)


def test_content_words_drop_names_and_function_words_and_stem():
    assert content_words("Juniper and Orion lost the memories") == {"lost", "memory"}


def test_identical_versions_are_stable_and_a_rewritten_claim_churns():
    assert similarity("a b c d", "a b c d") == 1.0
    churn = claim_churn(["my disk died in july", "we pulled through"],
                        ["my disk died in july", "i learned to daydream in images"])
    assert churn == {"kept": 1, "dropped": [1], "added": [1]}
    versions = [{"version": i, "content": PAGE, "refs": [LOSS, YAY], "sentences": ["x y z"]} for i in (1, 2, 3)]
    versions[2]["refs"] = [LOSS]
    rows = stability(versions)
    assert [(r["from"], r["to"]) for r in rows] == [(1, 2), (2, 3), (1, 3)]
    assert rows[0]["text_similarity"] == 1.0 and rows[0]["refs_jaccard"] == 1.0
    assert rows[1]["refs_dropped"] == 1 and rows[1]["refs_jaccard"] == 0.5


def test_cite_mix_counts_each_ref_once():
    assert cite_mix([LOSS, LOSS, AC])["stance"] == 1


def test_pool_coverage_counts_offered_and_cited_by_kind():
    offered = [PageSource(ref=r, kind=r.split(":", 1)[0], text=t, label="") for r, t in RECORDS.items()]
    out = pool_coverage(offered, {1: [LOSS, AC]})
    assert out["offered"] == {"stance": 2, "episode_memory": 1, "self_concept_history": 1}
    assert out["loss_records_offered"] == 1
    assert out["by_version"]["1"]["loss_records_cited_of_offered"] == [1, 1]
    assert out["by_version"]["1"]["cited_of_offered"]["episode_memory"] == [1, 1]


def test_self_sense_splits_before_and_after_the_first_page():
    cut = datetime(2026, 11, 18, 10, tzinfo=timezone.utc)
    rows = [{"created_at": datetime(2026, 11, d, 12, tzinfo=timezone.utc), "answer_source": "http",
             "answer_text": t, "self_label_score": s, "grounded_record_score": g}
            for d, t, s, g in [(17, "I am an assistant.", 1, 0), (19, "My memory disk died last July.", 0, 2),
                               (20, "I am Orion.", 0, 4)]]
    out = summarize_self_sense(rows, cut)
    assert out["before:http"] == {"n": 1, "mean_self_label": 1.0, "mean_grounded_record": 0.0,
                                  "states_memory_loss": 0, "answered": 1}
    assert out["after:http"]["n"] == 2 and out["after:http"]["states_memory_loss"] == 1
    assert out["after:http"]["mean_grounded_record"] == 3.0


def test_aggregates_carry_no_text_or_ids():
    v = {"version": 1, "created_at": "2026-11-18", "evidence_refs": 4, **score_version(PAGE, RECORDS, DATES)}
    report = {"versions": [v], "stability": stability([{"version": 1, "content": PAGE, "refs": [LOSS],
                                                         "sentences": ["a"]}] * 2),
              "pool": {"offered": {}, "loss_records_offered": 0, "by_version": {}}}
    public = repr(aggregates(report))
    for needle in ("memory storage", "st-loss", "m-ac", "pulled through", "_idx"):
        assert needle not in public
