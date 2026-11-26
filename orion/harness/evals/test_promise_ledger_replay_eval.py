"""Pure parts of promise_ledger_replay_eval (no DB, no model)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from orion.harness.evals.promise_ledger_replay_eval import (
    LABELS,
    ReplayLedger,
    beta_summary,
    build_request,
    quote_hash,
    score,
)
from orion.harness.tests.fixtures import make_appraisal, make_thought
from orion.schemas.harness_finalize import OpenPromiseRefV1

T0 = datetime(2026, 11, 4, 6, 0)


def test_ledger_shows_oldest_three_open_and_expires_at_72h() -> None:
    led = ReplayLedger()
    for h in range(4):
        led.add(quote=f"I'll do thing {h} later today", made_at=T0 + timedelta(hours=h), correlation_id=f"c{h}",
                offer=False)
    assert [p.id for p in led.open_at(T0 + timedelta(hours=5))] == ["p1", "p2", "p3"]
    led.keep(["p1"], correlation_id="later", at=T0 + timedelta(hours=6))
    assert [p.id for p in led.open_at(T0 + timedelta(hours=6))] == ["p2", "p3", "p4"]
    # p2 (made T0+1h) is past its 72 h horizon at T0+74h; p3, p4 are still open.
    assert [p.id for p in led.open_at(T0 + timedelta(hours=74))] == ["p3", "p4"]
    assert led.outcomes(T0 + timedelta(hours=74)) == {"kept": 1, "expired": 1, "expired_never_shown": 0,
                                                      "censored": 2}


def test_promise_never_shown_is_not_counted_as_broken() -> None:
    led = ReplayLedger()
    for h in range(4):
        led.add(quote=f"I'll do thing {h} later today", made_at=T0, correlation_id=f"c{h}", offer=False)
    led.open_at(T0 + timedelta(hours=1))  # shows p1..p3 only
    assert led.outcomes(T0 + timedelta(hours=80)) == {"kept": 0, "expired": 3, "expired_never_shown": 1,
                                                      "censored": 0}


def test_beta_summary_is_the_posterior_of_a_uniform_prior() -> None:
    s = beta_summary(1, 5)
    assert s["n"] == 6 and s["kept"] == 1
    assert abs(s["mean"] - 2 / 8) < 1e-9
    assert 0.03 < s["lo90"] < s["mean"] < s["hi90"] < 0.6  # Beta(2, 6): 90% ~ [0.06, 0.52]
    empty = beta_summary(0, 0)
    assert abs(empty["mean"] - 0.5) < 1e-9 and empty["lo90"] < 0.06 and empty["hi90"] > 0.94


def test_request_carries_offers_and_open_promises_with_the_promise_schema() -> None:
    turn = {"draft_text": "Sure. I'll send the list tomorrow morning.", "thought": make_thought(),
            "appraisal": make_appraisal(), "user_message": "Yes, go ahead.", "receipts": [], "notes": None}
    body = build_request(turn, backend="llamacpp", pending_offers=["If you want, I'll send the list tomorrow."],
                         open_promises=[OpenPromiseRefV1(id="p1", quote="I'll look at the logs tonight.")])
    system = body["messages"][0]["content"]
    assert '- "If you want, I\'ll send the list tomorrow."' in system
    assert '[p1] "I\'ll look at the logs tonight."' in system
    assert body["messages"][-1]["content"].endswith("promises_made, offers_accepted, promises_kept, promise_conditions_met.")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "promises_made" in json.dumps(body["response_format"])


def test_score_counts_precision_drops_and_recall_from_hash_labels() -> None:
    q_true, q_false = "I'll bring it back tomorrow.", "I'm reading it now."
    report = {"turns": [
        {"correlation_id": "c1", "promises_made": [{"quote": q_true, "offer": False},
                                                    {"quote": q_false, "offer": False}],
         "raw": {"promises_made": [{}, {}, {}]}, "dropped_not_in_draft": ["I will bring it back"]},
        {"correlation_id": "c2", "promises_made": [], "raw": {"promises_made": []}},
        {"correlation_id": "c3", "error": "http 502"},
    ], "ledger": [{"correlation_id": "c1"}], "outcomes": {"kept": 0, "expired": 1, "censored": 0}}
    labels = {
        "hand_found": [{"key": "bring-back", "correlation_id": "c1", "offer": False},
                       {"key": "missed", "correlation_id": "c2", "offer": True}],
        "labels": [{"correlation_id": "c1", "quote_sha": quote_hash(q_true), "promise": True,
                    "hand_found": "bring-back"},
                   {"correlation_id": "c1", "quote_sha": quote_hash(q_false), "promise": False}],
        "dropped": [{"correlation_id": "c1", "quote_sha": quote_hash("I will bring it back"),
                     "literal_in_reply": False}],
    }
    s = score(report, labels)
    assert s["precision"] == 0.5 and s["unlabelled"] == 0
    assert s["gate_dropped"] == 1 and s["gate_dropped_that_were_real_quotes"] == 0
    assert s["recall_hand_found"]["found"] == 1 and s["recall_hand_found"]["of"] == 2
    assert s["turns_errored"] == 1 and s["baseline_keep_rate"]["n"] == 1


def test_label_fixture_holds_hashes_not_text() -> None:
    data = json.loads(LABELS.read_text())
    for entry in data["labels"] + data["dropped"]:
        assert "quote" not in entry
        assert len(entry["quote_sha"]) == 16


def test_wilson_interval_matches_a_known_value_and_handles_empty() -> None:
    from orion.harness.evals.promise_ledger_replay_eval import wilson

    w = wilson(21, 29)  # the 11-12 baseline precision
    assert abs(w["p"] - 21 / 29) < 1e-9
    assert abs(w["lo95"] - 0.5428) < 1e-3 and abs(w["hi95"] - 0.8530) < 1e-3
    assert wilson(0, 0)["p"] is None
    assert wilson(5, 5)["hi95"] == 1.0


def test_score_ablates_the_doer_rule_and_pools_recall_by_promise_key() -> None:
    real, real_span, forecast, ask = ("I'll bring it back tomorrow.", "bring it back tomorrow, I promise",
                                      "The queue should clear by noon.", "Text me when you land.")
    report = {"turns": [
        {"correlation_id": "c1", "promises_made": [{"quote": real_span, "offer": False, "doer": "orion"}],
         "not_orion": [{"quote": forecast, "offer": False, "doer": "a_system"},
                       {"quote": ask, "offer": True, "doer": "juniper"},
                       {"quote": "not in the reply at all", "offer": False, "doer": "a_system"}],
         "ablation_kept": [{"quote": real_span, "offer": False, "doer": "orion"},
                           {"quote": forecast, "offer": False, "doer": "a_system"},
                           {"quote": ask, "offer": True, "doer": "juniper"}]},
        {"correlation_id": "c2", "promises_made": []},
        {"correlation_id": "c3", "error": "http 502"},
    ]}
    labels = {"hand_found": [], "dropped": [], "labels": [
        # An earlier run quoted a different span of the same promise: same key.
        {"correlation_id": "c1", "quote_sha": quote_hash(real), "promise": True, "key": "bring-back"},
        {"correlation_id": "c1", "quote_sha": quote_hash(real_span), "promise": True, "key": "bring-back"},
        {"correlation_id": "c1", "quote_sha": quote_hash(forecast), "promise": False},
        {"correlation_id": "c1", "quote_sha": quote_hash(ask), "promise": False},
        {"correlation_id": "c2", "quote_sha": "0" * 16, "promise": True, "key": "missed-one"},
        {"correlation_id": "c3", "quote_sha": "1" * 16, "promise": True, "key": "errored-turn"},
    ]}
    s = score(report, labels)
    assert s["precision_wilson95"]["k"] == 1 and s["precision_wilson95"]["n"] == 1
    # The ablation scores what the gate would keep without the doer rule.
    assert s["precision_without_doer_gate"]["k"] == 1 and s["precision_without_doer_gate"]["n"] == 3
    assert s["doer_dropped"]["by_doer"] == {"a_system": 2, "juniper": 1}
    # Unlabelled items are reported, not counted as false.
    report["turns"][0]["promises_made"].append({"quote": "I'll write it up tonight.", "offer": False, "doer": "orion"})
    s2 = score(report, labels)
    assert (s2["precision_wilson95"]["k"], s2["precision_wilson95"]["n"]) == (1, 1)
    assert s2["precision_wilson95"]["unlabelled"] == 1
    assert s["doer_dropped"]["real_promises_dropped"] == 0
    # c3 errored, so its promise is not in this run's pool; c2's is, and was missed.
    assert (s["recall_pooled"]["k"], s["recall_pooled"]["n"]) == (1, 2)
    assert s["recall_pooled"]["missed"] == ["c2:missed-one"]


def test_every_real_label_carries_a_promise_key() -> None:
    data = json.loads(LABELS.read_text())
    assert all(lab.get("key") for lab in data["labels"] if lab["promise"])


def test_ablation_runs_the_gate_without_the_doer_rule() -> None:
    from orion.harness.evals.promise_ledger_replay_eval import ablate_doer_rule
    from orion.harness.tests.test_promise_gate import _reflection
    from orion.schemas.harness_finalize import PromiseV1

    DW = "a later reply shows it done"

    draft = ("The queue should clear by noon. Text me when you land. I'll send the summary tonight. "
             "I'll also check the logs after lunch. I'll file the ticket tomorrow.")
    final = draft.replace(" Text me when you land.", "")  # repair removed the ask
    reflection = _reflection(promises_made=[
        PromiseV1(force="commits", quote="The queue should clear by noon.", doer="a_system", done_when=DW),
        PromiseV1(force="commits", quote="Text me when you land.", offer=True, doer="juniper", done_when=DW),
        PromiseV1(force="commits", quote="I'll send the summary tonight.", doer="orion", done_when=DW),
        PromiseV1(force="commits", quote="I'll also check the logs after lunch.", doer="orion", done_when=DW),
        PromiseV1(force="commits", quote="I'll file the ticket tomorrow.", doer="orion", done_when=DW),
        PromiseV1(force="commits", quote="not in the reply at all, invented", doer="a_system", done_when=DW)])
    kept = ablate_doer_rule(reflection, draft_text=draft, final_text=final)
    # As the live gate: the cap of 3 applies on the draft pass (the ask takes a slot), then
    # the final-text pass drops the ask that repair removed. Invented quotes never survive.
    assert [(k["quote"], k["doer"]) for k in kept] == [
        ("The queue should clear by noon.", "a_system"),
        ("I'll send the summary tonight.", "orion")]
    only_orion = _reflection(promises_made=[PromiseV1(force="commits", quote="I'll send the summary tonight.", doer="orion", done_when=DW)])
    assert ablate_doer_rule(only_orion, draft_text=draft, final_text=final) == []


def test_ablation_runs_the_gate_without_the_done_when_rule() -> None:
    from orion.harness.evals.promise_ledger_replay_eval import ablate_done_when_rule
    from orion.harness.tests.test_promise_gate import _reflection
    from orion.schemas.harness_finalize import PromiseV1

    draft = "I'm around all evening. The queue should clear by noon. I'll send the summary tonight."
    reflection = _reflection(promises_made=[
        PromiseV1(force="commits", quote="I'm around all evening.", doer="orion", act="be around"),
        PromiseV1(force="commits", quote="The queue should clear by noon.", doer="a_system"),
        PromiseV1(force="commits", quote="I'll send the summary tonight.", doer="orion", done_when="tonight's reply holds the summary")])
    kept = ablate_done_when_rule(reflection, draft_text=draft, final_text=draft)
    # Rule 0b off: the presence line comes back with its real (null) done_when; the doer
    # rule still drops the forecast.
    assert [(k["quote"], k["done_when"]) for k in kept] == [
        ("I'm around all evening.", None), ("I'll send the summary tonight.", "tonight's reply holds the summary")]
    only_real = _reflection(promises_made=[reflection.promises_made[2]])
    assert ablate_done_when_rule(only_real, draft_text=draft, final_text=draft) == []


def test_ablation_runs_the_gate_without_the_force_rule() -> None:
    from orion.harness.evals.promise_ledger_replay_eval import ablate_force_rule
    from orion.harness.tests.test_promise_gate import _reflection
    from orion.schemas.harness_finalize import PromiseV1

    draft = "Best to try the small model first. I'll send the summary tonight."
    dw = "tonight's reply holds the summary"
    reflection = _reflection(promises_made=[
        PromiseV1(force="suggests", quote="Best to try the small model first.", doer="orion", done_when=dw),
        PromiseV1(force="commits", quote="I'll send the summary tonight.", doer="orion", done_when=dw)])
    kept = ablate_force_rule(reflection, draft_text=draft, final_text=draft)
    assert [(k["quote"], k["force"]) for k in kept] == [
        ("Best to try the small model first.", "suggests"), ("I'll send the summary tonight.", "commits")]
    only_real = _reflection(promises_made=[reflection.promises_made[1]])
    assert ablate_force_rule(only_real, draft_text=draft, final_text=draft) == []


def test_ablation_reports_the_first_copy_of_a_duplicated_quote() -> None:
    """The gate keeps the first copy of a quote; the ablation must report that copy's force."""
    from orion.harness.evals.promise_ledger_replay_eval import ablate_force_rule
    from orion.harness.tests.test_promise_gate import _reflection
    from orion.schemas.harness_finalize import PromiseV1

    draft = "Best to try the small model first."
    dw = "a later reply reports the small-model result"
    reflection = _reflection(promises_made=[
        PromiseV1(force="suggests", quote=draft, doer="orion", done_when=dw),
        PromiseV1(force="commits", quote=draft, doer="orion", done_when=dw)])
    kept = ablate_force_rule(reflection, draft_text=draft, final_text=draft)
    assert [(k["quote"], k["force"]) for k in kept] == [(draft, "suggests")]


def test_hand_found_recall_matches_a_different_span_by_key() -> None:
    from orion.harness.evals.promise_ledger_replay_eval import quote_hash, score

    quote = "I'll bring the read back next time."
    report = {"turns": [{"correlation_id": "c1", "at": "2026-11-01T00:00:00",
                         "promises_made": [{"quote": quote, "offer": False, "doer": "orion"}]}]}
    labels = {"dropped": [], "hand_found": [{"key": "bring-back", "correlation_id": "c1", "offer": False}],
              "labels": [{"correlation_id": "c1", "quote_sha": quote_hash(quote), "promise": True,
                          "key": "bring-back"}]}
    assert score(report, labels)["recall_hand_found"]["found"] == 1


def test_merge_reports_joins_partial_runs_by_turn(tmp_path) -> None:
    from orion.harness.evals.promise_ledger_replay_eval import merge_reports

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"partial": True, "turns": [
        {"correlation_id": "c1", "at": "2026-11-01T00:00:00", "promises_made": []},
        {"correlation_id": "c2", "at": "2026-11-02T00:00:00", "error": "http 502"}]}))
    b.write_text(json.dumps({"turns": [
        {"correlation_id": "c2", "at": "2026-11-02T00:00:00", "promises_made": []},
        {"correlation_id": "c0", "at": "2026-10-31T00:00:00", "promises_made": []}]}))
    merged = merge_reports([a, b])
    assert [r["correlation_id"] for r in merged["turns"]] == ["c0", "c1", "c2"]
    assert "error" not in merged["turns"][2]  # the later file wins
    scored = score(merged, {"labels": [], "hand_found": [], "dropped": []})
    assert scored["turns_judged"] == 3 and scored["promises_counted"] is None
    # A good row is never replaced by a later error row.
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"turns": [{"correlation_id": "c1", "at": "2026-11-01T00:00:00", "error": "http 503"}]}))
    assert "error" not in merge_reports([a, b, c])["turns"][1]


def test_wilson_reproduces_its_pre_refactor_value_after_delegating_to_the_self_monitor() -> None:
    """Shared-bet-primitives Patch A: the eval's Wilson now calls orion.self_monitor.record.wilson.
    Golden: orion/prediction/tests/surprisal_golden.py (built from the pre-refactor tree)."""
    from orion.prediction.tests import surprisal_golden as golden

    want = golden.load()["expected"]["service"]["promise_wilson"]
    assert golden.diffs(golden.compute_promise_wilson(), want, "promise_wilson") == []
