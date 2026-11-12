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
    assert body["messages"][-1]["content"].endswith("promises_made, offers_accepted, promises_kept.")
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
