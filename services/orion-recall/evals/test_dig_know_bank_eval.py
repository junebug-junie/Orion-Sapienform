"""The dig-know bank eval's scoring, on made-up answers (no network)."""
from __future__ import annotations

from run_dig_know_bank_eval import EVAL_TURN_ID, run, score


def test_score_finds_targets_with_rank_and_strength():
    ans = {"status": "ok", "backend": "hybrid", "hits": [{"id": "a", "strength": "strong"}, {"id": "b", "strength": "weak"}]}
    got = score(ans, ["b", "z"])
    assert got["ok"] and got["found"] == [{"id": "b", "rank": 2, "strength": "weak"}]
    assert not score({"status": "below_bar", "hits": []}, ["b"])["ok"]


def test_run_only_items_with_words_and_tags_the_eval_turn():
    bank = {"referring": [{"label": f"l{i}", "targets": f"['t{i}']"} for i in range(7)]}
    words = {f"l{i}": f"words {i}" for i in range(6)}
    sent = []

    def post(url, payload):
        sent.append(payload)
        n = payload["words"].split()[-1]
        return {"status": "ok", "hits": [{"id": f"t{n}", "strength": "strong"}] if n != "0" else []}

    report = run(bank, words, "http://x/", post=post)
    assert report["items"] == 6 and report["found"] == 5 and report["pass"]
    assert all(p["turn_id"] == EVAL_TURN_ID for p in sent)
