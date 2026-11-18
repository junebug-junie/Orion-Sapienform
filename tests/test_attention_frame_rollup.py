"""orion.attention.frame_rollup: one frame's hourly counts (the rows `dig attention` reads). Pure."""
from datetime import datetime, timezone

from orion.attention import frame_rollup as R

WF = "world-first: not attended -- "


def t(tid, reason):
    return {"target_id": tid, "reasons": [reason]}


def test_ranked_contested_frame():
    frame = {
        "dominant_targets": [t("world:chat", "world-first: attended -- world busy for itself (high)"),
                             t("node:substrate.perception", "world-first: attended -- world busy for itself (high)")],
        "suppressed_targets": [
            t("world:reading", WF + "world quiet for itself: not busy; reading is quiet"),
            t("node:substrate.biometrics", WF + "body high for itself, not sustained (hot 100s of last 600s); x"),
            t("capability:graph", "Candidate B novelty-only salience: ..."),  # another policy's entry: not counted
        ],
    }
    inc = R.frame_increments(frame)
    assert inc[("frame", "", "")] == [1, 0, 1, 0]  # not rest, contested
    assert inc[("won", "world:chat", "")] == [1, 1, 1, 1]
    assert inc[("won", "node:substrate.perception", "")] == [1, 0, 1, 0]
    assert inc[("not", "world:reading", "quiet for itself")] == [1, 0, 0, 0]
    assert inc[("not", "node:substrate.biometrics", "not sustained")] == [1, 0, 0, 0]
    assert not any(k[1] == "capability:graph" for k in inc)


def test_rest_frame_and_uncontested_win():
    assert R.frame_increments({"dominant_targets": [], "suppressed_targets": []}) == {("frame", "", ""): [1, 1, 0, 0]}
    one = R.frame_increments({"dominant_targets": [t("world:chat", "world-first: attended -- x")]})
    assert one[("frame", "", "")] == [1, 0, 0, 0] and one[("won", "world:chat", "")] == [1, 1, 0, 0]


def test_reading_yield_to_chat_has_its_own_cause():
    assert R.classify(WF + "yields to Juniper (world:chat eligible)") == "yields to chat"


def test_upsert_params_are_hour_bucketed_and_key_ordered():
    rows = R.upsert_params({"dominant_targets": [t("b", "world-first: attended -- x"), t("a", "world-first: attended -- x")]},
                           "2026-11-17T08:59:59.9Z")
    assert {r["hour"] for r in rows} == {datetime(2026, 11, 17, 8, tzinfo=timezone.utc)}
    keys = [(r["part"], r["target"], r["cause"]) for r in rows]
    assert keys == sorted(keys)  # one lock order for every writer


def test_rank_is_among_world_first_entries_and_unnamed_targets_are_skipped():
    frame = {"dominant_targets": [t("capability:x", "Candidate B novelty-only salience"),
                                  t("world:chat", "world-first: attended -- x"),
                                  {"target_id": "", "reasons": ["world-first: attended -- x"]}],
             "suppressed_targets": [{"reasons": [WF + "body quiet for itself"]}]}
    inc = R.frame_increments(frame)
    assert inc[("won", "world:chat", "")] == [1, 1, 0, 0]  # first world-first entry is ranked first
    assert inc[("frame", "", "")] == [1, 0, 0, 0]
    assert all(k[1] for k in inc if k[0] != "frame")
