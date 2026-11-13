from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app import act_chain as C

T0 = datetime(2026, 11, 2, 7, 0, tzinfo=timezone.utc)   # Mon 00:00 Denver (MST after 11-01)
G = C.GRAIN


def _act(start: datetime, verb: str, lag: timedelta = timedelta(seconds=1)):
    return (start, start + lag, verb)


def _stream(seq, start=T0):
    """One act per bucket, at 1 min into the bucket; None = idle."""
    return [_act(start + i * G + timedelta(minutes=1), v) for i, v in enumerate(seq) if v]


def test_bucket_state_main_verb_tie_goes_to_latest_and_idle():
    acts = [_act(T0 + timedelta(minutes=1), "look"), _act(T0 + timedelta(minutes=2), "look"),
            _act(T0 + timedelta(minutes=3), "make"),                                # bucket 0: look (2 vs 1)
            _act(T0 + G + timedelta(minutes=1), "make"), _act(T0 + G + timedelta(minutes=5), "say"),  # tie -> say
            # reverse order: the later verb has the HIGHER index, so plain argmax would pick the wrong one
            _act(T0 + 2 * G + timedelta(minutes=1), "say"), _act(T0 + 2 * G + timedelta(minutes=5), "make"),
            _act(T0 + 3 * G, "frobnicate")]                                          # unknown verb: dropped
    s = C.Stream.build(acts, T0, T0 + 4 * G)
    assert [C.STATES[i] for i in s.final] == ["look", "say", "make", "idle"]


def test_late_rows_count_only_after_they_land():
    a = _act(T0 + timedelta(minutes=1), "look", lag=timedelta(minutes=20))   # lands 6 min after bucket end
    late = _act(T0 + G + timedelta(minutes=1), "make", lag=timedelta(days=2))  # a backfill
    s = C.Stream.build([a, late], T0, T0 + 3 * G)
    assert C.STATES[s.asof[0]] == "idle"          # not known when bucket 1's forecast is issued
    assert C.STATES[s.final[0]] == "look"         # but it is the outcome bucket 0 is scored against
    assert C.STATES[s.final[1]] == "idle"         # a backfill never counts


def test_clock_blocks_follow_juniper_rules_in_denver_time():
    def blk(utc):
        return C.BLOCKS[int(C.clock(np.array([C._ns(utc)]))[1][0])]
    assert blk(datetime(2026, 11, 4, 11, 0, tzinfo=timezone.utc)) == "asleep"          # Wed 04:00 MST
    assert blk(datetime(2026, 11, 7, 16, 0, tzinfo=timezone.utc)) == "weekend_morning"  # Sat 09:00
    assert blk(datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc)) == "day"             # Wed 09:00
    assert blk(datetime(2026, 11, 5, 8, 0, tzinfo=timezone.utc)) == "evening"          # Thu 01:00
    assert blk(datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)) == "asleep"           # 04:00 MDT


def test_forecast_uses_nothing_at_or_after_origin_and_matches_replay():
    rng = np.random.default_rng(0)
    seq = list(rng.choice(["look", "make", None], size=4 * 96))
    oi = 3 * 96
    seq[oi - 1] = None                               # bucket o-1 is idle as known at the origin
    for j in range(oi - 40, oi - 4, 6):              # and a few training condition buckets too
        seq[j] = None
    acts = _stream(seq)
    origin = T0 + oi * G
    f = C.forecast(acts, origin)
    assert f["prev_state"] == "idle"
    future = acts + [_act(origin + timedelta(minutes=k), "play") for k in range(0, 600, 5)]
    # rows landing late (after the next bucket opened) about past idle buckets must not leak, neither
    # into the condition nor into the training conditions (asof), though they are final outcomes
    late = timedelta(minutes=16)
    future.append(_act(origin - G + timedelta(minutes=1), "predict", lag=late))
    future += [_act(T0 + j * G + timedelta(minutes=1), "predict", lag=late) for j in range(oi - 40, oi - 4, 6)]
    g = C.forecast(future, origin)
    assert g["prev_state"] == "idle"
    assert g["probs"][C.MODEL]["predict"] > 0     # the late rows are real outcomes (landed within grace)
    no_train_leak = C.forecast([a for a in future if a[0] < origin - G], origin)
    assert g["probs"] == no_train_leak["probs"]
    row = next(r for r in C.replay(acts, origin - C.LOOKBACK - G, T0 + 4 * 96 * G, warmup=C.LOOKBACK)
               if r["origin"] == origin)
    assert np.allclose([f["probs"][C.MODEL][s] for s in C.STATES], row["probs"][C.MODEL])


def test_training_skips_buckets_not_settled_at_the_origin():
    seq = ["look"] * 95 + ["make", "make", "make"]   # bucket 95 is settled at the origin; 96, 97 are not
    s = C.Stream.build(_stream(seq), T0, T0 + 99 * G)
    p_make = C.predict(s, 98)["probs"]["act_marginal"][C.STATES.index("make")]
    seq2 = ["look"] * 96 + ["make", "make"]
    p_make2 = C.predict(C.Stream.build(_stream(seq2), T0, T0 + 99 * G), 98)["probs"]["act_marginal"][C.STATES.index("make")]
    assert p_make2 < 0.01 < p_make                   # o-3 trains, o-2 and o-1 do not


def test_replay_scores_only_settled_buckets():
    end = T0 + 3 * 96 * G + timedelta(minutes=7)
    rows = C.replay(_stream(["look"] * (3 * 96)), T0, end, warmup=timedelta(days=1))
    last = rows[-1]["origin"]
    assert last + G + C.SETTLE_GRACE <= end < last + 2 * G + C.SETTLE_GRACE


def test_markov_learns_a_deterministic_alternation():
    acts = _stream(["look", "make"] * (5 * 48))
    rows = C.replay(acts, T0, T0 + 5 * 96 * G, warmup=timedelta(days=2))
    v = C.verdict(rows)
    assert v["log_loss"]["act_markov"] < 0.05 < v["log_loss"]["act_marginal"]
    assert rows[-1]["k"]["act_markov"] <= 2.0        # LOO trusts a context that always says the same thing


def test_excess_surprise_rests_at_zero_when_the_world_matches_the_forecast():
    """iid draws from a fixed mix: every model is calibrated, so excess surprise is ~0 by theory,
    the rolling score returns below 0 often, and LOO backs the uninformative contexts off hard."""
    rng = np.random.default_rng(1)
    seq = list(rng.choice(["look", "make", "infer", None], p=[0.4, 0.3, 0.1, 0.2], size=12 * 96))
    rows = C.replay(_stream(seq), T0, T0 + 12 * 96 * G, warmup=timedelta(days=4))
    v = C.verdict(rows)
    e = v["excess"][C.MODEL]
    assert abs(e["mean"]) < 3 * e["se"] + 0.02
    assert 0.3 < v["rest"]["rolling_1h_share_le_0"] < 0.7
    assert rows[-1]["k"][C.MODEL] >= 64.0
    assert v["n_days"] >= C.MIN_DAYS and "vs" in v and v["served"] == C.MODEL


def test_silence_is_scored_and_surprising_after_a_busy_rhythm():
    seq = ["look"] * (6 * 96) + [None] * 8           # the act stream goes quiet (an outage)
    rows = C.replay(_stream(seq), T0, T0 + len(seq) * G + C.SETTLE_GRACE + G, warmup=timedelta(days=5))
    quiet = [r for r in rows if r["observed"] == "idle"]
    assert quiet and quiet[0]["scores"][C.MODEL]["surprise"] > 3.0


def test_verdict_withholds_gaps_below_min_days():
    rows = C.replay(_stream(["look", "make"] * 96 * 2), T0, T0 + 4 * 96 * G, warmup=timedelta(days=1))
    v = C.verdict(rows)
    assert v["n_days"] < C.MIN_DAYS and "vs" not in v


def test_training_conditions_use_what_was_known_not_the_final_outcome():
    """Idle-as-known buckets (whose act lands late) are always followed by 'look'; everything else is
    'make'. A chain that conditions training on the final view never sees idle -> look."""
    n, late = 4 * 96, timedelta(minutes=16)
    seq = ["make"] * n
    idle_j = list(range(10, n - 4, 12))
    acts = []
    for j in idle_j:
        seq[j], seq[j + 1] = None, "look"
        acts.append(_act(T0 + j * G + timedelta(minutes=1), "predict", lag=late))
    seq[n - 1] = None                                # origin's condition: idle as known
    acts += _stream(seq)
    s = C.Stream.build(acts, T0, T0 + (n + 1) * G)   # starts at T0: no all-idle padding before the data
    r = C.predict(s, n)
    look, make = (r["probs"]["act_markov"][C.STATES.index(v)] for v in ("look", "make"))
    assert r["prev_state"] == "idle"
    assert look > 0.5 > make
