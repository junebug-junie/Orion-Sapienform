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


class _Store:
    def __init__(self, acts):
        self.acts, self.rows = acts, {}

    def grammar_acts(self, start, end):
        return [a for a in self.acts if start <= a[0] < end]

    def act_chain_exists(self, origin):
        return any(r["origin"] == origin for r in self.rows.values())

    def insert_act_chain(self, row):
        if row["forecast_id"] in self.rows:
            return None
        self.rows[row["forecast_id"]] = {**row, "settled_at": None, "observed": None, "n_acts": None,
                                         "surprise": None, "excess": None}
        return row["forecast_id"]

    def pending_act_chain(self, before):
        return [r for r in self.rows.values() if r["settled_at"] is None and r["window_end"] <= before]

    def settle_act_chain(self, fid, observed, n, surprise, excess, now):
        r = self.rows[fid]
        if r["settled_at"] is not None:
            return 0
        r.update(observed=observed, n_acts=n, surprise=surprise, excess=excess, settled_at=now)
        return 1

    def act_chain_rows(self, since, settled_only):
        return sorted((r for r in self.rows.values() if r["origin"] >= since and (r["settled_at"] or not settled_only)),
                      key=lambda r: r["origin"])


def test_tick_issues_once_early_in_the_bucket_and_settles_after_grace():
    o = T0 + 2 * 96 * G
    acts = _stream(["look", "make"] * 96)
    store = _Store(acts + [_act(o + timedelta(minutes=2), "make"), _act(o + timedelta(minutes=4), "make"),
                           _act(o + timedelta(minutes=6), "play", lag=timedelta(hours=2))])   # lands too late
    ch = C.ActChain(store)
    assert ch.tick(o + timedelta(minutes=4))["issued"] is None                 # past the issue slack
    assert ch.tick(o + timedelta(seconds=10))["issued"] is None                # inside the commit margin
    assert ch.tick(o + timedelta(minutes=1))["issued"] == "act_chain:" + o.strftime("%Y-%m-%dT%H:%M")
    assert ch.tick(o + timedelta(minutes=2))["issued"] is None                 # already issued
    row = store.rows["act_chain:" + o.strftime("%Y-%m-%dT%H:%M")]
    assert row["prev_state"] == "make" and row["served"] == C.MODEL
    assert abs(sum(row["probs"][C.MODEL].values()) - 1) < 1e-9
    assert row["probs"] == C.forecast(store.acts, o)["probs"]                  # rows opened after o change nothing
    due = o + G + C.SETTLE_GRACE + C.COMMIT_MARGIN
    assert ch.tick(due - timedelta(seconds=1))["settled"] == 0
    assert ch.tick(due)["settled"] == 1
    assert (row["observed"], row["n_acts"]) == ("make", 2)                     # the late 'play' never counts
    p = row["probs"][C.MODEL]["make"]
    assert row["surprise"] == -np.log(p)
    ent = -sum(q * np.log(q) for q in row["probs"][C.MODEL].values())
    assert abs(row["excess"] - (-np.log(p) - ent)) < 1e-9


def test_settle_after_downtime_uses_the_same_cutoff():
    o = T0 + 96 * G
    store = _Store(_stream(["look"] * 96) + [_act(o + timedelta(minutes=1), "say", lag=timedelta(minutes=40)),
                                             _act(o + timedelta(minutes=2), "ask", lag=timedelta(minutes=23))])
    ch = C.ActChain(store)
    ch.tick(o + timedelta(minutes=1))
    ch.tick(o + timedelta(hours=6))                                            # settles hours late
    r = next(iter(store.rows.values()))
    # 'ask' lands 10 min after the bucket ends (inside the grace): counts. 'say' lands past it: never.
    assert (r["observed"], r["n_acts"]) == ("ask", 1)


def test_live_grades_stored_rows_like_the_replay():
    acts = _stream(["look", "make"] * (4 * 48))
    store = _Store(acts)
    ch = C.ActChain(store)
    t = T0 + 96 * G + timedelta(minutes=1)
    while t < T0 + 4 * 96 * G:
        ch.tick(t)
        t += G
    now = T0 + 4 * 96 * G + C.SETTLE_GRACE + C.COMMIT_MARGIN
    ch.tick(now)
    live = ch.live(now)
    g = live["grade"]
    assert g["n_buckets"] == len([r for r in store.rows.values() if r["settled_at"]]) > 250
    assert g["log_loss"]["act_markov"] < g["log_loss"]["act_marginal"]
    assert live["issuance"]["issue_lag_sec_max"] == 60.0
    assert live["current"] is None and live["newest_origin"] < C.bucket_floor(now)   # stale: not "current"
    # missed: the bucket at day 4 (never ticked) and the current one (now is past its issue slack)
    assert live["issuance"]["issued"] == live["issuance"]["buckets_since_first"] - 2
    rep = {r["origin"]: r for r in C.replay(acts, T0, now, warmup=timedelta(days=1))}
    stored = store.rows["act_chain:" + (T0 + 3 * 96 * G).strftime("%Y-%m-%dT%H:%M")]
    assert stored["observed"] == rep[stored["origin"]]["observed"]


def test_thin_night_cells_keep_the_night_mix():
    """Nights ask half the time, days almost never; the previous act says nothing. The served cell for
    (asleep, look) must back off toward the night mix (parallel backoff), not the all-day mix."""
    rng = np.random.default_rng(3)
    n = 10 * 96
    starts = T0 + G * np.arange(n)
    _, blk = C.clock(np.array([C._ns(t) for t in starts]))
    seq = [("ask" if rng.random() < 0.5 else "look") if b == 0 else ("ask" if rng.random() < 0.03 else "look")
           for b in blk]
    s = C.Stream.build(_stream(seq), T0, T0 + (n + 1) * G)
    night = next(i for i in range(n - 96, n) if blk[i] == 0 and C.STATES[s.asof[i - 1]] == "look")
    r = C.predict(s, night)
    p_ask = r["probs"][C.MODEL][C.STATES.index("ask")]
    assert r["block"] == "asleep" and r["k"]["lambda_prev"] < 1.0
    assert p_ask > 0.35 > r["probs"]["act_markov"][C.STATES.index("ask")]


def test_live_issue_equals_replay_once_history_covers_the_lookback():
    rng = np.random.default_rng(4)
    n = 31 * 96
    acts = _stream(list(rng.choice(["look", "make", "infer", None], p=[0.4, 0.3, 0.1, 0.2], size=n)))
    store = _Store(acts)
    ch = C.ActChain(store)
    origins = [T0 + (29 * 96 + j) * G for j in (0, 37, 90)]
    for o in origins:
        ch.tick(o + timedelta(minutes=1))
    rep = {r["origin"]: r for r in C.replay(acts, T0, T0 + n * G, warmup=timedelta(days=29))}
    for o in origins:
        live = store.rows["act_chain:" + o.strftime("%Y-%m-%dT%H:%M")]["probs"]
        for m in C.MODELS:
            assert np.allclose([live[m][s] for s in C.STATES], rep[o]["probs"][m], rtol=1e-12, atol=0)


def test_a_row_from_another_vocabulary_neither_blocks_settles_nor_breaks_the_grade():
    o = T0 + 96 * G
    store = _Store(_stream(["look"] * 98))
    ch = C.ActChain(store)
    ch.tick(o + timedelta(minutes=1))
    ch.tick(o + G + timedelta(minutes=1))
    first, second = sorted(store.rows.values(), key=lambda r: r["origin"])
    for m in first["probs"]:
        first["probs"][m].pop("look")                                          # issued before 'look' existed
    out = ch.tick(o + 3 * G + C.SETTLE_GRACE)
    assert out["settle_failed"] == 1 and out["settled"] == 1 and second["settled_at"] is not None
    first.update(observed="look", settled_at=o)                                # even if it got settled somehow
    live = ch.live(o + 3 * G + C.SETTLE_GRACE)
    assert live["ungradable_rows"] == 1 and live["grade"]["n_buckets"] == 1
