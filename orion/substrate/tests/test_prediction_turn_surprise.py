"""node:prediction.chat_turn fold (orion/substrate/prediction_turn_surprise.py): rest, window, staleness."""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from orion.substrate.prediction_error_freshness import is_retired_prediction_error_node
from orion.substrate.prediction_turn_surprise import (
    CHAT_TURN_NODE_ID,
    WINDOW,
    TurnSurpriseState,
)

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def _ev(cid, excess, var=1.0, settled=NOW, caller="chat"):
    return {"correlation_id": cid, "caller": caller, "settled_at": settled, "excess": excess, "calm_var": var,
            "bets": [{}]}


def test_calm_turns_rest_at_exactly_zero():
    st = TurnSurpriseState()
    for i in range(5):
        assert st.add(_ev(f"c{i}", -0.1, 0.2), now=NOW) == (CHAT_TURN_NODE_ID, "accepted")
    r = st.reading(CHAT_TURN_NODE_ID, now=NOW)
    assert r.z < 0 and r.value == 0.0 and r.n == 5


def test_no_turn_means_no_reading_never_calm():
    st = TurnSurpriseState()
    assert st.reading(CHAT_TURN_NODE_ID, now=NOW) is None
    st.add(_ev("c", 2.0), now=NOW)
    # After the window empties the node has no reading: the writer stops, observed_at ages to stale.
    assert st.reading(CHAT_TURN_NODE_ID, now=NOW + WINDOW + timedelta(seconds=1)) is None


def test_a_backlog_turn_never_writes_the_past_as_now():
    st = TurnSurpriseState()
    assert st.add(_ev("old", 5.0, settled=NOW - WINDOW - timedelta(minutes=1)), now=NOW)[1] == "older_than_window"
    assert st.reading(CHAT_TURN_NODE_ID, now=NOW) is None


def test_the_same_turn_twice_counts_once_and_bad_inputs_are_refused():
    st = TurnSurpriseState()
    st.add(_ev("c", 1.0), now=NOW)
    st.add(_ev("c", 1.0), now=NOW)
    assert st.reading(CHAT_TURN_NODE_ID, now=NOW).n == 1
    assert st.add(_ev("x", 1.0, caller="orion_curiosity"), now=NOW) == (None, "unknown_caller")
    assert st.add(_ev("y", float("nan")), now=NOW)[1] == "non_finite_or_no_variance"
    assert st.add(_ev("z", 1.0, var=0.0), now=NOW)[1] == "non_finite_or_no_variance"


def test_window_z_is_standard_normal_for_calibrated_turns():
    rng = random.Random(1)
    zs = []
    for k in range(3000):
        st = TurnSurpriseState()
        for i in range(rng.randint(1, 4)):
            q = rng.choice([0.05, 0.1, 0.2, 0.3])
            y = rng.random() < q
            lg = math.log(q / (1 - q))
            st.add(_ev(f"{k}-{i}", -(y - q) * lg, q * (1 - q) * lg * lg), now=NOW)
        zs.append(st.reading(CHAT_TURN_NODE_ID, now=NOW).z)
    m = sum(zs) / len(zs)
    sd = math.sqrt(sum((z - m) ** 2 for z in zs) / len(zs))
    assert abs(m) < 0.06 and abs(sd - 1) < 0.06


def test_successor_is_live_and_the_old_chat_node_is_retired():
    assert not is_retired_prediction_error_node(CHAT_TURN_NODE_ID)
    assert is_retired_prediction_error_node("node:substrate.chat")
