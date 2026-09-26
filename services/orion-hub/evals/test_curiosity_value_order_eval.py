"""Offline behavioural eval: does value ordering stop spending curiosity turns
on beliefs that cannot be learned?

SIMULATION, NOT LIVE EVIDENCE. The live question -- does the value arm buy
more net belief change per run than the uncertainty arm -- is answered by the
per-run randomized comparison on real `curiosity_run_outcomes` rows
(Acceptance check 5 of docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md).
This eval pins the mechanism the design claims, using the production
functions end to end (`select_priors`, `build_yield_model`, `kl_nats`), on a
population whose learnability is known:

- learnable: each test moves the belief halfway toward the truth;
- noisy: each test flips the belief between 0.3 and 0.7 (the "noisy TV" --
  surprising every time, learnable never);
- settled: at 0.95, tests do not move it.

Each simulated run tests the top offered prior (a stand-in for Orion's
choice). The noisy priors are MORE uncertain than a half-learned belief, so
most-uncertain-first keeps returning to them; measured progress should not.

What it found, and what it does NOT claim:

- Without the `stale_after` count rule, value order spends 8 of 24 runs on
  the flip-floppers against 20, and resolves all 4 learnable beliefs against 0.
- WITH the live rule (`HUB_CURIOSITY_STALE_PRIOR_TESTS=3`) the two orders tie
  on this population: the count rule caps every belief at 3 tests in the
  ordered list, learnable ones included, so it retires beliefs that were still
  moving before they resolve. That is the case for replacing the count rule
  with measured yield -- a separate, live-behaviour decision, not made here.
- Once nothing learnable is left, value order still spends runs on the best of
  bad options. Refusing to spend at all needs phase 2's floor.

"Learning" is scored only on the learnable beliefs (net KL toward the truth):
a flip-flopper that happens to end on the far side of its start is not
learning, and a whole-population net would count it as if it were.
"""

from __future__ import annotations

from orion.curiosity.value import PriorTestRecord, build_yield_model, kl_nats
from orion.curiosity.worldview import select_priors

RUNS = 24


def _population():
    priors = {}
    for i in range(4):
        priors[f"learn{i}"] = {"kind": "learnable", "conf": 0.55, "truth": 1.0 if i % 2 == 0 else 0.0}
    for i in range(4):
        priors[f"noise{i}"] = {"kind": "noisy", "conf": 0.7, "flip": 0}
    for i in range(2):
        priors[f"settled{i}"] = {"kind": "settled", "conf": 0.95}
    for p in priors.values():
        p["initial"] = p["conf"]
        p["tested"] = 0
    return priors


def _test(prior) -> float:
    kind = prior["kind"]
    if kind == "learnable":
        return prior["conf"] + 0.5 * (prior["truth"] - prior["conf"])
    if kind == "noisy":
        prior["flip"] += 1
        return 0.3 if prior["flip"] % 2 else 0.7
    return prior["conf"]


def _simulate(*, value_order: bool, stale_after: int = 0) -> dict:
    priors = _population()
    history: list[PriorTestRecord] = []
    runs_on = {"learnable": 0, "noisy": 0, "settled": 0}
    for run in range(RUNS):
        rows = [
            {"prior_id": pid, "claim": pid, "confidence": p["conf"], "status": "open",
             "times_tested": p["tested"], "formed_from": "", "last_tested_at": ""}
            for pid, p in priors.items()
        ]
        model = build_yield_model(history, window=3, pseudo_tests=2.0)
        offered, stale, _ = select_priors(
            rows,
            sample=3,
            stale_after=stale_after,
            rotate_seed=f"run{run}",
            expected_nats_for=(
                (lambda p: model.expected_nats(p.prior_id, p.confidence)) if value_order else None
            ),
        )
        pick = (offered or stale)[0].prior_id
        prior = priors[pick]
        before = prior["conf"]
        after = _test(prior)
        prior["conf"] = after
        prior["tested"] += 1
        history.append(PriorTestRecord(pick, before, after))
        runs_on[prior["kind"]] += 1
    learned = sum(
        kl_nats(p["conf"], p["initial"]) for p in priors.values() if p["kind"] == "learnable"
    )
    resolved = sum(
        1 for p in priors.values()
        if p["kind"] == "learnable" and abs(p["conf"] - p["truth"]) < 0.05
    )
    return {"runs_on": runs_on, "learned_nats": learned, "resolved": resolved}


def test_value_order_stops_feeding_the_noisy_tv() -> None:
    uncertainty = _simulate(value_order=False)
    value = _simulate(value_order=True)
    # Most-uncertain-first keeps returning to the flip-floppers.
    assert uncertainty["runs_on"]["noisy"] >= 8
    # Measured progress stops paying them after a few tests.
    assert value["runs_on"]["noisy"] <= uncertainty["runs_on"]["noisy"] // 2
    assert value["runs_on"]["learnable"] > uncertainty["runs_on"]["learnable"]


def test_value_order_buys_more_learning_from_the_same_runs() -> None:
    uncertainty = _simulate(value_order=False)
    value = _simulate(value_order=True)
    assert value["learned_nats"] > 2 * uncertainty["learned_nats"]
    assert value["resolved"] == 4
    assert uncertainty["resolved"] == 0


def test_under_the_live_stale_rule_the_orders_tie_and_learnable_beliefs_are_cut_off() -> None:
    # Recorded as a finding, not a win: with `stale_after=3` every belief gets
    # at most 3 tests in the ordered list, so over 24 runs both orders test the
    # same set -- and no learnable belief gets the 4th test it needed to
    # resolve. The count rule, not the ordering, decides what gets learned.
    uncertainty = _simulate(value_order=False, stale_after=3)
    value = _simulate(value_order=True, stale_after=3)
    assert value["runs_on"] == uncertainty["runs_on"]
    assert value["resolved"] == uncertainty["resolved"] == 0
    # The same value order with no count rule resolves all four.
    assert _simulate(value_order=True)["resolved"] == 4
