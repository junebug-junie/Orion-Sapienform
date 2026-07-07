"""The lock breaks: same high-pressure node over N ticks eventually loses the
coalition once habituation engages. Uses injected history (no clock/DB).

Note on the competitor salience (0.7): with the seed weights the net habituation
penalty for a fully saturated theme is ~-0.10, while the salience-driven terms
(evidence_strength + novelty_vs_known, combined weight 0.50) dominate. A stuck
loop at 0.9 only loses to a competitor whose intrinsic salience is >= ~0.66, so a
0.6 competitor would NOT break the lock (stuckN=0.45 vs freshN=0.43). 0.7 keeps
the stuck loop initially dominant (0.55 > 0.47) yet lets habituation flip it
(freshN=0.47 > stuckN=0.45). See the Task 11 report for the full analysis.
"""

from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1
from orion.substrate.attention.salience import SalienceHistory, compute_salience


def _loop(loop_id: str) -> OpenLoopV1:
    return OpenLoopV1(id=loop_id, target_type="anomaly", description=loop_id)


def _sig(loop_id: str, salience: float) -> AttentionSignalV1:
    return AttentionSignalV1(
        signal_id=f"s-{loop_id}", source="substrate_broadcast", target_text=loop_id,
        signal_kind="substrate_pressure", salience=salience, confidence=0.8,
        evidence_refs=[f"{loop_id}-n"],
    )


def test_habituation_demotes_stuck_loop_below_competitor():
    stuck = _loop("open-loop-stuck")
    fresh = _loop("open-loop-fresh")

    stuck0, _ = compute_salience(loop=stuck, signals=[_sig("open-loop-stuck", 0.9)],
                                 history=SalienceHistory(), apply_habituation=True)
    fresh0, _ = compute_salience(loop=fresh, signals=[_sig("open-loop-fresh", 0.7)],
                                 history=SalienceHistory(), apply_habituation=True)
    assert stuck0 > fresh0

    stuck_hist = SalienceHistory(dwell_ticks=8, recent_theme_counts={"open-loop-stuck": 8},
                                 resonance_theme_keys={"open-loop-stuck"})
    stuckN, feats = compute_salience(loop=stuck, signals=[_sig("open-loop-stuck", 0.9)],
                                     history=stuck_hist, apply_habituation=True)
    freshN, _ = compute_salience(loop=fresh, signals=[_sig("open-loop-fresh", 0.7)],
                                 history=SalienceHistory(), apply_habituation=True)
    assert feats.habituation > 0.5
    assert freshN > stuckN, f"lock did not break: {stuckN=} {freshN=}"
