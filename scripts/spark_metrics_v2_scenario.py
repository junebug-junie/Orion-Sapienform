"""
Deterministic scenario script for Spark metrics v2.

Sequence:
- 10 routine chat inputs → novelty stabilizes
- 1 article summary input → novelty spikes
- Return to routine → novelty returns toward baseline
"""

import os
from pathlib import Path

from orion.spark.spark_engine import SparkEngine


def main() -> None:
    # Ensure we don't load any persisted tissue snapshot
    os.environ["ORION_TISSUE_SNAPSHOT_PATH"] = "/tmp/spark_metrics_v2_scenario.npy"
    try:
        Path(os.environ["ORION_TISSUE_SNAPSHOT_PATH"]).unlink()
    except FileNotFoundError:
        pass

    engine = SparkEngine(H=8, W=8, C=4)
    routine_text = "Routine check-in about the day."
    summary_text = "Article summary: advances in reinforcement learning for robotics."

    novelty_track = []
    for _ in range(10):
        state = engine.record_chat(
            routine_text,
            agent_id="scenario",
            tags=["juniper", "chat", "phase:pre", "mode:chat"],
        )
        novelty_track.append(state["phi"]["novelty"])

    summary_state = engine.record_chat(
        summary_text,
        agent_id="scenario",
        tags=["juniper", "chat", "mode:summarize", "phase:pre"],
    )
    novelty_track.append(summary_state["phi"]["novelty"])

    back_to_routine = engine.record_chat(
        routine_text,
        agent_id="scenario",
        tags=["juniper", "chat", "phase:pre", "mode:chat"],
    )
    novelty_track.append(back_to_routine["phi"]["novelty"])

    print("Novelties:", novelty_track)
    print(f"Baseline final: {novelty_track[9]:.4f}")
    print(f"Summary spike: {novelty_track[10]:.4f}")
    print(f"Post-return:   {novelty_track[11]:.4f}")


if __name__ == "__main__":
    main()
