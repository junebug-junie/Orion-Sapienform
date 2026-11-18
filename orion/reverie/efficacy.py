"""Phase H — efficacy metrics for the reverie/dream weave.

Pure, deterministic reductions (§4) over already-emitted substrate facts. These
answer "is reverie doing anything useful?" without any LLM judgment:

  - pressure-discharge rate: did chains actually discharge the pressure that
    spawned them (SelfStateV1 pressure before vs after)?
  - recall delta: recall latency + memory-graph size before vs after compaction.

Each function degrades to a well-defined zero/None on empty input — an honest
"no evidence yet", never a fabricated score.
"""

from __future__ import annotations

from dataclasses import dataclass


def pressure_discharge_rate(before: list[float], after: list[float]) -> float | None:
    """Fraction of chains where post-chain pressure fell below pre-chain pressure.

    None on empty/misaligned input (no evidence). A high rate is the signal that
    chains resolve the loops that spawn them rather than feeding them.
    """
    n = min(len(before), len(after))
    if n == 0:
        return None
    discharged = sum(1 for b, a in zip(before[:n], after[:n]) if float(a) < float(b))
    return discharged / n


@dataclass(frozen=True)
class RecallDelta:
    """Before/after recall metrics around a compaction."""

    latency_ms_delta: float  # negative = faster after compaction (good)
    graph_size_delta: int  # negative = smaller graph after compaction (good)
    latency_ms_before: float
    latency_ms_after: float
    graph_size_before: int
    graph_size_after: int


def recall_delta(
    *,
    latency_ms_before: float,
    latency_ms_after: float,
    graph_size_before: int,
    graph_size_after: int,
) -> RecallDelta:
    """Deterministic before/after recall reduction. Negative deltas are wins:
    lower latency and a smaller graph after compaction."""
    return RecallDelta(
        latency_ms_delta=float(latency_ms_after) - float(latency_ms_before),
        graph_size_delta=int(graph_size_after) - int(graph_size_before),
        latency_ms_before=float(latency_ms_before),
        latency_ms_after=float(latency_ms_after),
        graph_size_before=int(graph_size_before),
        graph_size_after=int(graph_size_after),
    )
