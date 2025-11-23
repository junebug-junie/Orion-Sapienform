from __future__ import annotations

"""
Strategy Tracking (v0)
======================

This module introduces lightweight tracking for "strategy ids" and
their interaction with the self-field φ.

We’re not evolving anything yet, just creating the data spine you’ll
need for the breeder later.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import time


@dataclass
class StrategyConfig:
    """
    Minimal description of a reasoning strategy.

    For now, this is mostly metadata. You can expand this over time to
    encode richer "genomes" (ordering of agents, temps, etc.).
    """
    id: str
    description: str
    agents: List[str]


@dataclass
class StrategyUsageRecord:
    """
    A single use of a strategy on a specific interaction.
    """
    strategy_id: str
    timestamp: float
    context_label: str        # e.g. "chat", "dream", "cortex-task"
    phi_before: Dict[str, float]
    phi_after: Dict[str, float]
    predictions: Dict[str, float]
    outcomes: Dict[str, float]


class StrategyTracker:
    """
    In-memory strategy usage tracker.

    v0: fully functional, just not persisted yet.
    """

    def __init__(self) -> None:
        self.strategies: Dict[str, StrategyConfig] = {}
        self.records: List[StrategyUsageRecord] = []

    def register_strategy(self, config: StrategyConfig) -> None:
        """
        Add or update a strategy configuration.
        """
        self.strategies[config.id] = config

    def start_usage(
        self,
        strategy_id: str,
        *,
        context_label: str,
        phi_before: Dict[str, float],
        predictions: Optional[Dict[str, float]] = None,
    ) -> StrategyUsageRecord:
        """
        Create a StrategyUsageRecord with φ_before and predictions.

        You’ll later call `finalize_usage` with φ_after and outcomes.
        """
        if strategy_id not in self.strategies:
            self.strategies[strategy_id] = StrategyConfig(
                id=strategy_id,
                description="(auto-registered)",
                agents=[],
            )

        rec = StrategyUsageRecord(
            strategy_id=strategy_id,
            timestamp=time.time(),
            context_label=context_label,
            phi_before=dict(phi_before),
            phi_after={},
            predictions=predictions or {},
            outcomes={},
        )
        return rec

    def finalize_usage(
        self,
        record: StrategyUsageRecord,
        *,
        phi_after: Dict[str, float],
        outcomes: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Attach φ_after and outcomes to a record, and store it.
        """
        record.phi_after = dict(phi_after)
        record.outcomes = outcomes or {}
        self.records.append(record)

    def export_records(self) -> List[Dict[str, Any]]:
        """
        Export all records as plain dicts (for DB / CM ingestion).
        """
        out: List[Dict[str, Any]] = []
        for r in self.records:
            d = {
                "strategy_id": r.strategy_id,
                "timestamp": r.timestamp,
                "context_label": r.context_label,
                "phi_before": r.phi_before,
                "phi_after": r.phi_after,
                "predictions": r.predictions,
                "outcomes": r.outcomes,
            }
            out.append(d)
        return out

    def clear_records(self) -> None:
        """
        Clear the in-memory record buffer.
        """
        self.records.clear()
