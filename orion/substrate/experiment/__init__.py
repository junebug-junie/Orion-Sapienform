"""7-day substrate experiment harness.

The harness records lightweight daily metrics that answer one question:
*Did the shared substrate become more useful than bespoke organ state?*
"""

from .harness import SubstrateExperimentHarness
from .metrics import (
    DailyMetricsV1,
    GradientStats,
    OrganCoverage,
)
from .daily_rollup import compute_daily_rollup, write_daily_rollup
from .report import generate_week_report

__all__ = [
    "SubstrateExperimentHarness",
    "DailyMetricsV1",
    "GradientStats",
    "OrganCoverage",
    "compute_daily_rollup",
    "write_daily_rollup",
    "generate_week_report",
]
