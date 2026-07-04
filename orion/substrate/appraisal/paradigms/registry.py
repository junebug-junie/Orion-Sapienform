from __future__ import annotations

from typing import Callable

from .base import AppraisalParadigm
from .repair_pressure_v2 import RepairPressureV2Paradigm

PARADIGM_REGISTRY: dict[str, Callable[[], AppraisalParadigm]] = {
    "repair_pressure": RepairPressureV2Paradigm,
}
