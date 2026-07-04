from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .base import AppraisalParadigm
from .repair_pressure_v2 import RepairPressureV2Paradigm

_LLMCaller = Callable[[str], dict[str, Any] | Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ParadigmBuildContext:
    llm_caller: _LLMCaller
    weights_path: str


ParadigmFactory = Callable[[ParadigmBuildContext], AppraisalParadigm]


def _build_repair_pressure(ctx: ParadigmBuildContext) -> RepairPressureV2Paradigm:
    return RepairPressureV2Paradigm(
        llm_caller=ctx.llm_caller,
        weights_path=ctx.weights_path,
    )


PARADIGM_REGISTRY: dict[str, ParadigmFactory] = {
    "repair_pressure": _build_repair_pressure,
}
