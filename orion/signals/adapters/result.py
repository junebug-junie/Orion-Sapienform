from __future__ import annotations

from typing import List, Optional, Union

from orion.signals.models import OrionSignalV1

AdapterResult = Union[OrionSignalV1, List[OrionSignalV1], None]


def normalize_adapter_result(value: AdapterResult) -> List[OrionSignalV1]:
    if value is None:
        return []
    if isinstance(value, list):
        return [s for s in value if s is not None]
    return [value]
