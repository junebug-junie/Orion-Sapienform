from __future__ import annotations

import json
from typing import Any

from orion.schemas.reduction_receipt import ReductionReceiptV1


def parse_receipt_json(payload: Any) -> ReductionReceiptV1:
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ReductionReceiptV1.model_validate(payload)
