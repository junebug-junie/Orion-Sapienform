"""Bill drop directory: one JSON object per file.

Same contract whether Juniper types a bill in or the portal fetcher writes it:
`kind` is energy.bill.actual.v1 or energy.bill.forecast.v1, the other keys are that
schema's fields. `source` defaults to file_drop and `retrieved_at` to the drop time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Union

from orion.schemas.energy import (
    ENERGY_BILL_ACTUAL_KIND,
    ENERGY_BILL_FORECAST_KIND,
    EnergyBillActualV1,
    EnergyBillForecastV1,
)

from .inbox import replay_dir, scan_dir

Bill = Union[EnergyBillActualV1, EnergyBillForecastV1]
_MODELS = {ENERGY_BILL_ACTUAL_KIND: EnergyBillActualV1, ENERGY_BILL_FORECAST_KIND: EnergyBillForecastV1}


def parse_bill(raw: bytes, *, retrieved_at: datetime, source_file: str) -> Bill:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("bill file must be a JSON object")
    kind = data.pop("kind", None)
    model = _MODELS.get(kind) if isinstance(kind, str) else None
    if model is None:
        raise ValueError("bill file needs kind energy.bill.actual.v1 or energy.bill.forecast.v1")
    data.setdefault("source", "file_drop")
    data.setdefault("retrieved_at", retrieved_at.isoformat())
    data["source_file"] = source_file
    return model.model_validate(data)


def _parse(raw: bytes, retrieved_at: datetime, name: str) -> list[Bill]:
    return [parse_bill(raw, retrieved_at=retrieved_at, source_file=name)]


def scan_bills(inbox_dir: Path, processed_dir: Path, *, now: datetime) -> list[Bill]:
    return scan_dir(inbox_dir, processed_dir, now=now, suffix=".json", parse=_parse)


def load_processed_bills(processed_dir: Path) -> list[Bill]:
    return replay_dir(processed_dir, suffix=".json", parse=_parse)
