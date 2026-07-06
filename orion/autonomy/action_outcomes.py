from __future__ import annotations

import json
import os
from pathlib import Path

from orion.autonomy.models import ActionOutcomeRefV1

DEFAULT_STORE_PATH = "/tmp/orion-action-outcomes.json"
_MAX_OUTCOMES = 12


def _store_path() -> Path:
    return Path(os.getenv("ORION_ACTION_OUTCOME_STORE_PATH", DEFAULT_STORE_PATH))


def _load_raw() -> dict[str, list[dict]]:
    path = _store_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, list)}


def _save_raw(data: dict[str, list[dict]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_action_outcomes(subject: str) -> list[ActionOutcomeRefV1]:
    raw = _load_raw()
    bucket = raw.get(subject, [])
    out: list[ActionOutcomeRefV1] = []
    for item in bucket:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ActionOutcomeRefV1.model_validate(item))
        except Exception:
            continue
    return out


def append_action_outcome(subject: str, outcome: ActionOutcomeRefV1) -> None:
    data = _load_raw()
    bucket = data.get(subject, [])
    if not isinstance(bucket, list):
        bucket = []
    bucket.append(outcome.model_dump(mode="json"))
    data[subject] = bucket[-_MAX_OUTCOMES:]
    _save_raw(data)
