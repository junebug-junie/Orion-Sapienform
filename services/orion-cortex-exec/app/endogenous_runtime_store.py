from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Protocol

from orion.core.schemas.endogenous_runtime import EndogenousRuntimeExecutionRecordV1


class EndogenousRuntimeRecordStore(Protocol):
    def write(self, record: EndogenousRuntimeExecutionRecordV1) -> EndogenousRuntimeExecutionRecordV1:
        ...

    def list_recent(
        self,
        *,
        limit: int = 25,
        invocation_surface: str | None = None,
        workflow_type: str | None = None,
        outcome: str | None = None,
        subject_ref: str | None = None,
        mentor_invoked: bool | None = None,
        created_after: datetime | None = None,
    ) -> list[EndogenousRuntimeExecutionRecordV1]:
        ...


class InMemoryEndogenousRuntimeRecordStore:
    def __init__(self) -> None:
        self._records: list[EndogenousRuntimeExecutionRecordV1] = []
        self._lock = Lock()

    def write(self, record: EndogenousRuntimeExecutionRecordV1) -> EndogenousRuntimeExecutionRecordV1:
        with self._lock:
            self._records.append(record)
        return record

    def list_recent(
        self,
        *,
        limit: int = 25,
        invocation_surface: str | None = None,
        workflow_type: str | None = None,
        outcome: str | None = None,
        subject_ref: str | None = None,
        mentor_invoked: bool | None = None,
        created_after: datetime | None = None,
    ) -> list[EndogenousRuntimeExecutionRecordV1]:
        with self._lock:
            records = list(self._records)
        return _filter_records(
            records,
            limit=limit,
            invocation_surface=invocation_surface,
            workflow_type=workflow_type,
            outcome=outcome,
            subject_ref=subject_ref,
            mentor_invoked=mentor_invoked,
            created_after=created_after,
        )


class JsonlEndogenousRuntimeRecordStore:
    """Append-only durable JSONL record store for endogenous runtime records."""

    def __init__(self, *, path: str, max_records: int = 2000) -> None:
        self._path = Path(path)
        self._max_records = max(100, max_records)
        self._lock = Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    def write(self, record: EndogenousRuntimeExecutionRecordV1) -> EndogenousRuntimeExecutionRecordV1:
        line = json.dumps(record.model_dump(mode="json"), sort_keys=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._trim_if_needed()
        return record

    def list_recent(
        self,
        *,
        limit: int = 25,
        invocation_surface: str | None = None,
        workflow_type: str | None = None,
        outcome: str | None = None,
        subject_ref: str | None = None,
        mentor_invoked: bool | None = None,
        created_after: datetime | None = None,
    ) -> list[EndogenousRuntimeExecutionRecordV1]:
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        records: list[EndogenousRuntimeExecutionRecordV1] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                records.append(EndogenousRuntimeExecutionRecordV1.model_validate(payload))
            except Exception:
                continue
        return _filter_records(
            records,
            limit=limit,
            invocation_surface=invocation_surface,
            workflow_type=workflow_type,
            outcome=outcome,
            subject_ref=subject_ref,
            mentor_invoked=mentor_invoked,
            created_after=created_after,
        )

    def _trim_if_needed(self) -> None:
        lines = self._path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self._max_records:
            return
        kept = lines[-self._max_records :]
        self._path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _filter_records(
    records: list[EndogenousRuntimeExecutionRecordV1],
    *,
    limit: int,
    invocation_surface: str | None,
    workflow_type: str | None,
    outcome: str | None,
    subject_ref: str | None,
    mentor_invoked: bool | None,
    created_after: datetime | None,
) -> list[EndogenousRuntimeExecutionRecordV1]:
    out: list[EndogenousRuntimeExecutionRecordV1] = []
    for record in reversed(records):
        if invocation_surface and record.invocation_surface != invocation_surface:
            continue
        if workflow_type and record.decision.workflow_type != workflow_type:
            continue
        if outcome and record.decision.outcome != outcome:
            continue
        if subject_ref and record.subject_ref != subject_ref:
            continue
        if mentor_invoked is not None and record.mentor_invoked is not mentor_invoked:
            continue
        if created_after and record.created_at < created_after:
            continue
        out.append(record)
        if len(out) >= max(0, limit):
            break
    return out
