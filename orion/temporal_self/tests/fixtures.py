"""Small builders for reducer tests. Times are UTC; 2026-10-09 local (America/Denver, MDT,
UTC-6) runs 06:00Z 10-09 to 06:00Z 10-10."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from orion.schemas.temporal_self import TemporalSelfEventV1
from orion.temporal_self.broadcast import BroadcastTickView
from orion.temporal_self.day import day_id_for

T0 = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)  # 09:00 local
TICK = 30.0


def at(minutes: float, base: datetime = T0) -> datetime:
    return base + timedelta(minutes=minutes)


def ticks(refs: list[str | None], start: datetime = T0, step: float = TICK, prefix: str = "L") -> list[BroadcastTickView]:
    return [
        BroadcastTickView(log_id=f"{prefix}{i:05d}", generated_at=start + timedelta(seconds=i * step), ref=r, label=(r or ""))
        for i, r in enumerate(refs)
    ]


def ev(
    kind: str,
    ref: str,
    occurred: datetime,
    *,
    ended: datetime | None = None,
    table: str | None = None,
    subject: str | None = None,
    **fields: Any,
) -> TemporalSelfEventV1:
    return TemporalSelfEventV1(
        event_id=f"{kind}:{ref}", day_id=day_id_for(occurred), occurred_at=occurred, ended_at=ended,
        source_kind=kind, source_table=table or kind, source_ref=ref, subject_ref=subject, **fields,
    )


def run(ref: str, target: str, start: datetime, minutes: float, ticks_: int = 10, bar: int = 3) -> TemporalSelfEventV1:
    return ev("field_dominance_run", ref, start, ended=start + timedelta(minutes=minutes), subject=target,
              table="field_dominance_run", label=target,
              payload={"tick_count": ticks_, "min_streak_at_run": bar, "target_kind": "node", "left_censored": False})


def chat(ref: str, session: str, t: datetime, corr: str | None = None) -> TemporalSelfEventV1:
    return ev("chat_turn", ref, t, subject=session, table="chat_history_log", correlation_id=corr or f"c-{ref}",
              privacy_class="juniper_chat")


def metacog(ref: str, t: datetime) -> TemporalSelfEventV1:
    return ev("metacog_observation", ref, t, table="orion_metacog", verdict="degraded", payload={"trigger_kind": "transport"})


def attention_row(ref: str, t: datetime, process: str = "substrate_attention", reason: str = "bottom_up_salience",
                  corr: str | None = None) -> TemporalSelfEventV1:
    return ev("attention_row", ref, t, table="substrate_attention_schema", correlation_id=corr,
              payload={"process": process, "reason": reason})
