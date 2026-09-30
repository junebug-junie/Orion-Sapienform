"""Immutable stage-attempt bindings to the existing durable runner API."""

import json
from uuid import NAMESPACE_URL, uuid5

import httpx

from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.reading_turn import (
    READING_WORKFLOW,
    ReadingRunBriefV1,
    ReadingTurnResultV1,
)
from orion.schemas.resource_admission import ResourceRequirementV1


READING_DURABLE_SQL = """
CREATE TABLE IF NOT EXISTS reading_durable_turn (
    seed_id text NOT NULL REFERENCES world_pulse_read_seed(seed_id),
    stage integer NOT NULL CHECK (stage IN (1, 2)),
    attempt integer NOT NULL,
    run_id text NOT NULL UNIQUE,
    request_json jsonb NOT NULL,
    consumed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (seed_id, stage, attempt)
);
"""


# ResourceRequirementV1 fields deleted in GPU pool stage 4.6. Bindings committed before it were
# dumped with them (defaults: false / [] / null); the model is extra="forbid", so a stored row is
# read back without them. The stored prompt/run id stay authoritative; these keys never meant
# anything to the pool. Durable-runs' duplicate-receipt comparison ignores them too
# (orion/durable_runs/registry_store.py IGNORED_ADMISSION_FIELDS), so a resubmit is the same run.
PRE_4_6_ADMISSION_KEYS = ("allow_elastic_activation", "alternatives", "pinned_lane", "operator_override")


READING_RETRIEVAL_QUERY_SEPARATOR = " — "


def reading_retrieval_query(seed, claim: str | None = None) -> str | None:
    """What recall searches for during a reading turn: "<source title> — <stage-1 claim>".

    Stage 1 has no claim yet, so it searches the title alone. A seed with no
    title falls back to its URL, except a pinned document ref (an opaque hash
    is not something memory can match). Capped at RecallQueryV1's 1000 chars.
    """
    from orion.cognition.recall_query import cap_retrieval_query
    from orion.world_pulse_read.documents import is_document_ref

    source = " ".join(str(getattr(seed, "title", "") or "").split())
    if not source:
        url = str(getattr(seed, "url", "") or "").strip()
        source = "" if is_document_ref(url) else url
    claim_text = " ".join(str(claim or "").split())
    parts = [p for p in (source, claim_text) if p]
    return cap_retrieval_query(READING_RETRIEVAL_QUERY_SEPARATOR.join(parts))


def _stored_request(raw) -> DurableRunRequestV1:
    data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    admission = data.get("admission")
    if isinstance(admission, dict):
        data["admission"] = {k: v for k, v in admission.items() if k not in PRE_4_6_ADMISSION_KEYS}
    return DurableRunRequestV1.model_validate(data)


class ReadingPending(Exception):
    """No settled model result yet. Never charge a wallet or failed attempt."""


class ReadingCancelled(Exception):
    """Operator cancellation stops the seed without spending a retry."""


async def bind_turn(conn, brief: ReadingRunBriefV1, correlation_id: str):
    async with conn.transaction():
        found = await conn.fetchval(
            "SELECT seed_id FROM world_pulse_read_seed WHERE seed_id=$1 FOR UPDATE",
            brief.seed_id,
        )
        if found is None:
            raise ValueError("reading seed disappeared")
        active = await conn.fetchval(
            "SELECT request_json FROM reading_durable_turn WHERE seed_id=$1 AND stage=$2 AND consumed_at IS NULL ORDER BY attempt DESC LIMIT 1",
            brief.seed_id,
            brief.stage,
        )
        if active is not None:
            return _stored_request(active)
        attempt = await conn.fetchval(
            "SELECT COALESCE(MAX(attempt), -1) + 1 FROM reading_durable_turn WHERE seed_id=$1 AND stage=$2",
            brief.seed_id,
            brief.stage,
        )
        return await _insert_binding(conn, brief, correlation_id, attempt)


async def _insert_binding(conn, brief, correlation_id, attempt):
    run_id = "reading-" + str(
        uuid5(NAMESPACE_URL, f"{brief.seed_id}:{brief.stage}:{attempt}")
    )
    request = DurableRunRequestV1(
        run_id=run_id,
        workflow=READING_WORKFLOW,
        correlation_id=correlation_id,
        brief=brief,
        admission=ResourceRequirementV1(),
    )
    # The first committed prompt is authoritative across restarts/config changes.
    await conn.execute(
        """INSERT INTO reading_durable_turn(seed_id,stage,attempt,run_id,request_json)
        VALUES($1,$2,$3,$4,$5::jsonb) ON CONFLICT(seed_id,stage,attempt) DO NOTHING""",
        brief.seed_id,
        brief.stage,
        attempt,
        run_id,
        request.model_dump_json(),
    )
    raw = await conn.fetchval(
        "SELECT request_json FROM reading_durable_turn WHERE seed_id=$1 AND stage=$2 AND attempt=$3",
        brief.seed_id,
        brief.stage,
        attempt,
    )
    return _stored_request(raw)


async def poll_turn(request: DurableRunRequestV1, base_url: str) -> ReadingTurnResultV1:
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=10
        ) as client:
            submitted = await client.post("/runs", json=request.model_dump(mode="json"))
            submitted.raise_for_status()
            status = await client.get(f"/runs/{request.run_id}")
            status.raise_for_status()
            state = status.json()
    except (httpx.HTTPError, ValueError) as exc:
        # Acceptance may already have committed. Reuse this exact binding next
        # tick; never fall back to an unadmitted turn or mint a second run.
        raise ReadingPending(
            f"durable_unavailable:{request.run_id}:{type(exc).__name__}"
        ) from exc
    if (
        state.get("run_id") != request.run_id
        or state.get("workflow_kind") != READING_WORKFLOW
    ):
        raise ReadingPending(f"durable_identity_mismatch:{request.run_id}")
    if state["status"] == "completed":
        result = ReadingTurnResultV1.model_validate(state.get("reading_result"))
        if result.run_id != request.run_id or not result.ok or not result.text.strip():
            raise ValueError("invalid completed reading turn")
        return result
    if state["status"] in {"failed", "abandoned"}:
        if state.get("work_started") is False:
            raise ValueError(
                f"turn_deferred:reading_admission:{state.get('error') or state['status']}"
            )
        raise ValueError(state.get("error") or f"turn_error:{state['status']}")
    if state["status"] == "cancelled":
        raise ReadingCancelled(request.run_id)
    raise ReadingPending(f"durable:{request.run_id}:{state['status']}")


async def release_claim(conn, seed_id: str, stage: int):
    status, claimed = (
        ("status", "claimed_at")
        if stage == 1
        else ("stage2_status", "stage2_claimed_at")
    )
    await conn.execute(
        f"UPDATE world_pulse_read_seed SET {status}='pending', {claimed}=NULL "
        f"WHERE seed_id=$1 AND {status}='claimed'",
        seed_id,
    )


async def consume_turn(conn, run_id):
    await conn.execute(
        "UPDATE reading_durable_turn SET consumed_at=now() WHERE run_id=$1 AND consumed_at IS NULL",
        run_id,
    )


OPERATOR_CANCEL_REASON = "reading_cancelled_by_operator"


async def cancel_claim(conn, seed_id, stage):
    status, error = (
        ("status", "last_error") if stage == 1 else ("stage2_status", "stage2_error")
    )
    async with conn.transaction():
        await conn.execute(
            f"UPDATE world_pulse_read_seed SET {status}='skipped', {error}=$2 WHERE seed_id=$1",
            seed_id,
            OPERATOR_CANCEL_REASON,
        )
        await conn.execute(
            "UPDATE reading_durable_turn SET consumed_at=now() WHERE seed_id=$1 AND stage=$2 AND consumed_at IS NULL",
            seed_id,
            stage,
        )
