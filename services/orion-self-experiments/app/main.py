from __future__ import annotations

import logging
import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from orion.cognition.skills_manifest import load_skill_manifest


logger = logging.getLogger("orion-self-experiments")


class Settings(BaseSettings):
    service_name: str = Field("orion-self-experiments", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    experiments_store_path: str = Field("/tmp/orion-self-experiments/experiments.sqlite3", alias="EXPERIMENTS_STORE_PATH")
    experiments_allow_non_read_only: bool = Field(False, alias="EXPERIMENTS_ALLOW_NON_READ_ONLY")
    port: int = Field(7172, alias="PORT")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()


class ExperimentCreateRequest(BaseModel):
    skill_id: str = Field(min_length=1, max_length=120)
    provenance: dict[str, Any] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)


class ExperimentCreateResponse(BaseModel):
    ok: bool
    experiment_id: str
    status: str
    message: str | None = None


class ExperimentRecord(BaseModel):
    experiment_id: str
    skill_id: str
    status: str
    reason: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str


class ExperimentListResponse(BaseModel):
    ok: bool = True
    total: int
    items: list[ExperimentRecord] = Field(default_factory=list)


def _connect() -> sqlite3.Connection:
    db_path = Path(settings.experiments_store_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                skill_id TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                provenance_json TEXT NOT NULL,
                args_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            )
            """
        )
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    level_name = (settings.log_level or "INFO").upper()
    logging.basicConfig(level=getattr(logging, level_name, logging.INFO))
    _init_db()
    yield


app = FastAPI(title="orion-self-experiments", version=settings.service_version, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": settings.service_name, "version": settings.service_version}


@app.post("/v1/experiments", response_model=ExperimentCreateResponse)
def create_experiment(req: ExperimentCreateRequest) -> ExperimentCreateResponse:
    manifest = load_skill_manifest()
    entries = {item.skill_id: item for item in manifest}
    entry = entries.get(req.skill_id)
    if entry is None:
        status = "rejected"
        reason = "unknown_skill_id"
    elif not entry.read_only and not settings.experiments_allow_non_read_only:
        status = "rejected"
        reason = "non_read_only_skill_rejected"
    else:
        status = "validated"
        reason = None

    experiment_id = str(uuid4())
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO experiments (id, skill_id, status, reason, provenance_json, args_json, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                req.skill_id,
                status,
                reason,
                json.dumps(req.provenance, sort_keys=True),
                json.dumps(req.args, sort_keys=True),
                now_utc,
            ),
        )
        conn.commit()

    return ExperimentCreateResponse(
        ok=(status == "validated"),
        experiment_id=experiment_id,
        status=status,
        message=reason,
    )


@app.get("/v1/experiments", response_model=ExperimentListResponse)
def list_experiments(
    limit: int = Query(default=25, ge=1, le=200),
    correlation_id: str | None = Query(default=None),
    skill_id: str | None = Query(default=None),
    date: str | None = Query(default=None),
) -> ExperimentListResponse:
    where_clauses: list[str] = []
    values: list[Any] = []
    if correlation_id:
        where_clauses.append("provenance_json LIKE ?")
        values.append(f"%{str(correlation_id)}%")
    if skill_id:
        where_clauses.append("skill_id = ?")
        values.append(str(skill_id))
    if date:
        where_clauses.append("provenance_json LIKE ?")
        values.append(f"%{str(date)}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    query = f"""
        SELECT id, skill_id, status, reason, provenance_json, args_json, created_at_utc
        FROM experiments
        {where_sql}
        ORDER BY created_at_utc DESC
        LIMIT ?
    """
    values.append(int(limit))

    with _connect() as conn:
        rows = conn.execute(query, values).fetchall()
    items: list[ExperimentRecord] = []
    for row in rows:
        provenance_raw = row["provenance_json"] or "{}"
        args_raw = row["args_json"] or "{}"
        try:
            provenance = json.loads(provenance_raw)
        except Exception:
            provenance = {}
        try:
            args_payload = json.loads(args_raw)
        except Exception:
            args_payload = {}
        items.append(
            ExperimentRecord(
                experiment_id=str(row["id"]),
                skill_id=str(row["skill_id"]),
                status=str(row["status"]),
                reason=str(row["reason"]) if row["reason"] else None,
                provenance=provenance if isinstance(provenance, dict) else {},
                args=args_payload if isinstance(args_payload, dict) else {},
                created_at_utc=str(row["created_at_utc"]),
            )
        )
    return ExperimentListResponse(total=len(items), items=items)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port)
