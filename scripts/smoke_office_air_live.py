#!/usr/bin/env python3
"""Prove the live PAN-1 office-air path after an Athena deployment.

This smoke is deliberately read-only. It checks the Hub API and static asset, the
materialized Postgres row, the required air columns, and recent fallback-log debt.
It never reads or prints fallback payloads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


KIND = "office.env.sample.v1"
REQUIRED_COLUMNS = {"air_warming", "office_eco2_ppm", "office_tvoc_ppb"}
ASSET_MARKERS = ("CCS811", "officeEnvEco2", "officeEnvTvoc")
PAGE_MARKERS = ("officeEnvStrip", "officeEnvEco2", "officeEnvTvoc", "office-env.js")


class SmokeFailure(RuntimeError):
    """A failed live-path assertion with an operator-readable reason."""


@dataclass(frozen=True)
class Config:
    hub_base: str
    db_container: str
    db_name: str
    db_user: str
    max_age_sec: float
    fallback_window_hours: int
    allow_warming: bool
    timeout_sec: float


def fetch_text(url: str, timeout_sec: float) -> str:
    request = Request(url, headers={"Accept": "application/json, text/javascript"})
    try:
        with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - operator-supplied URL
            if response.status != 200:
                raise SmokeFailure(f"GET {url} returned HTTP {response.status}")
            return response.read().decode("utf-8")
    except (OSError, TimeoutError, URLError) as exc:
        raise SmokeFailure(f"GET {url} failed: {exc}") from exc


def fetch_json(url: str, timeout_sec: float) -> dict[str, Any]:
    try:
        payload = json.loads(fetch_text(url, timeout_sec))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeFailure(f"GET {url} did not return JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SmokeFailure(f"GET {url} returned a non-object JSON value")
    return payload


def run_psql(config: Config, sql: str) -> str:
    command = [
        "docker",
        "exec",
        config.db_container,
        "psql",
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        config.db_user,
        "-d",
        config.db_name,
        "-Atc",
        sql,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeFailure(f"Postgres check could not run: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise SmokeFailure(f"Postgres check failed: {detail[-1] if detail else 'unknown error'}")
    return result.stdout.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmokeFailure(f"{name} is missing or not numeric")
    return float(value)


def validate_reading(
    *,
    source: str,
    age_sec: Any,
    eco2: Any,
    tvoc: Any,
    stale: Any,
    warming: Any,
    max_age_sec: float,
    allow_warming: bool,
) -> dict[str, float | bool]:
    age = _number(age_sec, f"{source}.age_sec")
    if age < 0 or age > max_age_sec:
        raise SmokeFailure(f"{source} reading age {age:.1f}s exceeds {max_age_sec:.1f}s")
    if stale is not False:
        raise SmokeFailure(f"{source} reports stale={stale!r}")
    is_warming = warming is True
    if is_warming:
        if not allow_warming:
            raise SmokeFailure(f"{source} air sensor is still warming")
        return {"age_sec": age, "warming": True}

    eco2_value = _number(eco2, f"{source}.eco2")
    tvoc_value = _number(tvoc, f"{source}.tvoc")
    if not 400 <= eco2_value <= 29206:
        raise SmokeFailure(f"{source}.eco2 is outside the contract range: {eco2_value}")
    if not 0 <= tvoc_value <= 32768:
        raise SmokeFailure(f"{source}.tvoc is outside the contract range: {tvoc_value}")
    return {
        "age_sec": age,
        "eco2_ppm": eco2_value,
        "tvoc_ppb": tvoc_value,
        "warming": False,
    }


def validate_api(payload: dict[str, Any], config: Config) -> dict[str, float | bool]:
    if payload.get("ok") is not True:
        raise SmokeFailure(
            "Hub latest endpoint is not current "
            f"(sensor_stale={payload.get('sensor_stale')!r}, "
            f"publisher_stale={payload.get('publisher_stale')!r})"
        )
    sample = payload.get("sample")
    if not isinstance(sample, dict):
        raise SmokeFailure("Hub latest endpoint has no sample object")
    if sample.get("sensor_online") is not True:
        raise SmokeFailure("Hub latest endpoint reports sensor_online=false")
    return validate_reading(
        source="hub",
        age_sec=payload.get("age_sec"),
        eco2=sample.get("office_eco2_ppm"),
        tvoc=sample.get("office_tvoc_ppb"),
        stale=False,
        warming=sample.get("air_warming"),
        max_age_sec=config.max_age_sec,
        allow_warming=config.allow_warming,
    )


def _parse_json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{label} returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SmokeFailure(f"{label} returned no object")
    return value


def run_smoke(
    config: Config,
    *,
    json_get: Callable[[str, float], dict[str, Any]] = fetch_json,
    text_get: Callable[[str, float], str] = fetch_text,
    psql: Callable[[Config, str], str] = run_psql,
) -> dict[str, Any]:
    base = config.hub_base.rstrip("/")
    api = validate_api(json_get(f"{base}/api/office/env/latest", config.timeout_sec), config)

    page = text_get(f"{base}/", config.timeout_sec)
    missing_page_markers = [marker for marker in PAGE_MARKERS if marker not in page]
    if missing_page_markers:
        raise SmokeFailure(
            f"rendered Hub page is missing office UI markers: {', '.join(missing_page_markers)}"
        )

    asset = text_get(f"{base}/static/js/office-env.js", config.timeout_sec)
    missing_markers = [marker for marker in ASSET_MARKERS if marker not in asset]
    if missing_markers:
        raise SmokeFailure(f"Hub office asset is missing markers: {', '.join(missing_markers)}")

    columns_sql = """
        SELECT COALESCE(json_agg(column_name ORDER BY column_name)::text, '[]')
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'office_env_sample'
          AND column_name IN ('air_warming', 'office_eco2_ppm', 'office_tvoc_ppb');
    """
    columns_raw = psql(config, columns_sql)
    try:
        columns_value = json.loads(columns_raw)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"column check returned invalid JSON: {exc}") from exc
    if not isinstance(columns_value, list) or not all(isinstance(item, str) for item in columns_value):
        raise SmokeFailure("column check returned a non-string-list JSON value")
    columns = set(columns_value)
    if columns != REQUIRED_COLUMNS:
        raise SmokeFailure(f"office_env_sample air columns differ: {sorted(columns)}")

    latest_sql = """
        SELECT json_build_object(
            'age_sec', EXTRACT(EPOCH FROM (now() - ts)),
            'eco2', office_eco2_ppm,
            'tvoc', office_tvoc_ppb,
            'stale', stale,
            'warming', air_warming
        )::text
        FROM office_env_sample
        ORDER BY ts DESC
        LIMIT 1;
    """
    row = _parse_json_object(psql(config, latest_sql), "latest destination row")
    database = validate_reading(
        source="database",
        age_sec=row.get("age_sec"),
        eco2=row.get("eco2"),
        tvoc=row.get("tvoc"),
        stale=row.get("stale"),
        warming=row.get("warming"),
        max_age_sec=config.max_age_sec,
        allow_warming=config.allow_warming,
    )

    fallback_sql = f"""
        SELECT count(*)
        FROM bus_fallback_log
        WHERE kind = '{KIND}'
          AND created_at_ts > now() - interval '{config.fallback_window_hours} hours';
    """
    fallback_raw = psql(config, fallback_sql)
    try:
        fallback_count = int(fallback_raw)
    except ValueError as exc:
        raise SmokeFailure(f"fallback count is not an integer: {fallback_raw!r}") from exc
    if fallback_count:
        raise SmokeFailure(
            f"{fallback_count} {KIND} row(s) entered fallback in the last "
            f"{config.fallback_window_hours}h"
        )

    return {
        "ok": True,
        "hub": api,
        "database": database,
        "required_columns": sorted(columns),
        "fallback_count": fallback_count,
        "fallback_window_hours": config.fallback_window_hours,
    }


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-base", default="http://127.0.0.1:8080")
    parser.add_argument("--db-container", default="orion-athena-sql-db")
    parser.add_argument("--db-name", default="conjourney")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--max-age-sec", type=float, default=120.0)
    parser.add_argument("--fallback-window-hours", type=int, default=24)
    parser.add_argument("--allow-warming", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.max_age_sec <= 0 or args.timeout_sec <= 0:
        parser.error("--max-age-sec and --timeout-sec must be positive")
    if not 1 <= args.fallback_window_hours <= 168:
        parser.error("--fallback-window-hours must be between 1 and 168")
    return Config(
        hub_base=args.hub_base,
        db_container=args.db_container,
        db_name=args.db_name,
        db_user=args.db_user,
        max_age_sec=args.max_age_sec,
        fallback_window_hours=args.fallback_window_hours,
        allow_warming=args.allow_warming,
        timeout_sec=args.timeout_sec,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        result = run_smoke(config)
    except SmokeFailure as exc:
        print(f"OFFICE AIR LIVE SMOKE FAIL: {exc}", file=sys.stderr)
        return 1
    print("OFFICE AIR LIVE SMOKE PASS")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
