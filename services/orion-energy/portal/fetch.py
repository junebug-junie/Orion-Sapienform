"""One fetch attempt: usage XML + bills into the drop directories.

No credentials anywhere: a login redirect means the saved session died, and the
answer is `reauth_required` -- a human logs in once (MFA stays on), never a retry.
Anything empty or unparseable is an error with the raw artifact kept for debugging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from orion.energy.espi import EspiError, parse_espi

from .driver import PortalDriver
from .parse import bill_payload_from_fields, forecast_payload_from_fields, is_login_url

_STAMP = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class PortalOutcome:
    state: Literal["ok", "reauth_required", "error"]
    reason: str
    xml_file: Optional[Path] = None
    bill_files: tuple[Path, ...] = ()


def _atomic_write(directory: Path, name: str, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    part = directory / f".{name}.part"
    part.write_bytes(data)
    final = directory / name
    part.rename(final)
    return final


def _save_raw(raw_dir: Path, now: datetime, name: str, data: bytes) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{now.strftime(_STAMP)}-{name}").write_bytes(data)


async def _snapshot_html(driver: PortalDriver, raw_dir: Path, now: datetime) -> None:
    try:
        html = await driver.page_html()
    except Exception:  # noqa: BLE001 -- best effort debugging artifact
        return
    _save_raw(raw_dir, now, "billing.html", html.encode())


async def run_once(
    driver: PortalDriver,
    *,
    inbox_dir: Path,
    bill_inbox_dir: Path,
    raw_dir: Path,
    backfill_days: int,
    now: datetime,
) -> PortalOutcome:
    stamp = now.strftime(_STAMP)
    try:
        if is_login_url(await driver.open_usage()):
            return PortalOutcome("reauth_required", "session_expired")
        xml = await driver.download_green_button(days=backfill_days)
        try:
            intervals = parse_espi(xml, retrieved_at=now, source="rockymountain_power")
        except EspiError as exc:
            _save_raw(raw_dir, now, "green_button.xml", xml)
            return PortalOutcome("error", f"espi_invalid:{exc}"[:200])
        if not intervals:
            _save_raw(raw_dir, now, "green_button.xml", xml)
            return PortalOutcome("error", "empty_download")
        xml_file = _atomic_write(inbox_dir, f"rmp-portal-{stamp}.xml", xml)

        try:
            rows = await driver.billing_rows()
            forecast = await driver.forecast_fields()
        except Exception as exc:  # noqa: BLE001 -- any scrape failure is a visible error state
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", f"billing_scrape_failed:{type(exc).__name__}", xml_file=xml_file)
        if not rows:
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", "bill_rows_empty", xml_file=xml_file)
        try:
            payloads = [bill_payload_from_fields(row, retrieved_at=now) for row in rows]
            if forecast:
                payloads.append(forecast_payload_from_fields(forecast, retrieved_at=now))
        except ValueError as exc:
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", f"bill_parse_failed:{exc}"[:200], xml_file=xml_file)
        bill_files = tuple(
            _atomic_write(
                bill_inbox_dir,
                f"rmp-portal-{stamp}-{i:02d}.json",
                json.dumps(p).encode(),
            )
            for i, p in enumerate(payloads)
        )
        return PortalOutcome("ok", "fetched", xml_file=xml_file, bill_files=bill_files)
    except Exception as exc:  # noqa: BLE001 -- the loop must survive and report
        return PortalOutcome("error", type(exc).__name__)
