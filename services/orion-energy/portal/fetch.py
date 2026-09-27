"""One fetch attempt: usage XML + bills into the drop directories.

No credentials anywhere: a login redirect means the saved session died, and the
answer is `reauth_required` -- a human logs in once (MFA stays on), never a retry.
Anything empty or unparseable is an error with the raw artifact kept for debugging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from orion.energy.espi import EspiError, parse_espi

from .driver import PortalDriver
from .parse import PortalFieldError, bill_payload_from_fields, forecast_payload_from_fields, is_login_url

logger = logging.getLogger("orion-energy-portal")

_STAMP = "%Y%m%dT%H%M%SZ"
_ATTRS = r"""(?:[^>"']|"[^"]*"|'[^']*')*"""
_SCRIPT = re.compile(rf"(<script\b{_ATTRS}>).*?(</script\s*>|\Z)", re.IGNORECASE | re.DOTALL)
_INPUT = re.compile(rf"<input\b{_ATTRS}>", re.IGNORECASE)
_HIDDEN = re.compile(r"""(?<![\w-])type\s*=\s*["']?hidden\b""", re.IGNORECASE)
_VALUE = re.compile(r"""(?<![\w-])(value\s*=\s*)("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)
# Per-retrieval stamps; a bill whose other fields are unchanged is the same bill.
_VOLATILE_KEYS = ("retrieved_at", "as_of")


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


def scrub_html(html: str) -> str:
    """Drop inline script bodies and hidden-input values (CSRF/session tokens) before disk."""

    def _input(match: re.Match[str]) -> str:
        tag = match.group(0)
        return _VALUE.sub(r'\1""', tag) if _HIDDEN.search(tag) else tag

    return _INPUT.sub(_input, _SCRIPT.sub(lambda m: m.group(1) + m.group(2), html))


def _scrub_bytes(data: bytes) -> bytes:
    try:
        return scrub_html(data.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError:
        return data


def _save_raw(raw_dir: Path, now: datetime, name: str, data: bytes) -> None:
    """Best effort: a failed save is logged and never replaces the caller's error reason."""
    try:
        raw_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        raw_dir.chmod(0o700)
        path = raw_dir / f"{now.strftime(_STAMP)}-{name}"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(_scrub_bytes(data))
        path.chmod(0o600)
    except OSError as exc:
        logger.warning("energy_portal_raw_save_failed name=%s error=%s", name, type(exc).__name__)


async def _snapshot_html(driver: PortalDriver, raw_dir: Path, now: datetime) -> None:
    try:
        html = await driver.page_html()
    except Exception:  # noqa: BLE001 -- best effort debugging artifact
        return
    _save_raw(raw_dir, now, "billing.html", html.encode())


def _field_reason(prefix: str, exc: ValueError) -> str:
    if isinstance(exc, PortalFieldError):
        return f"{prefix}:{exc.cause}:{exc.field}"
    return f"{prefix}:{type(exc).__name__}"


def _natural_key(payload: dict[str, Any]) -> str:
    return f"{payload['kind']}:{payload['billing_period_start']}:{payload['billing_period_end']}"


def _content_hash(payload: dict[str, Any]) -> str:
    stable = {k: v for k, v in payload.items() if k not in _VOLATILE_KEYS}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _load_seen(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    seen = raw.get("seen") if isinstance(raw, dict) else None
    return {str(k): str(v) for k, v in seen.items()} if isinstance(seen, dict) else {}


def _store_seen(path: Optional[Path], seen: dict[str, str]) -> None:
    if path is None:
        return
    try:
        _atomic_write(path.parent, path.name, json.dumps({"version": 1, "seen": seen}, sort_keys=True).encode())
    except OSError as exc:
        logger.warning("energy_portal_bills_seen_write_failed error=%s", type(exc).__name__)


def _write_new_bills(
    payloads: list[dict[str, Any]], *, bill_inbox_dir: Path, seen_path: Optional[Path], stamp: str
) -> tuple[Path, ...]:
    """Write only bills/forecasts whose content changed since the last delivered copy."""
    seen = _load_seen(seen_path)
    written: list[Path] = []
    for i, payload in enumerate(payloads):
        key, digest = _natural_key(payload), _content_hash(payload)
        if seen.get(key) == digest:
            continue
        written.append(
            _atomic_write(bill_inbox_dir, f"rmp-portal-{stamp}-{i:02d}.json", json.dumps(payload).encode())
        )
        seen[key] = digest
    if written:
        _store_seen(seen_path, seen)
    return tuple(written)


async def run_once(
    driver: PortalDriver,
    *,
    inbox_dir: Path,
    bill_inbox_dir: Path,
    raw_dir: Path,
    backfill_days: int,
    now: datetime,
    seen_path: Optional[Path] = None,
) -> PortalOutcome:
    stamp = now.strftime(_STAMP)
    try:
        if is_login_url(await driver.open_usage()):
            return PortalOutcome("reauth_required", "session_expired")
        xml = await driver.download_green_button(days=backfill_days, now=now)
        if not xml:
            return PortalOutcome("error", "empty_download")
        try:
            parse_espi(xml, retrieved_at=now, source="rockymountain_power")
        except EspiError as exc:
            _save_raw(raw_dir, now, "green_button.xml", xml)
            return PortalOutcome("error", f"espi_invalid:{exc}"[:200])
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
        except ValueError as exc:
            await _snapshot_html(driver, raw_dir, now)
            return PortalOutcome("error", _field_reason("bill_parse_failed", exc), xml_file=xml_file)
        forecast_error: Optional[str] = None
        if forecast is not None:
            try:
                payloads.append(forecast_payload_from_fields(forecast, retrieved_at=now))
            except ValueError as exc:
                await _snapshot_html(driver, raw_dir, now)
                forecast_error = _field_reason("forecast_parse_failed", exc)
        bill_files = _write_new_bills(payloads, bill_inbox_dir=bill_inbox_dir, seen_path=seen_path, stamp=stamp)
        if forecast_error:
            return PortalOutcome("error", forecast_error, xml_file=xml_file, bill_files=bill_files)
        return PortalOutcome("ok", "fetched", xml_file=xml_file, bill_files=bill_files)
    except Exception as exc:  # noqa: BLE001 -- the loop must survive and report
        return PortalOutcome("error", type(exc).__name__)
