"""Stamped drop directories (Green Button XML, bill JSON).

A parsed file moves to processed/ with its retrieval time as a filename prefix, so
a restart replays the exact same rows with the exact same retrieved_at. A file that
fails to parse moves to <inbox>/failed/ (same stamp prefix, so a repeat failure never
overwrites an earlier one) and publishes nothing -- a broken export must never read as
a quiet day. Files the portal fetcher wrote start with `rmp-portal`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TypeVar

from orion.energy.espi import parse_espi
from orion.schemas.energy import EnergySource, EnergyUsageIntervalV1

logger = logging.getLogger("orion-energy.inbox")
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
SEP = "__"
PORTAL_PREFIX = "rmp-portal"

T = TypeVar("T")
Parse = Callable[[bytes, datetime, str], list[T]]


def _source_for(stamped_name: str) -> EnergySource:
    original = stamped_name.partition(SEP)[2] or stamped_name
    return "rockymountain_power" if original.startswith(PORTAL_PREFIX) else "file_drop"


def _parse_xml(raw: bytes, retrieved_at: datetime, stamped_name: str) -> list[EnergyUsageIntervalV1]:
    return parse_espi(raw, retrieved_at=retrieved_at, source=_source_for(stamped_name), source_file=stamped_name)


def scan_dir(inbox_dir: Path, processed_dir: Path, *, now: datetime, suffix: str, parse: Parse) -> list:
    if not inbox_dir.is_dir():
        return []
    stamp_time = now.astimezone(timezone.utc).replace(microsecond=0)
    rows: list = []
    for path in sorted(p for p in inbox_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix):
        target_name = f"{stamp_time.strftime(STAMP_FORMAT)}{SEP}{path.name}"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.warning("energy_inbox_io_failed file=%s op=read error=%s", path.name, exc)
            continue
        try:
            parsed = parse(raw, stamp_time, target_name)
        except ValueError as exc:
            failed = inbox_dir / "failed"
            failed.mkdir(parents=True, exist_ok=True)
            try:
                path.rename(failed / target_name)
            except OSError as rename_exc:
                logger.warning("energy_inbox_io_failed file=%s op=rename_to_failed error=%s", path.name, rename_exc)
                continue
            logger.warning("energy_inbox_parse_failed file=%s error=%s", path.name, exc)
            continue
        processed_dir.mkdir(parents=True, exist_ok=True)
        try:
            path.rename(processed_dir / target_name)
        except OSError as exc:
            logger.warning("energy_inbox_io_failed file=%s op=rename_to_processed error=%s", path.name, exc)
            continue
        logger.info("energy_inbox_parsed file=%s rows=%d", path.name, len(parsed))
        rows.extend(parsed)
    return rows


def _stamp_of(path: Path) -> Optional[datetime]:
    stamp, sep, _ = path.name.partition(SEP)
    if not sep:
        return None
    try:
        return datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def replay_dir(processed_dir: Path, *, suffix: str, parse: Parse) -> list:
    if not processed_dir.is_dir():
        return []
    rows: list = []
    for path in sorted(processed_dir.glob(f"*{suffix}")):
        retrieved = _stamp_of(path)
        if retrieved is None:
            logger.warning("energy_replay_skipped_unstamped file=%s", path.name)
            continue
        try:
            rows.extend(parse(path.read_bytes(), retrieved, path.name))
        except ValueError as exc:
            logger.warning("energy_replay_parse_failed file=%s error=%s", path.name, exc)
    return rows


def latest_processed_at(processed_dir: Path, suffix: str) -> Optional[datetime]:
    if not processed_dir.is_dir():
        return None
    stamps = [s for s in (_stamp_of(p) for p in processed_dir.glob(f"*{suffix}")) if s is not None]
    return max(stamps, default=None)


def scan_inbox(inbox_dir: Path, processed_dir: Path, *, now: datetime) -> list[EnergyUsageIntervalV1]:
    return scan_dir(inbox_dir, processed_dir, now=now, suffix=".xml", parse=_parse_xml)


def load_processed(processed_dir: Path) -> list[EnergyUsageIntervalV1]:
    return replay_dir(processed_dir, suffix=".xml", parse=_parse_xml)
