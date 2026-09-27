"""Drop directory for Green Button XML.

A parsed file moves to processed/ with its retrieval time as a filename prefix, so
a restart replays the exact same intervals with the exact same retrieved_at. A file
that fails to parse moves to inbox/failed/ and publishes nothing -- a broken export
must never read as a quiet day.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from orion.energy.espi import EspiError, parse_espi
from orion.schemas.energy import EnergyUsageIntervalV1

logger = logging.getLogger("orion-energy.inbox")
_STAMP = "%Y%m%dT%H%M%SZ"
_SEP = "__"


def scan_inbox(inbox_dir: Path, processed_dir: Path, *, now: datetime) -> list[EnergyUsageIntervalV1]:
    if not inbox_dir.is_dir():
        return []
    stamp_time = now.astimezone(timezone.utc).replace(microsecond=0)
    rows: list[EnergyUsageIntervalV1] = []
    for path in sorted(p for p in inbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xml"):
        target_name = f"{stamp_time.strftime(_STAMP)}{_SEP}{path.name}"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.warning("energy_inbox_io_failed file=%s op=read error=%s", path.name, exc)
            continue
        try:
            parsed = parse_espi(
                raw, retrieved_at=stamp_time, source="file_drop", source_file=target_name
            )
        except EspiError as exc:
            failed = inbox_dir / "failed"
            failed.mkdir(parents=True, exist_ok=True)
            try:
                path.rename(failed / path.name)
            except OSError as rename_exc:
                logger.warning(
                    "energy_inbox_io_failed file=%s op=rename_to_failed error=%s",
                    path.name,
                    rename_exc,
                )
                continue
            logger.warning("energy_inbox_parse_failed file=%s error=%s", path.name, exc)
            continue
        processed_dir.mkdir(parents=True, exist_ok=True)
        try:
            path.rename(processed_dir / target_name)
        except OSError as exc:
            logger.warning("energy_inbox_io_failed file=%s op=rename_to_processed error=%s", path.name, exc)
            continue
        logger.info("energy_inbox_parsed file=%s intervals=%d", path.name, len(parsed))
        rows.extend(parsed)
    return rows


def load_processed(processed_dir: Path) -> list[EnergyUsageIntervalV1]:
    if not processed_dir.is_dir():
        return []
    rows: list[EnergyUsageIntervalV1] = []
    for path in sorted(processed_dir.glob("*.xml")):
        stamp, sep, _ = path.name.partition(_SEP)
        if not sep:
            logger.warning("energy_replay_skipped_unstamped file=%s", path.name)
            continue
        retrieved = datetime.strptime(stamp, _STAMP).replace(tzinfo=timezone.utc)
        try:
            rows.extend(parse_espi(path.read_bytes(), retrieved_at=retrieved, source="file_drop", source_file=path.name))
        except EspiError as exc:
            logger.warning("energy_replay_parse_failed file=%s error=%s", path.name, exc)
    return rows
