from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class RefitDecision:
    should_refit: bool
    reason: str


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def decide_refit(
    *,
    policy: str,
    force_refit: bool,
    manifest_created_at: Optional[str],
    refit_ttl_hours: int,
    refit_doc_threshold: int,
    new_doc_count: int,
) -> RefitDecision:
    normalized = (policy or "never").lower().strip()
    if force_refit:
        return RefitDecision(True, "force_refit")

    if normalized == "manual":
        return RefitDecision(False, "manual_policy")
    if normalized == "never":
        return RefitDecision(False, "never_policy")

    if normalized == "ttl":
        created_at = _parse_iso(manifest_created_at)
        if created_at is None:
            return RefitDecision(True, "ttl_missing_manifest")
        ttl = timedelta(hours=refit_ttl_hours)
        if datetime.now(timezone.utc) - created_at >= ttl:
            return RefitDecision(True, "ttl_expired")
        return RefitDecision(False, "ttl_not_expired")

    if normalized == "count":
        if new_doc_count >= refit_doc_threshold:
            return RefitDecision(True, "doc_threshold")
        return RefitDecision(False, "doc_threshold_not_met")

    return RefitDecision(False, "unknown_policy")
