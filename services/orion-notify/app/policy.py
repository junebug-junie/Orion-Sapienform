from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

from orion.schemas.notify import NotificationRequest


@dataclass
class ThrottleRule:
    max_per_window: int
    window_seconds: int


@dataclass
class PolicyDecision:
    allowed: bool
    channels: List[str]
    recipient_group: str
    dedupe_window_seconds: Optional[int]
    throttle: Optional[ThrottleRule]
    action: str
    reason: Optional[str] = None
    require_ack: bool = False
    ack_deadline_minutes: Optional[int] = None
    escalation_channels: List[str] | None = None
    require_read_receipt: bool = False
    read_receipt_deadline_minutes: Optional[int] = None


class Policy:
    def __init__(self, raw: Dict[str, Any]) -> None:
        self.raw = raw
        self.recipient_groups = raw.get("recipient_groups", {})
        self.rules = raw.get("rules", [])
        self.quiet_hours = raw.get("quiet_hours", {})

    @classmethod
    def load(cls, path: str) -> "Policy":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(data)

    def evaluate(
        self,
        request: NotificationRequest,
        now: datetime,
        *,
        quiet_hours_enabled: bool = True,
        quiet_hours_override: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        action = self._match_rule(request)
        allowed = action.get("allowed", True)
        channels = list(action.get("channels", []))
        recipient_group = action.get("recipient_group", request.recipient_group)
        dedupe_window_seconds = action.get("dedupe_window_seconds")
        require_ack = bool(action.get("require_ack", False))
        ack_deadline_minutes = action.get("ack_deadline_minutes")
        escalation_channels = list(action.get("escalation_channels", [])) if action.get("escalation_channels") else []
        require_read_receipt = bool(action.get("require_read_receipt", False))
        read_receipt_deadline_minutes = action.get("read_receipt_deadline_minutes")
        throttle_cfg = action.get("throttle")
        throttle = None
        if throttle_cfg:
            throttle = ThrottleRule(
                max_per_window=int(throttle_cfg.get("max_per_window", 0)),
                window_seconds=int(throttle_cfg.get("window_seconds", 0)),
            )

        if request.channels_requested is not None:
            if channels:
                channels = [c for c in request.channels_requested if c in channels]
            else:
                channels = []

        quiet_cfg = quiet_hours_override if quiet_hours_override is not None else self.quiet_hours
        if quiet_hours_enabled and self._is_quiet_hours(now, quiet_cfg) and request.severity.lower() not in {"critical", "error"}:
            return PolicyDecision(
                allowed=True,
                channels=[],
                recipient_group=recipient_group,
                dedupe_window_seconds=dedupe_window_seconds,
                throttle=throttle,
                action="quiet_hours",
                reason="quiet_hours",
                require_ack=require_ack,
                ack_deadline_minutes=ack_deadline_minutes,
                escalation_channels=escalation_channels,
                require_read_receipt=require_read_receipt,
                read_receipt_deadline_minutes=read_receipt_deadline_minutes,
            )

        return PolicyDecision(
            allowed=allowed,
            channels=channels,
            recipient_group=recipient_group,
            dedupe_window_seconds=dedupe_window_seconds,
            throttle=throttle,
            action=action.get("name", "policy_rule"),
            require_ack=require_ack,
            ack_deadline_minutes=ack_deadline_minutes,
            escalation_channels=escalation_channels,
            require_read_receipt=require_read_receipt,
            read_receipt_deadline_minutes=read_receipt_deadline_minutes,
        )

    def is_quiet_hours(self, now: datetime, quiet_hours_override: Optional[Dict[str, Any]] = None) -> bool:
        quiet_cfg = quiet_hours_override if quiet_hours_override is not None else self.quiet_hours
        return self._is_quiet_hours(now, quiet_cfg)

    def resolve_recipient_emails(self, group: str, env_lookup: Dict[str, str]) -> List[str]:
        group_cfg = self.recipient_groups.get(group, {})
        if isinstance(group_cfg, dict):
            direct = group_cfg.get("email_to")
            if isinstance(direct, list):
                return [str(item).strip() for item in direct if str(item).strip()]
            env_name = group_cfg.get("email_to_env")
            if env_name:
                raw = env_lookup.get(env_name, "")
                return [e.strip() for e in raw.split(",") if e.strip()]
        return []

    def _match_rule(self, request: NotificationRequest) -> Dict[str, Any]:
        for rule in self.rules:
            match = rule.get("match", {})
            if self._match_field(match.get("event_kind"), request.event_kind) and self._match_field(
                match.get("severity"), request.severity
            ):
                action = rule.get("action", {})
                action["name"] = rule.get("name", "policy_rule")
                return action
        return {"allowed": True, "channels": [], "name": "fallback"}

    @staticmethod
    def _match_field(pattern: Any, value: str) -> bool:
        if pattern is None:
            return True
        if isinstance(pattern, str):
            return pattern == "*" or pattern == value
        if isinstance(pattern, list):
            return value in pattern or "*" in pattern
        return False

    def _is_quiet_hours(self, now: datetime, quiet_hours: Dict[str, Any]) -> bool:
        if not quiet_hours:
            return False
        timezone = quiet_hours.get("timezone", "America/Denver")
        start = quiet_hours.get("start", "22:00")
        end = quiet_hours.get("end", "07:00")
        tz = ZoneInfo(timezone)
        local_now = now.astimezone(tz)
        start_time = _parse_time(start)
        end_time = _parse_time(end)
        if start_time <= end_time:
            return start_time <= local_now.time() <= end_time
        return local_now.time() >= start_time or local_now.time() <= end_time


def _parse_time(value: str) -> time:
    parts = value.split(":")
    if len(parts) != 2:
        return time(22, 0)
    return time(hour=int(parts[0]), minute=int(parts[1]))
