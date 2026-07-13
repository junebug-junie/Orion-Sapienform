"""Declarative journal notification dispatch policy, keyed off `trigger_kind`.

Context (2026-07-13): two independent codepaths in
`services/orion-actions/app/main.py` used to email the same journal `entry_id` for
`trigger_kind=daily_summary, source_kind=scheduler` entries -- one fired inline right
after compose (`_build_scheduler_daily_journal_email_request`, dedupe key
`actions:journal:daily:scheduler:{entry_id}`), the other fired from a generic
post-persist consumer on every persisted journal entry
(`_build_post_persist_journal_email_request`, dedupe key
`actions:journal:persisted:{entry_id}`). Two different dedupe-key namespaces meant
neither path ever saw the other's delivery, so every scheduler daily journal was
emailed twice.

This registry replaces both ad-hoc codepaths with a single table, keyed by
`trigger_kind`, consumed by exactly one dispatch function
(`_dispatch_journal_notifications` in services/orion-actions/app/main.py) using a
single dedupe-key namespace (`actions:journal:notify:{entry_id}`) regardless of which
channel would fire -- structurally impossible to double-send across two namespaces
because there is only one namespace now.

Values are data-backed, not vibes:
- `in_app_enabled=False` everywhere: notify_requests.event_kind='orion.chat.message'
  shows 1796 sent / 14d with message_opened_at NULL on ALL of them -- zero measured
  engagement with in-app journal notifications as of 2026-07-13.
- `town_episode` gets `email_enabled=False` structurally. It was previously silenced
  only via a fragile `.env_example` CSV string-match on `source_kind='embodiment'`
  (~74 fires/day) -- moving it into the registry makes the exclusion typo-proof (a
  new town/embodiment source_kind spelling can no longer silently re-enable email).
- `collapse_response` / `notify_summary` show 0 fires in the last 14 days (dormant)
  but stay registered per the fail-closed philosophy below -- do not delete them.

Fail-closed by design: `resolve_policy` returns an all-disabled policy for any
trigger_kind with no registry row, rather than defaulting to "email everything". A new
trigger_kind must be deliberately registered before it can notify anyone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class JournalDispatchPolicy:
    trigger_kind: str
    email_enabled: bool
    in_app_enabled: bool
    dedupe_scope: Literal["entry_id"] = "entry_id"
    recall_profile_setting: str = ""


JOURNAL_DISPATCH_REGISTRY: dict[str, JournalDispatchPolicy] = {
    "daily_summary": JournalDispatchPolicy(
        "daily_summary",
        email_enabled=True,
        in_app_enabled=False,
        recall_profile_setting="actions_journal_scheduler_recall_profile",
    ),
    "metacog_digest": JournalDispatchPolicy(
        "metacog_digest",
        email_enabled=True,
        in_app_enabled=False,
        recall_profile_setting="actions_journal_metacog_recall_profile",
    ),
    "world_pulse_digest": JournalDispatchPolicy(
        "world_pulse_digest",
        email_enabled=True,
        in_app_enabled=False,
        recall_profile_setting="actions_journal_world_pulse_recall_profile",
    ),
    "notify_summary": JournalDispatchPolicy(
        "notify_summary",
        email_enabled=True,
        in_app_enabled=False,
        recall_profile_setting="actions_journal_notify_recall_profile",
    ),
    "autonomy_episode": JournalDispatchPolicy(
        "autonomy_episode",
        email_enabled=True,
        in_app_enabled=False,
        recall_profile_setting="",
    ),
    "collapse_response": JournalDispatchPolicy(
        "collapse_response",
        email_enabled=True,
        in_app_enabled=False,
        recall_profile_setting="",
    ),
    "town_episode": JournalDispatchPolicy(
        "town_episode",
        email_enabled=False,
        in_app_enabled=False,
        recall_profile_setting="",
    ),
    "manual": JournalDispatchPolicy(
        "manual",
        email_enabled=True,
        in_app_enabled=False,
        recall_profile_setting="",
    ),
}


def resolve_policy(trigger_kind: str) -> JournalDispatchPolicy:
    """Fail-closed lookup: an unregistered trigger_kind sends nothing."""
    return JOURNAL_DISPATCH_REGISTRY.get(
        trigger_kind,
        JournalDispatchPolicy(trigger_kind, email_enabled=False, in_app_enabled=False),
    )
