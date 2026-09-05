"""Every channel that names orion-sql-writer as a consumer must be subscribed.

`SQL_WRITER_SUBSCRIBE_CHANNELS` REPLACES the Python default wholesale rather
than merging the way `route_map` does, and `Hunter` issues a plain SUBSCRIBE on
exact names -- so a channel with a correct schema, a correct route, a real SQL
model, a created table and green tests still receives nothing if it is missing
from the subscribe list.

That has now happened at least three times in this service: the settings module
carries two separate comments describing it, and the routing-decision channel
shipped with the same defect anyway, because nothing checked. A reminder in a
comment is not a gate. This is the gate.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from app.settings import settings

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE = "orion-sql-writer"


def _declared_consumer_channels() -> list[str]:
    channels = yaml.safe_load((REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text())
    named: list[str] = []
    for entry in channels.get("channels") or []:
        consumers = entry.get("consumer_services") or []
        # "*" means any service may consume; it asserts nothing about this one.
        if SERVICE in consumers:
            named.append(str(entry["name"]))
    return named


#: Channels channels.yaml already declares this service as a consumer of, which
#: it does not subscribe to today. A ratchet, not an approval: the point is that
#: the set cannot GROW without someone saying so in a diff. Shrinking it is
#: always welcome. Some are plausibly deliberate -- "orion:effect:*" is a glob
#: this service never expands, and "orion:spark:state:snapshot" is appended only
#: when its feature toggle is on -- but none of them were verified as part of the
#: change that added this gate, so none are asserted to be fine.
KNOWN_UNSUBSCRIBED = {
    "orion:biometrics:sample",
    "orion:collapse:triage",
    "orion:debug:turn:dossier",
    "orion:effect:*",
    "orion:memory:drives:state",
    "orion:memory:tension:event",
    "orion:notify:config:preference",
    "orion:notify:config:recipient",
    "orion:recall:telemetry",
    "orion:spark:concepts:delta",
    "orion:spark:concepts:profile",
    "orion:spark:state:snapshot",
    "orion:world_pulse:digest:published",
}


def test_no_new_declared_consumer_channel_goes_unsubscribed() -> None:
    subscribed = set(settings.effective_subscribe_channels)
    declared = _declared_consumer_channels()

    assert declared, "channels.yaml names orion-sql-writer as a consumer nowhere -- check the parser"
    missing = {c for c in declared if c not in subscribed}
    new = sorted(missing - KNOWN_UNSUBSCRIBED)
    assert not new, (
        "channels.yaml declares orion-sql-writer as a consumer of these, but the "
        f"writer does not subscribe, so they will never arrive: {new}"
    )


def test_the_ratchet_does_not_hide_a_channel_that_got_fixed() -> None:
    """If a known gap is closed, remove it from the list rather than leaving it."""
    subscribed = set(settings.effective_subscribe_channels)
    stale = sorted(c for c in KNOWN_UNSUBSCRIBED if c in subscribed)
    assert not stale, f"now subscribed -- delete from KNOWN_UNSUBSCRIBED: {stale}"


# test_the_routing_decision_channel_survives_a_stale_operator_env() removed
# 2026-09-05: the always-append guarantee it tested (`orion:routing:decision`
# force-appended to effective_subscribe_channels) was removed along with the
# channel, schema, and RoutingDecisionSQL model -- all retired, see this
# change's PR description.
