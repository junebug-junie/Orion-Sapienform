from __future__ import annotations

from orion.journaler.schemas import JournalEntryDraftV1, JournalTriggerV1
from orion.journaler.worker import build_created_event_payload, build_write_payload


def _trigger(**overrides) -> JournalTriggerV1:
    defaults = dict(
        trigger_kind="daily_summary",
        source_kind="scheduler",
        source_ref="2026-07-13",
        summary="Daily cadence",
    )
    defaults.update(overrides)
    return JournalTriggerV1(**defaults)


def _draft() -> JournalEntryDraftV1:
    return JournalEntryDraftV1(mode="daily", title="Morning Reflection", body="Priorities and anchors.")


def test_build_write_payload_populates_trigger_kind_from_trigger() -> None:
    """Part A: JournalEntryWriteV1 must carry trigger_kind forward from the
    ephemeral, in-process-only JournalTriggerV1 -- this is the field a clean
    dispatch-policy registry keys on (see orion.journaler.dispatch_registry)."""
    trigger = _trigger()
    write = build_write_payload(
        _draft(),
        trigger=trigger,
        correlation_id="corr-1",
        author="orion",
    )
    assert write.trigger_kind == "daily_summary"


def test_trigger_kind_survives_model_dump_and_created_event_round_trip() -> None:
    """The bus-facing journal.entry.created.v1 payload is built via
    `write.model_dump(mode="json")` in build_created_event_payload -- since
    trigger_kind is now a declared field (not an extra key, which JournalEntryWriteV1's
    extra="forbid" would otherwise silently disallow on re-validation), it must survive
    the write -> model_dump -> created-event round trip with no extra propagation code."""
    trigger = _trigger(trigger_kind="metacog_digest", source_kind="metacog")
    write = build_write_payload(
        _draft(),
        trigger=trigger,
        correlation_id="corr-2",
        author="orion",
    )
    assert write.trigger_kind == "metacog_digest"

    dumped = write.model_dump(mode="json")
    assert dumped["trigger_kind"] == "metacog_digest"

    created_event_payload = build_created_event_payload(write)
    assert created_event_payload["trigger_kind"] == "metacog_digest"
    assert created_event_payload["entry_id"] == write.entry_id

    # Re-validating the created-event payload back into JournalEntryWriteV1 must not
    # raise, confirming trigger_kind round-trips as a real declared field rather than
    # an "extra" key that extra="forbid" would reject.
    from orion.journaler.schemas import JournalEntryWriteV1

    reparsed = JournalEntryWriteV1.model_validate(created_event_payload)
    assert reparsed.trigger_kind == "metacog_digest"


def test_trigger_kind_absent_when_no_trigger_kind_on_older_producer_payload() -> None:
    """Backward compatibility: a payload built before this patch (no trigger_kind key
    at all) must still validate, with trigger_kind defaulting to None."""
    from orion.journaler.schemas import JournalEntryWriteV1

    legacy_payload = {
        "entry_id": "entry-legacy-1",
        "author": "orion",
        "mode": "manual",
        "body": "Pre-existing entry with no trigger_kind key.",
    }
    parsed = JournalEntryWriteV1.model_validate(legacy_payload)
    assert parsed.trigger_kind is None
