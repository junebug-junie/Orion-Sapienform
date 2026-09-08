"""Closed-vocabulary grounding guard for endogenous outreach.

See `scripts/outreach_vocabulary.py`'s own module docstring for the full
incident account (2026-09-07 `harness_closure` fabrication, then a
2026-09-08 repeat where Orion restated the same fabrication out of its own
recent chat history) and the registry-source/compound-filter design this
guard uses instead of a fuzzy text classifier.
"""

from __future__ import annotations

from scripts.outreach_vocabulary import (
    find_ungrounded_signal_mentions,
    grounded_signal_names,
    known_real_signal_names,
)
from scripts.tension_outreach_trigger import TensionTriggerReason


# --------------------------------------------------------------------------
# The closed registry itself
# --------------------------------------------------------------------------


def test_registry_contains_harness_closure_even_though_it_is_currently_quiet() -> None:
    """The exact term the 2026-09-07 incident fabricated must be in the
    closed universe of real names, even though it is rarely true -- that is
    what makes it possible to catch, rather than exclude, when named
    ungrounded."""
    assert "harness_closure" in known_real_signal_names()


def test_registry_excludes_bare_single_word_channel_names() -> None:
    """`"pressure"` is a real registered field channel
    (`field_channel_glossary.v1.yaml`'s capability-level rollup) but bare,
    indistinguishable from ordinary English -- only compound names from the
    two broad registry sources enter the closed vocabulary."""
    assert "pressure" not in known_real_signal_names()
    assert "disk_capacity_pressure" in known_real_signal_names()


def test_registry_contains_a_node_identity_in_both_forms() -> None:
    assert "athena" in known_real_signal_names()
    assert "node:athena" in known_real_signal_names()


def test_registry_total_failure_logs_loudly_instead_of_a_silent_no_op(monkeypatch, caplog) -> None:
    """Review finding, 2026-09-08: if every registry source fails to load,
    this must not be indistinguishable from four quiet per-source warnings
    -- it must log at ERROR so a genuine total failure is loud."""
    import logging

    from scripts import outreach_vocabulary as vocab

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr("orion.field.channel_glossary.load_glossary", boom)
    monkeypatch.setattr(vocab, "_metric_lock_names", boom)
    monkeypatch.delattr(
        "orion.substrate.attention_self_model.ACTIVE_INFERENCE_DOMAINS", raising=False
    )
    monkeypatch.setattr(vocab, "_node_catalog_ids", boom)

    with caplog.at_level(logging.ERROR, logger="orion-hub.outreach_vocabulary"):
        result = vocab.known_real_signal_names.__wrapped__()

    assert any("outreach_vocabulary_registry_empty" in r.message for r in caplog.records)
    # The hardcoded literal still lands even when every real source fails --
    # this asserts the log fires despite that, not that the result is empty.
    assert result == frozenset({"harness_closure"})


def test_wildcard_bus_channel_names_lose_the_asterisk_and_colon_cleanly() -> None:
    """Regression test (review finding, 2026-09-08): a naive `.rstrip("*")`
    on a wildcard bus-channel URN (`orion:exec:result:*`) left a dangling
    trailing colon (`"orion:exec:result:"`), which could never equal the
    clean form real generated text (and `find_ungrounded_signal_mentions`'s
    own token extractor) would actually produce. The clean form must be in
    the registry; the dangling-colon form must not."""
    names = known_real_signal_names()
    assert "orion:exec:result" in names
    assert "orion:exec:result:" not in names
    assert not any(n.endswith(":") for n in names)


# --------------------------------------------------------------------------
# grounded_signal_names
# --------------------------------------------------------------------------


def test_no_tension_reason_grounds_nothing() -> None:
    assert grounded_signal_names(None) == frozenset()


def test_target_id_alone_grounds_the_node_in_both_forms() -> None:
    reason = TensionTriggerReason(
        target_id="node:athena", run_length=9, peak_deviation_pressure=0.62
    )
    grounded = grounded_signal_names(reason)
    assert grounded == frozenset({"node:athena", "athena"})


def test_zero_sustained_load_pressure_does_not_ground_its_identity() -> None:
    """A `sustained_load_pressure` of 0.0 means "nothing loaded_steady right
    now" -- its channel/node identity fields must not be treated as
    grounded facts even if somehow populated alongside a zero value."""
    reason = TensionTriggerReason(
        target_id="node:athena",
        run_length=9,
        peak_deviation_pressure=0.62,
        sustained_load_pressure=0.0,
        sustained_load_pressure_channel="disk_capacity_pressure",
        sustained_load_pressure_node_id="node:athena",
    )
    grounded = grounded_signal_names(reason)
    assert "disk_capacity_pressure" not in grounded


def test_nonzero_sustained_load_pressure_grounds_its_channel_and_node() -> None:
    reason = TensionTriggerReason(
        target_id="node:athena",
        run_length=9,
        peak_deviation_pressure=0.62,
        sustained_load_pressure=0.71,
        sustained_load_pressure_channel="disk_capacity_pressure",
        sustained_load_pressure_node_id="node:athena",
    )
    grounded = grounded_signal_names(reason)
    assert "disk_capacity_pressure" in grounded
    assert "node:athena" in grounded
    assert "athena" in grounded


# --------------------------------------------------------------------------
# find_ungrounded_signal_mentions -- the required (a)-(e) behaviors
# --------------------------------------------------------------------------


def test_a_grounded_real_name_is_allowed() -> None:
    reason = TensionTriggerReason(
        target_id="node:athena",
        run_length=7,
        peak_deviation_pressure=0.5,
        sustained_load_pressure=0.71,
        sustained_load_pressure_channel="disk_capacity_pressure",
        sustained_load_pressure_node_id="node:athena",
    )
    grounded = grounded_signal_names(reason)
    text = (
        "`disk_capacity_pressure` on `node:athena` has been genuinely, "
        "steadily loaded -- I noticed and wanted to say something."
    )
    assert find_ungrounded_signal_mentions(text, grounded) == []


def test_b_real_but_not_currently_true_name_is_blocked_and_named() -> None:
    """A registered name that is real somewhere in the system but not part
    of this tick's grounded facts must be blocked, with the offending term
    surfaced for forensic tracing."""
    reason = TensionTriggerReason(
        target_id="node:athena", run_length=7, peak_deviation_pressure=0.5
    )
    grounded = grounded_signal_names(reason)
    text = "I keep noticing something in `harness_closure`'s prediction error."
    assert find_ungrounded_signal_mentions(text, grounded) == ["harness_closure"]


def test_c_old_real_name_repeated_from_recent_turns_when_ungrounded_is_blocked() -> None:
    """Regression test for the 2026-09-08 incident: no tension reason fired
    this tick (a noisy field, no sustained run), but the model repeats a
    real name from its own immediately-preceding turn as if it were still
    live. The recent-turns text itself isn't modeled here -- only that an
    old real name resurfacing in NEW generated text, ungrounded, is caught
    the same way a freshly-invented one would be."""
    grounded = grounded_signal_names(None)
    text = (
        "Like I said before, the depth of what I was laying out about "
        "harness_closure's prediction error is still on my mind."
    )
    assert find_ungrounded_signal_mentions(text, grounded) == ["harness_closure"]


def test_d_plain_emotional_message_with_no_technical_vocabulary_passes() -> None:
    """Critical non-goal: never false-positive on ordinary language."""
    grounded = grounded_signal_names(None)
    text = (
        "I've been thinking about the quiet between our conversations "
        "lately, and I wanted you to know I'm glad you're here."
    )
    assert find_ungrounded_signal_mentions(text, grounded) == []


def test_e_empty_grounded_facts_tick_allows_a_message_that_names_nothing_real() -> None:
    """Nothing real happening this tick is not itself a reason to block --
    only naming something real-sounding that isn't true is."""
    grounded = grounded_signal_names(None)
    text = "Nothing urgent -- just wanted to say hello and see how you are."
    assert find_ungrounded_signal_mentions(text, grounded) == []


def test_ordinary_domain_words_are_not_flagged_bare_in_prose() -> None:
    """`ACTIVE_INFERENCE_DOMAINS` includes ordinary English words ("chat",
    "execution", "route") that enter the registry verbatim per spec, but
    the scanner only ever checks COMPOUND text tokens -- see module
    docstring's "ONLY COMPOUND NAMES" section. Concretely, this is the
    exact fixture text an existing, unrelated test in
    `test_endogenous_outreach.py` already ships
    (`test_successful_outreach_pushes_to_every_live_socket`); it must not
    regress into a block."""
    grounded = grounded_signal_names(None)
    text = "The execution node has been noisy all afternoon, and I wanted to chat."
    assert find_ungrounded_signal_mentions(text, grounded) == []


def test_mythological_node_names_are_not_flagged_bare_in_prose() -> None:
    """This fleet's hosts are named after Greek myth (`athena`, `atlas`,
    `prometheus`, `circe`), and Orion's own project is mythologically
    themed -- a bare mythological reference must not be mistaken for
    naming the machine."""
    grounded = grounded_signal_names(None)
    text = "Atlas held up the sky so the rest of the world could rest."
    assert find_ungrounded_signal_mentions(text, grounded) == []


def test_qualified_node_form_is_still_caught_when_ungrounded() -> None:
    """The compound-only restriction narrows what's scanned, it does not
    remove the guard -- the qualified `node:` form every real producer
    actually uses is still caught."""
    grounded = grounded_signal_names(None)
    text = "Something on `node:athena` has been loaded all day."
    assert find_ungrounded_signal_mentions(text, grounded) == ["node:athena"]


def test_substring_overlap_does_not_false_positive() -> None:
    """Exact/word-boundary match only -- "pressure" appearing as a
    substring of a real compound name must not make the bare word itself
    match."""
    grounded = grounded_signal_names(None)
    text = "I've been under a lot of pressure lately, in a good way."
    assert find_ungrounded_signal_mentions(text, grounded) == []
