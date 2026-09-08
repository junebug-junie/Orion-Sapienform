"""Closed vocabulary for endogenous outreach's post-generation grounding guard.

WHY THIS EXISTS (2026-09-08, ~1h42m after PR #2149 merged). PR #2149 threaded
real channel/node identity through `orion.field.significance` ->
`tension_outreach_trigger.py` -> `endogenous_outreach.build_outreach_prompt`
so a prompt states a real channel name instead of leaving a blank for the
model to guess into. Orion sent Juniper an unprompted message anyway, naming
"harness_closure prediction error" with zero real signal behind it. Traced
live: no genuine sustained-tension run existed that tick (the field was too
noisy -- the leading node flipped every tick -- so `ctx.tension_reason` was
absent). But `build_outreach_prompt`'s "The last thing the two of you said"
section fed the model its own immediately-preceding turn, in which Orion had
*defended* the same false claim after a different AI in the Hub room flagged
it as suspicious. With nothing fresh to say and its own recent argument
sitting right there, generation just restated the fabrication as if it were
still live.

Juniper's fix, verbatim: "if Orion is going to send me a message stating
they're seeing issues with telemetry, it better be real" -- solved as
**closed-vocabulary schema enforcement**, not a fuzzy "does this look
suspicious" guard on free text (rejected explicitly, in an earlier
conversation). The distinction: a deterministic lookup against real per-tick
validated data is legitimate schema validation; a heuristic text classifier
is not.

TWO SEPARATE SETS, NOT ONE:

  * `known_real_signal_names()` -- the closed UNIVERSE of every internal-
    signal identifier that is ever real anywhere in this system. Anything a
    generated message names that is OUTSIDE this set is definitely
    fabricated vocabulary (a made-up-sounding name, not a real one).
  * `grounded_signal_names(tension_reason)` -- the small subset of that
    universe that is ACTUALLY TRUE this specific tick. Anything inside the
    universe but outside this set is a real name being asserted about a
    moment it was not real for -- exactly the `harness_closure` incident:
    the name is genuine, the claim that it is happening RIGHT NOW is not.

Membership in the universe is necessary but not sufficient to say something
in the prompt; membership in the per-tick grounded set is what actually
licenses a claim of current fact. `find_ungrounded_signal_mentions` (used by
`endogenous_outreach._outreach_once` after `_generate()`, before `_deliver()`)
is the enforcement: a real-registry name in the generated text that is not
in this tick's grounded set blocks the send.

REGISTRY SOURCES (existing-mechanism check, CLAUDE.md 0A -- none of these
were built for this purpose; all four already exist for their own reasons):

  * `orion.field.channel_glossary.load_glossary()` -- the real field-channel
    registry backing Hub's Field Channel Glossary panel.
  * `config/metrics/metric_definitions.lock.json`'s `definitions` key -- the
    cross-registry metric/channel-name lock `scripts/check_definition_drift.
    py` already treats as "is this a real, registered name" for its own
    definition-change alert.
  * `orion.substrate.attention_self_model.ACTIVE_INFERENCE_DOMAINS`, plus the
    literal string `"harness_closure"` -- a real but currently-quiet domain
    (not in the active set; see that constant's own docstring) that is
    specifically the term the 2026-09-07 incident fabricated, so it MUST be
    in the closed universe even though it is rarely true.
  * `config/biometrics/node_catalog.yaml`'s node ids
    (`orion.biometrics.node_catalog.NodeCatalog`).

ONLY COMPOUND NAMES FROM THE TWO BROAD SOURCES, AND ONLY COMPOUND CANDIDATE
TOKENS ARE EVER SCANNED. `channel_glossary`/the metric lock are broad,
machine-generated registries, and both contain bare single-word entries
indistinguishable from ordinary English: confirmed live 2026-09-08,
`field_channel_glossary.v1.yaml` registers a literal channel named
`"pressure"` (the capability-level rollup), and the lock file separately
registers `"confidence"`, `"energy"`, `"tension"`, `"lane"`, `"level"`,
`"trend"`, `"availability"`, `"staleness"`, among others. Only COMPOUND
names (containing `"_"` or `":"`) from those two sources enter
`known_real_signal_names()` -- `"disk_capacity_pressure"` is unambiguous,
bare `"pressure"` is not (this is the literal example the spec for this
patch gave).

The same compound requirement is applied a second time, at SCAN time, to
every source -- including `ACTIVE_INFERENCE_DOMAINS` and node ids, which
enter the registry unfiltered because they are small, explicitly-named sets
rather than auto-generated ones. Concretely caught live while writing this
patch: `ACTIVE_INFERENCE_DOMAINS` includes `"chat"`, `"route"`, and
`"execution"` -- ordinary English words a plain outreach message uses
constantly (`test_endogenous_outreach.py::
test_successful_outreach_pushes_to_every_live_socket`'s own fixture text,
"The execution node has been noisy all afternoon.", would otherwise have
tripped this guard on an unrelated, pre-existing test). Node ids carry the
same risk from a different angle: this fleet's hosts are named after Greek
myth (`athena`, `atlas`, `prometheus`, `circe`), and Orion's own project is
mythologically themed, so a bare mention in a reflective, non-technical
message is a real possibility, not a hypothetical. Requiring the SCANNED
TEXT TOKEN to be compound (matching how these identities actually appear in
`build_outreach_prompt` and in real incidents -- `node:athena`,
`sustained_load_pressure`, `harness_closure`) means a bare "chat" or "athena"
in ordinary prose is never even a candidate, regardless of what the registry
contains.

This is a disclosed, deliberate trade: a message that names a real signal
using ONLY a bare, non-compound form it does not normally take (an unusually
mangled "athena" instead of "node:athena") could theoretically slip through
ungrounded. Every real incident and every producer of these names in this
repo uses the compound/qualified form, so this is a narrow gap, not an open
one -- and the alternative (flat bare-word matching) was measured, live,
against this repo's own pre-existing test suite, to produce a false block on
completely ordinary language.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any, Iterable

logger = logging.getLogger("orion-hub.outreach_vocabulary")

# Real but currently-quiet domain(s), not (yet) in ACTIVE_INFERENCE_DOMAINS,
# that are nonetheless real enough for Orion to legitimately name -- see
# module docstring. This literal set exists because of exactly one incident;
# do not fold ordinary domain-name discovery into it.
_EXTRA_KNOWN_DOMAINS: frozenset[str] = frozenset({"harness_closure"})

# A candidate text token (or a broad-source registry entry) must contain one
# of these to be treated as distinguishable-from-prose. See module docstring
# for the two live false-positive collisions ("pressure", "execution") this
# exists to prevent.
_COMPOUND_MARKERS = ("_", ":")


def _is_compound(token: str) -> bool:
    return any(marker in token for marker in _COMPOUND_MARKERS)


def _metric_lock_path():
    from scripts.service_logs import resolve_repo_root

    return resolve_repo_root() / "config" / "metrics" / "metric_definitions.lock.json"


def _metric_lock_names() -> set[str]:
    """Compound leaf names out of `metric_definitions.lock.json`'s URNs.

    A URN looks like `metric://field_channel/orion-field-digester/
    disk_capacity_pressure` or `metric://organ_signal/autonomy/
    autonomy_state#pressure_autonomy` -- the leaf segment after the last `/`,
    split again on `#` into a base name and an optional fragment, both of
    which are real, independently nameable identifiers. Bus-channel wildcard
    suffixes (`orion:exec:result:*`) have the trailing `*` stripped; it is
    not part of any literal channel name and would never appear verbatim in
    generated text.
    """
    path = _metric_lock_path()
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for urn in data.get("definitions", {}):
        tail = str(urn).rsplit("/", 1)[-1]
        base, _, frag = tail.partition("#")
        for candidate in (base, frag):
            candidate = candidate.rstrip("*").strip().lower()
            if candidate and _is_compound(candidate):
                names.add(candidate)
    return names


def _node_catalog_path():
    from scripts.service_logs import resolve_repo_root

    return resolve_repo_root() / "config" / "biometrics" / "node_catalog.yaml"


def _node_catalog_ids() -> set[str]:
    """Every canonical node id, bare and `node:`-qualified.

    Both forms enter the registry: `TensionTriggerReason.target_id` and
    `sustained_load_pressure_node_id` are always `node:`-qualified
    (`orion.attention.tension`'s own convention), but nothing stops
    generation from dropping the prefix, so the bare form is real vocabulary
    too.
    """
    from orion.biometrics.node_catalog import NodeCatalog

    path = _node_catalog_path()
    if not path.is_file():
        return set()
    catalog = NodeCatalog.load(path)
    ids: set[str] = set()
    for node_id in catalog.profiles:
        node_id = str(node_id).strip().lower()
        if not node_id:
            continue
        ids.add(node_id)
        ids.add(f"node:{node_id}")
    return ids


@functools.lru_cache(maxsize=1)
def known_real_signal_names() -> frozenset[str]:
    """The full closed universe of real internal-signal identifiers Orion
    could ever legitimately name. See module docstring for sources and the
    compound-only filter applied to the two broad ones.

    Cached for the process lifetime (matches `channel_glossary.
    load_glossary()`'s own convention) -- these are all deploy-time-static
    config/code artifacts, not live data.
    """
    names: set[str] = set()

    try:
        from orion.field.channel_glossary import load_glossary

        glossary = load_glossary()
        names.update(
            str(e.channel).strip().lower()
            for e in glossary["entries"]
            if _is_compound(str(e.channel))
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_vocabulary_channel_glossary_failed err=%s", exc)

    try:
        names.update(_metric_lock_names())
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_vocabulary_metric_lock_failed err=%s", exc)

    try:
        from orion.substrate.attention_self_model import ACTIVE_INFERENCE_DOMAINS

        names.update(str(d).strip().lower() for d in ACTIVE_INFERENCE_DOMAINS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_vocabulary_active_inference_domains_failed err=%s", exc)
    names.update(_EXTRA_KNOWN_DOMAINS)

    try:
        names.update(_node_catalog_ids())
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_vocabulary_node_catalog_failed err=%s", exc)

    return frozenset(n for n in names if n)


def _add_node_identity(facts: set[str], value: str | None) -> None:
    """Both the bare and `node:`-qualified forms of a real node identity."""
    if not value:
        return
    value = str(value).strip().lower()
    if not value:
        return
    facts.add(value)
    if value.startswith("node:"):
        facts.add(value.split(":", 1)[1])
    else:
        facts.add(f"node:{value}")


def _add_plain(facts: set[str], value: str | None) -> None:
    if not value:
        return
    value = str(value).strip().lower()
    if value:
        facts.add(value)


def grounded_signal_names(tension_reason: Any | None) -> frozenset[str]:
    """The exhaustive list of specific technical/telemetry claims Orion is
    allowed to assert as current fact THIS tick -- the small subset of
    `known_real_signal_names()` that is actually true right now, not merely
    real somewhere.

    Deliberately narrow: reads only what `TensionTriggerReason` (PR #2149,
    `scripts.tension_outreach_trigger`) already carries as a real, per-tick
    fact -- `target_id`, and the `sustained_load_pressure_channel`/
    `_node_id` identity pair, gated on `sustained_load_pressure > 0.0`
    exactly the way `build_outreach_prompt` itself already gates naming
    them. `tension_reason=None` (no real trigger fired this tick, or a
    forced debug trigger) grounds nothing -- an empty tick licenses no
    technical claims, which is the exact gap the 2026-09-08 incident fell
    through.

    Does NOT include a live `harness_closure` prediction-error reading.
    Checked live 2026-09-08: no such reading reaches Hub today.
    `reduce_attention_self_model`'s own `harness_closure_signal` argument
    (`orion/substrate/attention_self_model.py`) is a substrate-side reducer
    input built from a FalkorDB node's metadata, not something this tick's
    `OutreachContext` has access to -- threading it through is a real
    follow-up, not solved in this patch (see this patch's PR report).
    """
    facts: set[str] = set()
    if tension_reason is None:
        return frozenset()

    _add_node_identity(facts, getattr(tension_reason, "target_id", None))

    sustained_load_pressure = float(
        getattr(tension_reason, "sustained_load_pressure", 0.0) or 0.0
    )
    if sustained_load_pressure > 0.0:
        _add_plain(facts, getattr(tension_reason, "sustained_load_pressure_channel", None))
        _add_node_identity(facts, getattr(tension_reason, "sustained_load_pressure_node_id", None))

    return frozenset(facts)


# Identifier-shaped run: starts with a letter, continues through letters,
# digits, and the punctuation real internal names actually use
# (`sustained_load_pressure`, `node:athena`, `orion:system:error`). Trailing
# sentence punctuation a token can pick up (a period at the end of a
# sentence, a comma before a conjunction) is stripped separately below --
# this pattern would otherwise fold a sentence-final period into the token
# via its own `.` character class member.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_:./\-]*")
_TRAILING_PUNCT = ".,;:!?)('\"`"


def _extract_candidate_tokens(text: str) -> list[str]:
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).strip(_TRAILING_PUNCT)
        if token:
            tokens.append(token)
    return tokens


def find_ungrounded_signal_mentions(text: str, grounded: Iterable[str]) -> list[str]:
    """Real-registry internal-signal names in `text` that are NOT in
    `grounded` (this tick's actually-true facts) -- the closed-vocabulary
    enforcement itself.

    Exact/word-boundary match against `known_real_signal_names()`, never
    fuzzy: a candidate token must equal a registry entry exactly (case-
    insensitively), not merely contain one -- "pressure" inside an ordinary
    sentence never matches "disk_capacity_pressure". Only COMPOUND
    candidate tokens (containing "_" or ":") are ever checked, for the same
    reason the registry itself restricts its two broad sources to compound
    names -- see module docstring's "ONLY COMPOUND NAMES" section for the
    two live false-positive collisions ("pressure", "execution") this
    exists to prevent.

    Returns the sorted, de-duplicated, ORIGINAL-CASE offending terms (empty
    when `text` names nothing real, or names only things that are true this
    tick) -- kept in original case for forensic readability in the decision
    log, not because comparison is case-sensitive.
    """
    grounded_lower = {str(g).strip().lower() for g in grounded if g}
    known = known_real_signal_names()
    offenders: dict[str, str] = {}
    for token in _extract_candidate_tokens(text):
        if not _is_compound(token):
            continue
        lowered = token.lower()
        if lowered in known and lowered not in grounded_lower:
            offenders.setdefault(lowered, token)
    return sorted(offenders.values(), key=str.lower)
