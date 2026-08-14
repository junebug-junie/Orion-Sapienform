"""Load the structural field-channel direction map.

One bit per channel ("which way is worse"), and nothing else. See
`config/attention/channel_direction_map.yaml` for why there are no weights here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import yaml

from orion.field.pressure import HIGHER_IS_BETTER_CHANNELS

Worse = Literal["up", "down"]

_VALID = ("up", "down")
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "attention" / "channel_direction_map.yaml"
)


class DirectionMapError(ValueError):
    """Raised on a malformed map. Loud on purpose: a silently-empty direction map
    would make the whole gate inert while still reporting a clean run."""


@dataclass(frozen=True)
class DirectionMap:
    """Resolved channel -> worse-direction lookup.

    `exact` wins over `suffix_rules`; `unmapped` channels resolve to None and are
    excluded from admission entirely.
    """

    exact: Dict[str, Worse]
    suffixes: Tuple[Tuple[str, Worse], ...]
    unmapped: frozenset[str]

    def worse_for(self, channel: str) -> Optional[Worse]:
        """Return the worse-direction for `channel`, or None if it does not vote.

        Explicitly-unmapped channels and channels matching no rule both return
        None -- the caller cannot tell them apart, and does not need to, but the
        YAML distinguishes them so an omission reads as a decision.
        """
        if channel in self.unmapped:
            return None
        exact = self.exact.get(channel)
        if exact is not None:
            return exact
        # Longest suffix wins, so a more specific rule cannot be shadowed by a
        # shorter one that happens to be declared first.
        for suffix, worse in self.suffixes:
            if channel.endswith(suffix):
                return worse
        return None

    def votes(self) -> bool:
        return bool(self.exact or self.suffixes)


def load_direction_map(path: Path | str | None = None) -> DirectionMap:
    resolved = Path(path) if path is not None else _DEFAULT_PATH
    if not resolved.is_file():
        raise DirectionMapError(f"direction map not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DirectionMapError(f"{resolved}: top level must be a mapping, got {type(raw).__name__}")

    # Shape-check each section before iterating. Without this, `channels:` given
    # as a list raises a bare AttributeError, and -- caught in review 2026-08-14
    # -- `unmapped:` given as a scalar string silently becomes a frozenset of its
    # *characters*, so the channel it names stays mapped and keeps voting. A
    # malformed map must fail loudly; a quietly-wrong one is worse than a missing
    # one because the run still reports clean.
    for section, expected in (("channels", dict), ("suffix_rules", dict), ("unmapped", list)):
        value = raw.get(section)
        if value is not None and not isinstance(value, expected):
            raise DirectionMapError(
                f"{resolved}: '{section}' must be a {expected.__name__}, "
                f"got {type(value).__name__}"
            )

    # Seed the "a fall is the tension" set from the repo's existing polarity
    # constant rather than re-listing it here.
    #
    # WHY (found 2026-08-14, after Juniper flagged the assumption): the first
    # version of this loader hand-authored `availability`/`delivery_confidence`/
    # `stream_backlog_health` in the YAML -- a hand-re-derivation of
    # `orion.field.pressure.HIGHER_IS_BETTER_CHANNELS`, and a strict SUBSET of
    # it, silently missing `confidence` and `available_capacity`. That constant
    # is the polarity every existing cognition consumer already uses
    # (`orion/attention/field_attention/selectors.py`, `orion/field/
    # commensurability.py`), so a divergent private copy would have had this
    # module disagreeing with its own package neighbours about which way is
    # "worse" for two channels. Deriving it removes the drift surface entirely.
    exact: Dict[str, Worse] = {channel: "down" for channel in HIGHER_IS_BETTER_CHANNELS}

    for channel, worse in (raw.get("channels") or {}).items():
        if worse not in _VALID:
            raise DirectionMapError(f"channel {channel!r}: worse={worse!r} not in {_VALID}")
        channel = str(channel)
        # The YAML may add channels the constant does not cover, but may not
        # contradict it -- that would reintroduce the divergence above quietly.
        if channel in HIGHER_IS_BETTER_CHANNELS and worse != "down":
            raise DirectionMapError(
                f"channel {channel!r}: YAML says worse={worse!r} but it is in "
                f"HIGHER_IS_BETTER_CHANNELS (a fall is the tension). Fix the constant "
                f"in orion/field/pressure.py if the polarity really changed."
            )
        exact[channel] = worse

    suffixes: list[Tuple[str, Worse]] = []
    for pattern, worse in (raw.get("suffix_rules") or {}).items():
        text = str(pattern)
        if not text.startswith("*"):
            raise DirectionMapError(f"suffix rule {text!r} must start with '*'")
        if worse not in _VALID:
            raise DirectionMapError(f"suffix {text!r}: worse={worse!r} not in {_VALID}")
        suffixes.append((text[1:], worse))
    suffixes.sort(key=lambda kv: len(kv[0]), reverse=True)

    unmapped = frozenset(str(c) for c in (raw.get("unmapped") or []))
    overlap = unmapped.intersection(exact)
    if overlap:
        raise DirectionMapError(f"channels both mapped and unmapped: {sorted(overlap)}")

    # Guard on the YAML's OWN contribution, not on the merged map. `exact` is now
    # seeded from HIGHER_IS_BETTER_CHANNELS, so the merged map is never empty and
    # a `votes()` check here would be permanently true -- an inert gate. The
    # thing actually worth catching is a YAML that has stopped contributing:
    # suffix rules cover 28 of the 33 live channels, so losing them would gut the
    # map while every unit test still passed.
    if not raw.get("suffix_rules") and not raw.get("channels"):
        raise DirectionMapError(
            f"direction map {resolved} contributes no rules of its own "
            f"(no suffix_rules, no channels); refusing to load"
        )

    return DirectionMap(exact=exact, suffixes=tuple(suffixes), unmapped=unmapped)
