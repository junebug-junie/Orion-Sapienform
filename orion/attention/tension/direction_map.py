"""Load the structural field-channel direction map.

One bit per channel ("which way is worse"), and nothing else. See
`config/attention/channel_direction_map.yaml` for why there are no weights here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import yaml

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

    exact: Dict[str, Worse] = {}
    for channel, worse in (raw.get("channels") or {}).items():
        if worse not in _VALID:
            raise DirectionMapError(f"channel {channel!r}: worse={worse!r} not in {_VALID}")
        exact[str(channel)] = worse

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

    resolved_map = DirectionMap(exact=exact, suffixes=tuple(suffixes), unmapped=unmapped)
    if not resolved_map.votes():
        raise DirectionMapError(f"direction map {resolved} has no rules; refusing to load")
    return resolved_map
