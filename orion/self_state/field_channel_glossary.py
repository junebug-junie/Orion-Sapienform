"""Loader + live-liveness classifier for FieldStateV1's 29 raw channels.

Companion to config/field/field_channel_glossary.v1.yaml (the structured
channel index) and services/orion-field-digester/README.md's "Field channel
glossary" section (the prose reference). Built for Hub's Field Channel
Glossary observability panel
(services/orion-hub/scripts/field_channel_glossary_routes.py), which needs
to answer "is this metric alive" from real data, not from the README's
frozen prose verdicts -- those were already found stale once (see the
README's "Decay vs. injection-interval mismatch" section) and the repo's
own rule is that runtime truth beats config truth.

classify_channel_series() generalizes the liveness heuristic
scripts/analysis/measure_capability_channel_health.py already validated
(1e-100 subnormal-float cutoff, "live" if max - median > 0.05) from that
script's 8 capability channels / 2 targets to all 29 channels across every
node and capability, operating on the same merged, correctly-polarized
per-tick channel dict every cognition consumer reads
(orion.self_state.scoring.collect_field_channel_pressures) rather than
reimplementing merge/polarity logic here.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _glossary_path_candidates() -> list[Path]:
    """Resolve config/field/field_channel_glossary.v1.yaml from a monorepo
    checkout *or* Hub's Docker image.

    Hub's Dockerfile only `COPY orion /app/orion` -- `config/` (a sibling of
    `orion/` on disk, not a subdirectory of it) never lands under `/app`, so
    `Path(__file__).resolve().parents[2]` (repo root in a normal checkout)
    resolves to `/app` inside the Hub container and the file is genuinely
    missing there. Compose mounts the full repo read-only at `/repo` (and
    `/mnt/scripts/Orion-Sapienform`) specifically for cases like this -- same
    problem, same fix already established in
    services/orion-hub/scripts/drives_analytics.py's
    `_repo_root_candidates()`, mirrored here rather than re-derived.
    """
    seen: set[str] = set()
    roots: list[Path] = []

    def _add(root: Path | None) -> None:
        if root is None:
            return
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append(resolved)

    raw = os.getenv("ORION_REPO_ROOT", "").strip()
    if raw:
        _add(Path(raw))
    here = Path(__file__).resolve()
    if len(here.parents) >= 3:
        _add(here.parents[2])  # repo root in a normal monorepo checkout
    _add(Path("/repo"))
    _add(Path("/mnt/scripts/Orion-Sapienform"))
    return [root / "config" / "field" / "field_channel_glossary.v1.yaml" for root in roots]


def _resolve_glossary_path() -> Path:
    for candidate in _glossary_path_candidates():
        if candidate.is_file():
            return candidate
    # Fall back to the monorepo-checkout candidate so the resulting
    # FileNotFoundError names the expected path, not an arbitrary one.
    return _glossary_path_candidates()[0]


_GLOSSARY_PATH = _resolve_glossary_path()

# Same threshold scripts/analysis/measure_capability_channel_health.py uses:
# strictly finer-grained than the smallest downstream decision boundary
# (field_attention_policy.v1.yaml's min_salience=0.10) this data feeds, so a
# channel that swings by less than this can never meaningfully move a
# decision gated at those thresholds.
LIVE_VARIANCE_THRESHOLD: float = 0.05

# Below this magnitude, treat a value as "zero or numerically-decayed-to-dust
# subnormal", not real signal -- channels here have repeatedly decayed to
# values like 6.85e-322 instead of landing exactly on 0.0.
SUBNORMAL_CUTOFF: float = 1e-100

# Verdicts a Hub "clean channels" filter should keep.
CLEAN_VERDICTS: frozenset[str] = frozenset({"live", "quiet"})

# Minimum sample count before the monotonic-non-decreasing ratchet_suspect
# check is trusted. With only 2 points, "non-decreasing" is true for ~half
# of all noisy-but-healthy series (a coin flip, not a monotonicity signal) --
# below this count, fall through to the ordinary quiet/live spread check
# instead of flagging a possible one-way ratchet off a single up-step.
RATCHET_MIN_SAMPLES: int = 4


@dataclass(frozen=True)
class FieldChannelGlossaryEntry:
    channel: str
    level: tuple[str, ...]
    category: str
    meaning: str
    self_state_dimension: str | None = None
    evidence_dimension: str | None = None


@functools.lru_cache(maxsize=1)
def load_glossary(path: Path | None = None) -> dict[str, Any]:
    """Load config/field/field_channel_glossary.v1.yaml.

    Cached (the file only changes on deploy); pass an explicit path to
    bypass the cache in tests.
    """
    target = path or _GLOSSARY_PATH
    with open(target, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    entries = tuple(
        FieldChannelGlossaryEntry(
            channel=e["channel"],
            level=tuple(e.get("level", [])),
            category=e["category"],
            meaning=e["meaning"],
            self_state_dimension=e.get("self_state_dimension"),
            evidence_dimension=e.get("evidence_dimension"),
        )
        for e in raw.get("channels", [])
    )
    return {"entries": entries, "categories": dict(raw.get("categories", {}))}


def classify_channel_series(values: list[float]) -> str:
    """Classify one channel's per-tick value series into a liveness verdict.

    Verdicts:
    - never_produced: the channel never appeared in any tick this window
      (absent, not even at a reconciled-default 0.0).
    - dead: present but every value is subnormal/zero -- functionally no
      information (covers the README's "folded-away", "fully unproduced
      past reconcile", and "quiet-but-only-subnormal-noise" cases alike;
      from an observability standpoint all three mean the same thing: this
      metric isn't telling you anything).
    - ratchet_suspect: monotonically non-decreasing across the whole window
      with a real net climb, and at least RATCHET_MIN_SAMPLES points --
      consistent with a mode=add channel absent from NODE_DECAY_CHANNELS/
      CAPABILITY_DECAY_CHANNELS. A heuristic, not a structural fact (see
      module docstring); a channel that legitimately climbed once during a
      short sampled window will also trip this. Below RATCHET_MIN_SAMPLES,
      "non-decreasing" isn't a meaningful monotonicity signal (with 2 points
      it's true ~half the time for any noisy-but-healthy series), so short
      series fall through to the quiet/live spread check instead.
    - quiet: present, genuinely wired, but low-variance in this window
      (max - median <= LIVE_VARIANCE_THRESHOLD).
    - live: present with real variance this window.
    """
    if not values:
        return "never_produced"
    if all(abs(v) < SUBNORMAL_CUTOFF for v in values):
        return "dead"
    non_decreasing = all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    climbed = (values[-1] - values[0]) > LIVE_VARIANCE_THRESHOLD
    if non_decreasing and climbed and len(values) >= RATCHET_MIN_SAMPLES:
        return "ratchet_suspect"
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = (
        sorted_vals[n // 2]
        if n % 2
        else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )
    spread = max(values) - median
    if spread <= LIVE_VARIANCE_THRESHOLD:
        return "quiet"
    return "live"
