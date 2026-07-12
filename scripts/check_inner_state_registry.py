#!/usr/bin/env python3
"""Deterministic gate for orion/self_state/inner_state_registry.py.

Two distinct, independent checks:

1. Rot check (reliable): every InnerStateSignal.schema in REGISTRY must still
   import and be a real pydantic BaseModel subclass. Catches the registry
   silently going stale as schemas move/rename/get deleted.

2. New-duplicate heuristic (best-effort, NOT a proof): scans orion/bus/
   channels.yaml's schema_id fields and orion/schemas/registry.py's flat
   name->class dict for entries whose name matches a small, explicit,
   hand-maintained keyword list associated with "this looks like an
   inner-state-shaped signal" -- and fails if a match has no corresponding
   REGISTRY entry (by schema class name).

   Real, stated limitation: this is a naming-convention heuristic, not a
   formal proof. A new schema named around the keyword list can evade it.
   A shared marker base class across every inner-state schema was considered
   and rejected (see docs/superpowers/specs/2026-07-12-inner-state-
   unification-design.md) -- none of the existing six schemas currently
   share a common non-BaseModel parent, and retrofitting one purely to gain
   a slightly more precise heuristic is a real migration cost for a marginal
   precision gain over a maintained keyword list.

Usage:
    python scripts/check_inner_state_registry.py
    python scripts/check_inner_state_registry.py --channels-file <path>  # test override
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/check_inner_state_registry.py` puts scripts/ on
# sys.path[0], which shadows stdlib `platform` via scripts/platform/ and
# breaks pydantic (same issue documented in scripts/fit_phi_encoder.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.self_state.inner_state_registry import REGISTRY  # noqa: E402

# Deliberately small, explicit, hand-maintained -- grown by editing this list,
# not by a fuzzy classifier. Mirrors the same closed-list discipline already
# used by config/autonomy/signal_drive_map.yaml.
INNER_STATE_KEYWORDS: tuple[str, ...] = (
    "self_state",
    "selfstate",
    "drive",
    "autonomy_state",
    "autonomystate",
    "attention",
    "phi_reward",
    "phi_intrinsic",
    "mood",
    "felt_state",
    "cluster",
)

DEFAULT_CHANNELS_FILE = REPO_ROOT / "orion" / "bus" / "channels.yaml"

# schema class names deliberately excluded from the "needs its own REGISTRY
# entry" requirement, each with a real reason found while first running this
# gate (2026-07-12) -- not a blanket allowlist, every name here was traced.
_EXTRA_COVERED_SCHEMA_NAMES: dict[str, str] = {
    # --- companions/sub-objects of an existing REGISTRY entry ---
    "ProposalFrameV1": "l7_l11_ladder",
    "PolicyDecisionFrameV1": "l7_l11_ladder",
    "ExecutionDispatchFrameV1": "l7_l11_ladder",
    "FeedbackFrameV1": "l7_l11_ladder",
    "ConsolidationV1": "l7_l11_ladder",
    "PhiEncoderManifestV1": "phi_intrinsic_reward.v1 (training-artifact manifest, same lineage)",
    "SelfStateDimensionV1": "self_state.v1 (per-dimension row)",
    "FieldAttentionTargetV1": "field_attention_frame.v1 (per-target row, orion/schemas/field_attention_frame.py)",
    "DriveAuditV1": "drive_state.v1 (audit-trail companion, orion/core/schemas/drives.py)",
    "DriveNodeV1": "drive_state.v1 (cognitive-substrate graph-node projection, orion/core/schemas/cognitive_substrate.py -- static drive-kind node, not a live per-tick pressure reading)",
    # --- genuine keyword-collision false positives, unrelated domain ---
    "ArticleClusterV1": "false positive -- orion/schemas/world_pulse.py, news-article clustering, unrelated to node-health 'cluster'",
    # --- a distinct, real, adjacent subsystem NOT audited by this registry pass:
    #     conversational curiosity/attention-allocation (open loops, curiosity
    #     candidate actions, voluntary override), structurally different from
    #     field_attention_frame.v1's per-node/capability salience. Excluded
    #     here because this registry's scope (per the design spec) is felt-
    #     state signals, not attention-allocation mechanics -- flagged as
    #     worth its own future audit, not silently dismissed.
    "AttentionFrameV1": "distinct conversational-attention/curiosity subsystem (orion/schemas/attention_frame.py), not audited by this registry pass",
    "AttentionSignalV1": "distinct conversational-attention/curiosity subsystem (orion/schemas/attention_frame.py), not audited by this registry pass",
    "AttentionSalienceTraceV1": "distinct conversational-attention/curiosity subsystem (orion/schemas/attention_salience.py), not audited by this registry pass",
    "AttentionLoopOutcomeV1": "distinct conversational-attention/curiosity subsystem (orion/schemas/attention_salience.py), not audited by this registry pass",
    "PendingAttentionCardV1": "distinct conversational-attention/curiosity subsystem (orion/schemas/attention_salience.py), not audited by this registry pass",
    "ChatAttentionRequest": "distinct chat-notification-attention subsystem (orion/schemas/notify.py), not audited by this registry pass",
    "ChatAttentionAck": "distinct chat-notification-attention subsystem (orion/schemas/notify.py), not audited by this registry pass",
    "ChatAttentionState": "distinct chat-notification-attention subsystem (orion/schemas/notify.py), not audited by this registry pass",
}


def rot_check() -> list[str]:
    """Returns a list of failure messages; empty means all entries are healthy."""
    failures: list[str] = []
    for entry in REGISTRY:
        if entry.schema is None:
            continue
        if not isinstance(entry.schema, type) or not issubclass(entry.schema, BaseModel):
            failures.append(
                f"{entry.signal_id}: schema={entry.schema!r} is not a BaseModel subclass"
            )
    return failures


def _covered_schema_names() -> set[str]:
    names = {entry.schema.__name__ for entry in REGISTRY if entry.schema is not None}
    return names.union(_EXTRA_COVERED_SCHEMA_NAMES)


def _matches_keyword(name: str) -> bool:
    lowered = name.lower()
    return any(kw in lowered for kw in INNER_STATE_KEYWORDS)


def _channel_schema_ids(channels_file: Path) -> Iterable[str]:
    if not channels_file.is_file():
        return ()
    data = yaml.safe_load(channels_file.read_text(encoding="utf-8")) or {}
    channels = data.get("channels") or []
    seen: set[str] = set()
    for channel in channels:
        schema_id = channel.get("schema_id")
        if isinstance(schema_id, str) and schema_id not in seen:
            seen.add(schema_id)
            yield schema_id


def _registry_py_schema_names() -> Iterable[str]:
    from orion.schemas import registry as generic_registry  # local import: heavy module

    flat = getattr(generic_registry, "_REGISTRY", None)
    if isinstance(flat, dict):
        for name in flat:
            yield name


def new_duplicate_heuristic_check(*, channels_file: Path) -> list[str]:
    covered = _covered_schema_names()
    failures: list[str] = []

    for schema_id in _channel_schema_ids(channels_file):
        if _matches_keyword(schema_id) and schema_id not in covered:
            failures.append(
                f"orion/bus/channels.yaml declares schema_id={schema_id!r}, which matches an "
                f"inner-state keyword and has no orion/self_state/inner_state_registry.py entry"
            )

    for name in _registry_py_schema_names():
        if _matches_keyword(name) and name not in covered:
            failures.append(
                f"orion/schemas/registry.py declares {name!r}, which matches an inner-state "
                f"keyword and has no orion/self_state/inner_state_registry.py entry"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels-file",
        type=Path,
        default=DEFAULT_CHANNELS_FILE,
        help="override for testing against a fixture channels.yaml",
    )
    args = parser.parse_args(argv)

    failures = rot_check() + new_duplicate_heuristic_check(channels_file=args.channels_file)

    if failures:
        print("inner_state_registry gate FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"inner_state_registry gate OK ({len(REGISTRY)} entries checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
