"""Offline council parse-to-event replay of the observed entity-shape failure family.

The full failed model replies were not persisted. These synthetic reconstructions
use the field shapes exposed by the 2026-11-12 live validation errors; this eval
does not claim to measure live model compliance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1])]

from app.interpretation import parse_and_project  # noqa: E402
from orion.schemas.vision import VisionWindowPayload  # noqa: E402


def main() -> None:
    window = VisionWindowPayload(
        window_id="synthetic-walkway-vehicle",
        start_ts=1.0,
        end_ts=2.0,
        stream_id="walkway",
        summary={"evidence": {"hard_labels": ["vehicle"]}, "object_counts": {"vehicle": 1}},
        artifact_ids=["synthetic-artifact"],
    )
    base = {
        "schema_version": "1.0",
        "window_id": window.window_id,
        "scene_summary": "A vehicle is visible on the walkway.",
        "salient_observations": [],
        "uncertainties": [],
        "event_candidates": [{
            "event_type": "vehicle_present",
            "narrative": "A vehicle is visible on the walkway.",
            "entities": ["vehicle"],
        }],
        "evidence_refs": ["synthetic-artifact"],
    }
    cases = [
        ("strict", {}, "strict_v2", ["vehicle"]),
        ("grounded-object", {
            "entities": [{"id": "vehicle_1", "type": "vehicle", "confidence": 0.9}],
            "event_candidates": [{
                "event_type": "vehicle_present",
                "narrative": "A vehicle is visible on the walkway.",
                "entities": [{"id": "vehicle_1", "type": "vehicle"}],
            }],
        }, "salvaged_v2", ["vehicle"]),
        ("event-object-only", {
            "event_candidates": [{
                "event_type": "vehicle_present",
                "narrative": "A vehicle is visible on the walkway.",
                "entities": [{"type": "vehicle"}],
            }],
        }, "salvaged_v2", ["vehicle"]),
        ("unsupported-object", {
            "entities": [{"id": "dragon_1", "type": "dragon"}],
            "event_candidates": [{
                "event_type": "vehicle_present",
                "narrative": "A vehicle is visible on the walkway.",
                "entities": [{"type": "dragon"}],
            }],
        }, "parse_failed", None),
    ]
    for name, changes, mode, expected_entities in cases:
        interpretation, projected, outcome = parse_and_project(json.dumps({**base, **changes}), window)
        assert outcome.parse_mode == mode, (name, outcome.parse_mode)
        if expected_entities is None:
            assert interpretation is None and projected is None, name
        else:
            assert interpretation is not None and projected is not None, name
            assert len(projected.events) == 1, name
            assert projected.events[0].entities == expected_entities, name
            assert projected.events[0].narrative == base["event_candidates"][0]["narrative"], name
        print(f"{name}: {mode} events={len(projected.events) if projected else 0} entities={expected_entities}")


if __name__ == "__main__":
    main()
