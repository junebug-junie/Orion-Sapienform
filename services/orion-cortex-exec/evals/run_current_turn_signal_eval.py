"""Eval: precision floor for `current_turn_llm_signals.parse_current_turn_llm_signals`.

Context: `build_current_turn_llm_prompt` already instructs the model to exclude
filler/interjections (replacing the deleted regex `_PROPER_RE` detector -- see
`orion/substrate/attention/detectors/current_turn.py`'s docstring). Confirmed live
2026-08-21, AFTER that replacement had already been running for hours: the model
still returned bare single words as "concept"/"other" candidates ("bus", "Glad",
"Compact", "Interesting") -- unactionable garbage, one step removed from the old
regex's failure mode, not eliminated by it. `parse_current_turn_llm_signals` now
has a structural floor under the prompt's instruction (single bare token must be
typed person/place; multi-word phrases pass regardless of type).

This is a fixture-driven precision check, not a live LLM call -- deterministic,
no bus, no Docker, matches `orion/substrate/attention/evals/run_topdown_eval.py`'s
shape. It measures the floor's actual precision/recall on a labeled set rather
than asserting it by feel, per CLAUDE.md's "no regex swamp" / metric-quality-gate
spirit: a filter this thin only earns trust from a number that can regress.

Run: python services/orion-cortex-exec/evals/run_current_turn_signal_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Same two-path insertion as tests/conftest.py::_ensure_cortex_exec_paths (this
# script has no pytest conftest to do it automatically when run standalone via
# `python services/orion-cortex-exec/evals/run_current_turn_signal_eval.py`).
_SERVICE_DIR = str(Path(__file__).resolve().parents[1])
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
for _path in (_SERVICE_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("SERVICE_NAME", "cortex-exec")
os.environ.setdefault("SERVICE_VERSION", "0.2.0")
os.environ.setdefault("NODE_NAME", "athena")
os.environ.setdefault("ORION_BUS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ORION_BUS_ENABLED", "false")
os.environ.setdefault("ORION_BUS_ENFORCE_CATALOG", "false")

from app.current_turn_llm_signals import parse_current_turn_llm_signals  # noqa: E402

# (label, raw candidates as the model would return them, expected surviving phrases)
# "garbage_live" entries are the exact real rows confirmed in attention_salience_trace
# on 2026-08-21 -- this fixture is the regression test for that incident, not a
# hypothetical.
_FIXTURES: list[tuple[str, list[dict[str, str]], list[str]]] = [
    (
        "garbage_live_bus",
        [{"phrase": "bus", "type": "concept"}],
        [],
    ),
    (
        "garbage_live_glad",
        [{"phrase": "Glad", "type": "other"}],
        [],
    ),
    (
        "garbage_live_compact",
        [{"phrase": "Compact", "type": "activity"}],
        [],
    ),
    (
        "garbage_live_interesting",
        [{"phrase": "Interesting", "type": "belief"}],
        [],
    ),
    (
        "garbage_regex_precedent_heck",
        # The exact failure the deleted LegacyRegexSignalDetector had: a
        # sentence-initial interjection ("Heck yeah!") matched as a "proper noun".
        [{"phrase": "Heck", "type": "other"}],
        [],
    ),
    (
        "real_person_single_word",
        [{"phrase": "Sarah", "type": "person"}],
        ["Sarah"],
    ),
    (
        "real_place_single_word",
        [{"phrase": "Paris", "type": "place"}],
        ["Paris"],
    ),
    (
        "real_multiword_plan",
        [{"phrase": "the reactor rollout plan", "type": "plan"}],
        ["the reactor rollout plan"],
    ),
    (
        "real_multiword_concept",
        [{"phrase": "context compaction", "type": "concept"}],
        ["context compaction"],
    ),
    (
        "mixed_batch",
        [
            {"phrase": "bus", "type": "concept"},
            {"phrase": "Sarah", "type": "person"},
            {"phrase": "the reactor rollout plan", "type": "plan"},
            {"phrase": "Glad", "type": "other"},
        ],
        ["Sarah", "the reactor rollout plan"],
    ),
]


def run() -> int:
    print("\n=== current_turn_llm_signals structural-floor eval ===")
    total = len(_FIXTURES)
    passed = 0
    for label, candidates, expected in _FIXTURES:
        raw = json.dumps(candidates)
        parsed = parse_current_turn_llm_signals(raw) or []
        survivors = [p["phrase"] for p in parsed]
        ok = survivors == expected
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: survivors={survivors!r} expected={expected!r}")

    print(f"\nRESULT: {passed}/{total} fixtures correct")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
