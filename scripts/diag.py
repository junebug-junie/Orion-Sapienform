#!/usr/bin/env python3
"""Read-only phi corpus diagnostic: per-dimension variance + corpus gate status.

Referenced as an acceptance check in
docs/superpowers/specs/2026-07-09-phi-truthful-corpus-overview.md and
2026-07-09-phi-seedv4-feature-set-design.md ("scripts/diag.py on >=4h
seed-v4 corpus: >=8 dims var>1e-6"). Reuses fit_phi_encoder.py's row
loading/filtering/variance-gate logic directly rather than duplicating it --
this script never trains and never writes artifacts, it only reports.

Usage:
    python scripts/diag.py --corpus /mnt/telemetry/phi/corpus/inner_state.jsonl
    python scripts/diag.py --corpus <path> --features-version seed-v3

Exit code 0 when all corpus gates (rows, hours, variance) pass; 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/diag.py` puts scripts/ on sys.path[0], which
# shadows stdlib `platform` via scripts/platform/ and breaks pydantic (same
# guard as fit_phi_encoder.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from scripts.fit_phi_encoder import (  # noqa: E402
    DEFAULT_MIN_HOURS,
    DEFAULT_MIN_ROWS,
    DEFAULT_VARIANCE_EPS,
    DEFAULT_VARIANCE_FRACTION,
    _filter_training_rows,
    _hours_span,
    _load_jsonl,
    _variance_gate,
    features_version,
    input_features,
)

DEFAULT_CORPUS_PATH = Path("/mnt/telemetry/phi/corpus/inner_state.jsonl")
# Diagnostic default checks the currently-live corpus-write version, not
# fit_phi_encoder.py's own conservative DEFAULT_FEATURES_VERSION (which stays
# seed-v3 until a seed-v4 encoder has actually passed the promote gate).
DEFAULT_DIAG_FEATURES_VERSION = "seed-v4"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--features-version", type=str, default=DEFAULT_DIAG_FEATURES_VERSION)
    parser.add_argument("--legacy-corpus", action="store_true", help="seed-v1 felt-only rows")
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--min-hours", type=float, default=DEFAULT_MIN_HOURS)
    parser.add_argument("--variance-fraction", type=float, default=DEFAULT_VARIANCE_FRACTION)
    parser.add_argument("--variance-eps", type=float, default=DEFAULT_VARIANCE_EPS)
    return parser.parse_args(argv)


def run_diag(args: argparse.Namespace) -> dict:
    """Pure: load + gate-check a corpus, return the full report dict. Never raises."""
    fv = features_version(legacy_corpus=args.legacy_corpus, features_version_arg=args.features_version)
    feature_names = input_features(legacy_corpus=args.legacy_corpus, features_version=fv)

    if not args.corpus.is_file():
        return {"ok": False, "reason": f"corpus not found: {args.corpus}"}

    try:
        raw_rows = _load_jsonl(args.corpus)
    except Exception as exc:
        # _load_jsonl raises ValueError on a malformed JSONL line -- a
        # diagnostic must report that, not crash the process.
        return {"ok": False, "reason": f"corpus load failed: {exc}"}
    loaded, excluded = _filter_training_rows(
        raw_rows,
        feature_names=feature_names,
        features_version=fv,
        legacy_corpus=args.legacy_corpus,
    )
    matrix = np.stack([r.x for r in loaded], axis=0) if loaded else np.zeros((0, len(feature_names)))
    hours = _hours_span(loaded)
    variance_ok, var_ok_count, var_needed = _variance_gate(
        matrix, fraction=args.variance_fraction, eps=args.variance_eps, feature_names=feature_names
    )
    variances = np.var(matrix, axis=0) if matrix.size else np.zeros(len(feature_names))
    dims = [
        {"name": name, "variance": round(float(var), 8), "live": bool(var > args.variance_eps)}
        for name, var in zip(feature_names, variances)
    ]

    rows_ok = len(loaded) >= args.min_rows
    hours_ok = hours >= args.min_hours
    overall_ok = rows_ok and hours_ok and variance_ok

    return {
        "ok": overall_ok,
        "corpus": str(args.corpus),
        "features_version": fv,
        "rows_total": len(raw_rows),
        "rows_matching_version_and_healthy": len(loaded),
        "rows_excluded": excluded,
        "hours_span": round(hours, 3),
        "gates": {
            "min_rows": {"need": args.min_rows, "got": len(loaded), "ok": rows_ok},
            "min_hours": {"need": args.min_hours, "got": round(hours, 3), "ok": hours_ok},
            "variance": {"need": var_needed, "got": var_ok_count, "ok": variance_ok},
        },
        "dims": dims,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_diag(args)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
