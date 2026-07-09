"""Pure gate predicate for the phi (InnerStateFeaturesV1) training corpus.

Garbage-in should be rejected at the write/ingestion boundary, not discovered
later at fit time. This module holds the shared predicate both corpus writers
(offline backfill and the live per-tick worker) call before appending a row.

Must stay free of any dependency on services/* — services depend on orion/,
never the reverse. Callers supply the cognitive feature names they care about
rather than this module importing/hardcoding them.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1


def is_corpus_row_healthy(
    inner: InnerStateFeaturesV1,
    *,
    cognitive_feature_names: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str]]:
    """Pure predicate: should this InnerStateFeaturesV1 row enter the phi corpus?

    Returns (healthy, reasons) — reasons is empty when healthy, else one string
    per failed check (all applicable reasons are included, not just the first).
    Never raises: any unexpected/malformed shape degrades to a safe default
    rather than propagating.
    """
    reasons: List[str] = []

    try:
        phi_health = getattr(inner, "phi_health", "ok")
        if phi_health != "ok":
            reasons.append(f"phi_health:{phi_health}")
    except Exception:
        reasons.append("phi_health:unreadable")

    try:
        if getattr(inner, "grammar_truth_degraded", False):
            reasons.append("grammar_truth_degraded")
    except Exception:
        reasons.append("grammar_truth_degraded:unreadable")

    if cognitive_feature_names is not None:
        try:
            names = set(cognitive_feature_names)
        except Exception:
            names = set()
        if names:
            try:
                features: List[InnerFeatureV1] = getattr(inner, "features", None) or []
                if not isinstance(features, (list, tuple)):
                    features = []
                matched = []
                for feat in features:
                    try:
                        name = getattr(feat, "name", None)
                        if name in names:
                            matched.append(feat)
                    except Exception:
                        continue
                if matched:
                    all_none = True
                    for feat in matched:
                        source = getattr(feat, "source", "") or ""
                        try:
                            if not str(source).endswith(".none"):
                                all_none = False
                                break
                        except Exception:
                            all_none = False
                            break
                    if all_none:
                        reasons.append("cognitive_features_all_none")
            except Exception:
                # Defensive: never let a malformed features list raise.
                pass

    return (not reasons, reasons)
