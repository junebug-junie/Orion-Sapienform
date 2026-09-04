"""Read-only Hub API for the Mood Arc / Field Anomaly operator page.

Two views:

- GET /live: relays orion-field-digester's own `/health` status for its
  mood-arc encoder + live-enrichment coverage (app/anomaly_scorer.py
  ::status() in that service) -- this is genuinely that service's own
  runtime state, so it's fetched from the source rather than reconstructed
  from Postgres or the model directory's filesystem layout (see the
  `hub-mood-arc-status-ekg-traceability` plan's Approach section for why).
- GET /phi-v2-inventory: honest current-state snapshot of phi-v2's stubs --
  the two dead legacy `orion/inner_state_registry.py` entries plus whether
  the design doc's real (but unwired) successor pieces exist on disk. No
  progress bar, no fabricated completion percentage: phi-v2 itself is not
  implemented, and this says so plainly rather than implying otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from scripts.field_digester_client import FieldDigesterClientError, fetch_health
from scripts.service_logs import resolve_repo_root

router = APIRouter(prefix="/api/mood-arc-status", tags=["mood-arc-status"])

# `Path(__file__).resolve().parents[N]` breaks inside the Hub's own Docker
# image -- confirmed live (2026-09-04 docker up smoke test): the Dockerfile
# COPYs services/orion-hub flattened straight to /app (no `services/`,
# `docs/`, or repo-root level above it in the container), so a fixed
# `parents[3]` raised IndexError on every startup and crashed the whole
# service. `resolve_repo_root()` (scripts/service_logs.py) is the existing,
# already-used-by-two-other-routes mechanism for this exact problem: reads
# `ORION_REPO_ROOT` (the read-only `/repo` bind mount docker-compose.yml
# already sets up for grammar_atlas_routes.py/service_logs.py), falls back
# to walking up from this file, then cwd, then a bare `/repo` guess.
def _phi_v2_design_doc() -> Path:
    return resolve_repo_root() / "docs" / "superpowers" / "specs" / "2026-08-21-phi-v2-design.md"


def _phi_encoder_cli() -> Path:
    return resolve_repo_root() / "scripts" / "fit_phi_encoder.py"


# Both confirmed dead (2026-09-04 investigation): orion-spark-introspector,
# their shared producer, was deleted outright 2026-07-28. Hand-picked rather
# than a prefix scan over REGISTRY -- there are exactly two phi-tagged
# entries today and a silent new one showing up unlisted here is a real,
# worth-noticing gap, not something to paper over with a "phi" substring
# match that could also snag an unrelated future signal.
PHI_V2_LEGACY_SIGNAL_IDS: tuple[str, ...] = ("phi_heuristic.valence", "phi_intrinsic_reward.v1")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _first_sentences(text: str, count: int = 2) -> str:
    text = " ".join(text.split())  # collapse the registry's wrapped multi-line notes
    parts = _SENTENCE_SPLIT.split(text)
    return " ".join(parts[:count]).strip()


@router.get("/live")
async def live() -> dict[str, Any]:
    try:
        health = await fetch_health()
    except FieldDigesterClientError as exc:
        return {"reachable": False, "error": str(exc)}
    # `.get(..., default) or default`, not just `.get(..., default)`: review
    # finding (2026-09-04) -- the former only substitutes on a MISSING key,
    # so a hypothetical `"field_channel_anomaly": null` response would pass
    # None to `**`, raising TypeError instead of degrading gracefully.
    anomaly_block = health.get("field_channel_anomaly") or {"enabled": False}
    return {"reachable": True, **anomaly_block}


def _design_doc_status(doc_path: Path) -> str | None:
    if not doc_path.exists():
        return None
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Status:"):
            return line.removeprefix("Status:").strip()
    return None


@router.get("/phi-v2-inventory")
async def phi_v2_inventory() -> dict[str, Any]:
    from orion.inner_state_registry import REGISTRY

    repo_root = resolve_repo_root()
    signals = []
    by_id = {sig.signal_id: sig for sig in REGISTRY}
    for signal_id in PHI_V2_LEGACY_SIGNAL_IDS:
        sig = by_id.get(signal_id)
        if sig is None:
            signals.append({"signal_id": signal_id, "found_in_registry": False})
            continue
        # composition_status intentionally does NOT encode "retired" in this
        # registry's own convention (orion/inner_state_registry.md's
        # `field_attention_frame.v1` entry: RETIRED 2026-08-21 but stayed
        # COMPOSED) -- prose in `notes` carries that instead, and notes only
        # ever grow by dated append, so the most recent correction can land
        # well past what `last_note`'s first-2-sentences shows. A live,
        # code-verified check is the un-stale-able signal: does the claimed
        # producer_service still exist as a REAL, deployable service.
        #
        # NOT a bare `producer_dir.is_dir()` check -- confirmed live
        # (2026-09-04 docker smoke test) that `services/orion-spark-
        # introspector/` still physically exists on disk (app/, tests/,
        # train/, a gitignored .env) even though it was fully deleted from
        # git 2026-07-28 (commit 442e51ee2): `git rm` / the retirement PR
        # removed it from tracking but never `rm -rf`'d the directory
        # itself, so a bare-directory check reported "producer present" for
        # a service that is, by this repo's own convention, dead. Every
        # real service in this repo has its own docker-compose.yml (grepped
        # across services/*/); the leftover has none -- that's the reliable
        # signal, not mere directory presence.
        producer_dir = repo_root / "services" / sig.producer_service
        signals.append(
            {
                "signal_id": sig.signal_id,
                "found_in_registry": True,
                "producer_service": sig.producer_service,
                "producer_service_exists": (producer_dir / "docker-compose.yml").is_file(),
                "composition_status": sig.composition_status.value,
                "cognition_consumers": list(sig.cognition_consumers),
                "last_note": _first_sentences(sig.notes),
            }
        )

    design_doc = _phi_v2_design_doc()
    return {
        "legacy_signals": signals,
        "design_doc": {
            "path": str(design_doc.relative_to(repo_root)),
            "exists": design_doc.exists(),
            "status": _design_doc_status(design_doc),
        },
        "manual_cli_exists": _phi_encoder_cli().exists(),
    }
