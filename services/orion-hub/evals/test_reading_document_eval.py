"""Offline eval: can Orion actually read the repo's own specs under the default policy?

Runs every real ``docs/superpowers/specs/*.md`` through the default document
policy rooted at this checkout, then builds the real Stage 1 document prompt.
Scores: share of specs accepted, every refusal is only ``document_too_large``
(never a policy false-positive), and every accepted prompt stays under the
single-argv ceiling the reader is launched with.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.schemas.world_pulse_read import WorldPulseReadSeedV1
from orion.world_pulse_read.documents import DocumentPolicy, DocumentSourceError, read_document
from scripts.world_pulse_read_pipeline import _build_document_stage1_prompt

REPO = Path(__file__).resolve().parents[3]
MAX_ARG_STRLEN = 131072
# Measured 2026-09-28: 282 specs, median 15 KB, p90 31 KB, max 70 KB.
MIN_ACCEPT_SHARE = 0.9


def test_repo_specs_are_readable_under_the_default_policy() -> None:
    specs = sorted((REPO / "docs/superpowers/specs").glob("*.md"))
    if len(specs) < 20:
        pytest.skip("spec corpus not present in this checkout")
    defaults = DocumentPolicy.from_env({})
    policy = DocumentPolicy.from_values(roots=str(REPO), extensions=None, max_bytes=defaults.max_bytes)
    accepted, refusals, longest = 0, {}, 0
    for spec in specs:
        try:
            doc = read_document(str(spec), policy)
        except DocumentSourceError as exc:
            refusals[str(exc)] = refusals.get(str(exc), 0) + 1
            continue
        accepted += 1
        seed = WorldPulseReadSeedV1(seed_id="reading:eval", kind="reading", run_id="eval", url=doc.ref)
        prompt = _build_document_stage1_prompt(seed, "trace-eval", sha256=doc.sha256, text=doc.text)
        longest = max(longest, len(prompt.encode()))
    share = accepted / len(specs)
    print(f"reading_document_eval specs={len(specs)} accepted={accepted} share={share:.3f} "
          f"refusals={refusals} longest_prompt_bytes={longest}")
    assert share >= MIN_ACCEPT_SHARE
    assert set(refusals) <= {"document_too_large"}
    assert longest < MAX_ARG_STRLEN
