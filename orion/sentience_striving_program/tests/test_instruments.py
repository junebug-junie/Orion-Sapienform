"""Tests for the sentience-program instrument manifest and its gate.

Two lanes, deliberately kept apart:

* Contract tests over the REAL instruments.yaml -- these are what stop the
  manifest rotting into a keyword cathedral (every instrument must name a module
  that exists, a real outcome, and an entrypoint that resolves).
* Behaviour tests over synthetic manifests, for the drift/staleness logic itself.

The synthetic lane deliberately does NOT reuse the real file: a gate proven only
against the fixture it was written alongside proves nothing, and this repo has
been bitten by exactly that (a synthetic fixture hiding a gate inversion).
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.sentience_striving_program.instruments import (
    Claim,
    build_state,
    check_repo_presence,
    evaluate_claim,
    load_manifest,
    resolve_retention_hours,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Contract lane -- runs against the real manifest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_real_manifest_parses(manifest):
    assert manifest["instruments"], "manifest declares no instruments"
    assert set(manifest["outcomes"]) == {"O1", "O2", "O3", "O4"}


def test_every_instrument_module_exists(manifest):
    """The manifest may not name code that is not there.

    This caught two stale paths while the manifest was being authored:
    novelty_for_target had moved to scoring.py, and capability_policy.py lives
    under orion/autonomy/, not orion/policy/.
    """
    missing = []
    for inst in manifest["instruments"]:
        exists, entry_ok = check_repo_presence(inst, REPO_ROOT)
        if not exists:
            missing.append(f"{inst.id}: {inst.module}")
        elif entry_ok is False:
            missing.append(f"{inst.id}: {inst.module}::{inst.entrypoint}")
    assert not missing, f"manifest names code that does not exist: {missing}"


def test_every_outcome_has_at_least_one_instrument(manifest):
    """An outcome nothing ladders to is an outcome nobody is working toward."""
    claimed = {i.outcome for i in manifest["instruments"]}
    assert claimed == set(manifest["outcomes"]), (
        f"outcomes with no instrument: {set(manifest['outcomes']) - claimed}"
    )


def test_sql_claims_are_read_only(manifest):
    """No claim may mutate. The board is an observer of the program, not an actor."""
    forbidden = ("insert", "update", "delete", "drop", "truncate", "alter", "create")
    for inst in manifest["instruments"]:
        for claim in inst.claims:
            if not claim.sql:
                continue
            lowered = claim.sql.lower()
            assert lowered.strip().startswith("select"), (
                f"{inst.id}/{claim.id}: SQL must start with SELECT"
            )
            for word in forbidden:
                assert f" {word} " not in f" {lowered} ", (
                    f"{inst.id}/{claim.id}: SQL contains {word!r}"
                )


def test_every_claim_records_a_date(manifest):
    """A recorded value without a date cannot be reasoned about later."""
    for inst in manifest["instruments"]:
        for claim in inst.claims:
            assert claim.recorded_at, f"{inst.id}/{claim.id} has no recorded_at"


def test_retention_resolves_for_instruments_that_declare_one(manifest):
    """A declared retention setting must actually resolve to a number.

    Regression guard: this returned None for every instrument in a fresh
    worktree, because `.env` is gitignored and the lookup had no fallback -- so
    the board rendered no ceiling at all, silently, which is precisely the fact
    it exists to surface.
    """
    for inst in manifest["instruments"]:
        if not inst.storage.retention_setting:
            continue
        hours, source = resolve_retention_hours(inst, REPO_ROOT)
        assert hours and hours > 0, (
            f"{inst.id}: {inst.storage.retention_setting} did not resolve ({source})"
        )


# ---------------------------------------------------------------------------
# Behaviour lane -- synthetic manifests
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, *, last_reviewed: str, recorded: int) -> Path:
    path = tmp_path / "m.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            version: 1
            outcomes:
              O1: {{title: "t", claim: "c"}}
            review_max_age_days: 90
            instruments:
              - id: probe
                title: "Probe"
                theory: "t"
                program_ref: "r"
                module: orion/sentience_striving_program/instruments.py
                entrypoint: load_manifest
                outcome: O1
                unlock: "u"
                last_reviewed: {last_reviewed}
                storage: {{kind: none}}
                claims:
                  - id: absence
                    question: "q"
                    kind: absent_from_repo
                    target: "zzz_string_that_appears_nowhere_in_this_repo"
                    recorded: {recorded}
                    recorded_at: 2026-09-02
            """
        ).strip()
    )
    return path


def test_holding_claim_reports_holds(tmp_path):
    m = load_manifest(_write_manifest(tmp_path, last_reviewed="2026-09-02", recorded=0))
    states = build_state(m, conn=None, root=REPO_ROOT, with_consumers=False)
    assert [c.status for c in states[0].claims] == ["HOLDS"]


def test_drifted_claim_is_detected(tmp_path):
    """The recorded value says 3 files mention it; reality says 0."""
    m = load_manifest(_write_manifest(tmp_path, last_reviewed="2026-09-02", recorded=3))
    states = build_state(m, conn=None, root=REPO_ROOT, with_consumers=False)
    assert states[0].claims[0].status == "DRIFTED"
    assert states[0].claims[0].drifted


def test_stale_review_is_flagged(tmp_path):
    old = (datetime.now(timezone.utc).date() - timedelta(days=400)).isoformat()
    m = load_manifest(_write_manifest(tmp_path, last_reviewed=old, recorded=0))
    states = build_state(m, conn=None, root=REPO_ROOT, with_consumers=False)
    assert states[0].review_stale
    assert states[0].review_age_days >= 400


def test_fresh_review_is_not_flagged(tmp_path):
    """The complement of the test above -- a window that always fires is not a gate."""
    recent = (datetime.now(timezone.utc).date() - timedelta(days=5)).isoformat()
    m = load_manifest(_write_manifest(tmp_path, last_reviewed=recent, recorded=0))
    states = build_state(m, conn=None, root=REPO_ROOT, with_consumers=False)
    assert not states[0].review_stale


def test_manual_claim_is_never_auto_passed(tmp_path):
    """A human-run check must report MANUAL, not HOLDS.

    Reporting it as HOLDS would let a check nobody has actually run read as
    green -- the failure mode this whole program keeps rediscovering.
    """
    claim = Claim(
        id="m", question="q", kind="manual", recorded="PASS", recorded_at="2026-09-02"
    )
    assert evaluate_claim(claim, conn=None, root=REPO_ROOT).status == "MANUAL"


def test_sql_claim_without_connection_errors_not_passes(tmp_path):
    """No database must mean ERROR, never a silent HOLDS."""
    claim = Claim(
        id="s",
        question="q",
        kind="sql",
        recorded=1,
        recorded_at="2026-09-02",
        sql="SELECT 1",
    )
    assert evaluate_claim(claim, conn=None, root=REPO_ROOT).status == "ERROR"


def test_unknown_outcome_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            """
            version: 1
            outcomes:
              O1: {title: "t", claim: "c"}
            instruments:
              - id: probe
                title: "P"
                theory: "t"
                program_ref: "r"
                module: orion/sentience_striving_program/instruments.py
                outcome: O9
                unlock: "u"
                last_reviewed: 2026-09-02
                storage: {kind: none}
            """
        ).strip()
    )
    with pytest.raises(ValueError, match="unknown outcome"):
        load_manifest(path)


def test_duplicate_instrument_id_is_rejected(tmp_path):
    path = tmp_path / "dup.yaml"
    body = textwrap.dedent(
        """
              - id: probe
                title: "P"
                theory: "t"
                program_ref: "r"
                module: orion/sentience_striving_program/instruments.py
                outcome: O1
                unlock: "u"
                last_reviewed: 2026-09-02
                storage: {kind: none}
        """
    ).rstrip()
    path.write_text(
        "version: 1\noutcomes:\n  O1: {title: t, claim: c}\ninstruments:" + body + body
    )
    with pytest.raises(ValueError, match="duplicate instrument id"):
        load_manifest(path)
