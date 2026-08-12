"""Tests for `skills.runtime.builder_prune.v1`.

The gate tests are the load-bearing ones. This is the first action in this arc
with a real consequence, and the whole design turns on it acting ONLY when
acting would actually change something — an autonomous action that fires and
accomplishes nothing is the theater this repo keeps deleting.
"""
import asyncio
import os
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import verb_adapters as va  # noqa: E402
from orion.core.verbs.base import VerbContext  # noqa: E402

GB = 1024**3


def _ctx() -> VerbContext:
    return VerbContext(meta={"correlation_id": str(uuid4())})


def _payload(**args):
    """Matches what `_skill_args` actually reads: `payload.context["metadata"]
    ["skill_args"]` and `payload.args.extra["skill_args"]`. An earlier version
    of this helper guessed the shape and every gate test failed on an
    AttributeError rather than on the behaviour under test."""
    return SimpleNamespace(
        context={"metadata": {"skill_args": dict(args)}},
        args=SimpleNamespace(extra={}),
    )


def _run(verb, payload):
    return asyncio.get_event_loop().run_until_complete(verb.execute(_ctx(), payload))


def _result(out) -> dict:
    return out.metadata["skill_result"]


@pytest.fixture
def prune(monkeypatch):
    """The verb with its two real I/O calls stubbed, so the gate can be tested
    without a Docker daemon or a particular host."""
    verb = va.BuilderPruneVerb()
    calls: list[list[str]] = []

    class _Runner:
        def __init__(self, **_kw):
            pass

        def run(self, cmd):
            calls.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="deleted: sha256:abc\nTotal: 140GB\n")

    monkeypatch.setattr(va, "SafeCommandRunner", _Runner)
    verb._calls = calls  # type: ignore[attr-defined]
    return verb


def _set_disk(monkeypatch, used_pct: float, total_gb: int = 469):
    total = total_gb * GB
    used = int(total * used_pct / 100.0)
    monkeypatch.setattr(
        va,
        "_builder_prune_disk_usage",
        lambda path: {
            "mount_path": path,
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": total - used,
            "used_pct": used_pct,
        },
    )


def _set_cache(monkeypatch, **kw):
    async def _stats(_runner):
        return kw

    monkeypatch.setattr(va, "_builder_prune_cache_stats", _stats)


# --- size parsing (Docker's human output is the only source) -----------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("271.3GB", int(271.3 * 1000**3)),
        ("140.4GB (51%)", int(140.4 * 1000**3)),  # Reclaimable carries a percentage
        ("1.745GB", int(1.745 * 1000**3)),
        ("186.5MB", int(186.5 * 1000**2)),
        ("0B", 0),
        ("", 0),
        (None, 0),
        ("garbage", 0),
    ],
)
def test_parse_docker_size(raw, expected):
    assert va._parse_docker_size(raw) == expected


# --- the gate ----------------------------------------------------------------


def test_declines_when_disk_is_not_under_pressure(prune, monkeypatch):
    _set_disk(monkeypatch, used_pct=60.0)
    _set_cache(monkeypatch, available=True, reclaimable_bytes=200 * GB, total_bytes=271 * GB, entries=15037, active=0)
    out, _ = _run(prune, _payload(mode="execute"))
    r = _result(out)
    assert r["acted"] is False
    assert r["decision"] == "declined_no_pressure"
    assert r["escalate"] is False
    assert prune._calls == [], "must not touch docker when declining"


def test_declines_and_escalates_when_there_is_nothing_to_reclaim(prune, monkeypatch):
    """The branch that keeps this from being theater.

    Disk IS under pressure but pruning would change nothing, so the real
    condition is a capacity problem needing a human. Escalate rather than act.
    """
    _set_disk(monkeypatch, used_pct=88.0)
    _set_cache(monkeypatch, available=True, reclaimable_bytes=2 * GB, total_bytes=3 * GB, entries=40, active=0)
    out, _ = _run(prune, _payload(mode="execute"))
    r = _result(out)
    assert r["acted"] is False
    assert r["decision"] == "declined_nothing_to_reclaim"
    assert r["escalate"] is True
    assert prune._calls == []


def test_declines_when_the_cache_cannot_be_measured(prune, monkeypatch):
    """Unreadable is not zero. If the gate cannot be evaluated, the action
    declines rather than guessing in either direction."""
    _set_disk(monkeypatch, used_pct=88.0)
    _set_cache(monkeypatch, available=False, reason="docker_system_df_failed:boom")
    out, _ = _run(prune, _payload(mode="execute"))
    r = _result(out)
    assert out.ok is False
    assert r["acted"] is False
    assert r["decision"] == "declined_unmeasurable"
    assert r["escalate"] is True
    assert prune._calls == []


def test_preview_is_the_default_and_does_not_mutate(prune, monkeypatch):
    """A dry-run flag that does not actually gate the side effect is a failure
    mode this repo has shipped before. No mode arg at all must mean no prune."""
    _set_disk(monkeypatch, used_pct=88.0)
    _set_cache(monkeypatch, available=True, reclaimable_bytes=140 * GB, total_bytes=271 * GB, entries=15037, active=0)
    out, _ = _run(prune, _payload())
    r = _result(out)
    assert r["run_mode"] != "execute"
    assert r["acted"] is False
    assert r["decision"] == "would_act"
    assert prune._calls == [], "preview must never invoke docker builder prune"


def test_acts_only_when_both_conditions_hold(prune, monkeypatch):
    _set_disk(monkeypatch, used_pct=88.0)
    _set_cache(monkeypatch, available=True, reclaimable_bytes=140 * GB, total_bytes=271 * GB, entries=15037, active=0)
    out, _ = _run(prune, _payload(mode="execute"))
    r = _result(out)
    assert r["acted"] is True
    assert r["decision"] == "pruned"
    assert prune._calls, "execute mode must invoke docker"
    cmd = prune._calls[-1]
    assert cmd[:3] == ["docker", "builder", "prune"]
    assert "--filter" in cmd and f"until={va.BUILDER_PRUNE_DEFAULT_UNTIL_HOURS}h" in cmd
    assert "-f" in cmd


def test_until_filter_is_overridable_but_defaults_to_two_weeks(prune, monkeypatch):
    """The default exists so a prune keeps recent cache. Pruning everything is
    what forces the ~30-minute full rebuild that makes this chore painful."""
    assert va.BUILDER_PRUNE_DEFAULT_UNTIL_HOURS == 336
    _set_disk(monkeypatch, used_pct=88.0)
    _set_cache(monkeypatch, available=True, reclaimable_bytes=140 * GB, total_bytes=271 * GB, entries=15037, active=0)
    _run(prune, _payload(mode="execute", until_hours=168))
    assert "until=168h" in prune._calls[-1]


# --- the outcome, which is the point of the whole action ---------------------


def test_reports_a_real_measurable_outcome(prune, monkeypatch):
    """Continuous, per-action, externally verifiable, and measured by the actor
    itself — rather than inferred from a field delta shared across concurrent
    actions, which is why every existing outcome signal measured dead."""
    total = 469 * GB
    seq = [
        {"mount_path": "/mnt/docker", "total_bytes": total, "used_bytes": int(total * 0.88),
         "free_bytes": total - int(total * 0.88), "used_pct": 88.0},
        {"mount_path": "/mnt/docker", "total_bytes": total, "used_bytes": int(total * 0.58),
         "free_bytes": total - int(total * 0.58), "used_pct": 58.0},
    ]
    monkeypatch.setattr(va, "_builder_prune_disk_usage", lambda path: seq.pop(0))
    _set_cache(monkeypatch, available=True, reclaimable_bytes=140 * GB, total_bytes=271 * GB, entries=15037, active=0)

    out, _ = _run(prune, _payload(mode="execute"))
    r = _result(out)
    assert r["bytes_reclaimed"] > 0
    assert r["pct_reclaimed"] == pytest.approx(30.0, abs=0.01)
    assert r["disk_before"]["used_pct"] == 88.0
    assert r["disk_after"]["used_pct"] == 58.0


def test_thresholds_are_reported_so_a_decision_is_auditable(prune, monkeypatch):
    _set_disk(monkeypatch, used_pct=60.0)
    _set_cache(monkeypatch, available=True, reclaimable_bytes=140 * GB, total_bytes=271 * GB, entries=15037, active=0)
    out, _ = _run(prune, _payload())
    r = _result(out)
    assert r["thresholds"]["min_disk_pct"] == va.BUILDER_PRUNE_MIN_DISK_PCT
    assert r["thresholds"]["min_reclaimable_bytes"] == va.BUILDER_PRUNE_MIN_RECLAIMABLE_BYTES
    assert r["reason"]


def test_verb_is_registered_under_its_skill_name():
    """`executor.py`'s local-adapter branch resolves by this exact name; a
    mismatch is a silent no-op at dispatch time."""
    from orion.core.verbs.registry import registry

    assert registry.get("skills.runtime.builder_prune.v1") is va.BuilderPruneVerb
