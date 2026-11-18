"""The "Orion Daily Digest" email is retired (integration patch 4; Juniper Q7, 2026-11-15): kill means kill.
The daily organ roundup in the day letter carries its counts. The curiosity weekly and monthly compaction
digests are NOT retired (Juniper Q8): this file also pins that they are still there."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CODE_DIRS = ("services", "orion", "scripts", "config")
CODE_SUFFIXES = {".py", ".yml", ".yaml", ".sh", ".j2", ".toml"}


def _code_files():
    for d in CODE_DIRS:
        for p in (REPO / d).rglob("*"):
            if p.is_file() and (p.suffix in CODE_SUFFIXES or p.name == ".env_example") and "/tests/" not in str(p):
                yield p


def test_the_notify_digest_service_and_its_smokes_are_gone():
    assert not (REPO / "services" / "orion-notify-digest").exists()
    for smoke in ("smoke_digest.sh", "smoke_topic_digest.sh", "smoke_topic_drift_alert.sh"):
        assert not (REPO / "scripts" / smoke).exists(), smoke


def test_nothing_publishes_or_listens_for_the_daily_digest_event():
    literal = re.compile(r"""["']orion\.digest\.daily["']""")
    hits = [str(p.relative_to(REPO)) for p in _code_files()
            if literal.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert hits == []


def test_orion_actions_no_longer_journals_the_digest():
    main = (REPO / "services" / "orion-actions" / "app" / "main.py").read_text()
    assert "_handle_journal_notify" not in main and '"notify.notification.request.v1"' not in main
    for p in ("app/settings.py", ".env_example"):
        assert "orion:notify:persistence:request" not in (REPO / "services" / "orion-actions" / p).read_text(), p
    assert "ACTIONS_JOURNAL_NOTIFY_RECALL_PROFILE" not in (REPO / "services" / "orion-actions" / ".env_example").read_text()
    assert not (REPO / "orion" / "recall" / "profiles" / "journal.notify.grounded.v1.yaml").exists()


def test_the_curiosity_weekly_and_monthly_digests_stay():
    # Juniper Q8: keep them. The roundup sits beside them, not instead of them.
    for rel in ("orion/curiosity/compaction.py", "orion/curiosity/compaction_store.py",
                "services/orion-dream/app/curiosity_compaction_submit.py"):
        assert (REPO / rel).is_file(), rel
    from typing import get_args

    from orion.schemas.durable_run import DurableWorkflowV1

    assert "curiosity.compaction" in get_args(DurableWorkflowV1)
