"""PE suite rebuild step 5 (2026-11) removed fusion's meta.signal_kind exemption
from the low-info filter; it only existed for the retired text="" publish-gap
fragments. The causal-latency fragments recall still emits carry their own
English and must still survive a substantive query on their text alone."""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO / "services" / "orion-recall", _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.fusion import _is_low_info_social_candidate  # noqa: E402
from app.storage.falkor_bus_synaptic_adapter import _format_causal_anomaly_text  # noqa: E402


def test_causal_latency_text_is_not_low_info() -> None:
    now = time.time()
    row = {
        "source_organ": "orion-cortex-orch",
        "target_organ": "orion-cortex-exec",
        "zscore": 4.2,
        "count": 40,
        "last_seen_epoch": now - 60,
    }
    text = _format_causal_anomaly_text(row, now_epoch=now)
    assert text.strip()
    assert not _is_low_info_social_candidate(text)
