from datetime import datetime, timezone

from orion.signals.adapters.result import normalize_adapter_result
from orion.signals.models import OrganClass, OrionSignalV1


def test_normalize_single_signal() -> None:
    now = datetime.now(timezone.utc)
    sig = OrionSignalV1(
        signal_id="a" * 64,
        organ_id="biometrics",
        organ_class=OrganClass.exogenous,
        signal_kind="gpu_load",
        dimensions={"level": 0.5},
        observed_at=now,
        emitted_at=now,
    )
    out = normalize_adapter_result(sig)
    assert out == [sig]


def test_normalize_list() -> None:
    assert normalize_adapter_result([]) == []
    assert normalize_adapter_result(None) == []
