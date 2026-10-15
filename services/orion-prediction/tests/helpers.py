from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.series import parse_registry

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def spec_doc(**over) -> dict:
    target = {
        "series_id": "test.temp", "role": "target", "source": {"kind": "biometrics", "node": "n", "key": "t"},
        "unit": "C", "grain": "5min", "agg": "mean", "max_fill": "15min", "train_window": "10d",
        "seasonalities": [{"name": "daily", "period": "1d", "fourier_order": 3}],
        "horizons": ["5min", "1h", "6h"], "thresholds": [{"name": "hot", "value": 31.0, "window": "30min"}],
        "consumers": ["tests"],
    }
    target.update(over)
    return {"version": 1, "series": [target]}


def spec(**over):
    return parse_registry(spec_doc(**over)).get("test.temp")


def daily_series(days: float, grain_min: int = 5, amp: float = 1.0, noise: float = 0.05, walk: float = 0.0,
                 seed: int = 0, start: datetime = T0) -> pd.Series:
    n = int(days * 1440 // grain_min)
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=f"{grain_min}min", tz="UTC")
    minutes = (idx.hour * 60 + idx.minute).to_numpy()
    base = 28 + amp * np.sin(2 * np.pi * minutes / 1440)
    rw = np.cumsum(rng.normal(0, walk, n)) if walk else 0
    return pd.Series(base + rw + rng.normal(0, noise, n), index=idx)


def points_of(series: pd.Series, jitter_sec: int = 30) -> list[tuple[datetime, float]]:
    """Raw readings every `jitter_sec` inside each bucket, all equal to the bucket value."""
    out = []
    for ts, v in series.items():
        if not np.isfinite(v):
            continue
        for k in range(0, 300, jitter_sec):
            out.append(((ts + timedelta(seconds=k)).to_pydatetime(), float(v)))
    return out
