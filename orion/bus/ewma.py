"""Pure incremental EWMA mean/variance/z-score, shared across bus-observing
services.

Same math as ``services/orion-bus-mirror/app/graph_writer.py::compute_ewma_update``
(the bus-synaptic-graph arc's own EWMA mechanism), deliberately re-homed here
rather than imported cross-service -- ``orion/bus/`` is already the shared
home for bus utilities multiple services import (``census.py``, ``velocity.py``),
and ``graph_writer.py`` has had heavy concurrent edit activity this session
(Ideas C/D/2.1 all touched it, with a real merge conflict along the way);
adding a new shared module here avoids that hot file entirely rather than
risking another collision. Not a refactor of the existing implementation --
that file keeps its own copy, this is a fresh, independent user (Idea 6,
``bus_activity_zscore``, in ``services/orion-bus``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_MIN_VARIANCE = 1e-6


@dataclass(frozen=True)
class EwmaUpdate:
    ewma: float
    variance: float
    zscore: float | None  # None until a baseline exists (first observation)


def compute_ewma_update(
    *,
    prev_ewma: float,
    prev_variance: float,
    prev_count: int,
    value: float,
    alpha: float,
) -> EwmaUpdate:
    """Pure incremental EWMA mean/variance update, plus a z-score of ``value``
    against the *prior* baseline (computed before the baseline absorbs it).

    No z-score is returned on the first observation (``prev_count == 0``) --
    there is no baseline yet to be anomalous against. Returning 0.0 there
    would misrepresent "no data yet" as "measured, not anomalous" (this
    repo's own "no empty-shell cognition" rule).
    """
    if prev_count == 0:
        return EwmaUpdate(ewma=value, variance=0.0, zscore=None)

    variance_floor = max(prev_variance, _MIN_VARIANCE)
    zscore = (value - prev_ewma) / math.sqrt(variance_floor)
    new_ewma = alpha * value + (1 - alpha) * prev_ewma
    deviation = value - prev_ewma
    new_variance = alpha * (deviation * deviation) + (1 - alpha) * prev_variance
    return EwmaUpdate(ewma=new_ewma, variance=new_variance, zscore=zscore)
