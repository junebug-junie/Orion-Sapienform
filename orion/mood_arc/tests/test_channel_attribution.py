from __future__ import annotations

import numpy as np
import pytest

from orion.mood_arc.fit_encoder import (
    deviation_direction,
    init_weights,
    mean_signed_deviation,
    per_channel_reconstruction_error,
    recon_loss,
    top_channel_attribution,
)

_FIELDS = ("cpu_pressure", "gpu_pressure", "memory_pressure")
_WINDOW_SIZE = 4


def _weights() -> dict[str, np.ndarray]:
    d_in = _WINDOW_SIZE * len(_FIELDS)
    return init_weights(d_in, hidden_dim=6, latent_dim=3, seed=7, data_mean=np.zeros(d_in))


def _window(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(_WINDOW_SIZE * len(_FIELDS))


def test_per_channel_errors_average_to_the_same_total_recon_loss() -> None:
    """Regrouping (window_size, n_fields) into per-channel means must not
    change the total -- mean-of-per-channel-means over a rectangular grid
    equals the flat mean, so this is an exact identity, not an
    approximation. If this ever drifts, the reshape/axis logic is wrong."""
    weights = _weights()
    rng = np.random.default_rng(0)
    x = _window(rng)

    total = recon_loss(x, weights)
    per_channel = per_channel_reconstruction_error(
        x, weights, fields=_FIELDS, window_size=_WINDOW_SIZE
    )
    assert set(per_channel.keys()) == set(_FIELDS)
    assert np.mean(list(per_channel.values())) == pytest.approx(total, rel=1e-9)


def test_top_channel_attribution_ranks_highest_error_first() -> None:
    weights = _weights()
    rng = np.random.default_rng(1)
    x = _window(rng)

    ranked = top_channel_attribution(
        x, weights, fields=_FIELDS, window_size=_WINDOW_SIZE, limit=3
    )
    assert len(ranked) == 3
    # Parse back the values and confirm descending order.
    values = [float(entry.split("=")[1]) for entry in ranked]
    assert values == sorted(values, reverse=True)
    # Every channel name appears exactly once across the full (limit=3) ranking.
    names = {entry.split("=")[0] for entry in ranked}
    assert names == set(_FIELDS)


def test_top_channel_attribution_respects_limit() -> None:
    weights = _weights()
    rng = np.random.default_rng(2)
    x = _window(rng)

    ranked = top_channel_attribution(
        x, weights, fields=_FIELDS, window_size=_WINDOW_SIZE, limit=1
    )
    assert len(ranked) == 1


def test_a_deliberately_perturbed_channel_is_attributed_as_top_contributor() -> None:
    """Construct weights that reconstruct a known baseline well, then push
    one channel far from that baseline in the input -- it should dominate
    the attribution ranking."""
    d_in = _WINDOW_SIZE * len(_FIELDS)
    # b3 (the decoder's output bias) is the reconstruction target when the
    # rest of the network contributes ~0 -- init_weights seeds it to
    # data_mean, and scales W1/W2 down to 0.05, W3 down further to 0.005, so
    # xhat stays close to b3 regardless of x for a small random x.
    baseline = np.zeros(d_in)
    weights = init_weights(d_in, hidden_dim=6, latent_dim=3, seed=3, data_mean=baseline)

    x = np.zeros(d_in)
    # Push only the memory_pressure column (index 2 of 3 fields) far from
    # the zero baseline across all timesteps.
    channel_idx = _FIELDS.index("memory_pressure")
    for t in range(_WINDOW_SIZE):
        x[t * len(_FIELDS) + channel_idx] = 10.0

    ranked = top_channel_attribution(x, weights, fields=_FIELDS, window_size=_WINDOW_SIZE, limit=1)
    assert ranked[0].startswith("memory_pressure=")


def test_mean_signed_deviation_is_positive_when_real_values_run_higher_than_expected() -> None:
    d_in = _WINDOW_SIZE * len(_FIELDS)
    weights = init_weights(d_in, hidden_dim=6, latent_dim=3, seed=3, data_mean=np.zeros(d_in))

    x = np.zeros(d_in)
    channel_idx = _FIELDS.index("memory_pressure")
    for t in range(_WINDOW_SIZE):
        x[t * len(_FIELDS) + channel_idx] = 10.0  # real value >> expected (~0)

    signed = mean_signed_deviation(x, weights, window_size=_WINDOW_SIZE, n_fields=len(_FIELDS))
    assert signed > 0
    assert deviation_direction(signed) == "elevated"


def test_mean_signed_deviation_is_negative_when_real_values_run_lower_than_expected() -> None:
    d_in = _WINDOW_SIZE * len(_FIELDS)
    weights = init_weights(d_in, hidden_dim=6, latent_dim=3, seed=3, data_mean=np.zeros(d_in))

    x = np.zeros(d_in)
    channel_idx = _FIELDS.index("memory_pressure")
    for t in range(_WINDOW_SIZE):
        x[t * len(_FIELDS) + channel_idx] = -10.0  # real value << expected (~0)

    signed = mean_signed_deviation(x, weights, window_size=_WINDOW_SIZE, n_fields=len(_FIELDS))
    assert signed < 0
    assert deviation_direction(signed) == "depressed"


def test_deviation_direction_is_mixed_near_zero() -> None:
    assert deviation_direction(0.0) == "mixed"
    assert deviation_direction(1e-9) == "mixed"
    assert deviation_direction(-1e-9) == "mixed"
