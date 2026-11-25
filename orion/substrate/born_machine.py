"""Born machine on a matrix-product state (MPS): a small trained tensor network. Pure numpy, no I/O.

Spec: docs/superpowers/specs/2026-11-24-heartbeat-seats-field-demon-design.md (rev 3), "The model".

The model gives every configuration x of L discrete sites (d values each) the probability

    p(x) = psi(x)^2 / Z,   psi(x) = A_0[x_0] A_1[x_1] ... A_{L-1}[x_{L-1}]

where each A_i[s] is a small real matrix. The largest inner size is the bond dimension D.

What this module does:

* ``fit`` trains the chain by maximum likelihood with two-site sweeps (Han et al. 2018, "Unsupervised
  generative modeling using matrix product states"). Each step merges two neighbour tensors, takes
  a gradient step on the mean negative log-likelihood, and splits them again with an SVD that keeps
  at most D singular values. The chain stays in mixed canonical form, so Z is the squared norm of
  the merged tensor and costs nothing.
* ``select_and_fit`` picks D from {2, 4, 8, 16} and the number of sweeps by held-out
  log-likelihood on the last part of the training rows, then refits on all of them.
* ``log_prob`` is the exact log-likelihood. ``marginal_table`` is the exact marginal over any set of
  sites, so any conditional p(B | A) is exact too.
* ``entanglement_entropies`` is the von Neumann entropy at each bond. ``mutual_information`` is
  the classical mutual information between the two sides of each bond, which entanglement bounds
  from above (a pure state's Holevo bound).

Entropies and mutual information are in bits. Log-likelihoods are in nats unless a name says bits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

BOND_DIMS: tuple[int, ...] = (2, 4, 8, 16)
DEFAULT_SWEEPS = 10
DEFAULT_LR = 0.1
DEFAULT_STEPS_PER_BOND = 2
VAL_FRACTION = 0.2
# Held-out rows can land on a configuration the chain gives (almost) zero probability. Every
# log-likelihood mixes in this much of the uniform distribution, so one such row costs a finite
# amount instead of minus infinity.
LL_FLOOR = 1e-6
MAX_EXACT_CONFIGS = 3 ** 12
_PSI_FLOOR = 1e-150
_SV_CUTOFF = 1e-12


@dataclass(frozen=True)
class BornMachine:
    tensors: tuple[np.ndarray, ...]  # A_i, shape (D_left, d, D_right); D_left of site 0 = 1

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @property
    def d(self) -> int:
        return int(self.tensors[0].shape[1])

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return tuple(int(a.shape[2]) for a in self.tensors[:-1])


# --- contraction primitives -------------------------------------------------------------------


def norm(bm: BornMachine) -> float:
    """Z = sum over every configuration of psi(x)^2."""
    env = np.ones((1, 1))
    for a in bm.tensors:
        env = np.einsum("ab,asx,bsy->xy", env, a, a, optimize=True)
    return float(env[0, 0])


def amplitudes(tensors: Sequence[np.ndarray], data: np.ndarray) -> np.ndarray:
    """psi(x) for each row of an (N, L) integer array."""
    data = np.asarray(data, dtype=np.int64)
    v = np.ones((data.shape[0], 1))
    for i, a in enumerate(tensors):
        v = np.einsum("na,anb->nb", v, a[:, data[:, i], :])
    return v[:, 0]


def log_prob(bm: BornMachine, data: np.ndarray) -> np.ndarray:
    """Exact log p(x) in nats, per row (minus infinity where psi(x) = 0)."""
    psi = amplitudes(bm.tensors, data)
    with np.errstate(divide="ignore"):
        return 2.0 * np.log(np.abs(psi)) - np.log(norm(bm))


def floored_log_prob(bm: BornMachine, data: np.ndarray, floor: float = LL_FLOOR) -> np.ndarray:
    """log((1 - floor) p(x) + floor / d^L), always finite."""
    lp = log_prob(bm, data)
    log_uniform = -bm.n_sites * np.log(bm.d)
    return np.logaddexp(np.log1p(-floor) + lp, np.log(floor) + log_uniform)


def _left_envs(tensors: Sequence[np.ndarray]) -> list[np.ndarray]:
    """envs[j] = sum over sites 0..j-1 of the transfer product (D_j x D_j); envs[0] = [[1]]."""
    out = [np.ones((1, 1))]
    for a in tensors:
        out.append(np.einsum("ab,asx,bsy->xy", out[-1], a, a, optimize=True))
    return out


def _right_envs(tensors: Sequence[np.ndarray]) -> list[np.ndarray]:
    """envs[j] = sum over sites j..L-1 (D_left(j) x D_left(j)); envs[L] = [[1]]."""
    L = len(tensors)
    out: list[np.ndarray] = [np.ones((1, 1))] * (L + 1)
    for j in range(L - 1, -1, -1):
        a = tensors[j]
        out[j] = np.einsum("xy,asx,bsy->ab", out[j + 1], a, a, optimize=True)
    return out


def marginal_table(bm: BornMachine, sites: Sequence[int]) -> np.ndarray:
    """Exact p(x_S) for a set of sites S, shape (d,)*|S| with axes in increasing site order."""
    keep = sorted(set(int(s) for s in sites))
    if any(s < 0 or s >= bm.n_sites for s in keep):
        raise ValueError(f"marginal_table: sites {keep} outside 0..{bm.n_sites - 1}")
    keep_set = set(keep)
    state = np.ones((1, 1, 1))  # (configs so far, bra bond, ket bond)
    for j, a in enumerate(bm.tensors):
        if j in keep_set:
            state = np.einsum("cab,asx,bsy->csxy", state, a, a, optimize=True)
            state = state.reshape(-1, a.shape[2], a.shape[2])
        else:
            state = np.einsum("cab,asx,bsy->cxy", state, a, a, optimize=True)
    p = np.clip(state[:, 0, 0], 0.0, None)
    total = p.sum()
    if total <= 0:
        raise ValueError("marginal_table: the chain has zero norm")
    return (p / total).reshape((bm.d,) * len(keep))


def conditional_table(bm: BornMachine, given: Sequence[int], target: Sequence[int]) -> np.ndarray:
    """p(x_target | x_given), shape (d,)*|given| + (d,)*|target|, axes in the order passed.

    A given configuration the chain never produces (marginal 0) gets the uniform conditional.
    """
    given = [int(s) for s in given]
    target = [int(s) for s in target]
    if set(given) & set(target):
        raise ValueError("conditional_table: given and target sites overlap")
    order = sorted(given + target)
    table = marginal_table(bm, order)
    table = np.transpose(table, [order.index(s) for s in given + target])
    tail = tuple(range(len(given), len(given) + len(target)))
    denom = table.sum(axis=tail, keepdims=True)
    uniform = 1.0 / bm.d ** len(target)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = np.where(denom > 0, table / np.where(denom > 0, denom, 1.0), uniform)
    return cond


def full_probabilities(bm: BornMachine, *, max_configs: int = MAX_EXACT_CONFIGS) -> np.ndarray:
    """p(x) for every configuration, shape (d,)*L. Refuses more than ``max_configs`` entries."""
    if bm.d ** bm.n_sites > max_configs:
        raise ValueError(f"full_probabilities: {bm.d}^{bm.n_sites} configurations is too many")
    psi = np.ones((1, 1))
    for a in bm.tensors:
        psi = np.einsum("ca,asb->csb", psi, a).reshape(-1, a.shape[2])
    p = psi[:, 0] ** 2
    return (p / p.sum()).reshape((bm.d,) * bm.n_sites)


# --- canonical forms and entanglement ---------------------------------------------------------


def _right_canonical(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """Sites 1..L-1 right-canonical; the norm sits in site 0, scaled to 1."""
    ts = [t.copy() for t in tensors]
    for i in range(len(ts) - 1, 0, -1):
        dl, d, dr = ts[i].shape
        q, r = np.linalg.qr(ts[i].reshape(dl, d * dr).T)
        ts[i] = q.T.reshape(-1, d, dr)
        ts[i - 1] = np.einsum("asb,bc->asc", ts[i - 1], r.T)
    ts[0] = ts[0] / np.linalg.norm(ts[0])
    return ts


def entanglement_entropies(bm: BornMachine) -> np.ndarray:
    """Von Neumann entropy (bits) of the normalised state at each bond 0..L-2.

    Bond j separates sites 0..j from sites j+1..L-1.
    """
    ts = _right_canonical(list(bm.tensors))
    out = np.zeros(bm.n_sites - 1)
    for i in range(bm.n_sites - 1):
        dl, d, dr = ts[i].shape
        u, s, vt = np.linalg.svd(ts[i].reshape(dl * d, dr), full_matrices=False)
        ts[i] = u.reshape(dl, d, -1)
        ts[i + 1] = np.einsum("ab,bsc->asc", s[:, None] * vt, ts[i + 1])
        lam = s ** 2
        lam = lam[lam > 0] / lam.sum()
        out[i] = float(-(lam * np.log2(lam)).sum())
    return out


def mutual_information(
    bm: BornMachine,
    *,
    max_exact: int = MAX_EXACT_CONFIGS,
    n_samples: int = 20000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical I(left; right) in bits at each bond, and its standard error.

    Exact (standard error 0) when d^L <= ``max_exact``. Otherwise a Monte Carlo average of
    log p(x) - log p(left) - log p(right) over exact samples, with every term computed exactly.
    """
    L = bm.n_sites
    if bm.d ** L <= max_exact:
        p = full_probabilities(bm, max_configs=max_exact).ravel()
        mi = np.zeros(L - 1)
        for j in range(L - 1):
            P = p.reshape(bm.d ** (j + 1), -1)
            pl = P.sum(axis=1, keepdims=True)
            pr = P.sum(axis=0, keepdims=True)
            nz = P > 0
            mi[j] = float((P[nz] * np.log2(P[nz] / (pl @ pr)[nz])).sum())
        return mi, np.zeros(L - 1)
    rng = rng if rng is not None else np.random.default_rng()
    x = sample(bm, n_samples, rng)
    z = norm(bm)
    log_p = log_prob(bm, x)
    lenv, renv = _left_envs(bm.tensors), _right_envs(bm.tensors)
    lv = [np.ones((n_samples, 1))]
    for i, a in enumerate(bm.tensors):
        lv.append(np.einsum("na,anb->nb", lv[-1], a[:, x[:, i], :]))
    rv: list[np.ndarray] = [np.ones((n_samples, 1))] * (L + 1)
    for i in range(L - 1, -1, -1):
        a = bm.tensors[i]
        rv[i] = np.einsum("anb,nb->na", a[:, x[:, i], :], rv[i + 1])
    mi, se = np.zeros(L - 1), np.zeros(L - 1)
    for j in range(L - 1):
        p_left = np.einsum("na,ab,nb->n", lv[j + 1], renv[j + 1], lv[j + 1]) / z
        p_right = np.einsum("na,ab,nb->n", rv[j + 1], lenv[j + 1], rv[j + 1]) / z
        terms = (log_p - np.log(p_left) - np.log(p_right)) / np.log(2.0)
        mi[j] = float(terms.mean())
        se[j] = float(terms.std(ddof=1) / np.sqrt(n_samples))
    return mi, se


def sample(bm: BornMachine, n: int, rng: np.random.Generator) -> np.ndarray:
    """n exact samples from p(x), shape (n, L)."""
    renv = _right_envs(bm.tensors)
    v = np.ones((n, 1))
    out = np.zeros((n, bm.n_sites), dtype=np.int64)
    for j, a in enumerate(bm.tensors):
        w = np.einsum("na,asb->nsb", v, a)
        probs = np.clip(np.einsum("nsb,bc,nsc->ns", w, renv[j + 1], w), 0.0, None)
        probs /= probs.sum(axis=1, keepdims=True)
        u = rng.random(n)[:, None]
        s = np.minimum((np.cumsum(probs, axis=1) < u).sum(axis=1), bm.d - 1)
        out[:, j] = s
        v = w[np.arange(n), s]
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return out


# --- training ---------------------------------------------------------------------------------


def _product_init(data: np.ndarray, d: int) -> list[np.ndarray]:
    """Bond-1 chain whose amplitudes are sqrt of each site's (smoothed) marginal: the independent
    model. Every training row starts with psi > 0, and sweeps grow bonds where the data couple."""
    out = []
    for i in range(data.shape[1]):
        counts = np.bincount(data[:, i], minlength=d).astype(float) + 0.5
        out.append(np.sqrt(counts / counts.sum()).reshape(1, d, 1))
    return out


def _sweep_trace(
    data: np.ndarray,
    *,
    d: int,
    bond_dim: int,
    n_sweeps: int,
    lr: float,
    steps_per_bond: int,
    val: np.ndarray | None = None,
) -> tuple[BornMachine, list[float]]:
    """Fit, and the held-out mean log-likelihood after every sweep (empty when ``val`` is None)."""
    data = np.asarray(data, dtype=np.int64)
    N, L = data.shape
    if L < 2:
        raise ValueError("fit: need at least 2 sites")
    if data.min() < 0 or data.max() >= d:
        raise ValueError(f"fit: values outside 0..{d - 1}")
    ts = _right_canonical(_product_init(data, d))
    left: list[np.ndarray | None] = [None] * (L + 1)
    right: list[np.ndarray | None] = [None] * (L + 1)
    left[0] = np.ones((N, 1))
    right[L] = np.ones((N, 1))
    for i in range(L - 1, 0, -1):
        right[i] = np.einsum("anb,nb->na", ts[i][:, data[:, i], :], right[i + 1])
    pair_index = [data[:, i] * d + data[:, i + 1] for i in range(L - 1)]
    trace: list[float] = []

    def update(i: int, to_right: bool) -> None:
        b = np.einsum("asb,btc->astc", ts[i], ts[i + 1])
        dl, dr = b.shape[0], b.shape[3]
        b /= np.linalg.norm(b)
        lv, rv, pidx = left[i], right[i + 2], pair_index[i]
        for _ in range(steps_per_bond):
            bf = b.reshape(dl, d * d, dr)
            psi = np.einsum("na,anc,nc->n", lv, bf[:, pidx, :], rv)
            w = 1.0 / np.where(np.abs(psi) < _PSI_FLOOR, np.copysign(_PSI_FLOOR, psi), psi)
            g = np.zeros((d * d, dl, dr))
            # sum over rows of outer(L_n, R_n) / psi_n, scattered to that row's (s, t) slot.
            np.add.at(g, pidx, np.einsum("na,nc->nac", lv * w[:, None], rv))
            grad = b - np.transpose(g, (1, 0, 2)).reshape(dl, d, d, dr) / N
            b = b - lr * grad
            b /= np.linalg.norm(b)
        u, s, vt = np.linalg.svd(b.reshape(dl * d, d * dr), full_matrices=False)
        chi = max(1, min(bond_dim, int((s > _SV_CUTOFF * s[0]).sum())))
        u, s, vt = u[:, :chi], s[:chi] / np.linalg.norm(s[:chi]), vt[:chi]
        if to_right:
            ts[i] = u.reshape(dl, d, chi)
            ts[i + 1] = (s[:, None] * vt).reshape(chi, d, dr)
            left[i + 1] = np.einsum("na,anb->nb", left[i], ts[i][:, data[:, i], :])
        else:
            ts[i] = (u * s[None, :]).reshape(dl, d, chi)
            ts[i + 1] = vt.reshape(chi, d, dr)
            right[i + 1] = np.einsum("anb,nb->na", ts[i + 1][:, data[:, i + 1], :], right[i + 2])

    for _ in range(n_sweeps):
        for i in range(L - 1):
            update(i, True)
        for i in range(L - 2, -1, -1):
            update(i, False)
        if val is not None:
            trace.append(float(floored_log_prob(BornMachine(tuple(ts)), val).mean()))
    return BornMachine(tuple(t.copy() for t in ts)), trace


def fit(
    data: np.ndarray,
    *,
    d: int = 3,
    bond_dim: int = 4,
    n_sweeps: int = DEFAULT_SWEEPS,
    lr: float = DEFAULT_LR,
    steps_per_bond: int = DEFAULT_STEPS_PER_BOND,
) -> BornMachine:
    """Maximum-likelihood Born machine on an (N, L) integer array with values 0..d-1.

    Deterministic: it starts from the independent (product) model, not a random chain.
    """
    bm, _ = _sweep_trace(
        data, d=d, bond_dim=bond_dim, n_sweeps=n_sweeps, lr=lr, steps_per_bond=steps_per_bond
    )
    return bm


@dataclass(frozen=True)
class Selection:
    bond_dim: int
    n_sweeps: int
    val_ll: dict[int, float]  # best held-out mean log-likelihood (nats) per candidate D


def select_and_fit(
    data: np.ndarray,
    *,
    d: int = 3,
    bond_dims: Sequence[int] = BOND_DIMS,
    max_sweeps: int = DEFAULT_SWEEPS,
    val_fraction: float = VAL_FRACTION,
    lr: float = DEFAULT_LR,
    steps_per_bond: int = DEFAULT_STEPS_PER_BOND,
) -> tuple[BornMachine, Selection]:
    """Choose D and the sweep count by held-out log-likelihood on the last ``val_fraction`` of rows
    (time order kept), then refit on every row with that choice."""
    data = np.asarray(data, dtype=np.int64)
    n_val = int(round(val_fraction * len(data)))
    if n_val < 1 or len(data) - n_val < 2:
        raise ValueError(f"select_and_fit: {len(data)} rows is too few to hold out {val_fraction:.0%}")
    fit_rows, val_rows = data[:-n_val], data[-n_val:]
    best = (-np.inf, bond_dims[0], 1)
    val_ll: dict[int, float] = {}
    for D in bond_dims:
        _, trace = _sweep_trace(
            fit_rows, d=d, bond_dim=D, n_sweeps=max_sweeps, lr=lr, steps_per_bond=steps_per_bond,
            val=val_rows,
        )
        k = int(np.argmax(trace))
        val_ll[int(D)] = trace[k]
        if trace[k] > best[0] + 1e-9:  # ties go to the smaller D (listed first)
            best = (trace[k], int(D), k + 1)
    bm = fit(data, d=d, bond_dim=best[1], n_sweeps=best[2], lr=lr, steps_per_bond=steps_per_bond)
    return bm, Selection(bond_dim=best[1], n_sweeps=best[2], val_ll=val_ll)
