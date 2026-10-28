"""Action bets: "this body action, in this situation, at this place, works with probability p".

Spec: docs/superpowers/specs/2026-10-23-embodied-experience-learning-design.md
("Prediction service: the action bet").

The service only sees opaque strings (skill, context values, place). What a skill means and when it
succeeded is the body adapter's business.

Served model ``action_beta_backoff``: Beta-Bernoulli with empirical-Bayes backoff over settled
success/failure bets (void never counts), strength K = 4. At the skill and situation levels each place
counts as ONE unit (each of its n bets weighs 1/n; a bet without a place weighs 1): live 2026-10-23,
seven failed walks into one wall made every untried exit read 0.20 under raw pooled counts.

    skill      Beta(1 + s, 1 + f)                            pooled over every domain and body
    situation  Beta(K p_skill + s, K (1 - p_skill) + f)      same domain, same skill, exact context
    place      Beta(K p_situation + s, K (1 - p_situation) + f)  same domain, same place

p is the mean of the deepest level with at least one settled bet; no data at all -> level 'prior',
p = 0.5. Baseline ``action_skill_rate``: the skill-level mean only, stored on every bet as
``p_baseline``. The counts at placing are stored too, so the served model's p can be recomputed for
every bet even while the gate serves the baseline (the shadow never goes dark).

Gate: per bet, squared error of the model's p vs the baseline's; one pair per session (the session's
mean), >= 20 sessions, the overlap-aware paired test from backtest.paired_win_p. ``worse`` -> quotes
and new bets serve the baseline.

World-model shadow ``transition_mlp`` (WorldModelShadow): when a bet carries body-level ``features``
whose ``feature_version`` matches the loaded weights (orion/embodiment/transition.py), its p of the
skill's success is stored on the bet as ``shadow``. It is never served. ``shadow_verdict`` grades it
against the served model on the same settled bets, with the same per-session paired test.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import numpy as np
from scipy import special, stats

from app.backtest import paired_win_p
from orion.embodiment.transition import TransitionWeights, success_p
from app.grading import grade
from app.live import LIVE_ALPHA, MIN_INDEPENDENT

logger = logging.getLogger("orion-prediction.action_bets")

MODEL = "action_beta_backoff"
BASELINE = "action_skill_rate"
SHADOW_MODEL = "transition_mlp"
K = 4.0
EXPIRY = timedelta(hours=1)
NEVER_SETTLED = "never_settled"
MIN_REPORTED = 10          # Brier is not reported below this many settled bets
VERDICT_ROWS = 5000        # most recent settled bets read for the live verdict
LIST_CAP = 500
LIVE_LAST = 20
LIVE_TOP_PLACES = 20


# --- pure model -------------------------------------------------------------------------------------
def bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def beta_eig(a: float, b: float) -> float:
    """Mutual information (nats) between the next outcome and the rate under Beta(a, b):
    H(mean) - E[H(theta)], with E[H(theta)] = psi(a+b+1) - m psi(a+1) - (1-m) psi(b+1)."""
    m = a / (a + b)
    expected_h = float(special.digamma(a + b + 1) - m * special.digamma(a + 1) - (1 - m) * special.digamma(b + 1))
    return max(0.0, bernoulli_entropy(m) - expected_h)


def beta_interval80(a: float, b: float) -> tuple[float, float]:
    return float(stats.beta.ppf(0.1, a, b)), float(stats.beta.ppf(0.9, a, b))


def _sf(c: dict | None) -> tuple[int, int]:
    if not c:
        return 0, 0
    return float(c.get("successes", 0)), float(c.get("failures", 0))


def backoff(skill: dict | None, situation: dict | None, place: dict | None, *, has_place: bool) -> dict:
    """``skill`` / ``situation`` / ``place``: {successes, failures} or None (no settled bet). Returns
    p, p_baseline, interval80, eig_nats, level and the per-level counts list (skill, situation,
    place-if-any) -- the same shape stored on a bet."""
    s0, f0 = _sf(skill)
    s1, f1 = _sf(situation)
    s2, f2 = _sf(place)
    a, b = 1.0 + s0, 1.0 + f0
    p_skill = a / (a + b)
    levels = [("skill", a, b, s0 + f0)]
    a1, b1 = K * p_skill + s1, K * (1 - p_skill) + f1
    p_sit = a1 / (a1 + b1)
    levels.append(("situation", a1, b1, s1 + f1))
    if has_place:
        a2, b2 = K * p_sit + s2, K * (1 - p_sit) + f2
        levels.append(("place", a2, b2, s2 + f2))
    deepest = None
    for lv in levels:
        if lv[3] > 0:
            deepest = lv
    if deepest is None:
        level, pa, pb = "prior", 1.0, 1.0
    else:
        level, pa, pb = deepest[0], deepest[1], deepest[2]
    counts = [{"level": "skill", "successes": s0, "failures": f0},
              {"level": "situation", "successes": s1, "failures": f1}]
    if has_place:
        counts.append({"level": "place", "successes": s2, "failures": f2})
    return {"p": pa / (pa + pb), "p_baseline": p_skill, "interval80": beta_interval80(pa, pb),
            "eig_nats": beta_eig(pa, pb), "level": level, "counts": counts}


def skill_rate(skill: dict | None) -> dict:
    """The baseline served as a full quote: the skill level alone."""
    s0, f0 = _sf(skill)
    a, b = 1.0 + s0, 1.0 + f0
    return {"p": a / (a + b), "interval80": beta_interval80(a, b), "eig_nats": beta_eig(a, b),
            "level": "skill" if s0 + f0 else "prior"}


def backoff_p(skill: dict | None, situation: dict | None, place: dict | None, *, has_place: bool) -> float:
    """``backoff``'s p only (no quantiles, no digamma): the verdict and grading call it per bet."""
    s0, f0 = _sf(skill); s1, f1 = _sf(situation); s2, f2 = _sf(place)
    p = (1.0 + s0) / (2.0 + s0 + f0)
    p_sit = (K * p + s1) / (K + s1 + f1)
    p_place = (K * p_sit + s2) / (K + s2 + f2) if has_place else None
    if has_place and s2 + f2 > 0:
        return p_place
    if s1 + f1 > 0:
        return p_sit
    return p if s0 + f0 > 0 else 0.5


def model_p_from_counts(counts: list[dict]) -> float:
    """The served model's p, recomputed from a bet's stored counts (identical to what it was at placing)."""
    by = {c["level"]: c for c in counts or []}
    return backoff_p(by.get("skill"), by.get("situation"), by.get("place"), has_place="place" in by)


def live_verdict(rows: Iterable[dict]) -> dict:
    """``rows``: settled bets (result success|failure) with p_baseline, counts, session_id, placed_at.
    One pair per session (mean squared error of the model vs the baseline); a bet without a session
    is its own unit. Same words and rule as app/live.py: better (p < LIVE_ALPHA; one test, so Holm is
    the identity), worse (reverse p < LIVE_ALPHA), unclear, pending (< MIN_INDEPENDENT units)."""
    return _paired_verdict(rows, lambda r: model_p_from_counts(r["counts"]), lambda r: float(r["p_baseline"]),
                           MODEL, BASELINE)


def shadow_verdict(rows: Iterable[dict]) -> dict:
    """The world-model shadow vs the served model, on settled bets that carry a shadow p."""
    rows = [r for r in rows if isinstance(r.get("shadow"), dict) and r["shadow"].get("p") is not None]
    out = _paired_verdict(rows, lambda r: float(r["shadow"]["p"]), lambda r: model_p_from_counts(r["counts"]),
                          SHADOW_MODEL, MODEL)
    out["tags"] = sorted({str(r["shadow"].get("tag")) for r in rows})[-3:]
    return out


def _paired_verdict(rows: Iterable[dict], p_a: Callable[[dict], float], p_b: Callable[[dict], float],
                    name_a: str, name_b: str) -> dict:
    units: dict[str, list] = defaultdict(list)
    first: dict[str, datetime] = {}
    for r in rows:
        if r.get("result") not in ("success", "failure"):
            continue
        y = 1.0 if r["result"] == "success" else 0.0
        e_m = (p_a(r) - y) ** 2
        e_b = (p_b(r) - y) ** 2
        key = f"s:{r['session_id']}" if r.get("session_id") else f"b:{r['bet_id']}"
        units[key].append((e_m, e_b))
        t = r.get("placed_at")
        if t is not None and (key not in first or t < first[key]):
            first[key] = t
    keys = sorted(units, key=lambda k: (first.get(k) or datetime.min.replace(tzinfo=timezone.utc), k))
    m = np.array([np.mean([e[0] for e in units[k]]) for k in keys], dtype=float)
    b = np.array([np.mean([e[1] for e in units[k]]) for k in keys], dtype=float)
    n_bets = sum(len(v) for v in units.values())
    out = {"model": name_a, "baseline": name_b, "n_scored": n_bets, "n_independent": len(keys),
           "rel_mse": float(m.mean() / b.mean()) if len(keys) and b.mean() > 0 else None,
           "p_better": None, "p_worse": None, "verdict": "pending"}
    if len(keys) >= MIN_INDEPENDENT:
        out["p_better"], _ = paired_win_p(m, b, lag=1)
        out["p_worse"], _ = paired_win_p(b, m, lag=1)
        out["verdict"] = ("better" if out["p_better"] < LIVE_ALPHA else
                          "worse" if out["p_worse"] < LIVE_ALPHA else "unclear")
    return out


def grade_bets(rows: Iterable[dict]) -> dict:
    """Served p and baseline p of settled success/failure bets, each through grading.grade."""
    rows = [r for r in rows if r.get("result") in ("success", "failure")]
    hits = [r["result"] == "success" for r in rows]
    return {"served": grade([float(r["p"]) for r in rows], hits),
            "baseline": grade([float(r["p_baseline"]) for r in rows], hits),
            "model": grade([model_p_from_counts(r["counts"]) for r in rows], hits),
            "reportable": len(rows) >= MIN_REPORTED, "min_reported": MIN_REPORTED}


def _ctx_key(context: dict | None) -> tuple:
    return tuple(sorted((context or {}).items()))


def index_counts(rows: Iterable[dict]) -> dict:
    """Store count rows -> lookup dicts keyed (skill) / (domain, skill, context) / (domain, place)."""
    idx: dict[str, dict] = {"skill": {}, "situation": {}, "place": {}}
    for r in rows:
        c = {"successes": float(r["successes"]), "failures": float(r["failures"])}
        if r["level"] == "skill":
            idx["skill"][r["skill"]] = c
        elif r["level"] == "situation":
            idx["situation"][(r["domain"], r["skill"], _ctx_key(r["context"]))] = c
        elif r["level"] == "place":
            idx["place"][(r["domain"], r["place"])] = c
    return idx


class WorldModelShadow:
    """The exported TransitionModel weights at ``path``, reloaded when the file changes. A missing
    or unreadable file means no shadow (logged once per change), never a fake p."""

    def __init__(self, path: str | None):
        self.path = path
        self._mtime: float | None = None
        self.weights: TransitionWeights | None = None
        self.error: str | None = None

    def _refresh(self) -> None:
        if not self.path:
            return
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            if self.weights is not None or self.error != "missing":
                logger.info("world_model_shadow_unavailable path=%s reason=missing", self.path)
            self.weights, self.error, self._mtime = None, "missing", None
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            self.weights, self.error = TransitionWeights.load(self.path), None
            logger.info("world_model_shadow_loaded path=%s tag=%s", self.path, self.weights.tag)
        except Exception as exc:  # noqa: BLE001 -- a bad file must not stop bets
            self.weights, self.error = None, f"unreadable: {exc}"
            logger.warning("world_model_shadow_unavailable path=%s reason=%s", self.path, self.error)

    def score(self, bet: dict) -> dict | None:
        self._refresh()
        w, x = self.weights, bet.get("features")
        if w is None or not x or bet.get("feature_version") != w.version:
            return None
        heads = w.heads(x)
        p = success_p(bet["skill"], heads)
        if p is None:
            return None
        return {"model": SHADOW_MODEL, "tag": w.tag, "p": min(max(p, 0.0), 1.0),
                "heads": {k: round(v, 4) for k, v in heads.items()}}

    def status(self) -> dict:
        self._refresh()
        return {"model": SHADOW_MODEL, "path": self.path, "loaded": self.weights is not None,
                "tag": self.weights.tag if self.weights else None, "error": self.error}


# --- service ------------------------------------------------------------------------------------------
class ConflictError(Exception):
    def __init__(self, bet: dict):
        super().__init__(f"bet {bet['bet_id']} already settled {bet['result']}")
        self.bet = bet


class ActionBets:
    """Quote / place / settle / expire / live over a store (app/store.py PostgresStore or a fake)."""

    def __init__(self, store, clock: Callable[[], datetime] | None = None, *, expire_every_sec: float = 30.0,
                 verdict_ttl_sec: float = 60.0, monotonic: Callable[[], float] = time.monotonic,
                 shadow: WorldModelShadow | None = None):
        self.store = store
        self.shadow = shadow
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.expire_every_sec = expire_every_sec
        self.verdict_ttl_sec = verdict_ttl_sec
        self._mono = monotonic
        self._last_expire: float | None = None
        self._verdict: tuple[float, dict] | None = None

    # gate --------------------------------------------------------------------------------------------
    def verdict(self, force: bool = False) -> dict:
        now = self._mono()
        if not force and self._verdict is not None and now - self._verdict[0] < self.verdict_ttl_sec:
            return self._verdict[1]
        v = live_verdict(self.store.settled_action_bets(limit=VERDICT_ROWS))
        self._verdict = (now, v)
        return v

    def served_model(self) -> str:
        return BASELINE if self.verdict()["verdict"] == "worse" else MODEL

    # expiry ------------------------------------------------------------------------------------------
    def expire(self) -> list[str]:
        now = self.clock()
        ids = self.store.expire_action_bets(now - EXPIRY, now, NEVER_SETTLED)
        self._last_expire = self._mono()
        for bet_id in ids:
            logger.info("action_bet_expired bet_id=%s outcome=%s", bet_id, NEVER_SETTLED)
        return ids

    def maybe_expire(self) -> None:
        if self._last_expire is None or self._mono() - self._last_expire >= self.expire_every_sec:
            self.expire()

    # quote -------------------------------------------------------------------------------------------
    def _quote_options(self, domain: str, options: list[dict]) -> tuple[str, list[dict]]:
        skills = sorted({o["skill"] for o in options})
        places = sorted({o["place"] for o in options if o.get("place")})
        idx = index_counts(self.store.action_bet_counts(domain, skills, places))
        served = self.served_model()
        out = []
        for o in options:
            sk = idx["skill"].get(o["skill"])
            q = backoff(sk, idx["situation"].get((domain, o["skill"], _ctx_key(o.get("context")))),
                        idx["place"].get((domain, o["place"])) if o.get("place") else None,
                        has_place=bool(o.get("place")))
            if served == BASELINE:
                q.update(skill_rate(sk))
            q["model"] = served
            q["option_id"] = o.get("option_id")
            out.append(q)
        return served, out

    def quote(self, domain: str, options: list[dict]) -> dict:
        self.maybe_expire()
        served, quotes = self._quote_options(domain, options)
        return {"domain": domain, "served_model": served, "quotes": quotes}

    # place -------------------------------------------------------------------------------------------
    def place(self, bet: dict) -> tuple[dict, bool]:
        """Returns (stored bet, created). p and p_baseline are computed here, never taken from the caller."""
        self.maybe_expire()
        existing = self.store.get_action_bet(bet["bet_id"])
        if existing is not None:
            return existing, False
        _, (q,) = self._quote_options(bet["domain"], [bet])
        row = {"bet_id": bet["bet_id"], "domain": bet["domain"], "body": bet["body"], "skill": bet["skill"],
               "context": dict(sorted((bet.get("context") or {}).items())), "place": bet.get("place"),
               "action": bet.get("action") or {}, "model": q["model"], "p": q["p"], "p_baseline": q["p_baseline"],
               "level": q["level"], "counts": q["counts"], "chosen_rank": bet.get("chosen_rank"),
               "session_id": bet.get("session_id"), "placed_at": self.clock(),
               "shadow": self.shadow.score(bet) if self.shadow else None}
        stored = self.store.insert_action_bet(row)
        created = stored is not None
        if stored is None:                      # lost a race with the same bet_id
            stored = self.store.get_action_bet(bet["bet_id"])
        if created:
            logger.info("action_bet_placed bet_id=%s domain=%s skill=%s place=%s model=%s p=%.3f p_baseline=%.3f "
                        "level=%s chosen_rank=%s", row["bet_id"], row["domain"], row["skill"], row["place"],
                        row["model"], row["p"], row["p_baseline"], row["level"], row["chosen_rank"])
        return stored, created

    # settle ------------------------------------------------------------------------------------------
    def settle(self, bet_id: str, result: str, outcome: str, detail: dict | None) -> dict | None:
        """Open -> settled only. None: unknown bet. Same result again: the stored bet. Conflicting
        result (including a bet already expired void): ConflictError."""
        row = self.store.settle_action_bet(bet_id, result, outcome, detail or {}, self.clock())
        if row is not None:
            logger.info("action_bet_settled bet_id=%s result=%s outcome=%s p=%.3f", bet_id, result, outcome,
                        float(row["p"]))
            return row
        row = self.store.get_action_bet(bet_id)
        if row is None:
            return None
        if row["result"] == result:
            return row
        raise ConflictError(row)

    # reads -------------------------------------------------------------------------------------------
    def session_bets(self, session_id: str) -> list[dict]:
        return self.store.action_bets(session_id=session_id, limit=LIST_CAP)

    def health(self) -> dict:
        st = self.store.action_bet_status(None, None, self.clock() - timedelta(hours=24))
        return {"open": st["open"], "never_settled_24h": st["never_settled"], "served_model": self.served_model(),
                "world_model_shadow": self.shadow.status() if self.shadow else None}

    def live(self, domain: str | None, body: str | None) -> dict:
        self.maybe_expire()
        rows = self.store.action_bet_counts(domain, None, None, body=body, combos=True)
        idx = index_counts(rows)
        situations, places = [], []
        skills_here = set()
        for r in rows:
            if r["level"] == "situation":
                skills_here.add(r["skill"])
        skill_out = []
        for sk in sorted(skills_here if domain is not None else idx["skill"]):
            c = idx["skill"].get(sk, {"successes": 0, "failures": 0})
            a, b = 1.0 + c["successes"], 1.0 + c["failures"]
            skill_out.append({"skill": sk, **c, "mean": a / (a + b), "interval80": beta_interval80(a, b),
                              "eig_nats": beta_eig(a, b)})
        for (dom, sk, ctx), c in sorted(idx["situation"].items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            q = backoff(idx["skill"].get(sk), c, None, has_place=False)
            situations.append({"domain": dom, "skill": sk, "context": dict(ctx), **c, "mean": q["p"],
                               "interval80": q["interval80"], "eig_nats": q["eig_nats"]})
        for r in rows:
            if r["level"] != "combo":
                continue
            dom, sk, ctx, pl = r["domain"], r["skill"], _ctx_key(r["context"]), r["place"]
            pc = idx["place"].get((dom, pl))
            if not pc:
                continue
            q = backoff(idx["skill"].get(sk), idx["situation"].get((dom, sk, ctx)), pc, has_place=True)
            places.append({"domain": dom, "place": pl, "skill": sk, "context": dict(ctx), **pc, "mean": q["p"],
                           "interval80": q["interval80"], "level": q["level"]})
        places.sort(key=lambda x: (-(x["successes"] + x["failures"]), x["place"], x["skill"]))
        settled = self.store.settled_action_bets(domain=domain, body=body, limit=VERDICT_ROWS)
        st = self.store.action_bet_status(domain, body, self.clock() - timedelta(hours=24))
        return {"domain": domain, "body": body, "served_model": self.served_model(), "k": K,
                "skills": skill_out, "situations": situations, "places": places[:LIVE_TOP_PLACES],
                "grading": grade_bets(settled), "live_verdict": self.verdict(),
                "live_verdict_scope": "all domains (the gate is global)",
                "domain_verdict": live_verdict(settled),
                "shadow_verdict": shadow_verdict(settled),
                "world_model_shadow": self.shadow.status() if self.shadow else None,
                "open": st["open"], "never_settled_24h": st["never_settled"], "void_24h": st["void"],
                "settled_24h": st["settled"],
                "last_bets": self.store.action_bets(domain=domain, body=body, limit=LIVE_LAST)}
