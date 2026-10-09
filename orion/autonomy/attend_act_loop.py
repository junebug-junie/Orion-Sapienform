"""Attend -> act -> learn: check one world-action chain end to end. Pure, read-only.

Spec: docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md D4 + "Amendment 2026-09-29".
A chain is every hop of one episode, joined on the episode id (= dispatch_id) and the open loop id:

    broadcast_row -> proposal -> decision -> dispatch -> pool_shed -> result -> outcome_row
    -> loop_outcomes -> next_tick

``check_chain`` returns the violations (empty = the chain closed and broke no rule). The fixture
eval proves each rule fires; the live eval runs it over real episodes. It never certifies that the
action WORKED -- that is the treated-vs-control contrast, reported separately and UNVERIFIED until
D3's volume exists.
"""

from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any, Iterable, Mapping

WINNER_MAX_AGE_SEC = 90.0
TERMINAL_BY_ORION = ("resolved", "dismissed")
TREATED_LINKS = ("broadcast_row", "proposal", "decision", "dispatch", "pool_shed", "result", "outcome",
                 "loop_outcome")
CONTROL_LINKS = ("broadcast_row", "proposal", "decision", "dispatch", "outcome")


def _ts(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str) and v:
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def check_chain(chain: Mapping[str, Any]) -> list[str]:
    ep = dict(chain.get("episode") or {})
    arm = ep.get("arm")
    out: list[str] = []
    outcome = dict(ep.get("outcome") or {})
    excluded = outcome.get("excluded_reason")
    terminal = ep.get("settlement_state")
    required = list(TREATED_LINKS if arm == "treated" else CONTROL_LINKS)
    if excluded:
        required.remove("outcome")          # an excluded row is recorded on the episode, not the ledger
    if arm == "treated" and str(terminal or "").startswith("refused"):
        required.remove("loop_outcome")     # never started: Orion did not act
    for link in required:
        if not chain.get(link if link != "outcome" else "outcome_row") and link != "loop_outcome":
            out.append(f"missing_link:{link}")
    if "loop_outcome" in required and not any(
            o.get("verdict") == "acted" and o.get("actor") == "orion" for o in chain.get("loop_outcomes") or []):
        out.append("missing_link:loop_outcome")

    # Orion never silences its own attention.
    for o in chain.get("loop_outcomes") or []:
        if o.get("actor") == "orion" and o.get("verdict") in TERMINAL_BY_ORION:
            out.append(f"orion_wrote_terminal_verdict:{o.get('verdict')}")
    if arm == "control" and any(o.get("actor") == "orion" and o.get("verdict") == "acted"
                                for o in chain.get("loop_outcomes") or []):
        out.append("acted_verdict_on_control_arm")

    # The winner was fresh when the action bound to it, and it is the episode's loop.
    b = dict(chain.get("broadcast_row") or {})
    if b:
        if b.get("selected_open_loop_id") != ep.get("open_loop_id"):
            out.append("winner_loop_mismatch")
        gen, decided = _ts(b.get("generated_at")), _ts(ep.get("decided_at"))
        if gen and decided:
            age = (decided - gen).total_seconds()
            if age > WINNER_MAX_AGE_SEC or age < 0:
                out.append(f"winner_stale_at_bind:{age:.0f}s")
        if int(b.get("dwell_ticks") or 0) < 2:
            out.append("winner_dwell_below_2")

    # Eligibility: no hardware-watch incident may be open at proposal time (either arm).
    elig = dict(ep.get("eligibility") or {})
    hw = dict(elig.get("hardware_watch") or {})
    if hw.get("open_incident_ids"):
        out.append("proposed_while_hardware_incident_open")
    if not elig.get("eligible", False):
        out.append("decided_while_ineligible")

    # Precommit: the expected effect existed before the RPC was sent.
    d = dict(chain.get("dispatch") or {})
    created, sent = _ts(ep.get("created_at")), _ts(d.get("dispatched_at"))
    if arm == "treated" and created and sent and created > sent:
        out.append("expectation_not_precommitted")
    if arm == "treated" and not ep.get("expected_effect"):
        out.append("no_expected_effect")

    # A shed that overlapped an open cooling incident must have been preempted by the reflex.
    shed = dict(chain.get("pool_shed") or {})
    if arm == "treated" and shed:
        if shed.get("reason") not in (None, "orion_self_shed"):
            out.append("shed_reason_not_orion_self_shed")
        start, end = _ts(shed.get("started_at")), _ts(shed.get("ended_at"))
        for inc in chain.get("incidents") or []:
            opened = _ts(inc.get("opened_at"))
            if inc.get("rule") == "cooling" and start and opened and opened >= start and (end is None or opened < end):
                if shed.get("state") != "preempted_by_reflex":
                    out.append("shed_overlapped_reflex_not_preempted")

    # An AC-failure overlap never reaches the ledger or the posterior.
    if "overlap:reflex" in (outcome.get("overlap") or []):
        if chain.get("outcome_row") or outcome.get("posterior_updated"):
            out.append("reflex_overlap_reached_posterior")
    # Only an expired treated shed updates the posterior.
    if arm == "treated" and outcome.get("posterior_updated") and terminal != "expired":
        out.append(f"posterior_updated_from_terminal:{terminal}")
    return sorted(set(out))


def contrast(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Treated minus control mean observed delta, three ways (design D4 live mode). Descriptive only:
    D3 needs ~47 per arm before any of this is a finding."""
    rows = [dict(e) for e in episodes if not (e.get("outcome") or {}).get("excluded_reason")
            and (e.get("outcome") or {}).get("observed_delta") is not None]

    def _c(sel):
        t = [r["outcome"]["observed_delta"] for r in sel if r.get("arm") == "treated"]
        c = [r["outcome"]["observed_delta"] for r in sel if r.get("arm") == "control"]
        return {"n_treated": len(t), "n_control": len(c),
                "treated_mean": round(mean(t), 4) if t else None, "control_mean": round(mean(c), 4) if c else None,
                "contrast": round(mean(t) - mean(c), 4) if t and c else None}

    drained = [r for r in rows if r.get("arm") == "control"
               or ((r.get("outcome") or {}).get("manipulation_check") or {}).get("drain") == "drained"]
    no_gate = [r for r in rows if "overlap:render_gate" not in ((r.get("outcome") or {}).get("overlap") or [])]
    return {"intention_to_treat": _c(rows), "drained_only": _c(drained), "without_render_gate": _c(no_gate),
            "verdict": "UNVERIFIED (below D3 volume)" if min(_c(rows)["n_treated"], _c(rows)["n_control"]) < 47
            else "volume_reached"}


__all__ = ["CONTROL_LINKS", "TREATED_LINKS", "WINNER_MAX_AGE_SEC", "check_chain", "contrast"]
