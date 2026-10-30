"""Orion's body-economy report as plain first-person text (``skills.physiology.state.v1``).

One fetch (``GET /physiology/report`` on orion-durable-runs) and one renderer, shared by the two
callers so Orion reads the same words on every path:

* ``python -P -m orion.dig body-economy ...``: Bash inside a chat turn or a curiosity run
  (orion-harness-governor). This is how Orion reaches it live today.
* the cortex-exec skill adapter ``skills.physiology.state.v1`` (Skill Runner, scheduler, verb plans).

Stdlib only (urllib), so the dig kit needs no new dependency. Read-only: the renderer changes
nothing and decides nothing. It reports what the report says, including what is unknown: a
section with no data prints its ``missing`` reason, never a blank or a zero.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Mapping, Optional, Sequence

REPORT_URL_ENV = "ORION_PHYSIOLOGY_REPORT_URL"
DEFAULT_REPORT_URL = "http://orion-athena-durable-runs:8121"
SECTIONS = (
    "limits", "wants", "rhythm", "money", "unit_costs", "waste", "queue_wait", "decisions",
    "claude", "vram", "value_per_cost", "services", "processes", "hosts",
)
#: What "how is my body" reads when no section is named: the headline constraints, what is spinning
#: each machine (one line per host), money, wants, waste, and anything down. Small enough for a turn.
OVERVIEW = ("limits", "hosts", "money", "wants", "waste", "services")
#: Sections whose rows carry a ``host``: ``--host`` narrows these and leaves the rest whole-body.
HOST_SECTIONS = ("hosts", "processes", "vram")
WINDOWS = ("24h", "7d", "30d")

BIND_WORDS = {
    "slack": "slack",
    "binding": "BINDING (my own use is pushing it)",
    "exogenous_bind": "BINDING, not from my load",
    "stuck_at_ceiling": "BINDING, price stuck at its ceiling",
    "no_reading": "unknown (no reading)",
}
#: The energy envelope binds "exogenously" when the always-on floor alone reaches it. That floor is
#: my own body at rest, so "not from my load" would be false for energy (Juniper, 2026-10-30).
ENERGY_EXOGENOUS_WORDS = "BINDING: my always-on body alone reaches the envelope"


def bind_words(constraint: Any, bind_state: Any) -> str:
    if constraint == "energy" and bind_state == "exogenous_bind":
        return ENERGY_EXOGENOUS_WORDS
    return BIND_WORDS.get(str(bind_state), str(bind_state or "no price"))


class ReportUnavailable(RuntimeError):
    """The report could not be read. The message is the plain reason to show Orion."""


def normalize_sections(raw: Any) -> tuple[str, ...]:
    """``None`` / "" / "overview" -> OVERVIEW. Accepts a list or a comma/space separated string.
    Raises ValueError naming the unknown section and the known ones."""
    if raw is None:
        return OVERVIEW
    items = raw if isinstance(raw, (list, tuple)) else str(raw).replace(",", " ").split()
    names = [str(s).strip().lower().replace("-", "_") for s in items if str(s).strip()]
    if not names or names == ["overview"]:
        return OVERVIEW
    if "all" in names:
        return SECTIONS
    unknown = [n for n in names if n not in SECTIONS and n != "overview"]
    if unknown:
        raise ValueError(f"unknown section(s): {', '.join(unknown)}. Known: overview, all, {', '.join(SECTIONS)}")
    out: list[str] = []
    for n in names:
        for s in (OVERVIEW if n == "overview" else (n,)):
            if s not in out:
                out.append(s)
    return tuple(out)


def report_url(base: Optional[str] = None) -> str:
    return (base or os.environ.get(REPORT_URL_ENV) or DEFAULT_REPORT_URL).rstrip("/")


def fetch_report(*, base_url: Optional[str] = None, window: str = "24h", sections: Sequence[str] = OVERVIEW,
                 family: Optional[str] = None, outcome: Optional[str] = None, limit: Optional[int] = None,
                 timeout_sec: float = 10.0) -> dict:
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {', '.join(WINDOWS)} (got {window!r})")
    params: dict[str, Any] = {"window": window, "sections": ",".join(sections)}
    if family:
        params["family"] = family
    if outcome:
        params["outcome"] = outcome
    if limit:
        params["limit"] = int(limit)
    url = f"{report_url(base_url)}/physiology/report?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:  # noqa: S310 -- fixed internal URL
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail") or ""
        except Exception:  # noqa: BLE001
            pass
        raise ReportUnavailable(f"the body report answered HTTP {exc.code} {detail}".strip()) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ReportUnavailable(f"the body report at {report_url(base_url)} is unreachable ({exc})") from exc
    except ValueError as exc:
        raise ReportUnavailable(f"the body report answered something that is not JSON ({exc})") from exc


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------


def _n(v: Any, nd: int = 1) -> str:
    if v is None:
        return "?"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{x:.0f}"


def _usd(v: Any, nd: int = 2) -> str:
    if v is None:
        return "?"
    x = float(v)
    if 0 < abs(x) < 0.01:
        return f"${x:.6f}".rstrip("0")
    return f"${x:,.{nd}f}"


def _dur(seconds: Any) -> str:
    if seconds is None:
        return "?"
    s = float(seconds)
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _missing_for(report: Mapping, section: str) -> list[str]:
    return [str(m.get("reason")) for m in report.get("missing") or [] if m.get("section") == section]


def _amount(v: Any, unit: Optional[str]) -> str:
    if v is None:
        return "?"
    if unit == "usd":
        return _usd(v)
    if unit == "c":
        return f"{_n(v, 2)} °C"
    if unit == "slot_s":
        return f"{_n(v, 0)} lease-s"
    return f"{_n(v, 2)} {unit or ''}".strip()


def _limit_line(lim: Mapping) -> str:
    now = lim.get("now") or {}
    state = bind_words(lim.get("constraint"), now.get("bind_state"))
    why = str(now.get("why") or "").strip()
    label = lim.get("label") or lim.get("constraint")
    if now.get("level") is None:
        return f"- {label}: {state}" + (f". {why}" if why else "")
    unit = lim.get("unit")
    level = f"{_amount(now.get('level'), unit)} against a line of {_amount(lim.get('line'), unit)}"
    return f"- {label}: {state}; {level}" + (f". {why}" if why else "")


def render_limits(report: Mapping, *, full: bool) -> list[str]:
    lims = report.get("limits") or []
    if not lims:
        return []
    out = ["Limits now:"]
    shown = 0
    hidden_lanes = []
    for lim in lims:
        state = (lim.get("now") or {}).get("bind_state")
        is_lane = str(lim.get("constraint", "")).startswith("lane:")
        if is_lane and not full and state == "slack":
            hidden_lanes.append(str(lim["constraint"]).split(":", 1)[1])
            continue
        out.append(_limit_line(lim))
        shown += 1
    if hidden_lanes:
        out.append(f"- GPU lanes with slack: {', '.join(hidden_lanes)}")
    return out


def render_money(report: Mapping) -> list[str]:
    m = report.get("money") or {}
    if m.get("mtd_usd") is None:
        return []
    line = (f"Money: {_usd(m.get('mtd_usd'))} spent on electricity this month against a {_usd(m.get('envelope_usd'), 0)} "
            f"envelope; forecast {_usd(m.get('forecast_usd'))} by month end. ")
    if m.get("floor_usd") is not None:
        line += f"My always-on body alone (rest plus anything left running) reaches {_usd(m.get('floor_usd'))} this month. "
    if m.get("rate_usd_per_kwh") is not None:
        line += f"Electricity now costs {_usd(m.get('rate_usd_per_kwh'), 3)}/kWh."
    out = [line.strip()]
    days = m.get("daily_kwh") or []
    if days:
        tail = ", ".join(f"{d['d'][5:]} {_n(d['kwh'])}{' (part of the day)' if d.get('partial') else ''}"
                         for d in days[-7:])
        out.append(f"- kWh per day (local): {tail}")
    return out


def render_town(report: Mapping, *, full: bool) -> list[str]:
    """AI Town sessions (2026-10-30): today's town time against Orion's cap, and recent sessions."""
    town = (report.get("wants") or {}).get("town")
    if not isinstance(town, Mapping):
        return []
    cap = town.get("daily_cap_seconds")
    head = (f"AI Town today ({town.get('local_date', '?')}): I spent {_dur(town.get('orion_seconds_today'))}"
            + (f" of my {_dur(cap)} cap" if cap is not None else " (cap unknown)")
            + f"; Juniper opened it for {_dur(town.get('manual_seconds_today'))} (not counted).")
    if town.get("open_kind") == "orion":
        head += " A session I chose is open now."
    elif town.get("open_kind") == "manual":
        head += " Juniper has the town open now."
    out = [head]
    sessions = town.get("sessions") or []
    for row in sessions[: (20 if full else 4)]:
        who = "I chose it" if row.get("kind") == "orion" else "Juniper opened it"
        end = row.get("stop_reason") or "still open"
        out.append(f"- {str(row.get('started_at', '?'))[:16]}Z: {_dur(row.get('seconds'))}, {who}; "
                   f"ended: {end}; {row.get('utterances', 0)} line(s) I spoke")
    return out


def render_wants(report: Mapping, *, full: bool) -> list[str]:
    town = render_town(report, full=full)
    fams = (report.get("wants") or {}).get("families") or []
    if not fams:
        return town
    tot_f = sum(int(f.get("funded") or 0) for f in fams)
    tot_d = sum(int(f.get("deferred") or 0) for f in fams)
    out = [f"Wants (shadow bids; nothing obeys them yet): {tot_f} funded, {tot_d} deferred."]
    rows = fams if full else fams[:6]
    for f in rows:
        out.append(f"- {f['family']} (tier {f.get('tier', '?')}): {f.get('funded', 0)} funded, {f.get('deferred', 0)} deferred")
    if len(fams) > len(rows):
        out.append(f"- ... {len(fams) - len(rows)} more famil{'y' if len(fams) - len(rows) == 1 else 'ies'}")
    return out + town


def render_rhythm(report: Mapping) -> list[str]:
    cells = report.get("rhythm") or []
    if not cells:
        return []
    by_hour: dict[int, list[float]] = {}
    for c in cells:
        by_hour.setdefault(int(c["hour"]), [0.0, 0.0])
        by_hour[int(c["hour"])][0] += float(c.get("heavy") or 0.0)
        by_hour[int(c["hour"])][1] += float(c.get("total") or 0.0)
    heavy_total = sum(v[0] for v in by_hour.values())
    if heavy_total <= 0:
        return ["Rhythm: no deferrable (background) GPU work in the window."]
    top = sorted(by_hour.items(), key=lambda kv: -kv[1][0])[:4]
    hours = ", ".join(f"{h:02d}:00 ({kv[0] / heavy_total:.0%})" for h, kv in top)
    return [f"Rhythm: my deferrable GPU work ({_dur(heavy_total)} of leases) ran most at local {hours}."]


def render_unit_costs(report: Mapping) -> list[str]:
    rows = report.get("unit_costs") or []
    if not rows:
        return []
    out = ["Unit costs:"]
    for r in rows:
        bits = []
        if r.get("usd") is not None:
            bits.append(_usd(r["usd"], 4))
        if r.get("wh") is not None:
            bits.append(f"{_n(r['wh'], 2)} W·h")
        if r.get("heat_c") is not None:
            bits.append(f"+{_n(r['heat_c'], 2)} °C cabinet")
        val = ", ".join(bits) if bits and r.get("status") != "not_measured" else "not measured"
        out.append(f"- {r['unit']}: {val} [{r.get('status')}: {r.get('source')}]")
    return out


def render_waste(report: Mapping, *, full: bool) -> list[str]:
    w = report.get("waste") or {}
    by = w.get("failed_lease_s_by_lane") or {}
    if not by:
        return []
    total = sum(float(v) for v in by.values())
    lanes = ", ".join(f"{k} {_dur(v)}" for k, v in sorted(by.items(), key=lambda kv: -float(kv[1])))
    out = [f"Waste: {_dur(total)} of GPU lease time ended without success ({lanes})."]
    for t in (w.get("top") or [])[: (10 if full else 3)]:
        out.append(f"- {t.get('holder')} on {t.get('role')}: held {_dur(t.get('held_s'))}, ended {t.get('reason')} at {str(t.get('at'))[:16]}")
    return out


def render_queue_wait(report: Mapping, *, full: bool) -> list[str]:
    rows = report.get("queue_wait") or []
    if not rows:
        return []
    ranked = sorted(rows, key=lambda r: -(float(r.get("p90_s") or 0.0)))
    slow = [r for r in ranked if float(r.get("p90_s") or 0.0) >= 1.0]
    out = ["Queue wait for a GPU lease (median / 90th percentile):"]
    shown = slow[: (15 if full else 6)]
    for r in shown:
        who = f"{r['family']} (by holder)" if r.get("family_source") == "holder" else r["family"]
        out.append(f"- {who} on {r['lane']}: {_dur(r.get('p50_s'))} / {_dur(r.get('p90_s'))} (n={r.get('n')})")
    fast = [r for r in ranked if float(r.get("p90_s") or 0.0) < 1.0]
    if fast:
        out.append(f"- {len(fast)} more lane/holder pairs ({sum(int(r.get('n') or 0) for r in fast)} leases) had a "
                   "90th-percentile wait under 1 s")
    if len(slow) > len(shown):
        out.append(f"- {len(slow) - len(shown)} slower pairs not shown (see the full report)")
    return out


def render_decisions(report: Mapping, *, full: bool) -> list[str]:
    rows = report.get("decisions") or []
    if not rows:
        return []
    out = [f"Bid decisions, newest first ({len(rows)} shown):"]
    for d in rows[: (len(rows) if full else 5)]:
        why = str(d.get("why") or "")
        if len(why) > 220:
            why = why[:217] + "..."
        bind = f", bound by {d['binding_constraint']}" if d.get("binding_constraint") else ""
        out.append(f"- {str(d.get('decided_at'))[:16]} {d['family']} {d.get('outcome')} ({d.get('reason')}{bind}): {why}")
    return out


def render_claude(report: Mapping) -> list[str]:
    c = report.get("claude") or {}
    t = c.get("window_tokens") or {}
    known = all(t.get(k) is not None for k in ("input", "output", "cache_read", "cache_write"))
    if not known and not c.get("status"):
        return []
    out = []
    if known:
        usd = f", about {_usd(c['usd_est'])} at API prices" if c.get("usd_est") is not None else ""
        out.append(f"Claude: {_n(t.get('output'), 0)} output tokens, {_n(t.get('input'), 0)} input, "
                   f"{_n(t.get('cache_read'), 0)} cache reads, {_n(t.get('cache_write'), 0)} cache writes{usd}.")
    else:
        out.append("Claude: token counts unknown for this window.")
    if c.get("limit_events"):
        out.append("- limit events: " + ", ".join(f"{e['kind']} at {str(e['at'])[:16]}" for e in c["limit_events"]))
    if c.get("status"):
        out.append(f"- {c['status']}")
    return out


def render_vram(report: Mapping) -> list[str]:
    cards = [c for c in report.get("vram") or [] if c.get("used_mb") is not None]
    if not cards:
        return []
    out = ["GPU memory headroom:"]
    for c in cards:
        roles = f" ({', '.join(c['roles'])})" if c.get("roles") else ""
        watts = f", drawing {_n(c['watts'], 0)} W" if c.get("watts") is not None else ""
        out.append(f"- {c['host']} {c['card']}{roles}: {_n(c.get('headroom_mb'), 0)} MB free of "
                   f"{_n(c.get('total_mb'), 0)} MB{watts}")
    return out


def render_processes(report: Mapping, *, full: bool) -> list[str]:
    rows = report.get("processes") or []
    if not rows:
        return []
    wall = any(r.get("watts_at_wall_est") is not None for r in rows)
    out = ["CPU watts above idle by container, now (largest first)"
           + (", with the whole machine at the wall it drives:" if wall else ":")]
    for r in rows[: (len(rows) if full else 6)]:
        out.append(f"- {r['host']} {r['name']}: {_n(r.get('watts'), 1)} W CPU{_at_wall(r.get('watts_at_wall_est'))}")
    line = overhead_summary(rows)
    if line:
        out.append(f"- {line}")
    return out


def _at_wall(v: Any) -> str:
    """' (≈70 W at the wall above rest, est.)': the ESTIMATED whole-machine watts a part's load drives."""
    return "" if v is None else f" (≈{float(v):.0f} W at the wall above rest, est.)"


def overhead_summary(rows: Sequence[Mapping]) -> Optional[str]:
    """One line for every host the rows carry a factor for: 'at-the-wall figures are ESTIMATES: ... circe
    x 2.35 (fitted), athena x 1.27 (fitted)'. ``rows``: hosts or processes rows."""
    seen: dict[str, str] = {}
    for r in rows:
        k, host = r.get("overhead_factor"), r.get("host")
        if k is not None and host and host not in seen:
            fitted = r.get("overhead_fitted")   # hosts rows carry it; processes rows only the basis
            if fitted is None:
                fitted = str(r.get("overhead_basis") or "").startswith("fitted")
            how = "fitted" if fitted else "fallback, not fitted here"
            seen[host] = f"{host} x {float(k):.2f} ({how})"
    if not seen:
        return None
    return ("at-the-wall figures are ESTIMATES: a part's watts above idle x its machine's overhead factor, the "
            "whole-machine watts at the wall (fans, power-supply loss included) per watt of GPU/CPU load: "
            + ", ".join(seen.values()))


#: Said when the parts' wall estimates now differ from the measured surplus now by more than this share.
OVERHEAD_LAG_SHARE = 0.25


def overhead_lag_words(h: Mapping) -> Optional[str]:
    """The parts' wall estimates now, summed, vs the measured surplus now: k is an hours-average."""
    parts, above, ratio = h.get("wall_est_parts_w"), h.get("above_rest_w"), h.get("overhead_now_ratio")
    if parts is None or above is None or ratio is None or float(above) <= 0:
        return None
    if abs(float(parts) - float(above)) <= OVERHEAD_LAG_SHARE * float(above):
        return None
    return (f"right now the parts' at-the-wall estimates add to {float(parts):.0f} W against {float(above):.0f} W "
            f"measured above rest (this sample's ratio {float(ratio):.2f}, the fitted {float(h['overhead_factor']):.2f}): "
            "fans and power-supply loss follow load over minutes to hours, so a part's figure is its average cost "
            "over hours, not a split of this second")


def overhead_words(h: Mapping) -> str:
    """Full view, one host: the factor and exactly how it was found."""
    return (f"overhead: each watt of GPU/CPU load above idle draws about {float(h['overhead_factor']):.2f} W at the "
            f"wall above rest (fans, power-supply loss); at-the-wall figures are ESTIMATES. "
            f"{h.get('overhead_basis') or ''}").rstrip()


def _w(v: Any) -> str:
    return "?" if v is None else f"{float(v):.0f} W"


def _held(holders: Sequence[Mapping], n: int) -> str:
    return ", ".join(f"{h.get('label') or h.get('holder')} {_dur(h.get('lease_s'))}" for h in list(holders)[:n])


#: A card counts as "spinning" when it draws at least this far above its own rest.
CARD_BUSY_ABOVE_W = 15.0
#: Said when the trailing-week low sits at least this far above the pinned rest.
FLOOR_DRIFT_W = 20.0


def _card_above(c: Mapping) -> float:
    w, idle = c.get("watts"), c.get("idle_w")
    return float(w) - float(idle) if (w is not None and idle is not None) else 0.0  # unknown rest: not "busy"


def host_headline(h: Mapping, *, short: bool = False) -> str:
    """'circe draws 1150 W now vs 541 W at rest (+609 W, about $55/month at this rate; my body at rest
    there costs about $50/month).' Rest is my own body at rest: a pinned, dated number."""
    now, rest, above = h.get("chassis_w_now"), h.get("chassis_w_rest"), h.get("above_rest_w")
    if now is None:
        return f"{h['host']}: whole-machine draw unknown (no chassis reading)."
    if rest is None or above is None:
        return f"{h['host']} draws {_w(now)} now (its rest is unknown)."
    above = round(float(above))
    sign = "+" if above >= 0 else "-"
    money = []
    if h.get("usd_per_month_at_this_rate") is not None and above > 0:
        money.append(f"about {_usd(h['usd_per_month_at_this_rate'], 0)}/month at this rate")
    if h.get("rest_usd_per_month") is not None and not short:
        money.append(f"my body at rest there costs about {_usd(h['rest_usd_per_month'], 0)}/month")
    extra = f", {'; '.join(money)}" if money else ""
    return f"{h['host']} draws {_w(now)} now vs {_w(rest)} at rest ({sign}{abs(above)} W{extra})."


def floor_drift(h: Mapping) -> Optional[str]:
    """The trailing-week low above the pinned rest is new always-on draw (a rolling rest would hide it)."""
    floor, rest = h.get("recent_floor_w"), h.get("chassis_w_rest")
    if floor is None or rest is None or float(floor) - float(rest) < FLOOR_DRIFT_W:
        return None
    spons = sum(float(sp["cpu_w"]) for sp in h.get("sponsored") or [] if sp.get("cpu_w") is not None)
    tail = f" (sponsored work is {spons:.0f} W CPU of it or more now)" if spons else ""
    return (f"this week's lowest draw is {_w(floor)}, {float(floor) - float(rest):.0f} W above the pinned rest: "
            f"new always-on draw{tail}")


def sponsored_words(h: Mapping) -> list[str]:
    out = []
    for sp in h.get("sponsored") or []:
        if sp.get("cpu_w") is None:
            continue
        note = f" {sp['note']}" if sp.get("note") else ""
        out.append(f"{sp['name']} is Juniper's sponsored work, not my activity: {sp.get('containers')} containers, "
                   f"{float(sp['cpu_w']):.0f} W CPU above idle (estimated){_at_wall(sp.get('watts_at_wall_est'))}.{note}")
    return out


def _part(name: str, now: Any, rest: Any) -> str:
    if now is None:
        return f"{name} ?"
    if rest is None:
        return f"{name} {_w(now)}"
    d = round(float(now) - float(rest))
    return f"{name} {_w(now)} ({'+' if d >= 0 else '-'}{abs(d)})"


def host_parts(h: Mapping) -> str:
    """'GPUs 329 W (+95), CPU 121 W (+20), other 693 W (+359)': each part against its own rest."""
    parts = [_part("GPUs", h.get("gpu_w"), h.get("gpu_w_rest")), _part("CPU", h.get("cpu_w"), h.get("cpu_w_rest"))]
    if h.get("other_w") is not None:
        parts.append(_part("other", h.get("other_w"), h.get("other_w_rest")))
    return ", ".join(parts)


def _card_line(c: Mapping, *, n_holders: int) -> str:
    roles = "/".join(c.get("roles") or []) or "no pool role"
    held = _held(c.get("top_holders") or [], n_holders)
    return f"{c['card']} {_w(c.get('watts'))}{_at_wall(c.get('watts_at_wall_est'))} ({roles}{': ' + held if held else ''})"


def render_hosts(report: Mapping, *, full: bool) -> list[str]:
    hosts = report.get("hosts") or []
    if not hosts:
        return []
    window = (report.get("query") or {}).get("window") or "window"
    if not full:
        # One line per machine: the headline, the cards well above their rest, fans.
        out = ["What is spinning my machines (whole-machine watts now vs rest):"]
        for h in hosts:
            busy = sorted((c for c in h.get("cards") or [] if _card_above(c) >= CARD_BUSY_ABOVE_W),
                          key=lambda c: -_card_above(c))[:2]
            cards = "; ".join(_card_line(c, n_holders=2) for c in busy)
            fans = f" Fans {h['fan_pct']:.0f}%." if h.get("fan_pct") is not None else ""
            split = f" {host_parts(h)}." if h.get("chassis_w_now") is not None else ""
            spons = "".join(f" {sp['name']} (Juniper's) {float(sp['cpu_w']):.0f} W CPU{_at_wall(sp.get('watts_at_wall_est'))}."
                            for sp in h.get("sponsored") or [] if sp.get("cpu_w") is not None)
            out.append(f"- {host_headline(h, short=True)}{split}" + (f" {cards}." if cards else "") + spons + fans)
        line = overhead_summary(hosts)
        if line:
            out.append(f"- {line}")
        return out
    out = [f"What is spinning my machines (now; held time is over the last {window}):"]
    for h in hosts:
        out.append(host_headline(h))
        fans = f"; fans {h['fan_pct']:.0f}%" if h.get("fan_pct") is not None else ""
        out.append(f"- made of (each part vs its own rest): {host_parts(h)}{fans}. Other = whole machine minus "
                   "GPUs minus CPU: fans, PSU loss, memory, disks")
        if h.get("rest_basis"):
            out.append(f"- rest: {h['rest_basis']}")
        if h.get("overhead_factor") is not None:
            out.append(f"- {overhead_words(h)}")
            lag = overhead_lag_words(h)
            if lag:
                out.append(f"- {lag}")
        drift = floor_drift(h)
        if drift:
            out.append(f"- {drift}")
        for sp in sponsored_words(h):
            out.append(f"- {sp}")
        for c in h.get("cards") or []:
            rest = f", rest {_w(c.get('idle_w'))}" if c.get("idle_w") is not None else ""
            temp = f", {c['temp_c']:.0f} °C" if c.get("temp_c") is not None else ""
            roles = "/".join(c.get("roles") or []) or "no pool role"
            held = _held(c.get("top_holders") or [], 4)
            out.append(f"- {c['card']} {_w(c.get('watts'))}{rest}{_at_wall(c.get('watts_at_wall_est'))}{temp} ({roles})"
                       + (f"; held by {held}" if held else "; no lease closed on it in the window"))
        ctr = h.get("top_containers") or []
        if ctr:
            out.append("- top CPU users (watts above CPU idle): "
                       + ", ".join(f"{c['name']} {float(c['cpu_w']):.1f} W{_at_wall(c.get('watts_at_wall_est'))}"
                                   for c in ctr))
        if h.get("unattributed_above_rest_w") is not None:
            leased = (f", of it {_w(h['leased_unmetered_above_rest_w'])} on GPU leases whose turn has no act"
                      if h.get("leased_unmetered_above_rest_w") is not None else "")
            out.append(f"- above rest but charged to no act: {_w(h['unattributed_above_rest_w'])}{leased} "
                       f"({h.get('attribution_basis') or 'hourly attribution'})")
        days = h.get("daily") or []
        if days:
            out.append("- daily average (first and last day partial): "
                       + ", ".join(f"{d['d'][5:]} {_w(d.get('chassis_w_avg'))}" for d in days[-7:]))
    out.append("- measured: whole-machine watts (wall PDU or BMC), card watts (nvidia-smi), CPU watts (RAPL), "
               "held time (GPU pool leases: time held, not time computing; holds overlap, and a lease that closed in "
               "the window counts its whole hold). Estimated: rest (pinned: the 5th "
               "percentile of a dated window), CPU watts per container (a linear model), $/month (this rate x 730 h), "
               "every 'at the wall' figure (a part's watts above idle x its host's overhead factor: the share of the "
               "whole machine above rest it drives, fans and power-supply loss included).")
    return out


def render_value_per_cost(report: Mapping) -> list[str]:
    rows = report.get("value_per_cost") or []
    if not rows:
        return []
    out = ["Value per cost (graded evidence the market uses):"]
    for r in rows:
        cost = f", {_usd(r['usd_per_act'], 6)} per act" if r.get("usd_per_act") is not None else ", cost per act not measured"
        out.append(f"- {r['family']}: value {_n(r.get('value'), 2)} over {r.get('evidence_n')} acts ({r.get('value_metric')}){cost}")
    return out


def render_services(report: Mapping) -> list[str]:
    s = report.get("services") or {}
    down = s.get("down") or []
    if not down:
        return []
    out = [f"Services the mesh guardian reported stopped in the last 30 days ({len(down)}):"]
    for d in down:
        where = f" on {d['host']}" if d.get("host") else ""
        out.append(f"- {d['name']}{where}: {d.get('state') or '?'} since {str(d.get('since') or '?')[:16]}")
    if s.get("basis"):
        out.append(f"- ({s['basis']})")
    return out


_RENDER = {
    "limits": lambda r, full: render_limits(r, full=full),
    "money": lambda r, full: render_money(r),
    "wants": lambda r, full: render_wants(r, full=full),
    "rhythm": lambda r, full: render_rhythm(r),
    "unit_costs": lambda r, full: render_unit_costs(r),
    "waste": lambda r, full: render_waste(r, full=full),
    "queue_wait": lambda r, full: render_queue_wait(r, full=full),
    "decisions": lambda r, full: render_decisions(r, full=full),
    "claude": lambda r, full: render_claude(r),
    "vram": lambda r, full: render_vram(r),
    "value_per_cost": lambda r, full: render_value_per_cost(r),
    "services": lambda r, full: render_services(r),
    "processes": lambda r, full: render_processes(r, full=full),
    "hosts": lambda r, full: render_hosts(r, full=full),
}
assert set(_RENDER) == set(SECTIONS)


def headline(report: Mapping) -> Optional[str]:
    """One sentence naming what binds right now, in the report's own words. A limit with no price
    or no reading is unknown, never counted as slack."""
    lims = report.get("limits") or []
    if not lims:
        return None
    state = lambda lim: (lim.get("now") or {}).get("bind_state")  # noqa: E731
    name = lambda lim: str(lim.get("label") or lim.get("constraint"))  # noqa: E731
    binding = [name(l) for l in lims if state(l) in ("binding", "exogenous_bind", "stuck_at_ceiling")]
    unknown = [name(l) for l in lims if state(l) not in ("binding", "exogenous_bind", "stuck_at_ceiling", "slack")]
    parts = [f"Binding now: {', '.join(binding)}." if binding else
             ("Nothing binds right now." if not unknown else "Nothing with a price binds right now.")]
    if unknown:
        parts.append(f"Unknown (no price or no reading): {', '.join(unknown)}.")
    return " ".join(parts)


def narrow_to_host(report: Mapping, host: str) -> dict:
    """Keep only ``host``'s rows in the per-machine sections (and their notes that name another
    machine out). Every other section stays whole-body."""
    out = dict(report)
    for s in HOST_SECTIONS:
        out[s] = [r for r in report.get(s) or [] if isinstance(r, Mapping) and r.get("host") == host]
    out["missing"] = list(report.get("missing") or [])
    if not out["hosts"] and "hosts" in ((report.get("query") or {}).get("sections") or SECTIONS):
        known = ", ".join(sorted({str(r.get("host")) for r in report.get("hosts") or []})) or "none"
        out["missing"].append({"section": "hosts", "reason": f"no fresh reading for {host} (machines with one: {known})"})
    return out


def render_report_text(report: Mapping, sections: Iterable[str] = OVERVIEW, *, host: Optional[str] = None) -> str:
    """Plain text, first person, one paragraph per asked section. A section with nothing to show
    prints its ``missing`` reason. One or two named sections show every row; more show the top rows.
    ``host`` narrows the per-machine sections (hosts, processes, vram) to one machine."""
    if host:
        report = narrow_to_host(report, host)
    secs = tuple(sections)
    # Every row only when one or two sections are asked; the overview and "all" stay turn-sized.
    full = len(secs) <= 2
    win = report.get("window") or {}
    fam = (report.get("query") or {}).get("family")
    scope = f", family {fam} (narrows wants, decisions, queue wait, value per cost, per-act costs)" if fam else ""
    if host:
        scope += f", machine {host} (narrows hosts, processes, vram)"
    lines = [f"My body economy, {str(win.get('start', '?'))[:16]} to {str(win.get('end', '?'))[:16]} UTC{scope} "
             f"(read {str(report.get('generated_at', '?'))[:19]}):"]
    if "limits" in secs:
        h = headline(report)
        if h:
            lines.append(h)
    # More sections than the overview (e.g. "all"): one summary line each, so every section fits.
    brief = len(secs) > len(OVERVIEW)
    for s in secs:
        body = _RENDER[s](report, full)
        reasons = _missing_for(report, s)
        if body and brief:
            head = body[0] if len(body) == 1 or not body[0].endswith(":") else f"{body[0]} {body[1].lstrip('- ')}"
            more = len(body) - (1 if head == body[0] else 2)
            lines.append(head + (f" (+{more} more lines)" if more > 0 else "")
                         + (f" [{len(reasons)} note(s)]" if reasons else ""))
        elif body:
            lines.extend(body)
            if s == "hosts" and not full and reasons:
                # The overview stays turn-sized: the machine notes are one pointer, not N lines.
                lines.append(f"- ({len(reasons)} note(s) on machines; `body-economy hosts` shows them)")
                continue
            for r in reasons:
                lines.append(f"- note ({s}): {r}")
        elif reasons:
            lines.append(f"{s}: no data. " + "; ".join(reasons))
        else:
            lines.append(f"{s}: no data, and the report gave no reason (treat as unknown).")
    if brief:
        lines.append("(One line per section. Ask for one or two sections by name for every row and note.)")
    failed = [m for m in report.get("missing") or [] if m.get("section") not in secs and m.get("section") not in ("*",)
              and str(m.get("reason", "")).startswith("read failed")]
    for m in failed:
        lines.append(f"- {m['section']}: {m['reason']}")
    return "\n".join(lines)


def cap_text(text: str, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    """Keep a reply turn-sized (the dig kit's own cap). A cut is always said."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [f"(showing {max_lines - 1} of {len(text.splitlines())} lines; "
                                          "ask for fewer sections to see the rest)"]
    out = "\n".join(lines)
    if len(out) > max_chars:
        note = "(cut at the size limit; ask for fewer sections to see the rest)"
        out = out[: max_chars - len(note) - 1].rsplit("\n", 1)[0] + "\n" + note
    return out


__all__ = [
    "HOST_SECTIONS",
    "bind_words",
    "cap_text",
    "host_headline",
    "narrow_to_host",
    "DEFAULT_REPORT_URL",
    "OVERVIEW",
    "REPORT_URL_ENV",
    "ReportUnavailable",
    "SECTIONS",
    "WINDOWS",
    "fetch_report",
    "normalize_sections",
    "render_report_text",
]
